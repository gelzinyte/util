import os
import sys
import warnings
from copy import deepcopy
import numpy as np

from ase import Atoms
from ase.neb import NEB
from ase.optimize.precon import PreconLBFGS
from ase.io import read, write

from wfl.utils.at_copy_save_results import at_copy_save_results
from wfl.utils.parallel import construct_calculator_picklesafe
from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators import generic
from wfl.autoparallelize.autoparainfo import AutoparaInfo

from ase.constraints import FixBondLength
# from ase.optimize.precon import PreconLBFGS
from ase.optimize import FIRE, LBFGS
# from ase.neb import NEB
from ase import units
from ase.build import molecule

import util
from util.configs import check_geometry

def neb_ll(inputs, calculator, num_images=None, results_prefix="neb_", traj_step_interval=1, 
           fmax=1.0e-3, steps=1000, parallel=False, verbose=False, traj_subselect=None, skip_failures=True,  neb_precon=None, neb_method="aseneb", spring_const=0.1, climb=False, optimiser="PreconLBFGS", **opt_kwargs):
    """
    inputs - (Atoms_initial, Atoms_end) or list of (Atoms, Atoms) for first and last images.  
    or list of list of neb configs
    
    """


    if isinstance(calculator, tuple) and calculator[0] == "MACE":
        from mace.calculators.mace  import MACECalculator
        calculator = (MACECalculator, calculator[1], calculator[2])

    calculator = construct_calculator_picklesafe(calculator)    

    opt_kwargs_to_use = dict(logfile=None, master=True)
    opt_kwargs_to_use.update(opt_kwargs)

    if opt_kwargs_to_use.get('logfile') is None and verbose:
        opt_kwargs_to_use['logfile'] = '-'

    # import pdb; pdb.set_trace()

    if isinstance(inputs[0], Atoms):
        inputs = [inputs]
    
    all_nebs = []
    for idx, neb_configs in enumerate(inputs):
        if verbose== True:
            print(f"neb no: {idx}, num_configs: {len(neb_configs)}")
        
        interpolate=False
        if len(neb_configs) == 2:
            start = neb_configs[0]
            end = neb_configs[1]
            neb_configs = [start.copy() for _ in range(num_images - 1)] + [end]
            interpolate=True
        num_images = len(neb_configs)

        for idx, config in enumerate(neb_configs):
            config.info["neb_image_no"] = str(idx)
            config.info["opt_step"] = 0


        for at in neb_configs:
            at.calc = deepcopy(calculator)
            at.info["neb_optimize_config_type"] = "optimize_mid"

        neb = NEB(neb_configs, parallel=parallel, k=spring_const, climb=climb)

        if interpolate:
            neb.interpolate()

        if optimiser=="PreconLBFGS":
            opt = PreconLBFGS(neb, **opt_kwargs_to_use)
        elif optimiser == "LBFGS":
            opt = LBFGS(neb, **opt_kwargs_to_use)
        elif optimiser=="FIRE":
            opt = FIRE(neb, **opt_kwargs_to_use)

        traj = []
        cur_step = 0

        def process_step(interval):
            nonlocal cur_step

            # print(f"cur_step: {cur_step}")

            if cur_step % interval == 0 or cur_step == steps+1:
                # print('get config')
                all_mace_var = [] 
                for at in neb_configs:
                    new_config = at_copy_save_results(at, results_prefix=results_prefix)
                    new_config.info["opt_step"] = cur_step  
                    traj.append(new_config)
                    # if "mace_energy_var" not in new_config.info:
                        # import pdb; pdb.set_trace()
                    if "mace" in results_prefix:
                        all_mace_var.append(new_config.info[f"{results_prefix}energy_var"])
                if len(all_mace_var) > 0:
                    if np.max(all_mace_var) > 1e-1:
                        # import pdb; pdb.set_trace() 
                        raise RuntimeError("Too large of a variance, stopping nebs.")               
            
            cur_step += 1

        opt.attach(process_step, 1, traj_step_interval)

        # preliminary value
        final_status = 'unconverged'

        try:
            # print("optimize!!")
            opt.run(fmax=fmax, steps=steps)
        except Exception as exc:
            # label actual failed optimizations
            # when this happens, the atomic config somehow ends up with a 6-vector stress, which can't be
            # read by xyz reader.
            # that should probably never happen
            final_status = 'exception'
            if skip_failures:
                sys.stderr.write(f'Structure optimization failed with exception \'{exc}\'\n')
                sys.stderr.flush()
            else:
                raise

        # set for first config, to be overwritten if it's also last config
        for idx in range(num_images):
            traj[idx].info['neb_optimize_config_type'] = 'optimize_initial'

        if opt.converged():
            final_status = 'converged'

        for aaa in traj[-num_images:]:
            aaa.info['neb_optimize_config_type'] = f'neb_optimize_last_{final_status}'
            aaa.info['neb_optimize_n_steps'] = opt.get_number_of_steps()


        # Note that if resampling doesn't include original last config, later
        # steps won't be able to identify those configs as the (perhaps unconverged) minima.
        # Perhaps status should be set after resampling?
        traj = subselect_from_traj(traj, subselect=traj_subselect, num_images=num_images)

        all_nebs.append(traj)

    return all_nebs 


def run_neb(*args, **kwargs):
    # Normally each thread needs to call np.random.seed so that it will generate a different
    # set of random numbers.  This env var overrides that to produce deterministic output,
    # for purposes like testing
    def_autopara_info={ "num_inputs_per_python_subprocess":1}

    return autoparallelize(neb_ll, *args, 
        default_autopara_info=def_autopara_info, **kwargs)
# autoparallelize_docstring(run_neb, neb_ll, "Atoms")




# Just a placeholder for now. Could perhaps include:
#    equispaced in energy
#    equispaced in Cartesian path length
#    equispaced in some other kind of distance (e.g. SOAP)
# also, should it also have max distance instead of number of samples?
def subselect_from_traj(traj, subselect=None, num_images=None):
    """Sub-selects configurations from trajectory.

    Parameters
    ----------
    subselect: int or string, default None

        - None: full trajectory is returned
        - int: (not implemented) how many samples to take from the trajectory.
        - str: specific method

          - "last_converged": returns [last_config] if converged, or None if not.

    """
    if subselect is None:
        return traj

    elif subselect == "last":
        return traj[-num_images:]


    elif subselect == "last_converged":
        converged_configs = [at for at in traj if at.info["optimize_config_type"] == "optimize_last_converged"]
        if len(converged_configs) == 0:
            warnings.warn("no converged configs have been found")
            return None
        else:
            return converged_configs

    raise RuntimeError(f'Subselecting confgs from trajectory with rule '
                       f'"subselect={subselect}" is not yet implemented')



def make_end_images(sub, H_idx, separation):
    '''Makes end images for H abstraction. Quite a specific case. '''

    methanol = molecule('CH3OH')
    # methanol.set_calculator(dftb)
    # opt = PreconLBFGS(methanol)
    # opt.run(fmax=1e-3)

    idx_shift = len(methanol)

    meth_O_idx = np.where(methanol.get_atomic_numbers()==8)[0][0]
    # meth_C_idx = np.where(methanol.get_atomic_numbers()==6)[0][0]
    meth_OH_H_idx = 3

    dists = sub.get_all_distances()[H_idx]
    sub_C_idx = np.argmin([d if d!= 0 else np.inf for d in dists])

    # OC = methanol.get_distance(meth_O_idx, meth_C_idx, vector=True)
    HO = methanol.get_distance(meth_OH_H_idx, meth_O_idx, vector=True)
    CH = sub.get_distance(sub_C_idx, H_idx, vector=True)

    # methanol.rotate(OC, CH, center=methanol.positions[1])
    methanol.rotate(HO, CH, center=methanol.positions[1])

    # methanol.positions -= methanol.positions[meth_O_idx]
    methanol.positions -= methanol.positions[meth_OH_H_idx]
    sub.positions -= sub.positions[sub_C_idx]

    at = methanol + sub

    sub = util.remove_energy_force_containing_entries(sub)

    for key, val in sub.info.items():
        at.info[key] = val

    at_mid = at.copy()

    unit_dir = CH / np.linalg.norm(CH)

    at_mid.positions[:idx_shift] += methanol.get_distance(meth_OH_H_idx, meth_O_idx) * unit_dir

    at.positions[:idx_shift] += separation * unit_dir

    dists = at.get_all_distances()[meth_O_idx]
    tmp_H_idx = np.argmin([d if d != 0 else np.inf for d in dists])
    tmp_H_pos = at.positions[tmp_H_idx]
    del at[tmp_H_idx]

    at_init = at.copy()
    at_final = at.copy()
    at_final.positions[H_idx + idx_shift - 1] = tmp_H_pos
    del at_mid[tmp_H_idx]

    return [at_init, at_mid, at_final]


def make_mid_image(orig_sub, H_idx):
    '''Makes end images for H abstraction. Quite a specific case. '''

    orig_methanol = molecule('CH3OH')

    idx_shift = len(orig_methanol)

    meth_O_idx = np.where(orig_methanol.get_atomic_numbers()==8)[0][0]
    meth_C_idx = np.where(orig_methanol.get_atomic_numbers()==6)[0][0]
    meth_OH_H_idx = 3

    dists = orig_sub.get_all_distances()[H_idx]
    sub_C_idx = np.argmin([d if d!= 0 else np.inf for d in dists])

    # OC = methanol.get_distance(meth_O_idx, meth_C_idx, vector=True)
    HO = orig_methanol.get_distance(meth_OH_H_idx, meth_O_idx, vector=True)
    CH = orig_sub.get_distance(sub_C_idx, H_idx, vector=True)

    # methanol.rotate(OC, CH, center=methanol.positions[1])
    orig_methanol.rotate(HO, CH, center=orig_methanol.positions[1])

    orig_sub = util.remove_energy_force_containing_entries(orig_sub)

    trials = []
    angles = np.arange(0, 360, 30)
    for angle in angles:

        sub = orig_sub.copy()

        methanol = orig_methanol.copy()
        methanol.rotate(a=angle, v=CH, center=orig_methanol.positions[1])
        methanol.info["angle"] = angle

        # methanol.positions -= methanol.positions[meth_O_idx]
        methanol.positions -= methanol.positions[meth_OH_H_idx]
        sub.positions -= sub.positions[sub_C_idx]

        at = methanol + sub

        for key, val in sub.info.items():
            at.info[key] = val

        at_mid = at.copy()

        unit_dir = CH / np.linalg.norm(CH)

        at_mid.positions[:idx_shift] += methanol.get_distance(meth_OH_H_idx, meth_O_idx) * unit_dir

        dists = at.get_all_distances()[meth_O_idx]
        tmp_H_idx = np.argmin([d if d != 0 else np.inf for d in dists])
        del at_mid[tmp_H_idx]

        at_mid.info["clash_dist"] = get_clash_dist(at_mid, idx_shift, meth_OH_H_idx, meth_O_idx, meth_C_idx, sub_C_idx)

        trials.append(at_mid)


    energies = [at.info["clash_dist"] for at in trials]
    idx = np.argmax(energies)
    # trials[idx].info["selected_as_min_energy"] = "selected"
    # return trials 
    return trials[idx]


def get_clash_dist(at_mid, idx_shft, meth_OH_H_idx, meth_O_idx, meth_C_idx, sub_C_idx):
    # finds the minimum istance between any of he atoms on the MeO
    # and any of the substrate atoms. 
    # Exclude the removed H. 

    # import pdb; pdb.set_trace()

    all_distances = at_mid.get_all_distances()
    # select only distances to the MeO atoms 
    all_meoh_distances = all_distances[:idx_shft] 
    # mask away all distances within the meoh wit a large number
    all_meoh_distances[:, :idx_shft] = 100
    # mask the remved H's, meo C andO distances with something large
    all_meoh_distances[[meth_OH_H_idx, meth_O_idx, meth_C_idx], :] = 100
    # mask distances toth sub_C
    all_meoh_distances[:, sub_C_idx] = 100

    min_dist =  np.min(all_meoh_distances)
    return min_dist




def make_ends_from_mid(at, separation):

    num_methanol_at = 6

    methanol_O_idx = 1
    dists = at.get_all_distances()[methanol_O_idx]
    methanol_OH_H_idx = np.argmin([d if d!=0 else np.inf for d in dists])

    dists = at.get_all_distances()[methanol_OH_H_idx] 
    masked_dists = []
    for dist, sym in zip(dists, at.symbols):
        if dist == 0 or sym != "C":
            masked_dists.append(np.inf)
        else:
            masked_dists.append(dist)

    closest_C_idx = np.argmin([d if d!=0 else np.inf for d in masked_dists])

    at.info['methanol_OH_H_idx'] = methanol_OH_H_idx
    at.info["Closest_C_idx"] = closest_C_idx
    
    CH = at.get_distance(closest_C_idx, methanol_OH_H_idx, vector=True)
    unit_dir = CH / np.linalg.norm(CH) 

    end = at.copy()
    end.positions[:num_methanol_at] += separation * unit_dir

    start = end.copy()
    start.positions[methanol_OH_H_idx] -= separation * unit_dir

    return [start, at, end]



def guess_init_neb_from_2(ats, num_copies=20):

    start = ats[0]
    end = ats[1]


    half_1 = [start.copy() for _ in range(num_copies)] + [end.copy()]

    neb_1 = NEB(half_1)
    neb_1.interpolate()

    return half_1
    # check geometries



def guess_init_neb_from_3(ats, num_copies=20):

    start = ats[0]
    mid = ats[1]
    end = ats[2]

    num_copies = int(num_copies/2)

    half_1 = [start.copy() for _ in range(num_copies)] + [mid.copy()]

    neb_1 = NEB(half_1)
    neb_1.interpolate()

    half_2 = [mid.copy() for _ in range(num_copies)] + [end.copy()]
    neb_2 = NEB(half_2)
    neb_2.interpolate()

    all_neb = half_1 + half_2[1:]
    return all_neb


def pick_two_ends(start, mid, end):


    # start = ats[0]
    # mid = ats[1]
    # end = ats[2]


    ch_c_idx = start.info["Closest_C_idx"]
    removed_h_idx = start.info["methanol_OH_H_idx"]
    methanol_O_idx = 1 

    start_ch_dist = start.get_distance(ch_c_idx, removed_h_idx)
    start_oh_dist = start.get_distance(methanol_O_idx, removed_h_idx)
    end_ch_dist = end.get_distance(ch_c_idx, removed_h_idx)
    end_oh_dist = end.get_distance(methanol_O_idx, removed_h_idx)
    mid_ch_dist = mid.get_distance(ch_c_idx, removed_h_idx)
    mid_oh_dist = mid.get_distance(methanol_O_idx, removed_h_idx)


    # check which CH or OH distance is shorter

    if mid_oh_dist < mid_ch_dist:
        # middle is methanol
        return "good", [start, mid]
    else:
        # middle is methoxy
        return "good", [mid, end]


    #--------
    # checks which distance changed more. Too complicated, unreliable 
    #------------
    # # how much going to/from mid the bond lenghts have changed
    # ch_diff_to_start = np.abs(start_ch_dist - mid_ch_dist)
    # ch_diff_to_end = np.abs(end_ch_dist - mid_ch_dist)
    # oh_dist_to_start = np.abs(start_oh_dist - mid_oh_dist)
    # oh_dist_to_end = np.abs(end_oh_dist - mid_oh_dist)

    # # mid more similar to start
    # if ch_diff_to_start < ch_diff_to_end:
    #     # oh should also be more similar to start
    #     if oh_dist_to_start > oh_dist_to_end:
    #         return "bad", ats
    #     return "good", [mid, end]
    # else:
    #     if oh_dist_to_start < oh_dist_to_end:
    #         return "bad", ats
    #     return "good", [start, mid]

