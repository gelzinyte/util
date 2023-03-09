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
# from ase.optimize import FIRE
# from ase.neb import NEB
from ase import units
from ase.build import molecule

import util

def neb_ll(inputs, calculator, num_images=None, results_prefix="neb_", traj_step_interval=1, 
           fmax=1.0e-3, steps=1000, parallel=False, verbose=False, traj_subselect=None, skip_failures=True, **opt_kwargs):
    """
    inputs - (Atoms_initial, Atoms_end) or list of (Atoms, Atoms) for first and last images.  
    or list of list of neb configs
    
    """

    calculator = construct_calculator_picklesafe(calculator)    

    opt_kwargs_to_use = dict(logfile=None, master=True)
    opt_kwargs_to_use.update(opt_kwargs)

    if opt_kwargs_to_use.get('logfile') is None and verbose:
        opt_kwargs_to_use['logfile'] = '-'

    if isinstance(inputs[0], Atoms):
        inputs = [inputs]
    
    all_nebs = []
    for neb_configs in inputs:
        
        interpolate=False
        if len(neb_configs) == 2:
            start = neb_configs[0]
            end = neb_configs[1]
            neb_configs = [start.copy() for _ in range(num_images - 1)] + [end]
            interpolate=True
        num_images = len(neb_configs)

        for idx, config in enumerate(neb_configs):
            configs.info["neb_image_no"] = str(idx)


        for at in neb_configs:
            at.calc = deepcopy(calculator)
            at.info["neb_optimize_config_type"] = "optimize_mid"

        neb = NEB(neb_configs, parallel=parallel)

        if interpolate:
            neb.interpolate()

        opt = PreconLBFGS(neb, **opt_kwargs_to_use)

        traj = []

        def process_step():
            for at in neb_configs:
                new_config = at_copy_save_results(at, results_prefix=results_prefix)
                traj.append(new_config)

        opt.attach(process_step, interval=traj_step_interval)

        # preliminary value
        final_status = 'unconverged'

        try:
            print("optimize!!")
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
            traj[idx].info['optimize_config_type'] = 'optimize_initial'

        if opt.converged():
            final_status = 'converged'

        for aaa in traj[-num_images:]:
            aaa.info['optimize_config_type'] = f'optimize_last_{final_status}'
            aaa.info['optimize_n_steps'] = opt.get_number_of_steps()


        # Note that if resampling doesn't include original last config, later
        # steps won't be able to identify those configs as the (perhaps unconverged) minima.
        # Perhaps status should be set after resampling?
        traj = subselect_from_traj(traj, subselect=traj_subselect)

        all_nebs.append(traj)

    return all_nebs 


def run_neb(*args, **kwargs):
    # Normally each thread needs to call np.random.seed so that it will generate a different
    # set of random numbers.  This env var overrides that to produce deterministic output,
    # for purposes like testing
    if 'WFL_DETERMINISTIC_HACK' in os.environ:
        initializer = (None, [])
    else:
        initializer = (np.random.seed, [])
    def_autopara_info={"initializer":initializer, "num_inputs_per_python_subprocess":1,
            "hash_ignore":["initializer"]}

    return autoparallelize(neb_ll, *args, 
        def_autopara_info=def_autopara_info, **kwargs)
# autoparallelize_docstring(run_neb, neb_ll, "Atoms")




# Just a placeholder for now. Could perhaps include:
#    equispaced in energy
#    equispaced in Cartesian path length
#    equispaced in some other kind of distance (e.g. SOAP)
# also, should it also have max distance instead of number of samples?
def subselect_from_traj(traj, subselect=None):
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


def make_mid_image(orig_sub, H_idx, calculator=None):
    '''Makes end images for H abstraction. Quite a specific case. '''

    orig_methanol = molecule('CH3OH')
    # methanol.set_calculator(dftb)
    # opt = PreconLBFGS(methanol)
    # opt.run(fmax=1e-3)

    idx_shift = len(orig_methanol)

    meth_O_idx = np.where(orig_methanol.get_atomic_numbers()==8)[0][0]
    # meth_C_idx = np.where(methanol.get_atomic_numbers()==6)[0][0]
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

        trials.append(at_mid)

    inputs = ConfigSet(trials)
    outputs = OutputSpec()
    inputs = generic.run(
        inputs = inputs,
        outputs=outputs,
        calculator=calculator,
        properties=["energy"],
        output_prefix="test_",
        autopara_info=AutoparaInfo(
            remote_info=None,
            num_inputs_per_python_subprocess=1,
        )
    )

    energies = [at.info["test_energy"] for at in inputs]
    idx = np.argmin(energies)
    inputs = list(inputs)
    # inputs[idx].info["selected_as_min_energy"] = "selected"
    # return inputs 
    return inputs[idx]


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




def run_old_neb(neb, fname, steps_fire, fmax_fire, steps_lbfgs, fmax_lbfgs):
    "Runs NEB with fire/lbfgs for steps/fmax on given NEB object, returns NEB object"

    opt_fire = FIRE(neb, trajectory=f'structure_files/{fname}.traj')
    opt_lbfgs = PreconLBFGS(neb, precon=None, use_armijo=False, \
                            trajectory=f'structure_files/{fname}.traj')

    print('\n----NEB\n')
    if steps_fire:
        opt_fire.run(fmax=fmax_fire, steps=steps_fire)

    if steps_lbfgs:
        opt_lbfgs.run(fmax=fmax_lbfgs, steps=steps_lbfgs)
    return neb


def prepare_do_neb(atoms, calc_name, no_images, fname, steps_fire, fmax_fire, steps_lbfgs, fmax_lbfgs):
    """Sets up and NEB given end images"""
    images = [atoms[0]]
    images += [atoms[0].copy() for _ in range(no_images - 2)]
    images += [atoms[1].copy()]

    neb = NEB(images)
    neb.interpolate()

    for i, image in enumerate(images):
        print(f'setting {i} image calculator')
        if calc_name == 'dftb':
            image.set_calculator(
                Potential(args_str='TB DFTB', param_filename='/home/eg475/reactions/tightbind.parms.DFTB.mio-0-1.xml'))
        else:
            image.set_calculator(Potential(param_filename=calc_name))

    neb = run_neb(neb, fname, steps_fire, fmax_fire, steps_lbfgs, fmax_lbfgs)
    return neb

