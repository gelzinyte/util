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



def neb_ll(inputs, calculator, num_images, results_prefix="neb_", traj_step_interval=1, 
           fmax=1.0e-3, steps=1000, verbose=False, traj_subselect=None, skip_failures=True, **opt_kwargs):
    """
    inputs - (Atoms_initial, Atoms_end) or list of (Atoms, Atoms) for first and last images.  
    
    """

    calculator = construct_calculator_picklesafe(calculator)    

    opt_kwargs_to_use = dict(logfile=None, master=True)
    opt_kwargs_to_use.update(opt_kwargs)

    if opt_kwargs_to_use.get('logfile') is None and verbose:
        opt_kwargs_to_use['logfile'] = '-'

    if len(inputs) == 2 and isinstance(inputs[0], Atoms):
        inputs = [inputs]
    
    all_nebs = []
    for pair in inputs:
        assert len(pair) == 2
        start = pair[0]
        end = pair[1]

        neb_configs = [start.copy() for _ in range(num_images - 1)] + [end]

        for at in neb_configs:
            at.calc = deepcopy(calculator)
            at.info["neb_optimize_config_type"] = "optimize_mid"

        traj = []

        neb = NEB(neb_configs)
        neb.interpolate()

        opt = PreconLBFGS(neb, **opt_kwargs_to_use)

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




import sys
import numpy as np
from quippy.potential import Potential

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io import read, write
from ase.md.verlet import VelocityVerlet
from ase.constraints import FixBondLength
# from ase.optimize.precon import PreconLBFGS
from ase.optimize import FIRE
from ase.neb import NEB
from ase import units
from ase.build import molecule


def make_end_images(sub, H_idx, separation):
    '''Makes end images for H abstraction. Quite a specific case. '''
    # dftb = Potential(args_str='TB DFTB', param_filename='/home/eg475/reactions/tightbind.parms.DFTB.mio-0-1.xml')

    # print('\n----Relaxing substrate and methanol\n')
    # sub.set_calculator(dftb)
    # opt = PreconLBFGS(sub)
    # opt.run(fmax=1e-3)

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

    unit_dir = CH / np.linalg.norm(CH)
    at.positions[:idx_shift] += separation * unit_dir

    dists = at.get_all_distances()[meth_O_idx]
    tmp_H_idx = np.argmin([d if d != 0 else np.inf for d in dists])
    tmp_H_pos = at.positions[tmp_H_idx]
    del at[tmp_H_idx]

    at_init = at.copy()
    at_final = at.copy()
    at_final.positions[H_idx + idx_shift - 1] = tmp_H_pos

    return at_init, at_final


def run_neb(neb, fname, steps_fire, fmax_fire, steps_lbfgs, fmax_lbfgs):
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

