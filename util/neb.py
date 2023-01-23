import sys
import warnings
from copy import deepcopy
from ase import Atoms
from ase.neb import NEB
from ase.optimize.precon import PreconLBFGS
from ase.io import read, write

from wfl.utils.at_copy_save_results import at_copy_save_results
from wfl.utils.parallel import construct_calculator_picklesafe



def neb_ll(inputs, calculator, num_images, results_prefix="neb_", traj_step_interval=1, 
           fmax=1.0e-3, steps=1000, verbose=False, skip_failures=True, **opt_kwargs):
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

    return neb_ll(_run_autopara_wrappable, *args, 
        def_autopara_info=def_autopara_info, **kwargs)
autoparallelize_docstring(run, _run_autopara_wrappable, "Atoms")




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


