import logging

from ase import Atoms

from wfl.generate import optimize
from wfl.autoparallelize.base import autoparallelize


logger = logging.getLogger(__name__)


# def optimise(inputs, outputs, calculator, output_prefix,  num_inputs_per_python_subprocess=1,
#              traj_step_interval=None,num_python_subprocesses=None, info_for_logfile=None, remote_info=None):
#     return autoparallelize(iterable=inputs, outputspec=outputs,
#                          calculator=calculator, op=optimise_autopara_wrappable,
#                          num_inputs_per_python_subprocess=num_inputs_per_python_subprocess,
#                          traj_step_interval=traj_step_interval,
#                          output_prefix=output_prefix, num_python_subprocesses=num_python_subprocesses, 
#                          info_for_logfile=info_for_logfile, remote_info=remote_info)


def optimise_autopara_wrappable(atoms, calculator, output_prefix, traj_step_interval=None, info_for_logfile=None, 
                                steps = 500, precon=None, skip_failures=True, verbose=False):
    """traj_step_interval: if None, only the last converged config will be
    taken. Otherwise take all that get sampled. + the last

    """

    if isinstance(atoms, Atoms): 
        atoms = [atoms]


    if info_for_logfile is not None:
        label = atoms[0].info[info_for_logfile]

        if len(atoms) == 1:
            logfile = label + ".opt_log"
        else:
            logger.warn(f"got `info_for_logfile`, but have more than one Atoms object")
            logfile = label + ".opt_log"
    else:
        logfile = None

    logfile = None
    label = None

    opt_kwargs = {'logfile': logfile, 'master': True, 'precon': precon,
                  'use_armijo': False, 'steps':steps}

    if traj_step_interval is None:
        opt_kwargs["traj_subselect"] = "last_converged"
    if traj_step_interval is not None:
        opt_kwargs['traj_step_interval'] = traj_step_interval

    all_trajs = optimize._run_autopara_wrappable(atoms=atoms, calculator=calculator,
                             keep_symmetry=False, update_config_type=False,
                             results_prefix=output_prefix,
                             fmax=1e-2, 
                             #traj_fn_prefix = label,
                             skip_failures=skip_failures,
                             verbose=verbose,
                             **opt_kwargs)

    return all_trajs


def optimise(*args, **kwargs):
    return autoparallelize(optimise_autopara_wrappable, *args, **kwargs)
