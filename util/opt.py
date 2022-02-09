import logging

from ase import Atoms

from wfl.generate_configs import minim
from wfl.pipeline.base import iterable_loop


logger = logging.getLogger(__name__)


def optimise(inputs, outputs, calculator, prop_prefix,  chunksize=1,
             traj_step_interval=None,npool=None):
    return iterable_loop(iterable=inputs, configset_out=outputs,
                         calculator=calculator, op=optimise_op,
                         chunksize=chunksize,
                         traj_step_interval=traj_step_interval,
                         prop_prefix=prop_prefix, npool=npool)


def optimise_op(atoms, calculator, prop_prefix, traj_step_interval=None):
    """traj_step_interval: if None, only the last converged config will be
    taken. Otherwise take all that get sampled. + the last

    """

    opt_kwargs = {'logfile': None, 'master': True, 'precon': None,
                  'use_armijo': False, 'steps':500}

    if traj_step_interval is None:
        opt_kwargs["traj_subselect"] = "last_converged"
    if traj_step_interval is not None:
        opt_kwargs['traj_step_interval'] = traj_step_interval

    all_trajs = minim.run_op(atoms=atoms, calculator=calculator,
                             keep_symmetry=False, update_config_type=False,
                             results_prefix=prop_prefix,
                             fmax=1e-2, **opt_kwargs)

    return all_trajs


