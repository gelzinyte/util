import logging

from ase import Atoms

from wfl.generate_configs import minim
from wfl.pipeline.base import iterable_loop


logger = logging.getLogger(__name__)


def optimise(inputs, outputs, calculator, chunksize=1,
             traj_step_interval=None):
    return iterable_loop(iterable=inputs, configset_out=outputs,
                         calculator=calculator, op=optimise_op,
                         chunksize=chunksize,
                         traj_step_interval=traj_step_interval)


def optimise_op(atoms, calculator, traj_step_interval=None):
    """traj_step_interval: if None, only the last converged config will be
    taken. Otherwise take all that get sampled. + the last

    """

    opt_kwargs = {'logfile': None, 'master': True, 'precon': None,
                  'use_armijo': False, 'steps':500}
    if traj_step_interval is not None:
        opt_kwargs['traj_step_interval']:traj_step_interval

    all_trajs = minim.run_op(atoms=atoms, calculator=calculator,
                             keep_symmetry=False, update_config_type=False,
                             fmax=1e-2, **opt_kwargs)

    if traj_step_interval is None:
        ats_out = []
        for traj in all_trajs:
            last_at = traj[-1]
            assert isinstance(last_at, Atoms)
            if last_at.info["minim_config_type"] == 'minim_last_converged':
                logger.info(f'optimisaition converged after {last_at.info["minim_n_steps"]}')
                ats_out.append(last_at)
            else:
                logger.info(f'optimisation hasn\'t converged. atoms.info:'
                            f' {last_at.info}')

        return ats_out
    else:
        return all_trajs

