import logging

from ase import Atoms

from wfl.generate_configs import minim
from wfl.pipeline import iterable_loop


logger = logging.getLogger(__name__)


def optimise(inputs, outputs, calculator, chunksize=10):
    return iterable_loop(iterable=inputs, configset_out=outputs,
                         calculator=calculator, op=optimise_op,
                         chunksize=chunksize)


def optimise_op(atoms, calculator):

    opt_kwargs = {'logfile': None, 'master': True, 'precon': None,
                  'use_armijo': False}

    all_trajs = minim.run_op(atoms=atoms, calculator=calculator,
                             keep_symmetry=False, update_config_type=False,
                             fmax=1e-2, **opt_kwargs)

    ats_out = []
    for traj in all_trajs:
        last_at = traj[-1]
        assert isinstance(last_at, Atoms)
        if last_at.info["minim_config_type"] == 'minim_last_converged':
            logger.info(f'optimisaition converged after {last_at.info["minim_n_steps"]}')
            ats_out.append(last_at)
        else:
            logger.info(f'optimisation hasn\'t converged. atoms.info:'
                        f' {atoms.info}')

    return ats_out

