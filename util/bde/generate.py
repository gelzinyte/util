import os
import random
import logging

from os.path import join as pj

import numpy as np

from ase.io import read, write
from ase import Atoms

from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import generic
from wfl.calculators import orca
from wfl.generate_configs import minim

from util.util_config import Config

logger = logging.getLogger(__name__)


def gap_isolated_h(calculator, dft_prop_prefix, gap_prop_prefix, output_fname,
                   wdir='gap_bde_wdir'):

    dft_h = Atoms('H', positions=[(0, 0, 0)])
    dft_h.info['config_type'] = 'H'


    output_prefix=dft_prop_prefix
    inputs = ConfigSet_in(input_configs=dft_h)
    dft_outputs = ConfigSet_out()
    orca_kwargs = setup_orca_kwargs()
    orca.evaluate(inputs=inputs,
                  outputs=dft_outputs,
                  base_rundir=pj(wdir, 'orca_outputs'),
                  orca_kwargs=orca_kwargs,
                  output_prefix=output_prefix)

    outputs = ConfigSet_out(output_files=output_fname)
    generic.run(inputs=dft_outputs.output_configs,
                outputs=outputs,
                calculator=calculator, properties=['energy'],
                output_prefix=gap_prop_prefix)



def everything(calculator, dft_bde_filename, output_fname_prefix,
               dft_prop_prefix, gap_prop_prefix, wdir='gap_bde_wdir'):
    """

    # 1. evaluate dft structures with gap
    #     - should have (dft) hashses
    #     - should have dft_opt_positions
    # 2. gap-optimise and evaluate with gap
          - save optimised positions
    # 3. evaluate with dft
    # 4. add isolated hydrogen atom (optionally?)

    by the end should have
    hash

    {dft_prefix}opt_positions

    {dft_prefix}opt_{dft_prefix}forces
    {dft_prefix}opt_{dft_prefix}energy

    {dft_prefix}opt_{gap_prefix}forces
    {dft_prefix}opt_{gap_prefix}energy


    {gap_prefix}opt_positions

    {gap_prefix}opt_{gap_prefix}forces
    {gap_prefix}opt_{gap_prefix}energy

    {gap_prefix}opt_{dft_prefix}forces
    {gap_prefix}opt_{dft_prefix}energy

    and for isolated_h simply:
    {gap_prefix}energy
    {dft_prefix}energy

    Parameters
    ----------
    calculator - (calculator, calc_args, calc_kwargs) construct for wfl

    """
    random_at = random.choice(read(dft_bde_filename, ':'))
    assert f'{dft_prop_prefix}opt_mol_positions_hash' in random_at.info.keys()
    assert f'{dft_prop_prefix}opt_positions' in random_at.arrays.keys()

    if not os.path.exists(wdir):
        os.mkdir(wdir)

    dft_bde_with_gap_fname = pj(wdir, output_fname_prefix + '_gap.xyz')
    gap_reopt_fname = pj(wdir, output_fname_prefix + '_gap_reopt_trajectories.xyz')
    gap_reopt_with_gap_fname = pj(wdir, output_fname_prefix + '_gap_reopt_w_gap.xyz')
    gap_reopt_with_dft_fname = output_fname_prefix + 'gap_bde.xyz'


    logger.info('Evaluating GAP on DFT structures')
    output_prefix = f'{dft_prop_prefix}opt_{gap_prop_prefix}'
    inputs = ConfigSet_in(input_files=dft_bde_filename)
    outputs_gap_energies = ConfigSet_out(output_files=dft_bde_with_gap_fname)
    generic.run(inputs=inputs, outputs=outputs_gap_energies, calculator=calculator,
                properties=['energy', 'forces'], output_prefix=output_prefix)


    logger.info('GAP-optimising DFT structures')
    outputs_gap_reopt = ConfigSet_out(output_files=gap_reopt_fname)
    gap_reoptimised = gap_optimise(inputs=outputs_gap_energies.to_ConfigSet_in(),
                                    outputs=outputs_gap_reopt,
                                   calculator=calculator,
                                   wdir=wdir)


    logger.info('Evaluating GAP-optimised structures with GAP')
    output_prefix = f'{gap_prop_prefix}opt_{gap_prop_prefix}'
    inputs_to_gap_opt_regap = ConfigSet_in(input_configs=gap_reoptimised)
    outputs_gap_opt_w_gap = ConfigSet_out(output_files=gap_reopt_with_gap_fname)
    generic.run(inputs=inputs_to_gap_opt_regap,
                outputs=outputs_gap_opt_w_gap,
                calculator=calculator, properties=['energy', 'forces'],
                output_prefix=output_prefix)

    logger.info('Labeling GAP-optimised positions')
    atoms = read(gap_reopt_with_gap_fname, ':')
    for at in atoms:
        at.arrays[f'{gap_prop_prefix}opt_positions'] = at.positions.copy()

    logger.info('Re-evaluating gap-optimised structures with DFT')
    output_prefix=f'{gap_prop_prefix}opt_{dft_prop_prefix}'
    inputs_to_dft_reeval = ConfigSet_in(input_configs=atoms)
    final_outputs = ConfigSet_out(output_files=gap_reopt_with_dft_fname)
    orca_kwargs = setup_orca_kwargs()
    orca.evaluate(inputs=inputs_to_dft_reeval,
                  outputs=final_outputs,
                  base_rundir=pj(wdir, 'orca_outputs'),
                  orca_kwargs=orca_kwargs,
                  output_prefix=output_prefix)



def setup_orca_kwargs():


    cfg=Config.load()
    default_kw = Config.from_yaml(
        os.path.join(cfg['util_root'], 'default_kwargs.yml'))

    smearing = default_kw['orca']['smearing']
    orcasimpleinput = default_kw['orca']['orcasimpleinput']
    orcablocks = f'%scf Convergence Tight\nSmearTemp {smearing}\nmaxiter ' \
                 f'500\nend'

    orca_kwargs = {'orcasimpleinput': orcasimpleinput,
                   'orcablocks': orcablocks}
    print(orca_kwargs)

    return orca_kwargs


def gap_optimise(inputs, outputs, calculator, wdir):

    logfile = os.path.join(wdir, 'optimisation.log')

    opt_kwargs = {'logfile':logfile, 'master':True, 'precon':None,
                    'use_armijo':False}

    optimised_configset = minim.run(inputs, outputs,  calculator,
                                    keep_symmetry=False,
                             update_config_type=False, **opt_kwargs)

    atoms_opt = [at for at in optimised_configset
                    if at.info['minim_config_type']=='minim_last_converged']

    return atoms_opt



