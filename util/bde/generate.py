import os
import random

from os.path import join as pj

import numpy as np

from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import generic
from wfl.calculators import orca



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
    assert f'{dft_prop_prefix}opt_positions_hash' in random_at.info.keys()
    assert f'{dft_prop_prefix}opt_positions' in random_at.arrays.keys()

    dft_bde_with_gap_fname = pj(wdir, output_fname_prefix + '_gap.xyz')
    gap_reopt_fname = pj(wdir, output_fname_prefix + '_gap_reopt_trajectories.xyz')
    gap_reopt_with_gap_fname = pj(wdir, output_fname_prefix + '_gap_reopt_w_gap.xyz')
    gap_reopt_with_dft_fname = output_fname_prefix + 'gap_bde.xyz'

    # get gap energies on dft structures
    output_prefix = f'{dft_prop_prefix}opt_{gap_prop_prefix}'
    inputs = ConfigSet_in(input_files=dft_bde_filename)
    outputs_gap_energies = ConfigSet_out(output_files=dft_bde_with_gap_fname)
    generic.run(inputs=inputs, outputs=outputs_gap_energies, calculator=calculator,
                properties=['energy', 'forces'], output_prefix=output_prefix)

    # gap-optimise
    outputs_gap_reopt = ConfigSet_out(output_files=gap_reopt_fname)
    gap_reoptimised = gap_optimise(inputs=outputs_gap_energies.to_ConfigSet_in(),
                                    outputs=outputs_gap_reopt,
                                   calculator=claculator)
    # evaluate with gap
    output_prefix = f'{gap_prop_prefix}opt_{gap_prop_prefix}'
    inputs_to_gap_opt_regap = ConfigSet_in(input_configs=gap_reoptimised)
    outputs_gap_opt_w_gap = ConfigSet_out(output_files=gap_reopt_with_gap_fname)
    generic.run(inputs=inputs_to_gap_opt_regap,
                outputs=outputs_gap_opt_w_gap,
                calculator=calculator, properties=['energy', 'forces'],
                output_prefix=output_prefix)

    # label positions
    atoms = read(gap_reopt_with_gap_fname, ':')
    for at in atoms:
        at.arrays[f'{gap_prop_prefix}opt_positions'] = at.positions.copy()
        
    # dft re-evaluate
    output_prefix=f'{gap_prop_prefix}opt_{dft_prop_prefix}'
    inputs_to_dft_reeval = ConfigSet_in(input_configs=atoms)
    final_outputs = ConfigSet_out(output_configs=gap_reopt_with_dft_fname)
    orca_kwargs = setup_orca_kwargs()
    orca.evaluate(inputs=inputs_to_dft_reeval,
                  outputs=final_outputs,
                  base_rundir=pj(wdir, 'orca_outputs'),
                  orca_kwargs=orca_kwargs,
                  output_prefix=output_prefix)



def setup_orca_kwargs():

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


def gap_optimise(inputs, outputs, calculator):

    opt_kwargs = {'logfile':'log.txt', 'master':True, 'precon':None,
                    'use_armijo':False}

    minim.run(inputs, outputs,  calculator, keep_symmetry=False,
                             update_config_type=False, **opt_kwargs)

    atoms_opt = [traj[-1] for traj in outputs.output_configs]

    return atoms_opt



