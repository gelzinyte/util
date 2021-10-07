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
from util import opt

logger = logging.getLogger(__name__)


def ip_isolated_h(calculator, dft_prop_prefix, ip_prop_prefix, output_fname,
                   wdir='ip_bde_wdir'):

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
    generic.run(inputs=dft_outputs.to_ConfigSet_in(),
                outputs=outputs,
                calculator=calculator, properties=['energy'],
                output_prefix=ip_prop_prefix)



def everything(calculator, dft_bde_filename, output_fname_prefix,
               dft_prop_prefix, ip_prop_prefix, wdir='ip_bde_wdir',
               chunksize=10):
    """

    # 1. evaluate dft structures with ip
    #     - should have (dft) hashses
    #     - should have dft_opt_positions
    # 2. ip-optimise and evaluate with ip
          - save optimised positions
    # 3. evaluate with dft
    # 4. add isolated hydrogen atom (optionally?)

    by the end should have
    hash

    {dft_prefix}opt_positions

    {dft_prefix}opt_{dft_prefix}forces
    {dft_prefix}opt_{dft_prefix}energy

    {dft_prefix}opt_{ip_prefix}forces
    {dft_prefix}opt_{ip_prefix}energy


    {ip_prefix}opt_positions

    {ip_prefix}opt_{ip_prefix}forces
    {ip_prefix}opt_{ip_prefix}energy

    {ip_prefix}opt_{dft_prefix}forces
    {ip_prefix}opt_{dft_prefix}energy

    and for isolated_h simply:
    {ip_prefix}energy
    {dft_prefix}energy

    Parameters
    ----------
    calculator - (calculator, calc_args, calc_kwargs) construct for wfl

    """
    random_at = random.choice(read(dft_bde_filename, ':50'))
    assert f'{dft_prop_prefix}opt_mol_positions_hash' in random_at.info.keys()
    assert f'{dft_prop_prefix}opt_positions' in random_at.arrays.keys()

    if not os.path.exists(wdir):
        os.mkdir(wdir)

    dft_fname_prefix = os.path.splitext(os.path.basename(dft_bde_filename))[0]

    dft_bde_with_ip_fname = pj(wdir, dft_fname_prefix + 'with_ip.xyz')
    ip_reopt_fname = pj(wdir, output_fname_prefix + 'ip_reopt_only.xyz')
    ip_reopt_with_ip_fname = pj(wdir, output_fname_prefix +
                                'ip_bde_without_dft.xyz')
    ip_reopt_with_dft_fname = output_fname_prefix + 'ip_bde.xyz'


    at = read(dft_bde_filename)
    output_prefix = f'{dft_prop_prefix}opt_{ip_prop_prefix}'
    energy_key = f'{output_prefix}energy'
    if energy_key not in at.info.keys():
        logger.info(f'Evaluating GAP on DFT structures to '
                    f'{dft_bde_with_ip_fname}')
        inputs = ConfigSet_in(input_files=dft_bde_filename)
        outputs_ip_energies = ConfigSet_out(output_files=dft_bde_with_ip_fname)
        inputs = generic.run(inputs=inputs, outputs=outputs_ip_energies,
                     calculator=calculator,
                     properties=['energy', 'forces'],
                             output_prefix=output_prefix)
    else:
        logger.info(f'found {energy_key} in at.info.keys(), not evaluating '
                    f'GAP on sturctures')
        inputs = ConfigSet_in(input_files=dft_bde_filename)



    logger.info('IP-optimising DFT structures')
    outputs = ConfigSet_out(output_files=ip_reopt_fname,
                            force=True, all_or_none=True)
    inputs = opt.optimise(inputs=inputs,
                           outputs=outputs,
                           calculator=calculator,
                          chunksize=chunksize)


    logger.info('Evaluating IP-optimised structures with IP')
    output_prefix = f'{ip_prop_prefix}opt_{ip_prop_prefix}'
    inputs_to_ip_opt_reip = ConfigSet_in(input_configs=inputs)
    outputs_ip_opt_w_ip = ConfigSet_out(output_files=ip_reopt_with_ip_fname,
                                        force=True, all_or_none=True)
    generic.run(inputs=inputs_to_ip_opt_reip,
                outputs=outputs_ip_opt_w_ip,
                calculator=calculator, properties=['energy', 'forces'],
                output_prefix=output_prefix,
                chunksize=chunksize)

    atoms = read(ip_reopt_with_ip_fname, ':')
    if f'{ip_prop_prefix}opt_positions' not in atoms[0].arrays.keys():
        logger.info('Labeling IP-optimised positions')
        for at in atoms:
            at.arrays[f'{ip_prop_prefix}opt_positions'] = at.positions.copy()
        write(ip_reopt_with_ip_fname, atoms)

    logger.info('Re-evaluating ip-optimised structures with DFT')
    output_prefix=f'{ip_prop_prefix}opt_{dft_prop_prefix}'
    inputs_to_dft_reeval = ConfigSet_in(input_configs=atoms)
    final_outputs = ConfigSet_out(output_files=ip_reopt_with_dft_fname,
                                  force=True, all_or_none=True)
    orca_kwargs = setup_orca_kwargs()
    orca.evaluate(inputs=inputs_to_dft_reeval,
                  outputs=final_outputs,
                  base_rundir=pj(wdir, 'orca_outputs'),
                  orca_kwargs=orca_kwargs,
                  output_prefix=output_prefix,
                  keep_files=False)



def setup_orca_kwargs():

    cfg=Config.load()
    default_kw = Config.from_yaml(
        os.path.join(cfg['util_root'], 'default_kwargs.yml'))

    return default_kw['orca']





