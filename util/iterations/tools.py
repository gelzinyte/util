import os
import logging

import pandas as pd
import numpy as np

from wfl.calculators import generic
from wfl.configset import ConfigSet_in, ConfigSet_out

from util import smiles, radicals
from util import configs

logger = logging.getLogger(__name__)


def make_dirs(dir_names):
    for dir_name in dir_names:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

def make_structures(smiles_csv, iter_no, num_smi_repeat, outputs):

    atoms_out = []

    logger.info(f"writing to {outputs.output_files}")

    # generate molecules
    df = pd.read_csv(smiles_csv)
    for smi, name in zip(df['SMILES'], df['Name']):
        for _ in range(num_smi_repeat):
            mol = smiles.smi_to_atoms(smi)
            mol.info['config_type'] = name

            interim_outputs = ConfigSet_out()
            radicals.abstract_sp3_hydrogen_atoms(mol, outputs=interim_outputs)
            num_mols_and_rads = len(interim_outputs.output_configs)

            for idx in range(num_mols_and_rads):
                # make a new conformer for each molecule/radical I take
                mol = smiles.smi_to_atoms(smi)
                mol.info['config_type'] = name

                interim_outputs = ConfigSet_out()
                radicals.abstract_sp3_hydrogen_atoms(mol, outputs=interim_outputs)

                atoms_out.append(interim_outputs.output_configs[idx])

    logger.info(f'length of output atoms: {len(atoms_out)}')

    for at in atoms_out:
        at.cell = [50, 50, 50]
        at.info['iter_no'] = iter_no
        outputs.write(at)

    outputs.end_write()
    return outputs.to_ConfigSet_in()

def filter_configs(inputs, outputs, bad_structures_fname,
                   gap_prefix,
                   e_threshold,
                   f_threshold,
                   dft_prefix='dft_'):

    atoms = configs.filter_insane_geometries(inputs, mult=1,
                                bad_structures_fname=bad_structures_fname)

    for at in atoms:
        e_error = at.info[f'{gap_prefix}energy'] - \
                  at.info[f'{dft_prefix}energy']
        if np.abs(e_error) > e_threshold:
            outputs.write(at)
            continue

        if f_threshold is not None:
            f_error = at.arrays[f'{gap_prefix}forces'] - \
                      at.arrays[f'{dft_prefix}forces']
            if np.max(np.abs(f_error.flatten())) > f_threshold:
                outputs.write(at)

    outputs.end_write()
    return outputs.to_ConfigSet_in()







