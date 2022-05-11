import pandas as pd
import os
import sys
from wfl.configset import ConfigSet, ConfigSet_out
try:
    from wfl.generate_configs import smiles, radicals
except ImportError:
    from util import smiles, radicals
from ase.optimize.precon import PreconLBFGS
from wfl.autoparallelize import iterable_loop
from wfl.utils.parallel import construct_calculator_picklesafe
from wfl.utils.at_copy_spc import at_copy_SPC
from wfl.utils.misc import atoms_to_list
from ase.io import write, read
from util.util_config import Config
from wfl.calculators import orca
import os

from util import error_table as et
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from wfl.generate_configs import vib

def testing_set_from_gap_normal_modes(iter_no, temperature, no_samples):

    nm_fname = f'xyzs/normal_modes_reference_{iter_no}.xyz'
    inputs = ConfigSet(input_files=nm_fname)
    outputs = ConfigSet_out()
    info_to_keep = ['config_type', 'iter_no', 'minim_n_steps']


    vib.sample_normal_modes(inputs=inputs,
                            outputs=outputs,
                            temp=temperature,
                            sample_size=no_samples,
                            info_to_keep=info_to_keep,
                            prop_prefix='gap_')

    return outputs.output_configs


def filter_by_error(atoms, gap_prefix='gap_', dft_prefix='dft_',
                    e_threshold=0.05, f_threshold=0.1):
    atoms_out = []
    for at in atoms:

        e_error = at.info[f'{gap_prefix}energy'] - at.info[f'{dft_prefix}energy']
        if np.abs(e_error) > e_threshold:
            atoms_out.append(at)
            continue

        if f_threshold is not None:
            f_error = at.arrays[f'{gap_prefix}forces'] - at.arrays[f'{dft_prefix}forces']
            if np.max(np.abs(f_error.flatten())) > f_threshold:
                atoms_out.append(at)

    return atoms_out


def make_structures(smiles_csv, iter_no, num_smi_repeat, output_fname):

    atoms_out = []

    # generate molecules
    df = pd.read_csv(smiles_csv)
    for smi, name in zip(df['SMILES'], df['Name']):
        for _ in range(num_smi_repeat):
            mol = smiles.smi_to_atoms(smi)
            mol.info['config_type'] = name

            outputs = ConfigSet_out()
            radicals.abstract_sp3_hydrogen_atoms(mol, outputs=outputs)
            num_mols_and_rads = len(outputs.output_configs)

            for idx in range(num_mols_and_rads):
                # make a new conformer for each molecule/radical I take
                mol = smiles.smi_to_atoms(smi)
                mol.info['config_type'] = name

                outputs = ConfigSet_out()
                radicals.abstract_sp3_hydrogen_atoms(mol, outputs=outputs)

                atoms_out.append(outputs.output_configs[idx])

    for at in atoms_out:
        at.cell = [50, 50, 50]
        at.info['iter_no'] = iter_no

    write(output_fname, atoms_out)



