import pandas as pd
import os
import sys
from wfl.configset import ConfigSet_in, ConfigSet_out
try:
    from wfl.generate_configs import smiles, radicals
except ImportError:
    pass
from ase.optimize.precon import PreconLBFGS
from wfl.pipeline import iterable_loop
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
    inputs = ConfigSet_in(input_files=nm_fname)
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


def make_structures(smiles_csv, iter_no, num_smi_repeat, opt_starts_fname):

    # generate molecules
    df = pd.read_csv(smiles_csv)
    smiles_to_convert = []
    smi_names = []
    for smi, name in zip(df['SMILES'], df['Name']):
        smiles_to_convert += [smi] * num_smi_repeat
        smi_names += [name] * num_smi_repeat


    molecules = ConfigSet_out()
    smiles.run(outputs=molecules, smiles=smiles_to_convert, extra_info={'iter_no':iter_no})
    for at, name in zip(molecules.output_configs, smi_names):
        at.info['config_type'] = name
        at.cell = [40, 40, 40]

    # generate radicals
    mols_rads = ConfigSet_out(output_files=opt_starts_fname)
    radicals.abstract_sp3_hydrogen_atoms(molecules.output_configs, outputs=mols_rads)





