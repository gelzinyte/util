from ase.io import read, write
import logging
import pandas as pd
from util import smiles
from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.generate_configs.vib import Vibrations
from ase import units
from ase import Atoms
import warnings
from ase import neighborlist
import numpy as np
from util import grouper
import os
import hashlib

logger = logging.getLogger(__name__)

def into_dict_of_labels(ats, info_label):
    if info_label == None:
        return {"no_label":ats}
    data = {}
    for at in ats:
        if info_label not in at.info.keys():
            label = "no_label"
        else:
            label = at.info[info_label]
            
        if label not in data.keys():
            data[label] = []
        data[label].append(at)

    return data

def strip_info_arrays(atoms, info_to_keep, arrays_to_keep):

    if info_to_keep is None:
        info_to_keep = []

    if arrays_to_keep is None:
        arrays_to_keep = []

    if isinstance(atoms, Atoms):
        atoms = [atoms]

    for at in atoms:

        info_keys = list(at.info.keys())
        for key in info_keys:
            if key not in info_to_keep:
                at.info.pop(key)

        arrays_keys = list(at.arrays.keys())
        for key in arrays_keys:
            if key in ['numbers', 'positions']:
                continue
            elif key not in arrays_to_keep:
                at.arrays.pop(key)

    if len(atoms) == 1:
        atoms = atoms[0]

    return atoms



#
# def smiles_csv_to_molecules(smiles_csv, outputs, repeat=1,
#                             smiles_col='smiles',
#                             name_col='zinc_id'):
#
#     df = pd.read_csv(smiles_csv, delim_whitespace=True)
#     smi_names = []
#     smiles_to_convert = []
#     for smi, name in zip(df[smiles_col], df[name_col]):
#         smiles_to_convert += [smi] * repeat
#         smi_names += [name] * repeat
#
#     smiles.run(outputs=outputs, smiles=smiles_to_convert)
#     for at, name in zip(outputs.to_ConfigSet_in(), smi_names):
#         at.info['config_type'] = name
#         at.info['compound'] = name
#         at.cell = [50, 50, 50]
#
#     return outputs.to_ConfigSet_in()





def batch_configs(in_fname, num_tasks, batch_in_fname_prefix='in_',
                  count_from=1, dir_prefix=None):

    all_atoms = read(in_fname, ':')
    batch_size = int(len(all_atoms) / num_tasks) + 1

    for idx, batch in enumerate(grouper(all_atoms, batch_size)):

        batch = [b for b in batch if b is not None]

        output_filename = f'{batch_in_fname_prefix}{idx+count_from}.xyz'
        if dir_prefix is not None:
            dir_name = dir_prefix + str(idx+count_from)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            output_filename = os.path.join(dir_name, output_filename)

        write(output_filename, batch)


def collect_configs(out_fname, num_tasks, batch_out_fname_prefix='out_',
                    dir_prefix='job_', count_from=1):

    ats_out = []
    for idx in range(num_tasks):
        if dir_prefix is None:
            this_dir_prefix=""
        else:
            this_dir_prefix=f'{dir_prefix}{idx+count_from}/'
        ats = read(f'{this_dir_prefix}{batch_out_fname_prefix}{idx+count_from}.xyz', ':')
        ats_out += ats

    write(out_fname, ats_out)

def cleanup_configs(num_tasks=8, batch_in_fname_prefix='in_',
                    batch_out_fname_prefix='out_', count_from=1):

    for idx in range(num_tasks):

        in_fname = f'{batch_in_fname_prefix}{idx+count_from}.xyz'
        if os.path.exists(in_fname):
            os.remove(in_fname)

        out_fname = f'{batch_out_fname_prefix}{idx+count_from}.xyz'
        if os.path.exists(out_fname):
            os.remoe(out_fname)


def filter_insane_geometries(atoms_list, mult=1.2):

    bad_atoms = []
    atoms_out = []
    skipped_idx = []
    for idx, atoms in enumerate(atoms_list):
        if len(atoms) == 1:
            atoms_out.append(atoms)


        natural_cutoffs = neighborlist.natural_cutoffs(atoms,
                                                       mult=mult)
        neighbor_list = neighborlist.NeighborList(natural_cutoffs,
                                                  self_interaction=False,
                                                  bothways=True)
        _ = neighbor_list.update(atoms)

        

        for at in atoms:

            indices, offsets = neighbor_list.get_neighbors(at.index)
            if at.symbol == 'H':
                if len(indices) != 1:
                    skipped_idx.append(idx)
                    bad_atoms.append(atoms)
                    break

            elif at.symbol == 'C':
                if len(indices) < 2:
                    skipped_idx.append(idx)
                    bad_atoms.append(atoms)
                    break
            elif at.symbol == 'O':
                if len(indices) == 0:
                    skipped_idx.append(idx)
                    bad_atoms.append(atoms)
                    break

        else:
            atoms_out.append(atoms)

    num_skipped = len(skipped_idx)
    if num_skipped > 0:
        logger.warning(f'skipped {num_skipped} structures ({num_skipped/len([at for at in atoms_list])*100:.1f}%), because couldn\'t find '
                    f'a H whithin reasonable cutoff. Nos: {skipped_idx}')
    else: 
        logger.info("Found no unreasonable geometries")

    return {'good_geometries':atoms_out, 'bad_geometries':bad_atoms}


def process_config_info(fname_in, fname_out):

    ats = read(fname_in, ':')
    ats = process_config_info_on_atoms(ats)

    write(fname_out, ats)

def assign_info_entries(atoms, config_type, compound, mol_or_rad, rad_no):
    atoms.info["config_type"] = config_type
    atoms.info["compound"] = compound
    atoms.info["mol_or_rad"] = mol_or_rad
    atoms.info["rad_num"] = rad_no
    atoms.info["graph_name"] = f'{compound}_{rad_no}' if rad_no == 'mol' \
        else f'{compound}_rad{rad_no}'


def process_config_info_on_atoms(ats, verbose=True):

    all_mol_or_rad_entries = []
    all_compound_entries = []

    for at in ats:

        if len(at) == 1:
            continue

        cfg = at.info['config_type']
        words = cfg.split('_')

        mol_or_rad = words[-1]

        if 'mol' not in mol_or_rad and 'rad' not in mol_or_rad:
            raise RuntimeError(
                f'{mol_or_rad} isn\'t eiter molecule or radical')

        all_mol_or_rad_entries.append(mol_or_rad)
        at.info['mol_or_rad'] = mol_or_rad

        compound = '-'.join(words[:-1])
        all_compound_entries.append(compound)
        at.info['compound'] = compound

    if verbose:
        print(f'all mol_or_rad entries: {set(all_mol_or_rad_entries)}')
        print(f' all compound entries: {set(all_compound_entries)}')

    return ats

#creates unique hash for a matrix of numbers
def hash_array(v):
    return hashlib.md5(np.array2string(v, precision=8, sign='+', floatmode='fixed').encode()).hexdigest()

#creates unique hash for Atoms from atomic numbers and positions
def hash_atoms(at):
    v = np.concatenate((at.numbers.reshape(-1,1), at.positions),axis=1)
    return hash_array(v)
