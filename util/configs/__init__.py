from ase.io import read, write
import logging
import pandas as pd
from util import smiles
from wfl.configset import ConfigSet, OutputSpec
from ase import units
from ase import Atoms
import warnings
from ase import neighborlist
import numpy as np
from util import grouper
import util
import os
import hashlib
import random
from util import radicals
from util import distances_dict

logger = logging.getLogger(__name__)

def min_max_data(atoms):
    dd = distances_dict(atoms)
    for key, vals in dd.items():
        print(f"{key}: min: {min(vals)}, max: {max(vals)}")


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


def filter_insane_geometries(atoms_list, mult=1.2, mark_elements=False, skin=0):

    bad_atoms = []
    atoms_out = []
    skipped_idx = []
    for idx, atoms in enumerate(atoms_list):

        if len(atoms) == 1:
            atoms_out.append(atoms)
        
        geometry_ok = check_geometry(atoms, mult=mult, skin=skin, mark_elements=mark_elements)

        if not geometry_ok:
            skipped_idx.append(idx)
            bad_atoms.append(atoms)
        else:
            atoms_out.append(atoms)

    num_skipped = len(skipped_idx)
    if num_skipped > 0:
        logger.warning(f'skipped {num_skipped} structures ({num_skipped/len([at for at in atoms_list])*100:.1f}%), because couldn\'t find '
                    f'enough neighbours for some atoms. Nos: {skipped_idx}')
    else: 
        logger.info("Found no unreasonable geometries")

    return {'good_geometries':atoms_out, 'bad_geometries':bad_atoms}


def check_geometry(atoms, mult=1.2, mark_elements=False, skin=0, ignore_idx=None):

    if ignore_idx is None:
        ignore_idx = []

    natural_cutoffs = neighborlist.natural_cutoffs(atoms, mult=mult)
    neighbor_list = neighborlist.NeighborList(natural_cutoffs,
                                                self_interaction=False,
                                                skin=skin,
                                                bothways=True)
    _ = neighbor_list.update(atoms)

    for at_idx, at in enumerate(atoms):

        if at_idx in ignore_idx:
            continue

        indices, offsets = neighbor_list.get_neighbors(at.index)
        if at.symbol == 'H':
            if len(indices) != 1:
                if mark_elements:
                    at.symbol = "He"
                    atoms.info["bad_atom_id"] = at_idx
                    atoms.info["bad_atom_neighbrous"] = indices
                    for bad_idx in indices:
                        if atoms[bad_idx].symbol == "C":
                            atoms[bad_idx].symbol = "Co"
                        if atoms[bad_idx].symbol == "H":
                            atoms[bad_idx].symbol = "Hg"

                return False

        elif at.symbol == 'C':
            if len(indices) < 2:
                if mark_elements:
                    for bad_idx in indices:
                        if atoms[bad_idx].symbol == "C":
                            atoms[bad_idx].symbol = "Co"
                        if atoms[bad_idx].symbol == "H":
                            atoms[bad_idx].symbol = "Hg"
                    atoms.info["bad_atom_id"] = at_idx
                    atoms.info["bad_atom_neighbrous"] = indices
                    at.symbol = "Ca"
                return False
        elif at.symbol == 'O':
            if len(indices) == 0 or len(indices)>3:
                if mark_elements:
                    atoms.symbol = "Os"
                # bad_atoms.append(atoms)
                return False

    else:
        return True

    
def process_config_info(fname_in, fname_out):

    ats = read(fname_in, ':')
    ats = process_config_info_on_atoms(ats)

    write(fname_out, ats)


def assign_info_entries(atoms, mol_or_rad, rad_no, config_type=None, compound=None, radical_c_idx=None):
    if config_type is not None:
        atoms.info["config_type"] = config_type
    if compound is not None:
        atoms.info["compound"] = compound
    atoms.info["mol_or_rad"] = mol_or_rad
    atoms.info["rad_num"] = rad_no
    if compound is not None:
        atoms.info["graph_name"] = f'{compound}_{rad_no}' if rad_no == 'mol' \
        else f'{compound}_rad{rad_no}'
    if radical_c_idx is not None:
        atoms.info["radical_c_idx"] = radical_c_idx


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

def hash_smiles(smi):
    return hashlib.md5(smi.encode()).hexdigest()


def find_closest_c(at, h_idx):
    distances = at.get_all_distances()[h_idx]
    distances[distances == 0] = np.inf
    closest_id = np.argmin(distances)
    if at.symbols[closest_id] != "C":
        write("bad_at.xyz", at)
        print(f"h_idx: {h_idx}, closest: {closest_id}")
        print(distances)
        raise RuntimeError
    return closest_id


def mark_sp3_CH(at, info_key="sp3_ch"):

    is_sp3 = np.empty(len(at))
    is_sp3.fill(False)
    # print(is_sp3)

    sp3_H_numbers = radicals.get_sp3_h_numbers(at)
    closest_carbons = [find_closest_c(at, h_idx) for h_idx in sp3_H_numbers] 
    sp3_ch = closest_carbons + sp3_H_numbers 

    is_sp3[sp3_ch] = True
    is_sp3 = [bool(val) for val in is_sp3]

    at.arrays[info_key] = np.array(is_sp3)
    return at


def mark_mol_rad_envs(at, info_key):

    is_sp3 = np.empty(len(at))
    is_sp3.fill(False)
    # print(is_sp3)

    sp3_H_numbers = radicals.get_sp3_h_numbers(at)
    closest_carbons = [find_closest_c(at, h_idx) for h_idx in sp3_H_numbers] 
    sp3_ch = closest_carbons + sp3_H_numbers 

    is_sp3[sp3_ch] = True
    is_sp3 = [bool(val) for val in is_sp3]

    at.arrays[info_key] = np.array(is_sp3)
    return at

def assign_bde_to_C_atoms (inputs, outputs,bde_label):

    if outputs.all_written():
        print(f"{outputs} written, not reassigning")
        return outputs.to_ConfigSet()

    ch_cutoff = 1.5
    ats_out = []
    for at_idx, at in enumerate(inputs):
        syms = list(at.symbols)

        new_bde_label = "C_lowest_" + bde_label
        new_bde_array = np.empty(len(at))
        new_bde_array.fill(np.nan)

        if bde_label not in at.arrays:
            continue
        
        old_bde_array = at.arrays[bde_label]

        at = mark_sp3_CH(at)

        for idx, is_sp3  in enumerate(at.arrays["sp3_ch"]):

            if not is_sp3 or syms[idx] != "C":
                continue
            
            # find closest H 
            distances = at.get_all_distances()[idx]
            distances[distances == 0] = np.inf

            closest_H_id = [neigh_idx for neigh_idx, dist in enumerate(distances) if dist < ch_cutoff and syms[neigh_idx] == "H"]
            # print(f"atidx: {at_idx}, cidx: {idx}, num neighbours: {len(closest_H_id)}, neighbours: {closest_H_id}")
            if at.info["compound"] != "methane":
                assert len(closest_H_id) < 4 and len(closest_H_id) > 0

            # check that that C atom doesn't have an entry
            assert np.isnan(old_bde_array[idx])

            # pick lowes of thee bdes
            lowest_H_bde = np.min(old_bde_array[closest_H_id])
            new_bde_array[idx] = lowest_H_bde 

        # save to atoms
        at.arrays[new_bde_label] = new_bde_array
        outputs.store(at)
    outputs.close()
    return outputs.to_ConfigSet()

def find_closest_h(at, c_idx, cutoff=1.5):
    distances = at.get_all_distances()[c_idx]
    distances[distances == 0] = np.inf
    closest_at_idx = np.where(distances < cutoff)[0]

    # check it's sp3 carbon
    if len(closest_at_idx) != 4:
        print("at info", at.info)
        print("c_idx", c_idx)
        print("closest_at_idx",closest_at_idx)
        raise RuntimeError(f"Found {len(closest_at_idx)} neighbours, expected 4.")

    idcs_out = []
    for idx in closest_at_idx:
        if at.symbols[idx] == "H":
            idcs_out.append(idx)

    if len(idcs_out) == 0:
        raise RuntimeError(f"Found no H neighbours.")
    return idcs_out 
