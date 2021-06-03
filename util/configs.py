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




def smiles_csv_to_molecules(smiles_csv, repeat=1):

    df = pd.read_csv(smiles_csv)
    smi_names = []
    smiles_to_convert = []
    for smi, name in zip(df['SMILES'], df['Name']):
        smiles_to_convert += [smi] * repeat
        smi_names += [name] * repeat

    molecules = ConfigSet_out()
    smiles.run(outputs=molecules, smiles=smiles_to_convert)
    for at, name in zip(molecules.output_configs, smi_names):
        at.info['config_type'] = name
        at.cell = [50, 50, 50]

    return molecules.output_configs




def batch_configs(in_fname, num_tasks, batch_in_fname_prefix='in_',
                  count_from=1):

    all_atoms = read(in_fname, ':')
    batch_size = int(len(all_atoms) / num_tasks) + 1

    for idx, batch in enumerate(grouper(all_atoms, batch_size)):

        batch = [b for b in batch if b is not None]

        write(f'{batch_in_fname_prefix}{idx+count_from}.xyz', batch)


def collect_configs(out_fname, num_tasks, batch_out_fname_prefix='out_', 
                    count_from=1):

    ats_out = []
    for idx in range(num_tasks):
        ats = read(f'{batch_out_fname_prefix}{idx+count_from}.xyz', ':')
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


def filter_insane_geometries(atoms_list, mult=1.2, bad_structures_fname=None):

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

    if bad_structures_fname is not None and len(bad_atoms) > 0:
        print(type(bad_atoms))
        write(bad_structures_fname, bad_atoms)
    logger.info(f'skipped {len(skipped_idx)} structures, because couldn\'t find '
                  f'a H whithin reasonable cutoff. Nos: {skipped_idx}')

    return atoms_out



def sample_downweighted_normal_modes(inputs, outputs, temp, sample_size, prop_prefix,
                        info_to_keep=None, arrays_to_keep=None):
    """Multiple times displace along normal modes for all atoms in input

    Parameters
    ----------

    inputs: Atoms / list(Atoms) / ConfigSet_in
        Structures with normal mode information (eigenvalues &
        eigenvectors)
    outputs: ConfigSet_out
    temp: float
        Temperature for normal mode displacements
    sample_size: int
        Now many perturbed structures per input structure to return
    prop_prefix: str / None
        prefix for normal_mode_frequencies and normal_mode_displacements
        stored in atoms.info/arrays
    info_to_keep: str, default "config_type"
        string of Atoms.info.keys() to keep
    arrays_to_keep: str, default None
        string of Atoms.arrays.keys() entries to keep

    Returns
    -------
      """

    if isinstance(inputs, Atoms):
        inputs = [inputs]

    for atoms in inputs:
        at_vib = Vibrations(atoms, prop_prefix)
        energies_into_modes = energies_for_weighting_normal_modes(at_vib.frequencies,
                                                                  temp)
        try:
            sample = at_vib.sample_normal_modes(energies_for_modes=energies_into_modes[6:],
                                                sample_size=sample_size,
                                                info_to_keep=info_to_keep,
                                                arrays_to_keep=arrays_to_keep)
        except TypeError as e:
            print(f'config type: {at_vib.atoms.info["config_type"]}')
            raise(e)
        outputs.write(sample)

    outputs.end_write()


def energies_for_weighting_normal_modes(frequencies, temp, threshold_invcm=200,
                              threshold_eV=None):
    """Frequencies in eV, threshold in eV or invcm"""

    assert threshold_invcm is None or threshold_eV is None

    if threshold_invcm is not None:
        threshold_eV = threshold_invcm * units.invcm

    return np.array([units.kB * temp * (
                freq / threshold_eV) ** 2 if freq < threshold_eV else
                     units.kB * temp
                     for freq in frequencies])



def process_config_info(fname_in, fname_out):

    ats = read(fname_in, ':')
    ats = process_config_info_on_atoms(ats)

    write(fname_out, ats)


def process_config_info_on_atoms(ats, verbose=True):

    all_mol_or_rad_entries = []
    all_compound_entries = []

    for at in ats:

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
