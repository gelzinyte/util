import warnings
import logging

import numpy as np
from ase import neighborlist
from wfl.utils.misc import atoms_to_list

from wfl.generate_configs.utils import config_type_append

from util import smiles, configs

logger = logging.getLogger(__name__)

def get_sp3_h_numbers(atoms):
    natural_cutoffs = neighborlist.natural_cutoffs(atoms)
    neighbor_list = neighborlist.NeighborList(natural_cutoffs,
                                              self_interaction=False,
                                              bothways=True)
    _ = neighbor_list.update(atoms)

    symbols = np.array(atoms.symbols)
    sp3_hs = []
    for at in atoms:
        if at.symbol == 'H':
            h_idx = at.index

            indices, offsets = neighbor_list.get_neighbors(h_idx)
            if len(indices) != 1:
                raise RuntimeError("Got more than hydrogen neighbor")

            # find index of the atom H is bound to
            h_neighbor_idx = indices[0]

            if symbols[h_neighbor_idx] != 'C':
                continue

            # count number of neighbours of the carbon H is bound to
            indices, offsets = neighbor_list.get_neighbors(h_neighbor_idx)

            no_carbon_neighbors = len(indices)

            if no_carbon_neighbors == 4:
                sp3_hs.append(h_idx)

    if len(sp3_hs) == 0:
        warnings.warn("No sp3 hydrogens were found; no radicals returned")

    return sp3_hs




def abstract_sp3_hydrogen_atoms(inputs, label_config_type=True,
                                return_mol=True):
    """ Removes molecules' sp3 hydrogen atoms one at a time to give a number
    of corresponding unsaturated (radical) structures.

    Method of determining sp3: carbon has 4 neighbors of any kind. Only
    removes from Carbon atoms.

    Parameters
    ----------

    inputs: Atoms / list(Atoms) / ConfigSet_in
        structure to remove sp3 hydrogen atoms from
    outputs: ConfigSet_out:
        where to write structures to
    label_config_type: bool, default True
        whether to append config_type with 'mol' or 'rad{idx}'
    return_mol: bool, default True
        returns the original molecule too

    Returns
    -------

    """
    all_output_atoms = []
    for atoms in atoms_to_list(inputs):

        sp3_hs = get_sp3_h_numbers(atoms)

        radicals = []
        for h_idx in sp3_hs:
            at = atoms.copy()
            del at[h_idx]
            radicals.append(at)

        if label_config_type:

            atoms_out = []

            if return_mol:
                mol = atoms.copy()
                mol.info['compound'] = mol.info['config_type']
                config_type_append(mol, 'mol')
                mol.info['mol_or_rad'] = 'mol'
                atoms_out.append(mol)

            for rad, h_id in zip(radicals, sp3_hs):
                rad.info['compound'] = rad.info['config_type']
                config_type_append(rad, f'rad{h_id}')
                rad.info['mol_or_rad'] = f'rad{h_id}'
                atoms_out.append(rad)

        else:
            if return_mol:
                atoms_out = [atoms.copy()] + radicals
            else:
                atoms_out = radicals
        all_output_atoms += atoms_out
    return all_output_atoms

def rad_conformers_from_smi(smi, compound, num_radicals):
    """given molecule, make "num_radicals" of radicals"""
    try:
        mol = smiles.smi_to_atoms(smi)
    except IndexError:
        # means converting from smiles to xyz didn't work, but it
        # doesn't raise an error
        raise RuntimeError("Couldn't make xyz from smiles")

    configs.assign_info_entries(mol, config_type="rdkit",
                                compound=compound, mol_or_rad="mol",
                                rad_no="mol")

    selected_sp3_h_nos = get_sp3_h_numbers(mol)
    if num_radicals is not None:
        if len(selected_sp3_h_nos) == 0:
            logger.warning(f"No sp3 radicals found, id: {compound}")
            return []
        elif num_radicals > len(selected_sp3_h_nos):
            logger.warning(f"Asked for number of radicals per molecule "
                           f"({num_radicals}) is larger than the number of "
                           f"radiclas found ({len(selected_sp3_h_nos)}), "
                           f"selecting all radicals. Compound id: {compound}")
        else:
            selected_sp3_h_nos = np.random.choice(selected_sp3_h_nos, size=num_radicals,
                                        replace=False)

    output_ats = [mol]
    for idx in selected_sp3_h_nos:
        # make a new conformer for each molecule/radical I take
        rad = smiles.smi_to_atoms(smi)
        del rad[idx]
        configs.assign_info_entries(rad, config_type='rdkit',
                                    compound=compound, mol_or_rad="rad",
                                    rad_no=idx)
        output_ats.append(rad)
    return output_ats

