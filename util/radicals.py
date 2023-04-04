import warnings
import logging

import numpy as np
from ase import neighborlist
from wfl.utils.misc import atoms_to_list

from wfl.generate.utils import config_type_append

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



def abstract_sp3_hydrogen_atoms(inputs, outputs, label_config_type=True,
                                return_mol=True):
    """ Removes molecules' sp3 hydrogen atoms one at a time to give a number
    of corresponding unsaturated (radical) structures.

    Method of determining sp3: carbon has 4 neighbors of any kind. Only
    removes from Carbon atoms.

    Parameters
    ----------

    inputs: Atoms / list(Atoms) / ConfigSet
        structure to remove sp3 hydrogen atoms from
    outputs: OutputSpec:
        where to write structures to
    label_config_type: bool, default True
        whether to append config_type with 'mol' or 'rad{idx}'
    return_mol: bool, default True
        returns the original molecule too

    Returns
    -------

    """
    if outputs.all_written():
        print('found file with abstracted sp3 hydrogens, not repeated')
        return outputs.to_ConfigSet()
    all_output_atoms = []
    for atoms in atoms_to_list(inputs):

        sp3_hs = get_sp3_h_numbers(atoms)

        radicals = []
        for h_idx in sp3_hs:
            at = atoms.copy()
            del at[h_idx]
            configs.assign_info_entries(at, mol_or_rad="rad", rad_no=h_idx, compound=at.info["compound"])
            radicals.append(at)

        if label_config_type:

            atoms_out = []

            if return_mol:
                mol = atoms.copy()
                # configs.assign_info_entries(mol, "mol", "mol", config_type=None,
                # compound=None)
                # mol.info['compound'] = mol.info['config_type']
                # config_type_append(mol, 'mol')
                # mol.info['mol_or_rad'] = 'mol'
                atoms_out.append(mol)

            for rad, h_id in zip(radicals, sp3_hs):
                # configs.assign_info_entires(rad, "rad", h_id)
                # rad.info['compound'] = rad.info['config_type']
                # config_type_append(rad, f'rad{h_id}')
                # rad.info['mol_or_rad'] = f'rad{h_id}'
                atoms_out.append(rad)

        else:
            if return_mol:
                atoms_out = [atoms.copy()] + radicals
            else:
                atoms_out = radicals
        
        outputs.store(atoms_out)
    outputs.close()
    return outputs.to_ConfigSet() 

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
    elif num_radicals == 0:
        selected_sp3_h_nos = []

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





def generate_radicals_from_optimsied_molecules(ci, co, number_of_radicals, copy_mol=True):

    if co.is_done():
        logger.info("returning because outputs are done")
        return co.to_ConfigSet()

    orig_num_or_radicals = number_of_radicals

    for at in ci:
        if copy_mol:
            # save molecule
            at = util.remove_energy_force_containing_entries(at)
            co.write(at)

        #make a radical
        rad = at.copy()
        comp = rad.info["compound"]
        sp3_Hs = radicals.get_sp3_h_numbers(rad.copy())
        if orig_num_or_radicals > len(sp3_Hs):
            logger.warning(f"Asking for more radicals ({number_of_radicals}) than there are sp3 hydrogens ({len(sp3_Hs)}), returning all of radicals ({len(sp3_Hs)})for {at.info}")
            number_of_radicals = len(sp3_Hs)
        else:
            number_of_radicals = orig_num_or_radicals

        # print(f'len(sp3_Hs): {len(sp3_Hs)}; num_rads: {number_of_radicals}')
        
        all_H_to_remove = random.sample(sp3_Hs, number_of_radicals)

        for h_to_remove in all_H_to_remove: 
            atoms = rad.copy()
            del atoms[h_to_remove]

            atoms.info["mol_or_rad"] = "rad"
            atoms.info["rad_num"] = h_to_remove
            atoms.info["graph_name"] = str(comp) + '_rad' + str(h_to_remove)

            co.write(atoms)

    co.end_write()
    return co.to_ConfigSet()
