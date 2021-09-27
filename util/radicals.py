import warnings

import numpy as np
from ase import neighborlist
from wfl.utils.misc import atoms_to_list

from wfl.generate_configs.utils import config_type_append


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
