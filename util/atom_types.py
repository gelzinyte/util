from ase import neighborlist
import numpy as np
import warnings

def assign_aromatic(inputs, outputs, mult=1):

    bad_configs = []
    for idx, atoms in enumerate(inputs):

        if len(atoms) == 1:
            continue

        natural_cutoffs = neighborlist.natural_cutoffs(atoms,
                                                       mult=mult)
        neighbor_list = neighborlist.NeighborList(natural_cutoffs,
                                                  self_interaction=False,
                                                  bothways=True)
        _ = neighbor_list.update(atoms)

        symbols = np.array(atoms.symbols)
        for at in atoms:

            if at.symbol == 'H':
                h_idx = at.index

                indices, offsets = neighbor_list.get_neighbors(h_idx)
                if len(indices) != 1:
                    bad_configs.append(idx)
                    warnings.warn("Got more than one hydrogen neighbor")
                    continue

                # find index of the atom H is bound to
                h_neighbor_idx = indices[0]

                if symbols[h_neighbor_idx] != 'C':
                    continue

                c_idx = h_neighbor_idx

            elif at.symbol == 'C':
                c_idx = at.index
            else:
                continue

            indices, offsets = neighbor_list.get_neighbors(c_idx)

            no_C_neighbours = len(indices)
            if no_C_neighbours == 4:
                pass

            elif no_C_neighbours == 3:
                if at.symbol == 'H':
                    # symbols[h_idx] = 'HAr'
                    at.symbol = 'He'
                if at.symbol == 'C':
                    # symbols[c_idx] = 'CAr'
                    at.symbol = 'Li'

            else:
                print(at)
                print(atoms.info)
                print(f'conifg index {idx}')
                warnings.warn(f'Got carbon with weird number of '
                                   f'neighbours: {no_C_neighbours}')
                bad_configs.append(idx)
                continue

        # atoms.arrays['atom_types'] = symbols
        outputs.write(atoms)

    print(f'bad_config idx ({len(bad_configs)}):', bad_configs)
    outputs.end_write()




