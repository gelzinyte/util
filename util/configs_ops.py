from ase.io import read, write
import warnings
from ase import neighborlist
import numpy as np
from util import grouper
import os


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


def filter_expanded_geometries(atoms_list):

    atoms_out = []
    skipped_idx = []
    for idx, atoms in enumerate(atoms_list):
        if len(atoms) == 1:
            atoms_out.append(atoms)


        natural_cutoffs = neighborlist.natural_cutoffs(atoms,
                                                       mult=2)
        neighbor_list = neighborlist.NeighborList(natural_cutoffs,
                                                  self_interaction=False,
                                                  bothways=True)
        _ = neighbor_list.update(atoms)

        for at in atoms:
            if at.symbol == 'H':

                indices, offsets = neighbor_list.get_neighbors(at.index)
                if len(indices) == 0:
                    skipped_idx.append(idx)
                    # print(f'skipped {idx} because of atom {at.index}')
                    break

        else:
            atoms_out.append(atoms)
    warnings.warn(f'skipped {len(skipped_idx)} atoms, because couldn\'t find '
                  f'a H whithin reasonable cutoff. Nos: {skipped_idx}')

    return atoms_out





