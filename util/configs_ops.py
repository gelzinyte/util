from ase.io import read, write
from util import grouper


def batch_configs(in_fname, num_tasks, batch_in_fname_prefix='in_'):

    all_atoms = read(in_fname, ':')
    batch_size = int(len(all_atoms) / num_tasks) + 1

    for idx, batch in enumerate(grouper(all_atoms, batch_size)):

        batch = [b for b in batch if b is not None]

        write(f'{batch_in_fname_prefix}{idx}.xyz', batch)


def collect_configs(out_fname, num_tasks, batch_out_fname_prefix='out_'):

    ats_out = []
    for idx in range(num_tasks):
        ats = read(f'{batch_out_fname_prefix}{idx}.xyz', ':')
        ats_out += ats

    write(out_fname, ats_out)

def cleanup_configs(num_tasks=8, batch_in_fname_prefix='in_',
                    batch_out_fname_prefix='out_'):

    for idx in range(num_tasks):

        in_fname = f'{batch_in_fname_prefix}{idx}.xyz'
        if os.path.exists(in_fname):
            os.remove(in_fname)

        out_fname = f'{batch_out_fname_prefix}{idx}.xyz'
        if os.path.exists(out_fname):
            os.remoe(out_fname)
