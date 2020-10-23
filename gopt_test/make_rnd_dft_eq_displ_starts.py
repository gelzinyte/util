import os
from util import urdkit
import click
from ase.io import read, write
from os.path import join as pj
import time


@click.command()
@click.option('--no_structures', type=int, default=25, show_default=True,
              help='Number of structures per smiles string')
@click.option('--stds', type=str,  help='list of standard deviations to test')
@click.option('--dft_min_fname', type=click.Path(), help='Name of xyz with named dft minima')
def generate_starts(no_structures, stds, dft_min_fname):
    '''generates structures to be optimised with GAP for test'''

    stds = stds.strip('][').split(', ')
    stds = [float(std) for std in stds]

    db_path = '/home/eg475/programs/my_scripts/gopt_test/'


    dft_min_fname = pj(db_path, 'dft_minima', dft_min_fname)
    dft_min = read(dft_min_fname, ':')

    run_seed = int(time.time()) % (2 ** 8 - 1)

    for dft_idx, dft_at in enumerate(dft_min):

        dft_name = dft_at.info['name']

        for std_idx, std in enumerate(stds):

            starts_name = pj(db_path, f'starts/starts_{dft_name}_{std}A_std.xyz')

            if not os.path.exists(starts_name):
                starts = []
                for idx in range(no_structures):

                    seed = run_seed + dft_idx + std_idx + idx

                    at = dft_at.copy()
                    at.rattle(stdev=std, seed=seed)
                    at.info['config_type'] = f'start_{idx}'
                    starts.append(at)

                print(f'writing {starts_name}')
                write(starts_name, starts, 'extxyz', write_results=False)



if __name__ == '__main__':
    generate_starts()