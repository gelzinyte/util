import os
from util import urdkit
import click
from ase.io import read, write
from os.path import join as pj
from util import itools
import time


@click.command()
@click.option('--no_dpoints', type=int, default=25, show_default=True,
              help='Number of structures per smiles string')
@click.option('--stds', type=str,  help='list of standard deviations to test')
@click.option('--in_fname', type=click.Path(), help='input xyzs which to rattle')
@click.option('--prefix', type=str, help='prefix for the file to save structures to')
@click.option('--config_type', type=str, help='config_type to add to (all) of the atoms')
def generate_starts(no_dpoints, stds, in_fname, config_type, prefix):
    '''generates structures to be optimised with GAP for test'''

    stds = stds.strip('][').split(', ')
    stds = [float(std) for std in stds]

    atoms_list = read(in_fname, ':')
    run_seed = int(time.time()) % (2 ** 8 - 1)

    for std_idx, std in enumerate(stds):
        output_fname = f'{prefix}_{std}A_std.xyz'
        if not os.path.exists(output_fname):
            atoms_out = []
            for dft_idx, ref_at in enumerate(atoms_list):

                for idx in range(no_dpoints):

                    seed = run_seed + dft_idx + std_idx + idx
                    at = ref_at.copy()
                    at.rattle(stdev=std, seed=seed)
                    atoms_out.append(at)

            print('Calculating DFT energies and forces')
            atoms_out = itools.get_dft_energies(atoms_out)
            
            at_info = {'config_type':config_type}
            atoms_out = itools.add_my_decorations(atoms_out, at_info)

            print(f'writing {output_fname}')
            write(output_fname, atoms_out, 'extxyz', write_results=False)


if __name__ == '__main__':
    generate_starts()