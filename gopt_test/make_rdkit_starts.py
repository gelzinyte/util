import os
from util import urdkit
import click
from ase.io import read, write
from os.path import join as pj




@click.command()
@click.option('--no_structures', type=int, default=25, show_default=True,
              help='Number of structures per smiles string')
@click.argument('smiles', nargs=-1, required=True)
def generate_starts(no_structures, smiles):
    '''generates structures to be optimised with GAP for test'''

    db_path = '/home/eg475/scripts/gopt_test/'


    for smi in smiles:

        starts_name = pj(db_path, f'starts/rdkit_starts_{smi}.xyz')
        if not os.path.exists(starts_name):
            starts = []
            for idx in range(no_structures):
                at = urdkit.smi_to_xyz(smi, useBasicKnowledge=False, useExpTorsionAnglePrefs=False)
                # at = urdkit.smi_to_xyz(smi)
                at.info['smiles'] = smi
                at.info['config_type'] = f'start_{idx}'
                starts.append(at)

        write(starts_name, starts, 'extxyz', write_results=False)



if __name__ == '__main__':
    generate_starts()