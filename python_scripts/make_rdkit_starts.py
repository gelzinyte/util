import os
from util import urdkit
import click
from ase.io import read, write
from os.path import join as pj




@click.command()
@click.option('--no_structures', type=int, default=25, show_default=True,
              help='Number of structures per smiles string')
@click.option('--rad_at', type=int, help='Atom number from which to make a radical')
@click.argument('smiles', nargs=-1, required=True)
def generate_starts(no_structures, rad_at, smiles):
    '''generates structures to be optimised with GAP for test'''


    for smi in smiles:

        smi_label = smi
        if rad_at:
            smi_label += f'_rad{rad_at}'
        starts_name = f'rdkit_starts_{smi_label}.xyz'
        if not os.path.exists(starts_name):
            starts = []
            for idx in range(no_structures):
                at = urdkit.smi_to_xyz(smi, useBasicKnowledge=False, useExpTorsionAnglePrefs=False)
                # at = urdkit.smi_to_xyz(smi)
                if rad_at:
                    del at[rad_at]
                at.info['smiles'] = smi_label
                at.info['config_type'] = f'start_{idx}'
                starts.append(at)

        write(starts_name, starts, 'extxyz', write_results=False)



if __name__ == '__main__':
    generate_starts()
