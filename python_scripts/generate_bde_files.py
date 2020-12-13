import click
from ase import Atoms
from util import urdkit
from util import qm
import util
import os
from ase.io import write



@click.command()
@click.option('--smiles', type=str, help='smiles to generate structures from')
@click.option('--h_list', type=str, help='list of hydrogen indices to remove')
@click.option('--prefix', type=click.Path())
def generate_bde_files(smiles, h_list, prefix):


    h_list = util.str_to_list(h_list, type=int)
    no_cores = int(os.environ['OMP_NUM_THREADS'])

    mol_start = urdkit.smi_to_xyz(smiles)
    mol_start.info['parent_smiles'] = smiles

    mol = mol_start.copy()
    mol.info['config_type'] = 'mol'

    h = Atoms('H', positions=[(0, 0, 0)])
    h.info['config_type'] = 'iso_at'

    to_optimise = [mol, h]
    for h_idx in h_list:
        at = mol_start.copy()
        del at[h_idx]
        at.info['config_type'] = f'rad_{h_idx}'
        to_optimise.append(at)


    write(f'{prefix}_non-optimised.xyz', to_optimise, 'extxyz')

    optimised = qm.orca_par_opt(to_optimise, no_cores)
    print('Optimised with orca, getting dft energies')
    optimised = qm.get_dft_energies(optimised)
    print('Got dft energies, almost there')
    optimised = qm.add_my_decorations(optimised)

    write(f'{prefix}_optimised.xyz', optimised, 'extxyz')




if __name__=='__main__':
    generate_bde_files()


