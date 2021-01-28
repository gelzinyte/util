import click
from ase import Atoms
try:
    from util import urdkit
except ImportError:
    pass
from util import qm
from util.archive import qm as ar_qm
import util
import os
from ase.io import write, read



@click.command()
@click.option('--smiles', type=str, help='smiles to generate structures from')
@click.option('--in_fname', type=click.Path(), help='xyz to read non-optimised structure from')
@click.option('--h_list', type=str, help='list of hydrogen indices to remove')
@click.option('--prefix', type=click.Path())
@click.option('--orca_wdir', default='orca_optimisations')
def generate_bde_files(smiles, in_fname, h_list, prefix, orca_wdir='orca_optimisations'):


    h_list = util.str_to_list(h_list, type=int)
    no_cores = int(os.environ['OMP_NUM_THREADS'])

    if smiles:
        mol_start = urdkit.smi_to_xyz(smiles)
        mol_start.info['parent_smiles'] = smiles
    elif in_fname:
        mol_start = read(in_fname)


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
    at_infos = [{'config_type': at.info["config_type"]} for at in to_optimise]

    optimised = qm.orca_par_opt(to_optimise, no_cores, orca_wdir=orca_wdir)
    print('Optimised with orca, getting dft energies')
    optimised = ar_qm.get_dft_energies(optimised)
    print('Got dft energies, almost there')
    optimised = qm.add_my_decorations(optimised, at_info=at_infos)

    write(f'{prefix}_optimised.xyz', optimised, 'extxyz')




if __name__=='__main__':
    generate_bde_files()


