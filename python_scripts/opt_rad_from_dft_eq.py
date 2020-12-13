import click
from ase import Atoms
from util import urdkit
from util import qm
import util
import os
from ase.io import write, read
import opt_with_orca as opt



@click.command()
@click.option('--in_fname', type=str, help='dft equilibrium geometry')
@click.option('--h_list', type=str, help='list of hydrogen indices to remove')
@click.option('--prefix', type=click.Path())
def optimise_radicals(in_fname, h_list, prefix):


    h_list = util.str_to_list(h_list, type=int)

    mol_start = read(in_fname)
    no_cores = int(os.environ["OMP_NUM_THREADS"])


    mol = mol_start.copy()
    mol.info['config_type'] = 'mol'

    h = Atoms('H', positions=[(0, 0, 0)])
    h.info['config_type'] = 'iso_at'

    to_optimise = []
    for h_idx in h_list:
        at = mol_start.copy()
        del at[h_idx]
        at.info['config_type'] = f'rad{h_idx}'
        to_optimise.append(at)


    non_opt = f'{prefix}_non_optimised_rads.xyz'
    write(non_opt, to_optimise, 'extxyz')

    opt_trajs = qm.orca_par_opt(to_optimise, no_cores, at_or_traj='traj')
    print('Optimised with orca, getting dft energies')
    opt_ats = []
    for h_idx, traj in zip(h_list, opt_trajs):
        print(f'calculating dft energies for trajectory {h_idx}')
        traj_name = f'{prefix}_rad{h_idx}.xyz'
        traj = qm.get_dft_energies(traj)
        traj = qm.add_my_decorations(traj, at_info={'config_type':f'{prefix}_rad{h_idx}'})
        write(traj_name, traj)
        opt_ats.append(traj[-1])
    write(f'{prefix}_opt_rads.xyz', opt_ats, 'extxyz')

    all_trajs = []
    for h_idx in h_list:
        traj = read(f'{prefix}_rad{h_idx}.xyz', ':')
        all_trajs += traj
    write(f'{prefix}_all_trajectories.xyz', all_trajs)








if __name__=='__main__':
    optimise_radicals()


