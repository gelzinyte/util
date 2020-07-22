
import os
from ase.io import read, write
import util
from util import ugap
from util.vib import Vibrations
import subprocess
from quippy.potential import Potential
from copy import deepcopy
from ase.optimize.precon import PreconLBFGS


def get_more_data(source_atoms, iter_no, n_dpoints, n_rattle_atoms, stdev, calc, template_path):
    '''source_atoms - list of atoms to be rattled and added to dataset'''
    atoms = []
    for source_at in source_atoms:
        for _ in range(n_dpoints):
            at = source_at.copy()
            at = util.rattle(at, stdev=stdev, natoms=n_rattle_atoms)
            calc.reset()
            at.set_calculator(calc)
            at.info['dft_energy'] = at.get_potential_energy()
            at.arrays['dft_forces'] = at.get_forces()
            if not util.has_converged(template_path=template_path, molpro_out_path='MOLPRO/molpro.out'):
                raise RuntimeError('Molpro has not converged')
            at.info['config_type'] = f'iter_{iter_no}'
            at.set_cell([20, 20, 20])
            atoms.append(at)
    return atoms


def fit_gap(idx, descriptors, default_sigma):
    train_file = f'xyzs/dset_{idx}.xyz'
    gap_fname = f'gaps/gap_{idx}.xml'
    out_fname = f'gaps/out_{idx}.txt'

    desc = deepcopy(descriptors)

    command = ugap.make_gap_command(gap_filename=gap_fname, training_filename=train_file, descriptors_dict=desc,
                                  default_sigma=default_sigma, output_filename=out_fname, glue_fname='glue_orca.xml')

    print(f'\n-------GAP {idx} command\n')
    print(command)
    out = subprocess.run(command, shell=True)


def optimise_structure(iter_no, atoms, fmax=1e-2, steps=500):
    gap_name = f'gaps/gap_{iter_no}.xml'
    gap = Potential(param_filename=gap_name)
    # copy first guess so that first_guess stays non-optimised.
    guess = atoms.copy()
    guess.set_calculator(gap)
    print(f'Fmax = {fmax}')
    opt = PreconLBFGS(guess, trajectory=f'xyzs/optimisation_{iter_no}.traj')
    converged = opt.run(fmax=fmax, steps=steps)
    return guess



def extend_dset(iter_no, n_dpoints, n_rattle_atoms, stdev, calc, template_path, stride=5):
    # take every stride-th structure from trajectory, and always the last one
    opt_traj_name = f'xyzs/optimisation_{iter_no}'
    traj = read(opt_traj_name+'.traj', ':')
    # skipping first structure, which is just first guess
    source_atoms = traj[stride::stride]
    if len(traj) % stride != 1:
        source_atoms.append(traj[-1])

    more_atoms = get_more_data(source_atoms, iter_no=iter_no+1, n_dpoints=n_dpoints, n_rattle_atoms=n_rattle_atoms, stdev=stdev, calc=calc, template_path=template_path)
    old_dset = read(f'xyzs/dset_{iter_no}.xyz', index=':')
    write(f'xyzs/more_atoms_{iter_no+1}.xyz', more_atoms, 'extxyz', write_results=False)
    new_dset = old_dset + more_atoms
    write(f'xyzs/dset_{iter_no+1}.xyz', new_dset, 'extxyz', write_results=False)

