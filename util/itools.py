
import os
from ase.io import read, write
import util
from util import ugap
from util.vib import Vibrations
import subprocess
from quippy.potential import Potential
from ase.optimize.precon import PreconLBFGS




def get_more_data(min_atoms, iter_no, n_dpoints, stdev, calc, template_path):
    atoms = []
    for _ in range(n_dpoints):
        at = min_atoms.copy()
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


def optimise_structure(iter_no, atoms):
    gap_name = f'gaps/gap_{iter_no}.xml'
    gap = Potential(param_filename=gap_name)
    # copy first guess so that first_guess stays non-optimised.
    guess = atoms.copy()
    guess.set_calculator(gap)
    steps = 1000
    fmax = max(10/10**iter_no, 0.01)
    print(f'Fmax = {fmax}')
    opt = PreconLBFGS(guess, trajectory=f'xyzs/optimisation_{iter_no}.traj')
    converged = opt.run(fmax=fmax, steps=steps)
    return guess

def check_NM(opt_atoms, iter_no, molpro_calc, first_guess):
    #TODO implement new NM scheme
    # TODO only half-check for intermediate calcs and do full check at the end
    at = first_guess.copy()
    at.set_positions(opt_atoms.get_positions())
    gap_name = f'gaps/gap_{iter_no}.xml'
    gap = Potential(param_filename=gap_name)
    atoms = at.copy()
    atoms.set_calculator(gap)
    vib_gap = Vibrations(atoms, name=f'gap_iter_{iter_no}')
    if not os.path.isfile(f'gap_iter_{iter_no}.all.pckl'):
        atoms_gap = vib_gap.run()
    atoms = at.copy()
    molpro_calc.reset()
    atoms.set_calculator(molpro_calc)
    vib_dft = Vibrations(atoms, name=f'dft_iter_{iter_no}')
    vib_dft_fname = f'xyzs/dft_iter_{iter_no}_vib.xyz'
    if not os.path.isfile(f'dft_iter_{iter_no}.all.pckl'):
        atoms_dft = vib_dft.run()
        atoms_dft = util.set_dft_vals(atoms_dft)
        write(vib_dft_fname, atoms_dft, 'extxyz', write_results=False)
    gap_plots.make_scatter_plots_from_file(param_fname=gap_name, train_fname=vib_dft_fname, \
                                           output_dir='pictures/', prefix=f'NM{iter_no}')
    util.evec_plot(vib_dft.evals, vib_dft.evecs, vib_gap.evals, vib_gap.evecs, f'iteration_{iter_no}')
    # TODO fix eval_plot
    # util.eval_plot(vib_dft.evals, vib_gap.evals, f'iteration_{iter_no}')


def extend_dset(opt_atoms, iter_no):
    more_atoms = get_more_data(opt_atoms, iter_no=iter_no+1)
    old_dset = read(f'xyzs/dset_{iter_no}.xyz', index=':')
    write(f'xyzs/more_atoms_{iter_no+1}.xyz', more_atoms, 'extxyz', write_results=False)
    new_dset = old_dset + more_atoms
    write(f'xyzs/dset_{iter_no+1}.xyz', new_dset, 'extxyz', write_results=False)

