
import os
from ase.io import read, write
import util
from util import ugap
from util.vib import Vibrations
import subprocess
from quippy.potential import Potential
from copy import deepcopy
from ase.optimize.precon import PreconLBFGS
import matplotlib.pyplot as plt
import matplotlib as mpl
from util import shift0 as sft


def get_more_data(source_atoms, iter_no, n_dpoints, n_rattle_atoms, stdev, calc, template_path=None):
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
            if template_path is not None:
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
    traj = read(f'xyzs/optimisation_{iter_no}.traj', ':')
    write(f'xyzs/optimisation_{iter_no}.xyz', traj, 'extxyz', write_results=False)
    return guess



def extend_dset(iter_no, n_dpoints, n_rattle_atoms, stdev, calc, template_path=None, stride=5):
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


def do_opt(at, gap_fname, dft_calc, traj_name, dft_stride=5, fmax=0.01,
           steps=200):
    gap = Potential(param_filename=gap_fname)

    at.set_calculator(gap)
    opt = PreconLBFGS(at, use_armijo=False, trajectory=f'{traj_name}.traj')
    opt.run(fmax=fmax, steps=steps)
    traj = read(f'{traj_name}.traj', ':')
    for idx, trj_at in enumerate(traj):

        if idx % dft_stride == 0:
            print('dft_step:', idx)
            trj_at.set_calculator(dft_calc)
            try:
                trj_at.info['dft_energy'] = trj_at.get_potential_energy()
                trj_at.arrays['dft_forces'] = trj_at.get_forces()
            except Exception:
                print('couldn\'t get dft energy, skipping')

        trj_at.set_calculator(gap)
        trj_at.info['gap_energy'] = trj_at.get_potential_energy()
        trj_at.arrays['gap_forces'] = trj_at.get_forces()

    print('writing atoms for', traj_name)
    write(f'{traj_name}.xyz', traj, 'extxyz', write_results=False)
    write(f'{traj_name}_at.xyz', at, 'extxyz', write_results=False)


def get_data(opt_fnames):
    all_data = []
    all_es = []
    for file in opt_fnames:
        this_dict = {}
        print('reading:', file)
        if os.path.isfile(file):
            print('file found by os.path.isfile')
        else:
            print('file not found by os.path.isfile')
        ats = read(file, ':')
        this_dict['gap_es'] = [at.info['gap_energy'] / len(at) for at in ats]
        this_dict['gap_fmaxs'] = [max(at.arrays['gap_forces'].flatten()) for
                                  at in ats]
        dft_es = []
        dft_fmaxs = []
        dft_idx = []
        for idx, at in enumerate(ats):
            if 'dft_energy' in at.info.keys():
                dft_idx.append(idx)
                dft_es.append(at.info['dft_energy'] / len(at))
                dft_fmaxs.append(max(at.arrays['dft_forces'].flatten()))
        this_dict['dft_es'] = dft_es
        this_dict['dft_fmaxs'] = dft_fmaxs
        this_dict['dft_idx'] = dft_idx

        all_es += this_dict['gap_es']
        all_es += dft_es

        all_data.append(this_dict)

    shift_by = min(all_es)

    return all_data, shift_by


def plot_opt_plot(opt_fnames):
    all_data, shift_by = get_data(opt_fnames)

    fig1 = plt.figure(figsize=(7, 5))
    ax1 = plt.gca()

    fig2 = plt.figure(figsize=(7, 5))
    ax2 = plt.gca()

    for idx, dt in enumerate(all_data):
        ax1.plot(range(len(dt['gap_es'])), sft(dt['gap_es'], shift_by),
                 label=f'GAP {idx}')
        ax1.scatter(dt['dft_idx'], sft(dt['dft_es'], shift_by), marker='x',
                    label=f'DFT {idx}')

        ax2.plot(range(len(dt['gap_fmaxs'])), dt['gap_fmaxs'],
                 label=f'GAP {idx}')
        ax2.scatter(dt['dft_idx'], dt['dft_fmaxs'], marker='x',
                    label=f'DFT {idx}')

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_xlabel('optimisation step')
        ax.grid(color='lightgrey')
        ax.legend()
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        formatter = mpl.ticker.LogFormatter(labelOnlyBase=False,
                                            minor_thresholds=(2, 0.4))
        ax.get_yaxis().set_minor_formatter(formatter)

    ax1.set_ylabel('energy wrt min E / eV/atom')
    ax2.set_ylabel('Fmax / eV/A')

    fig1.savefig('optg_gap_tests/gap_energy_optg.pdf')
    fig2.savefig('optg_gap_tests/gap_fmax_optg.pdf')


def gap_optg_test(gap_fname, dft_calc, first_guess='xyzs/first_guess.xyz',
                  no_runs=4, fmax=0.01, dft_stride=5):
    if not os.path.isdir('optg_gap_tests'):
        os.makedirs('optg_gap_tests')

    fg = read(first_guess)
    if 'dft_energy' in fg.info.keys():
        del fg.info['dft_energy']
        del fg.arrays['dft_forces']

    for run_idx in range(no_runs):
        print(f'\n---RUN: {run_idx}\n')
        at = fg.copy()
        if run_idx == 0:
            traj_name = f'optg_gap_tests/opt_{run_idx}_original_fg'
        else:
            traj_name = f'optg_gap_tests/opt_{run_idx}_rattled_fg'
            at.rattle(stdev=0.1, seed=run_idx + 592)

        if not os.path.isfile(f'{traj_name}.xyz'):
            print(f'\n---optimisation for {traj_name}\n')
            do_opt(at, gap_fname, dft_calc, traj_name, dft_stride, fmax)
        else:
            print(f'found file: {traj_name}')

    fnames = ['optg_gap_tests/opt_0_original_fg.xyz']
    fnames += [f'optg_gap_tests/opt_{idx}_rattled_fg.xyz' for idx in
               range(1, no_runs)]

    plot_opt_plot(fnames)
