
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
from util import urdkit
import re
import time


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


def fit_gap(idx, descriptors, default_sigma, config_type_sigma=None):
    train_file = f'xyzs/dset_{idx}.xyz'
    gap_fname = f'gaps/gap_{idx}.xml'
    out_fname = f'gaps/out_{idx}.txt'

    desc = deepcopy(descriptors)

    command = ugap.make_gap_command(gap_filename=gap_fname, training_filename=train_file, descriptors_dict=desc,
                                  default_sigma=default_sigma, output_filename=out_fname, glue_fname='glue_orca.xml',
                                    config_type_sigma=config_type_sigma)

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
           steps=199):
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
        # print('reading:', file)
        # if os.path.isfile(file):
        #     print('file found by os.path.isfile')
        # else:
        #     print('file not found by os.path.isfile')
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


def plot_opt_plot(opt_fnames, prefix=None):
    all_data, shift_by = get_data(opt_fnames)
    print(shift_by)

    fig1 = plt.figure(figsize=(7, 5))
    ax1 = plt.gca()

    fig2 = plt.figure(figsize=(7, 5))
    ax2 = plt.gca()

    for idx, dt in enumerate(all_data):

        label_gap = None
        label_dft = None
        if idx==0:
            label_gap = 'GAP trajectory'
            label_dft = 'DFT re-evaluation'

        ax1.plot(range(len(dt['gap_es'])), dt['gap_es'],
                 label=label_gap)
        ax1.scatter(dt['dft_idx'], dt['dft_es'], marker='x',
                    label=label_dft, )

        ax2.plot(range(len(dt['gap_fmaxs'])), dt['gap_fmaxs'],
                 label=label_gap)
        ax2.scatter(dt['dft_idx'], dt['dft_fmaxs'], marker='x',
                    label=label_dft)

    for ax in [ax1, ax2]:
        # ax.set_yscale('log')
        ax.set_xlabel('optimisation step')
        ax.grid(color='lightgrey')
        ax.legend()
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        formatter = mpl.ticker.LogFormatter(labelOnlyBase=False,
                                            minor_thresholds=(2, 0.4))
        ax.get_yaxis().set_minor_formatter(formatter)

    ax2.set_yscale('log')

    ax1.set_ylabel('energy / eV/atom')
    ax2.set_ylabel('Fmax / eV/A')

    if prefix is None: 
        prefix = 'gap'

    ax1.set_title(f'geometry optimisation of perturbed structures')
    ax2.set_title(f'geometry optimisation of perturbed structures')

    fig1.savefig(f'{prefix}_energy_optg.png', dpi=300)
    fig2.savefig(f'{prefix}_fmax_optg.png', dpi=300)


def gap_optg_test(gap_fname, dft_calc, first_guess='xyzs/first_guess.xyz',
                  no_runs=4, fmax=0.01, dft_stride=5, output_dir='optg_gap_tests', seed_shift=483):
    print("run's random seed:", seed_shift)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    gap_title = os.path.splitext(os.path.basename(gap_fname))[0]
    mol_title = os.path.splitext(os.path.basename(first_guess))[0]
    prefix = os.path.join(output_dir,  gap_title + '_' + mol_title)

    fg = read(first_guess)
    if 'dft_energy' in fg.info.keys():
        del fg.info['dft_energy']
        del fg.arrays['dft_forces']

    for run_idx in range(no_runs):
        print(f'\n---RUN: {run_idx}\n')
        at = fg.copy()
        if run_idx == 0:
            traj_name = f'{output_dir}/opt_{run_idx}_original_fg'
        else:
            traj_name = f'{output_dir}/opt_{run_idx}_rattled_fg'
            at.rattle(stdev=0.1, seed=run_idx + seed_shift)

        if not os.path.isfile(f'{traj_name}.xyz'):
            print(f'\n---optimisation for {traj_name}\n')
            do_opt(at, gap_fname, dft_calc, traj_name, dft_stride, fmax)
        else:
            print(f'found file: {traj_name}')

    fnames = [f'{output_dir}/opt_0_original_fg.xyz']
    fnames += [f'{output_dir}/opt_{idx}_rattled_fg.xyz' for idx in
               range(1, no_runs)]

    plot_opt_plot(fnames, prefix)


def get_no_cores(sub_script='sub.sh'):

    with open(sub_script, 'r') as f:
        for line in f:
            if '-pe smp' in line:
                no_cores = int(re.findall(r'\d+', line)[0])
                return no_cores


def read_args(arg_file='fit_args.txt'):
    arguments = []
    with open(arg_file, 'r') as f:
        for line in f:
            line = line.split(' #')[0]
            arguments.append(line)
    return arguments


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

def get_structure_to_optimise(smi, seed, stdev=0.1):
    urdkit.smi_to_xyz(smi, 'xyzs/mol_3d.xyz')
    at = read('xyzs/mol_3d.xyz')
    if seed is not None:
        at.rattle(stdev=stdev, seed=seed)
    return at

