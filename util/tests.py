from util import itools
import shutil
import os
from ase.io import read, write
import matplotlib.pyplot as plt
import matplotlib as mpl
from util import urdkit
from util import qm
import util
import subprocess
from quippy.potential import Potential
from ase.optimize.precon import PreconLBFGS
import numpy as np

def make_structures_to_optimise(smiles, no_runs, output_dir,  seed_shift):
    '''prepares all of the structures that are later optimised with GAP and DFT'''
    names = []
    for idx in range(no_runs):
        name = os.path.join(output_dir,  f'start_at_{idx}.xyz')
        if not os.path.isfile(name):
            at = itools.get_structure_to_optimise(smi=smiles, seed=seed_shift+idx)
            write(name, at, 'extxyz')
        else:
            print(f'found structure {name}, not creating')


def dft_optimise_start_ats(no_runs, output_dir, molpro_wdir):
    '''makes molpro optg templates and submits them'''
    mp_command = '/opt/molpro/bin/molpro'
    source_template = '../../template_molpro.txt'

    orig_dir = os.getcwd()
    os.chdir(output_dir)

    if not os.path.isdir(molpro_wdir):
        os.makedirs(molpro_wdir)
    os.chdir(molpro_wdir)

    for idx in range(no_runs):
        optg_xyz_output_fname = f'../dft_opt_at_{idx}.xyz'
        if not os.path.isfile(optg_xyz_output_fname):
            at_name = f'start_at_{idx}.xyz'
            # print(f'Making optg {idx} template')
            optg_template_fname = f'optg_template_{idx}.txt'
            xyz_input_fname = os.path.join('../', at_name)

            qm.make_optg_template(source_template=source_template,
                                  optg_template=optg_template_fname,
                                  input_fname=xyz_input_fname,
                                  output_fname=optg_xyz_output_fname)

            print(f'writing submission script {idx}')
            submission_script_fname = f'dft_geo_opt_{idx}.sh'
            job_name = f'dft_optg_{idx}'
            output_fname =  f'opt_{idx}.out'
            command = f'{mp_command} {optg_template_fname} -o {output_fname}'

            util.write_generic_submission_script(
                script_fname=submission_script_fname, job_name=job_name,
                command=command)

            subprocess.run(f'qsub {submission_script_fname}', shell=True)
        else:
            print(f'found structure {optg_xyz_output_fname}, not optimising')

    os.chdir(orig_dir)

def check_if_needed(no_runs):

    for idx in range(no_runs):
        if os.path.isfile(f'opt_traj_{idx}.xyz') and os.path.isfile(f'opt_traj_{idx}_dft.xyz'):
            continue
        else:
            return True

    return False

def submit_gap_geo_opt_test(gap_no, no_runs, fmax, steps, dft_stride, no_cores):
    '''if needed, submits the test, otherwise just plots the plots. TODO: change name'''

    is_needed = check_if_needed(no_runs)

    if is_needed:
        job_name = f'gopt_gap{gap_no}'
        command = f'python ../test_gopt_main.py {gap_no} {no_runs} {fmax} {steps} {dft_stride} {no_cores}'
        script_fname = 'gap_gopt_main.sh'
        util.write_generic_submission_script(script_fname=script_fname, job_name=job_name,
                                            command=command, no_cores=no_cores)

        subprocess.run(f'qsub {script_fname}', shell=True)

    else:
        print(f'Found all {no_runs} gap trajectory and dft re-evaluation files, just re-plotting figure')
        plot_gap_optg_results(gap_no, no_runs)




def gap_optg_test(gap_no, mp_template,  no_runs, fmax, steps, dft_stride, no_cores):
    '''main function to do multiple gap optimisation runs and re-evaluates them. '''

    gap_fname = f'../gaps/gap_{gap_no}.xml'


    for run_idx in range(no_runs):
        print(f'\n---RUN: {run_idx}\n')

        at = read(f'start_at_{run_idx}.xyz')
        traj_name = f'opt_traj_{run_idx}'

        if not os.path.isfile(f'{traj_name}.xyz'):
            print(f'\n---optimisation for {traj_name}\n')
            do_gap_optimisation(at, gap_fname=gap_fname, traj_name=traj_name, fmax=fmax, steps=steps)
        else:
            print(f'found gap trajectory: {traj_name}.xyz')

        if not os.path.isfile(f'{traj_name}_dft.xyz'):
            do_dft_reevaluation(traj_name, dft_stride, mp_template, no_cores)
        else:
            print(f'found dft reevaluation of trajectory: {traj_name}_dft.xyz')

    plot_gap_optg_results(gap_no, no_runs)



def do_dft_reevaluation(traj_name, dft_stride, mp_template, no_cores):
    '''given gap-optimised trajectory, selects every dft_stride-th structure,
    recalculates energies/forces with molpro parallel-y and saves in a separate file'''

    traj = read(f'{traj_name}.xyz', ':')
    dft_atoms = []
    dft_at_idx = []
    for idx, at in enumerate(traj):
        if idx % dft_stride == 0:
            positions = at.arrays['positions']
            numbers = at.arrays['numbers']

            at.info.clear()
            at.arrays.clear()
            at.arrays['positions'] = positions
            at.arrays['numbers'] = numbers

            dft_atoms.append(at)
            dft_at_idx.append(idx)

    dft_out_atoms = qm.get_parallel_molpro_energies_forces(atoms=dft_atoms,
        no_cores=no_cores, mp_template=mp_template)

    for idx, dft_in_at, dft_out_at in zip(dft_at_idx, dft_atoms, dft_out_atoms):
        if not (dft_in_at.positions == dft_out_at.positions).all():
            write('mixup_in.xyz', dft_in_at)
            write('mixup_out.xyz', dft_out_at)
            print(f'dft_in_at.positions:\n{dft_in_at.positions}')
            print(f'dft_out_at.positions:\n{dft_out_at.positions}')
            print(f'comparison result: {not (dft_in_at.positions == dft_out_at.positions).all()}')
            raise RuntimeError('mp parallel in and out atom positions do not match, somethings mixed up')

        dft_out_at.info['gap_optg_step'] = idx
        dft_out_at.info['dft_energy'] = dft_out_at.info['energy']
        dft_out_at.arrays['dft_forces'] = dft_out_at.arrays['forces']

    write(f'{traj_name}_dft.xyz', dft_out_atoms, 'extxyz', write_results=False)


def do_gap_optimisation(at, traj_name, fmax, steps, gap_fname=None, gap=None):

    if gap is None:
        if gap_fname is not None:
            raise Exception('Need to give either a gap potential or filename, not both')
        gap = Potential(param_filename=gap_fname)

    at.set_calculator(gap)
    opt = PreconLBFGS(at, use_armijo=False, trajectory=f'{traj_name}.traj')
    opt.run(fmax=fmax, steps=steps)
    traj = read(f'{traj_name}.traj', ':')

    for idx, trj_at in enumerate(traj):

        trj_at.set_calculator(gap)
        trj_at.info['gap_energy'] = trj_at.get_potential_energy()
        trj_at.arrays['gap_forces'] = trj_at.get_forces()

    print('writing atoms for', traj_name)
    write(f'{traj_name}.xyz', traj, 'extxyz', write_results=False)


def plot_gap_optg_results(gap_no, no_runs):

    colors = np.arange(10)
    cmap = mpl.cm.get_cmap('tab10')

    fig1 = plt.figure(figsize=(12, 7))
    ax1 = plt.gca()

    fig2 = plt.figure(figsize=(12, 7))
    ax2 = plt.gca()

    for idx in range(no_runs):

        dt = get_data(idx)

        # label_gap = None
        # label_dft = None
        # if idx==0:
        #     label_gap = 'GAP trajectory'
        #     label_dft = 'DFT re-evaluation'
        label_gap = f'gap {idx}'
        label_dft = f'dft {idx}'

        c = cmap(colors[idx])
        lw = 0.5

        ax1.plot(dt['idx_gap'], dt['es_gap'], label=label_gap, color=c)
        ax1.plot(dt['idx_dft'], dt['es_dft'], marker='o', label=label_dft, color=c, linewidth=lw)

        ax2.plot(dt['idx_gap'], dt['fmax_gap'], label=label_gap, color=c)
        ax2.plot(dt['idx_dft'], dt['fmax_dft'], marker='o', label=label_dft, color=c, linewidth=lw)


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


    ax1.set_title(f'geometry optimisation of perturbed structures')
    ax2.set_title(f'geometry optimisation of perturbed structures')

    fig1.savefig(f'gap_{gap_no}_energy_optg.png', dpi=300)
    fig2.savefig(f'gap_{gap_no}_fmax_optg.png', dpi=300)


def get_data(idx):
    '''from gap and dft-reevaluated trajectories extract energies and max force
    components to be plotted'''


    data = {}

    traj_gap = read(f'opt_traj_{idx}.xyz', ':')
    traj_dft = read(f'opt_traj_{idx}_dft.xyz', ':')

    es_gap = [at.info['gap_energy']/len(at) for at in traj_gap]
    idx_gap = np.arange(len(traj_gap))
    fmax_gap = [max(at.arrays['gap_forces'].flatten()) for at in traj_gap]

    es_dft = [at.info['dft_energy']/len(at) for at in traj_dft]
    idx_dft = [at.info['gap_optg_step'] for at in traj_dft]
    fmax_dft = [max(at.arrays['dft_forces'].flatten()) for at in traj_dft]

    data['es_gap'] = es_gap
    data['idx_gap'] = idx_gap
    data['fmax_gap'] = fmax_gap
    data['es_dft'] = es_dft
    data['idx_dft'] = idx_dft
    data['fmax_dft'] = fmax_dft

    return data







