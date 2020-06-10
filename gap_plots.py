#!/usr/bin/env python3

# Standard imports
import sys, os, subprocess
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl

# My imports
sys.path.append("/home/eg475/reactions")
import util

# Specific imports
import click

# Atomistic imports
from quippy.potential import Potential
from ase.io import read, write

# TODO do logger everywhere
# TODO check if exceptions are correct
# TODO make getting forces optional (??)
# TODO add printing of GAP command somehow
# TODO add printing summary table of accuracies somehow

def get_E_F_dict(atoms, calc_type, param_filename=None):
    # TODO add check for 'dft_energy' or 'energy'
    calc_type = calc_type.upper()
    data = dict()
    data['energy'] = []
    data['forces'] = dict()

    if calc_type == 'DFT':
        if param_filename:
            print("WARNING: calc_type selected as DFT, but gap filename is given, are you sure?")
        energy_name = 'dft_energy'
        if energy_name not in atoms[0].info.keys():
            print("WARNING: 'dft_energy' not found, using 'energy', which might be anything")
            energy_name = 'energy'
        force_name = 'dft_forces'
        if force_name not in atoms[0].arrays.keys():
            print("WARNING: 'dft_forces' not in found, using 'forces', which might be anything")
            force_name = 'forces'

    elif calc_type == 'GAP':
        if param_filename:
            gap = Potential(param_filename=param_filename)
        else:
            print('GAP filename is not given, but GAP energies requested.')
            raise NameError
    else:
        print('calc_type should be either "GAP" or "DFT"')
        raise NameError

    for atom in atoms:
        at = atom.copy()
        if len(at) != 1:
            if calc_type == 'DFT':
                data['energy'].append(at.info[energy_name])
                forces = at.arrays[force_name]

            elif calc_type == 'GAP':
                at.set_calculator(gap)
                data['energy'].append(at.get_potential_energy())
                forces = at.get_forces()


            sym_all = at.get_chemical_symbols()
            for j, sym in enumerate(sym_all):
                try:
                    data['forces'][sym].append(forces[j])
                except KeyError:
                    data['forces'][sym] = []
                    data['forces'][sym].append(forces[j])

    # TODO make it append np array in the loop
    data['energy'] = np.array(data['energy'])
    for key, value in data['forces'].items():
        data['forces'][key] = np.array(value)

    return data


def do_plot(ref_values, pred_values, ax, label):
    rmse = util.get_rmse(ref_values, pred_values)
    std = util.get_std(ref_values, pred_values)

    label = f'{label} {rmse:.3f} $\pm$ {std:.3f}'

    ax.scatter(ref_values, pred_values, label=label, s=8, alpha=0.5)

def make_scatter_plots(param_filename, train_filename, test_filename=None, output_dir=None, prefix=None):

    test_set=False
    if test_filename:
        test_set=True

    train_ats = read(train_filename, index=':')
    train_ref_data = get_E_F_dict(train_ats, calc_type='dft')
    train_pred_data = get_E_F_dict(train_ats, calc_type='gap', param_filename=param_filename)

    if test_set:
        test_ats = read(test_filename, index=':')
        test_ref_data = get_E_F_dict(test_ats, calc_type='dft')
        test_pred_data = get_E_F_dict(test_ats, calc_type='gap', param_filename=param_filename)

    # TODO include automatic selection for figsize
    plt.figure(figsize=(10, 16))
    gs = gridspec.GridSpec(4, 2)
    ax = [plt.subplot(g) for g in gs]

    # Energy plots
    this_ax = ax[0]
    train_ref_es = train_ref_data['energy']
    train_pred_es = train_pred_data['energy']
    do_plot(train_ref_es, train_pred_es, this_ax, 'Training:')
    for_limits = np.concatenate([train_ref_es, train_pred_es])

    if test_set:
        test_ref_es = test_ref_data['energy']
        test_pred_es = test_pred_data['energy']
        do_plot(test_ref_es, test_pred_es, this_ax, 'Test:      ')
        for_limits = np.concatenate([for_limits, test_ref_es, test_pred_es])

    flim = (for_limits.min() - 0.5, for_limits.max() + 0.5)
    this_ax.plot(flim, flim, c='k', linewidth=0.8)
    this_ax.set_xlim(flim)
    this_ax.set_ylim(flim)
    this_ax.set_xlabel('reference energy / eV')
    this_ax.set_ylabel('predicted energy / eV')
    this_ax.set_title('Energies')
    this_ax.legend(title='Set: RMSE $\pm$ STD, eV')

    this_ax = ax[1]
    do_plot(train_ref_es, train_pred_es - train_ref_es, this_ax, 'Training:')
    if test_set:
        do_plot(test_ref_es, test_pred_es - test_ref_es, this_ax, 'Test:      ')
    this_ax.set_xlabel('reference energy / eV')
    this_ax.set_ylabel('E$_{pred}$ - E$_{ref}$ / eV')
    this_ax.axhline(y=0, c='k', linewidth=0.8)
    this_ax.set_title('Energy errors')


    # Force plots
    for idx, sym in enumerate(train_ref_data['forces'].keys()):

        this_ax = ax[2 * (idx + 1)]

        train_ref_fs = train_ref_data['forces'][sym]
        train_pred_fs = train_pred_data['forces'][sym]
        do_plot(train_ref_fs, train_pred_fs, this_ax, 'Training:')
        for_limits = np.concatenate([train_ref_fs, train_pred_fs])

        if test_set:
            test_ref_fs = test_ref_data['forces'][sym]
            test_pred_fs = test_pred_data['forces'][sym]
            do_plot(test_ref_fs, test_pred_fs, this_ax, 'Test:      ')
            for_limits = np.concatenate([for_limits, test_ref_fs, test_pred_fs])


        this_ax.set_xlabel('reference force / eV')
        this_ax.set_ylabel('predicted force / eV')
        flim = (for_limits.min() - 0.5, for_limits.max() + 0.5)
        this_ax.plot(flim, flim, c='k', linewidth=0.8)
        this_ax.set_xlim(flim)
        this_ax.set_ylim(flim)
        this_ax.set_title(f'Forces on {sym}')
        this_ax.legend(title='Set: RMSE $\pm$ STD, eV/Å')

        this_ax = ax[2 * (idx + 1) + 1]
        do_plot(train_ref_fs, train_pred_fs - train_ref_fs, this_ax, 'Training:')
        if test_set:
            do_plot(test_ref_fs, test_pred_fs - test_ref_fs, this_ax, 'Test:      ')
        this_ax.set_xlabel('reference force / eV/Å')
        this_ax.set_ylabel('F$_{pred}$ - F$_{ref}$ / eV/Å')
        this_ax.axhline(y=0, c='k', linewidth=0.8)
        this_ax.set_title(f'Force errors on {sym}')

    if not prefix:
        prefix = os.path.basename(param_filename)
        prefix = os.path.splitext(prefix)[0]
    picture_fname = f'{prefix}_scatter.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.suptitle(prefix)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(picture_fname, dpi=300)

def make_dimer_plot(dimer_name, ax, param_filename):

    # TODO make this more robust
    corr_desc = {'HH': 1, 'CH': 2, 'OH': 3, 'CC': 4, 'CO': 5}

    dimer = read(f'/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz', index=':')
    distances = [at.get_distance(0, 1) for at in dimer]

    ref_data = get_E_F_dict(dimer, calc_type='dft')
    # pred_data = get_E_F_dict(dimer, calc_type='gap', param_filename=param_filename)


    command = f"quip E=T F=T atoms_filename=/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz param_filename={param_filename} calc_args={{only_descriptor={corr_desc[dimer_name]}}} \
                    | grep AT | sed 's/AT//' > ./tmp_atoms.xyz"

    subprocess.run(command, shell=True)
    atoms = read('./tmp_atoms.xyz', index=':')
    os.remove('./tmp_atoms.xyz')
    es = [at.info['energy'] for at in atoms]
    ax.plot(distances, es, label='GAP 2b')

    ax.plot(distances, ref_data['energy'], label='reference', linestyle='--', color='k')
    # plt.plot(distances, pred_data['energy'], label='gap',  linestyle=':')

    ax.set_title(dimer_name)
    ax.set_xlabel('distance (Å)')
    ax.set_ylabel('energy (eV)')
    ax.legend()


def make_2b_plots(param_filename, output_dir=None, prefix=None):

    dimers = ['HH', 'CH', 'CC', 'OH', 'CO']

    plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(3, 2)
    axes = [plt.subplot(g) for g in gs]

    for ax, dimer in zip(axes, dimers):
        make_dimer_plot(dimer, ax, param_filename)

    plt.tight_layout()

    if not prefix:
        prefix = os.path.basename(param_filename)
        prefix = os.path.splitext(prefix)[0]
    picture_fname = f'{prefix}_2body.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)
    plt.savefig(picture_fname, dpi=300)



@click.command()
@click.option('--param_filename',  type=click.Path(exists=True), required=True, help='GAP xml to test')
@click.option('--train_filename',  type=click.Path(exists=True), required=True, help='.xyz file used for training')
@click.option('--test_filename',  type=click.Path(exists=True), help='.xyz file to test GAP on')
@click.option('--output_dir', type=click.Path(), help='directory for figures. Create if not-existent')
@click.option('--prefix', help='prefix to label plots')
def make_plots(param_filename, train_filename, test_filename=None, output_dir=None, prefix=None):
    """Makes energy and force scatter plots"""
    # TODO make optional directory where to save stuff
    # TODO maybe include dftb???
    # TODO add option to include filename
    # TODO get .xyz files from GAP xml file!!

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    make_scatter_plots(param_filename=param_filename, train_filename=train_filename, test_filename=test_filename, \
                       output_dir=output_dir, prefix=prefix)
    make_2b_plots(param_filename=param_filename, output_dir=output_dir, prefix=prefix)



if __name__=='__main__':
    make_plots()
    print('\n\n-----------------CONGRATS YOU MADE IT')
