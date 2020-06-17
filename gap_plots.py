#!/usr/bin/env python3

# Standard imports
import sys, os, subprocess
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
from collections import OrderedDict

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
    data['energy'] = OrderedDict()
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
        config_type='no_config_type'
        if 'config_type' in at.info.keys():
           config_type = at.info['config_type']



        if len(at) != 1:
            if calc_type == 'DFT':
                try:
                    data['energy'][config_type].append(at.info[energy_name])
                except KeyError:
                    data['energy'][config_type] = []
                    data['energy'][config_type].append(at.info[energy_name])

                forces = at.arrays[force_name]

            elif calc_type == 'GAP':
                at.set_calculator(gap)
                try:
                    data['energy'][config_type].append(at.get_potential_energy())
                except KeyError:
                    data['energy'][config_type] = []
                    data['energy'][config_type].append(at.get_potential_energy())

                forces = at.get_forces()


            sym_all = at.get_chemical_symbols()
            for j, sym in enumerate(sym_all):
                if sym not in data['forces'].keys():
                    data['forces'][sym] = OrderedDict()
                try:
                    data['forces'][sym][config_type].append(forces[j])
                except KeyError:
                    data['forces'][sym][config_type] = []
                    data['forces'][sym][config_type].append(forces[j])

    # TODO make it append np array in the loop
    for config_type, values in data['energy'].items():
        data['energy'][config_type] = np.array(values)
    for sym in data['forces'].keys():
        for config_type, values in data['forces'][sym].items():
            data['forces'][sym][config_type] = np.array(values)


    return data

def dict_to_vals(my_dict):
    all_values = []
    for type, values in my_dict.items():
        all_values.append(values)
    all_values = np.concatenate(all_values)
    return all_values

def do_plot(ref_values, pred_values, ax, label, by_config_type=False):

    if not by_config_type:
        ref_vals = dict_to_vals(ref_values)
        pred_vals = dict_to_vals(pred_values)

        rmse = util.get_rmse(ref_vals, pred_vals)
        std = util.get_std(ref_vals, pred_vals)
        # TODO make formatting nicer
        print_label = f'{label}: {rmse:.3f} $\pm$ {std:.3f}'
        ax.scatter(ref_vals, pred_vals, label=print_label, s=8, alpha=0.7)

    else:
        n_colours = len(ref_values.keys())
        cmap = mpl.cm.get_cmap('plasma')
        colour_idx = np.linspace(0, 1, n_colours)
        # TODO maybe colour by eigenvalue?
        for ref_config_type, pred_config_type, cidx in zip(ref_values.keys(), pred_values.keys(), colour_idx):
            if ref_config_type != pred_config_type:
                raise ValueError('Reference and predicted config_types do not match')
            ref_vals = ref_values[ref_config_type]
            pred_vals = pred_values[pred_config_type]

            rmse = util.get_rmse(ref_vals, pred_vals)
            std = util.get_std(ref_vals, pred_vals)
            # print_label = f'{label}, config type {ref_config_type}, {rmse:.3f} $\pm$ {std:.3f}'
            print_label = f'{ref_config_type}: {rmse:.3f} $\pm$ {std:.3f}'
            # print_label = ref_config_type
            # if cidx != 1.0 and cidx != 0.0:
            #     print_label = None
            # ax.scatter(ref_vals, pred_vals, label=print_label, s=8, alpha=0.7, color=cmap(cidx))
            ax.scatter(ref_vals, pred_vals, label=print_label, s=8, alpha=0.7)


def error_dict(pred, ref):
    errors = OrderedDict()
    for pred_type, ref_type in zip(pred.keys(), ref.keys()):
        if pred_type != ref_type:
            raise ValueError('Reference and predicted config_types do not match')

        errors[pred_type] = pred[pred_type] - ref[ref_type]

    return errors

def make_scatter_plots_from_file(param_filename, train_filename, test_filename=None, output_dir=None, prefix=None, \
                                 by_config_type=False):

    train_ats = read(train_filename, index=':')
    test_ats = None
    if test_filename:
        test_ats = read(test_filename, index=':')

    make_scatter_plots(param_filename, train_ats, test_ats=test_ats, output_dir=output_dir, prefix=prefix, \
                       by_config_type=by_config_type)


def make_scatter_plots(param_filename, train_ats, test_ats=None, output_dir=None, prefix=None, by_config_type=False):

    test_set=False
    if test_ats:
        test_set=True

    train_ref_data = get_E_F_dict(train_ats, calc_type='dft')
    train_pred_data = get_E_F_dict(train_ats, calc_type='gap', param_filename=param_filename)

    if test_set:
        test_ref_data = get_E_F_dict(test_ats, calc_type='dft')
        test_pred_data = get_E_F_dict(test_ats, calc_type='gap', param_filename=param_filename)


    counts = util.get_counts(train_ats[0])
    no_unique_elements = len(counts.keys())
    width = 10
    height = width * 0.6
    height *= no_unique_elements

    plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(no_unique_elements+1, 2)
    ax = [plt.subplot(g) for g in gs]

    # Energy plots
    this_ax = ax[0]
    train_ref_es = train_ref_data['energy']
    train_pred_es = train_pred_data['energy']
    do_plot(train_ref_es, train_pred_es, this_ax, 'Training', by_config_type)
    for_limits = np.concatenate([dict_to_vals(train_ref_es), dict_to_vals(train_pred_es)])

    if test_set:
        test_ref_es = test_ref_data['energy']
        test_pred_es = test_pred_data['energy']
        do_plot(test_ref_es, test_pred_es, this_ax, 'Test', by_config_type)
        for_limits = np.concatenate([for_limits, dict_to_vals(test_ref_es), dict_to_vals(test_pred_es)])

    flim = (for_limits.min() - 0.5, for_limits.max() + 0.5)
    this_ax.plot(flim, flim, c='k', linewidth=0.8)
    this_ax.set_xlim(flim)
    this_ax.set_ylim(flim)
    this_ax.set_xlabel('reference energy / eV')
    this_ax.set_ylabel('predicted energy / eV')
    this_ax.set_title('Energies')
    this_ax.legend(title='Set: RMSE $\pm$ STD, eV')


    this_ax = ax[1]
    # do_plot(train_ref_es, train_pred_es - train_ref_es, this_ax, 'Training:')
    do_plot(train_ref_es, error_dict(train_pred_es, train_ref_es), this_ax, 'Training', by_config_type)
    if test_set:
        # do_plot(test_ref_es, test_pred_es - test_ref_es, this_ax, 'Test:      ')
        do_plot(test_ref_es, error_dict(test_pred_es, test_ref_es), this_ax, 'Test', by_config_type)
    this_ax.set_xlabel('reference energy / eV')
    this_ax.set_ylabel('E$_{pred}$ - E$_{ref}$ / eV')
    this_ax.axhline(y=0, c='k', linewidth=0.8)
    this_ax.set_title('Energy errors')



    # Force plots
    for idx, sym in enumerate(train_ref_data['forces'].keys()):

        this_ax = ax[2 * (idx + 1)]

        train_ref_fs = train_ref_data['forces'][sym]
        train_pred_fs = train_pred_data['forces'][sym]
        do_plot(train_ref_fs, train_pred_fs, this_ax, 'Training', by_config_type)
        for_limits = np.concatenate([dict_to_vals(train_ref_fs), dict_to_vals(train_pred_fs)])

        if test_set:
            test_ref_fs = test_ref_data['forces'][sym]
            test_pred_fs = test_pred_data['forces'][sym]
            do_plot(test_ref_fs, test_pred_fs, this_ax, 'Test', by_config_type)
            for_limits = np.concatenate([for_limits, dict_to_vals(test_ref_fs), dict_to_vals(test_pred_fs)])


        this_ax.set_xlabel('reference force / eV')
        this_ax.set_ylabel('predicted force / eV')
        flim = (for_limits.min() - 0.5, for_limits.max() + 0.5)
        this_ax.plot(flim, flim, c='k', linewidth=0.8)
        this_ax.set_xlim(flim)
        this_ax.set_ylim(flim)
        this_ax.set_title(f'Forces on {sym}')
        this_ax.legend(title='Set: RMSE $\pm$ STD, eV/Å')


        this_ax = ax[2 * (idx + 1) + 1]
        # do_plot(train_ref_fs, train_pred_fs - train_ref_fs, this_ax, 'Training:')
        do_plot(train_ref_fs, error_dict(train_pred_fs, train_ref_fs), this_ax, 'Training', by_config_type)
        if test_set:
            # do_plot(test_ref_fs, test_pred_fs - test_ref_fs, this_ax, 'Test:      ')
            do_plot(test_ref_fs, error_dict(test_pred_fs, test_ref_fs), this_ax, 'Test', by_config_type)
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
    pred_data = get_E_F_dict(dimer, calc_type='gap', param_filename=param_filename)


    command = f"quip E=T F=T atoms_filename=/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz param_filename={param_filename} calc_args={{only_descriptor={corr_desc[dimer_name]}}} \
                    | grep AT | sed 's/AT//' > ./tmp_atoms.xyz"

    subprocess.run(command, shell=True)
    atoms = read('./tmp_atoms.xyz', index=':')
    os.remove('./tmp_atoms.xyz')
    es = [at.info['energy'] for at in atoms]
    ax.plot(distances, es, label='GAP 2b')

    ax.plot(distances, dict_to_vals(ref_data['energy']), label='reference', linestyle='--', color='k')
    plt.plot(distances, dict_to_vals(pred_data['energy']), label='gap')

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
@click.option('--by_config_type', type=bool, help='if structures should be coloured by config_type in plots')
def make_plots(param_filename, train_filename, test_filename=None, output_dir=None, prefix=None, by_config_type=False):
    """Makes energy and force scatter plots"""
    # TODO make optional directory where to save stuff
    # TODO maybe include dftb???
    # TODO add option to include filename
    # TODO get .xyz files from GAP xml file!!

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    make_scatter_plots_from_file(param_filename=param_filename, train_filename=train_filename, test_filename=test_filename, \
                       output_dir=output_dir, prefix=prefix, by_config_type=by_config_type)
    make_2b_plots(param_filename=param_filename, output_dir=output_dir, prefix=prefix)



if __name__=='__main__':
    make_plots()
    print('\n\n-----------------CONGRATS YOU MADE IT')

