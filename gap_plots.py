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
import util

# Specific imports
import click
from tqdm import tqdm


# Atomistic imports
from quippy.potential import Potential
from ase.io import read, write
from ase import Atom, Atoms

# TODO do logger everywhere
# TODO check if exceptions are correct
# TODO make getting forces optional (??)
# TODO add printing of GAP command somehow
# TODO add printing summary table of accuracies somehow

def get_E_F_dict(atoms, calc_type, param_fname=None):
    # TODO add check for 'dft_energy' or 'energy'
    # calc_type = calc_type.upper()
    data = dict()
    data['energy'] = OrderedDict()
    data['forces'] = OrderedDict()

    if calc_type.upper() == 'GAP':
        if param_fname:
            gap = Potential(param_filename=param_fname)
        else:
            raise NameError('GAP filename is not given, but GAP energies requested.')
    # else:
    #     raise NameError('calc_type should be either "GAP" or "DFT"')

    else: # calc_type == 'DFT':
        if param_fname:
            print("WARNING: calc_type selected as DFT, but gap filename is given, are you sure?")
        energy_name = f'{calc_type}_energy'
        if energy_name not in atoms[0].info.keys():
            print(f"WARNING: '{calc_type}_energy' not found, using 'energy', which might be anything")
            energy_name = 'energy'
        force_name = f'{calc_type}_forces'
        if force_name not in atoms[0].arrays.keys():
            print("WARNING: 'dft_forces' not in found, using 'forces', which might be anything")
            force_name = 'forces'



    for atom in atoms:
        at = atom.copy()
        config_type='no_config_type'
        if 'config_type' in at.info.keys():
           config_type = at.info['config_type']


        if len(at) != 1:

            if calc_type.upper() == 'GAP':
                at.set_calculator(gap)
                try:
                    data['energy'][config_type].append(at.get_potential_energy())
                except KeyError:
                    data['energy'][config_type] = []
                    data['energy'][config_type].append(at.get_potential_energy())

                forces = at.get_forces()
            else:
                try:
                    data['energy'][config_type].append(at.info[energy_name])
                except KeyError:
                    data['energy'][config_type] = []
                    data['energy'][config_type].append(at.info[energy_name])

                forces = at.arrays[force_name]



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
        n_groups = len(ref_values.keys())

        colors = np.arange(10)
        cmap = mpl.cm.get_cmap('tab10')

        for ref_config_type, pred_config_type, idx in zip(ref_values.keys(), pred_values.keys(), range(n_groups)):
            if ref_config_type != pred_config_type:
                raise ValueError('Reference and predicted config_types do not match')
            ref_vals = ref_values[ref_config_type]
            pred_vals = pred_values[pred_config_type]

            rmse = util.get_rmse(ref_vals, pred_vals)
            std = util.get_std(ref_vals, pred_vals)
            print_label = f'{ref_config_type}: {rmse:.3f} $\pm$ {std:.3f}'
            if idx < 10:
                kws = {'marker': 'o', 'facecolors': 'none', 'edgecolors': cmap(colors[idx % 10])}
            elif idx < 20 and idx >=10:
                kws = {'marker': 'x', 'facecolors': cmap(colors[idx % 10])}
            elif idx < 30 and idx >=20:
                kws = {'marker': '+', 'facecolors': cmap(colors[idx % 10])}
            else:
                kws = {'marker': '*', 'facecolors': 'none', 'edgecolors': cmap(colors[idx % 10])}

            ax.scatter(ref_vals, pred_vals, label=print_label, linewidth=0.9, **kws)


def error_dict(pred, ref):
    errors = OrderedDict()
    for pred_type, ref_type in zip(pred.keys(), ref.keys()):
        if pred_type != ref_type:
            raise ValueError('Reference and predicted config_types do not match')

        errors[pred_type] = pred[pred_type] - ref[ref_type]

    return errors

def make_scatter_plots_from_file(param_fname, train_fname, test_fname=None, output_dir=None, prefix=None, \
                                 by_config_type=False, ref_name='dft'):

    train_ats = read(train_fname, index=':')
    test_ats = None
    if test_fname:
        test_ats = read(test_fname, index=':')

    make_scatter_plots(param_fname, train_ats, test_ats=test_ats, output_dir=output_dir, prefix=prefix, \
                       by_config_type=by_config_type, ref_name=ref_name)


def make_scatter_plots(param_fname, train_ats, test_ats=None, output_dir=None, prefix=None, by_config_type=False, ref_name='dft'):

    test_set=False
    if test_ats:
        test_set=True

    train_ref_data = get_E_F_dict(train_ats, calc_type=ref_name)
    train_pred_data = get_E_F_dict(train_ats, calc_type='gap', param_fname=param_fname)

    if test_set:
        test_ref_data = get_E_F_dict(test_ats, calc_type=ref_name)
        test_pred_data = get_E_F_dict(test_ats, calc_type='gap', param_fname=param_fname)


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
    this_ax.set_xlabel(f'{ref_name.upper()} energy / eV')
    this_ax.set_ylabel(f'GAP energy / eV')
    this_ax.set_title('Energies')
    lgd = this_ax.legend(title='Set: RMSE $\pm$ STD, eV', bbox_to_anchor=(2.9, 1.05))


    this_ax = ax[1]
    do_plot(train_ref_es, error_dict(train_pred_es, train_ref_es), this_ax, 'Training', by_config_type)
    if test_set:
        # do_plot(test_ref_es, test_pred_es - test_ref_es, this_ax, 'Test:      ')
        do_plot(test_ref_es, error_dict(test_pred_es, test_ref_es), this_ax, 'Test', by_config_type)
    this_ax.set_xlabel(f'{ref_name.upper()} energy / eV')
    this_ax.set_ylabel(f'E$_{{GAP}}$ - E$_{{{ref_name.upper()}}}$ / eV')
    this_ax.axhline(y=0, c='k', linewidth=0.8)
    this_ax.set_title('Energy errors')
    # lgd = this_ax.legend(title='Set: RMSE $\pm$ STD, eV', bbox_to_anchor=(1.1, 1.05))



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


        this_ax.set_xlabel(f'{ref_name.upper()} force / eV/Å')
        this_ax.set_ylabel('GAP force / eV/Å')
        flim = (for_limits.min() - 0.5, for_limits.max() + 0.5)
        this_ax.plot(flim, flim, c='k', linewidth=0.8)
        this_ax.set_xlim(flim)
        this_ax.set_ylim(flim)
        this_ax.set_title(f'Force components on {sym}')
        this_ax.legend(title='Set: RMSE $\pm$ STD, eV/Å', bbox_to_anchor=(2.9, 1.05))


        this_ax = ax[2 * (idx + 1) + 1]
        do_plot(train_ref_fs, error_dict(train_pred_fs, train_ref_fs), this_ax, 'Training', by_config_type)
        if test_set:
            # do_plot(test_ref_fs, test_pred_fs - test_ref_fs, this_ax, 'Test:      ')
            do_plot(test_ref_fs, error_dict(test_pred_fs, test_ref_fs), this_ax, 'Test', by_config_type)
        this_ax.set_xlabel(f'{ref_name.upper()} force / eV/Å')
        this_ax.set_ylabel(f'F$_{{GAP}}$ - F$_{{{ref_name.upper()}}}$ / eV/Å')
        this_ax.axhline(y=0, c='k', linewidth=0.8)
        this_ax.set_title(f'Force component errors on {sym}')
        # this_ax.legend(title='Set: RMSE $\pm$ STD, eV/Å', bbox_to_anchor=(1.1, 1.05))


    if not prefix:
        prefix = os.path.basename(param_fname)
        prefix = os.path.splitext(prefix)[0]
    picture_fname = f'{prefix}_scatter.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.suptitle(prefix)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    # plt.title(prefix)
    plt.savefig(picture_fname, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')

def make_2b_only_plot(dimer_name, ax, param_fname):

    # TODO make this more robust
    corr_desc = {'HH': 1, 'CH': 2, 'HO': 3, 'CC': 4, 'CO': 5}
    # atoms_fname =f'xyzs/dftb_{dimer_name}_dimer.xyz'
    atoms_fname = f'/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz'
    dimer = read(atoms_fname, index=':')
    distances = [at.get_distance(0, 1) for at in dimer]

    # command = f"quip E=T F=T atoms_filename=/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz param_filename={param_fname} calc_args={{only_descriptor={corr_desc[dimer_name]}}} \
    #                 | grep AT | sed 's/AT//' > ./tmp_atoms.xyz"

    command = f"quip E=T F=T atoms_filename={atoms_fname} param_filename={param_fname} calc_args={{only_descriptor={corr_desc[dimer_name]}}} \
                         | grep AT | sed 's/AT//' > ./tmp_atoms.xyz"

    subprocess.run(command, shell=True)
    atoms = read('./tmp_atoms.xyz', index=':')
    os.remove('./tmp_atoms.xyz')
    es = [at.info['energy'] for at in atoms]
    ax.plot(distances, es, label='GAP: only 2b', color='tab:orange')


def make_dimer_plot(dimer_name, ax, calc, label, isolated_atoms_fname=None):
    init_dimer = Atoms(dimer_name, positions=[(0, 0, 0), (0, 0, 2)])
    if dimer_name=='HH':
        distances = np.linspace(0.4, 5, 50)
    else:
        distances = np.linspace(0.8, 5, 50)
    dimer = []
    for d in distances:
        at = init_dimer.copy()
        at.set_distance(0, 1, d)
        dimer.append(at)

    energies = []
    for at in dimer:
        at.set_calculator(calc)
        energies.append(at.get_potential_energy())

    #clean this up majorly
    color='tab:blue'
    if 'Glue' in label:
        if isolated_atoms_fname!='none':
            isolated_atoms = read(isolated_atoms_fname, ':')
            e_shift = 0
            for sym in dimer_name:
                for iso_at in isolated_atoms:
                    if sym in iso_at.symbols:
                        e_shift +=iso_at.info['dft_energy']
            energies = [e + e_shift for e in energies]
        color='tab:green'



    ax.plot(distances, energies, label=label, color=color)

def make_ref_plot(dimer_name, ax):
    # atoms_fname = f'xyzs/dftb_{dimer_name}_dimer.xyz'
    atoms_fname=f'/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz'
    # dimer = read(f'/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz', index=':')
    dimer = read(atoms_fname, ':')
    distances = [at.get_distance(0, 1) for at in dimer]
    # TODO maybe deal with optional calc_type name
    ref_data = get_E_F_dict(dimer, calc_type='dft')
    ax.plot(distances, dict_to_vals(ref_data['energy']), label='(RKS) reference', linestyle='--', color='k')

def make_dimer_curves(param_fname, train_fname, output_dir=None, prefix=None, glue_fname=None, plot_2b_contribution=True, \
                      plot_ref_curve=True, isolated_atoms_fname=None, ref_name='dft', dimer_scatter=None):

    train_ats = read(train_fname, index=':')
    distances_dict = util.distances_dict(train_ats)
    dimers = [key for key, item in distances_dict.items() if len(item)!=0]

    if len(dimers)%2==0:
        no_vert = int(len(dimers)/2)
    else:
        no_vert = int((len(dimers)+1)/2)

    plt.figure(figsize=(12, no_vert*5))
    gs1 = gridspec.GridSpec(no_vert, 2)
    axes_main = []
    axes_hist = []
    for gs in gs1:
        gs2 = gs.subgridspec(2, 1, height_ratios=[2,1])
        ax1 = plt.subplot(gs2[0])
        axes_main.append(ax1)
        axes_hist.append(plt.subplot(gs2[1], sharex=ax1))


    print('Plotting complete GAP on dimers')
    gap = Potential(param_filename=param_fname)
    for ax, dimer in zip(axes_main, dimers):
        make_dimer_plot(dimer, ax, calc=gap, label='GAP')


    if plot_2b_contribution:
        print('Plotting 2b of GAP on dimers')
        for ax, dimer in zip(axes_main, tqdm(dimers)):
            make_2b_only_plot(dimer, ax, param_fname)


    if glue_fname:
        print('Plotting Glue')
        glue = Potential('IP Glue', param_filename=glue_fname)
        for ax, dimer in zip(axes_main, dimers):
            make_dimer_plot(dimer, ax, calc=glue, label='Glue', isolated_atoms_fname=isolated_atoms_fname)

    if plot_ref_curve:
        print('Plotting reference dimer curves (to be fixed still)')
        for ax, dimer in zip(axes_main, dimers):
            make_ref_plot(dimer, ax)

    if dimer_scatter:
        dimer_scatter_ats = read(dimer_scatter, ':')
        for ax, dimer in zip(axes_main, dimers):
            x_vals = []
            y_vals = []
            for at in dimer_scatter_ats:
                if dimer in ''.join(at.get_chemical_symbols()):
                    x_vals.append(at.get_distance(0, 1))
                    y_vals.append(at.info[f'{ref_name}_energy'])
            ax.scatter(x_vals, y_vals, color='tab:red', marker='x', label='training points')

    for ax, dimer in zip(axes_main, dimers):
        ax.legend()
        ax.set_title(dimer)
        # ax.set_xlabel('distance (Å)')
        ax.set_ylabel('energy (eV)')

    print('Plotting distance histogram')
    for ax, dimer in zip(axes_hist, dimers):
        data = distances_dict[dimer]
        # print(data)
        ax.hist(data, bins=np.arange(min(data), max(data)+0.1, 0.1))
        # plot_histogram(dimer_name, ax, distance_dict)
        ax.set_xlabel('distance (Å)')
        ax.set_ylabel('count in training set')





    if not prefix:
        prefix = os.path.basename(param_fname)
        prefix = os.path.splitext(prefix)[0]
    plt.suptitle(prefix)
    picture_fname = f'{prefix}_dimer.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)
    plt.tight_layout()
    plt.savefig(picture_fname, dpi=300)


@click.command()
@click.option('--param_fname',  type=click.Path(exists=True), required=True, help='GAP xml to test')
@click.option('--train_fname', type=click.Path(exists=True), required=True, help='.xyz file used for training')
@click.option('--test_fname', type=click.Path(exists=True), help='.xyz file to test GAP on')
@click.option('--output_dir', default='pictures', show_default=True, type=click.Path(), help='directory for figures. Create if not-existent')
@click.option('--prefix', help='prefix to label plots')
@click.option('--by_config_type', type=bool, help='if structures should be coloured by config_type in plots')
@click.option('--glue_fname', type=click.Path(exists=True), help='glue potential\'s xml to be evaluated for dimers')
@click.option('--plot_2b_contribution', type=bool, default='True', show_default=True, help='whether to plot the 2b only bit of gap')
@click.option('--plot_ref_curve', type=bool, default='True', show_default=True, help='whether to plot the reference DFT dimer curve')
@click.option('--isolated_atoms_fname',  default='xyzs/isolated_atoms.xyz', show_default=True, help='isolated atoms to shift glue')
@click.option('--ref_name', default='dft', show_default=True, help='prefix to \'_forces\' and \'_energy\' to take as a reference')
# TODO take this out maybe
@click.option('--dimer_scatter', help='dimer data in training set to be scattered on top of dimer curves')
def make_plots(param_fname, train_fname, test_fname=None, output_dir=None, prefix=None, by_config_type=False, glue_fname=False, \
               plot_2b_contribution=True, plot_ref_curve=True, isolated_atoms_fname=None, ref_name='dft', dimer_scatter=None):
    """Makes energy and force scatter plots and dimer curves"""
    # TODO make optional directory where to save stuff
    # TODO maybe include dftb???
    # TODO add option to include filename
    # TODO get .xyz files from GAP xml file!!

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    print('Scatter plotting')
    make_scatter_plots_from_file(param_fname=param_fname, train_fname=train_fname, test_fname=test_fname, \
                       output_dir=output_dir, prefix=prefix, by_config_type=by_config_type, ref_name=ref_name)
    print('Ploting dimers')
    make_dimer_curves(param_fname=param_fname, train_fname=train_fname, output_dir=output_dir, prefix=prefix,\
                      glue_fname=glue_fname, plot_2b_contribution=plot_2b_contribution, plot_ref_curve=plot_ref_curve,\
                      isolated_atoms_fname=isolated_atoms_fname, ref_name=ref_name, dimer_scatter=dimer_scatter)



if __name__=='__main__':
    make_plots()
    print('\n\n-----------------CONGRATS YOU MADE IT')

