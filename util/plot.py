
# Standard imports
import os, subprocess, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
from collections import OrderedDict

# My imports
import util
from util import ugap
sys.path.append('/home/eg475/programs/ASAP')
from scripts import kpca_for_projection_viewer as kpca


# Specific imports
import click
from tqdm import tqdm
from lxml import etree as et

# Atomistic imports
from quippy.potential import Potential
from ase.io import read, write
from ase import Atom, Atoms
from ase.io.extxyz import key_val_str_to_dict
from ase.io.extxyz import key_val_dict_to_str




def get_E_F_dict(atoms, calc_type, param_fname=None):

    data = dict()
    data['energy'] = OrderedDict()
    data['forces'] = OrderedDict()

    # select which energies and forces to extract
    if calc_type.upper() == 'GAP':
        if param_fname:
            gap = Potential(param_filename=param_fname)
        else:
            raise NameError('GAP filename is not given, but GAP energies requested.')

    else:
        if param_fname:
            print(f"WARNING: calc_type selected as {calc_type}, but gap filename is given, are you sure?")
        energy_name = f'{calc_type}_energy'
        if energy_name not in atoms[0].info.keys():
            print(f"WARNING: '{calc_type}_energy' not found, using 'energy', which might be anything")
            energy_name = 'energy'
        force_name = f'{calc_type}_forces'
        if force_name not in atoms[0].arrays.keys():
            print(f"WARNING: '{calc_type}_forces' not in found, using 'forces', which might be anything")
            force_name = 'forces'


    for atom in atoms:
        at = atom.copy()
        config_type='no_config_type'
        if 'config_type' in at.info.keys():
            config_type = at.info['config_type']


        if len(at) != 1:
            if calc_type.upper() == 'GAP':
                at.set_calculator(gap)
                energy = at.get_potential_energy()
                try:
                    data['energy'][config_type] = np.append(data['energy'][config_type], energy)
                except KeyError:
                    data['energy'][config_type] = np.array([])
                    data['energy'][config_type] = np.append(data['energy'][config_type], energy)
                forces = at.get_forces()

            else:
                try:
                    data['energy'][config_type] = np.append(data['energy'][config_type], at.info[energy_name])
                except KeyError:
                    data['energy'][config_type] = np.array([])
                    data['energy'][config_type] = np.append(data['energy'][config_type], at.info[energy_name])
                forces = at.arrays[force_name]

            sym_all = at.get_chemical_symbols()
            for j, sym in enumerate(sym_all):
                if sym not in data['forces'].keys():
                    data['forces'][sym] = OrderedDict()
                try:
                    data['forces'][sym][config_type] = np.append(data['forces'][sym][config_type], forces[j])
                except KeyError:
                    data['forces'][sym][config_type] = np.array([])
                    data['forces'][sym][config_type] = np.append(data['forces'][sym][config_type], forces[j])

    return data


def dict_to_vals(my_dict):
    ''' concatenates dictionary of multiple dictionary:value  to dictionary:[value1, value2, ...]'''
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

    make_scatter_plots(param_fname=param_fname, train_ats=train_ats, test_ats=test_ats, output_dir=output_dir,\
                       prefix=prefix, by_config_type=by_config_type, ref_name=ref_name)


def scatter_plot(param_fname, train_ats, ax, test_ats=None, by_config_type=False, ref_name='dft'):

    '''ax - axes list'''
    test_set = False
    if test_ats:
        test_set = True

    train_ref_data = get_E_F_dict(train_ats, calc_type=ref_name)
    train_pred_data = get_E_F_dict(train_ats, calc_type='gap', param_fname=param_fname)

    if test_set:
        test_ref_data = get_E_F_dict(test_ats, calc_type=ref_name)
        test_pred_data = get_E_F_dict(test_ats, calc_type='gap', param_fname=param_fname)

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
        if len(train_ref_data['forces'][sym]) == 0:
            print(f'skipping {train_ref_data["forces"][sym]}, which only has one element')
            continue

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
            do_plot(test_ref_fs, error_dict(test_pred_fs, test_ref_fs), this_ax, 'Test', by_config_type)
        this_ax.set_xlabel(f'{ref_name.upper()} force / eV/Å')
        this_ax.set_ylabel(f'F$_{{GAP}}$ - F$_{{{ref_name.upper()}}}$ / eV/Å')
        this_ax.axhline(y=0, c='k', linewidth=0.8)
        this_ax.set_title(f'Force component errors on {sym}')
        # this_ax.legend(title='Set: RMSE $\pm$ STD, eV/Å', bbox_to_anchor=(1.1, 1.05))
    return lgd


def make_scatter_plots(param_fname, train_ats, test_ats=None, output_dir=None, prefix=None, by_config_type=False, ref_name='dft'):

    counts = util.get_counts(train_ats[0])
    no_unique_elements = len(counts.keys())
    width = 10
    height = width * 0.6
    height *= no_unique_elements

    plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(no_unique_elements+1, 2)
    ax = [plt.subplot(g) for g in gs]

    lgd = scatter_plot(param_fname=param_fname, train_ats=train_ats, ax=ax, test_ats=test_ats, by_config_type=by_config_type, ref_name=ref_name)

    if not prefix:
        prefix = os.path.basename(param_fname)
        prefix = os.path.splitext(prefix)[0]
    picture_fname = f'{prefix}_scatter.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.suptitle(prefix)
    plt.savefig(picture_fname, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.savefig(picture_fname, dpi=300)


def make_2b_only_plot(dimer_name, ax, param_fname, label=None, color=None):

    corr_desc = ugap.get_gap_2b_dict(param_fname)
    atoms_fname = f'/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz'
    dimer = read(atoms_fname, index=':')
    distances = [at.get_distance(0, 1) for at in dimer]

    command = f"quip E=T F=T atoms_filename={atoms_fname} param_filename={param_fname} calc_args={{only_descriptor={corr_desc[dimer_name]}}} \
                         | grep AT | sed 's/AT//' > ./tmp_atoms.xyz"

    subprocess.run(command, shell=True)
    atoms = read('./tmp_atoms.xyz', index=':')
    os.remove('./tmp_atoms.xyz')
    es = [at.info['energy'] for at in atoms]
    if label is none:
        label = 'GAP: only 2b'
    if color is none:
        color='tab:orange'
    ax.plot(distances, es, label=label, color=color)


def make_dimer_plot(dimer_name, ax, calc, label, color=None, isolated_atoms_fname=None):
    init_dimer = Atoms(dimer_name, positions=[(0, 0, 0), (0, 0, 2)])

    distances = np.linspace(0.2, 6, 50)

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
    if color is None:
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


def make_ref_plot(dimer_name, ax, calc_type='dft'):
    atoms_fname=f'/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz'
    dimer = read(atoms_fname, ':')
    distances = [at.get_distance(0, 1) for at in dimer]
    ref_data = get_E_F_dict(dimer, calc_type=calc_type)
    ax.plot(distances, dict_to_vals(ref_data['energy']), label='(RKS) reference', linestyle='--', color='k')


def make_dimer_curves(param_fnames, train_fname, output_dir=None, prefix=None, glue_fname=None, plot_2b_contribution=True, \
                      plot_ref_curve=True, isolated_atoms_fname=None, ref_name='dft', dimer_scatter=None, ylim=(15, 15)):
    # param_fname - list of param_fnames, most often one

    train_ats = read(train_fname, index=':')
    distances_dict = util.distances_dict(train_ats)
    dimers = [key for key, item in distances_dict.items() if len(item)!=0]

    # get number of subplots
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

    if len(param_fnames) == 1:
        # means only one thing in param_fname
        param_fname = param_fnames[0]
        print('Plotting complete GAP on dimers')
        gap = Potential(param_filename=param_fname)
        for ax, dimer in zip(axes_main, dimers):
            make_dimer_plot(dimer, ax, calc=gap, label='GAP')

    elif len(param_fnames) > 1 and plot_2b_contribution==False:
        print('WARNING: did not ask to evaluate 2b contributions on dimers, so evaluating multiple full gaps instead')
        # only evaluate full gap on dimers if not asked to evaluate 2b contribution only
        # means param_fname is actually a list of
        cmap = mpl.cm.get_cmap('Blues')
        colors = np.linspace(0.2, 1, len(param_fnames))

        for color, param_fname in zip(colors, param_fnames):
            # print(param_fname)
            label = os.path.basename(param_fname)
            label = os.path.splitext(label)[0]
            gap = Potential(param_filename=param_fname)
            for ax, dimer in zip(axes_main, dimers):
                make_dimer_plot(dimer, ax, calc=gap, label=label, color=cmap(color))


    if plot_2b_contribution:
        if len(param_fnames) == 1:
            param_fname = param_fnames[0]
            print('Plotting 2b of GAP on dimers')
            for ax, dimer in zip(axes_main, tqdm(dimers)):
                make_2b_only_plot(dimer, ax, param_fname)

        elif len(param_fnames) > 1:
            print('evaluating 2b contributions from all gaps')
            cmap = mpl.cm.get_cmap('Oranges')
            colors = np.linspace(0.2, 1, len(param_fnames))
            for color, param_fname in zip(colors, param_fnames):
                label = os.path.basename(param_fnames)
                label = os.path.splitext(label)[0]
                for ax, dimer in zip(axes_main, dimers):
                    make_2b_only_plot(dimer, ax, param_fname=param_fname, label=label, color=cmap(color))

    if glue_fname:
        print('Plotting Glue')
        glue = Potential('IP Glue', param_filename=glue_fname)
        for ax, dimer in zip(axes_main, dimers):
            make_dimer_plot(dimer, ax, calc=glue, label='Glue', isolated_atoms_fname=isolated_atoms_fname)

    if plot_ref_curve:
        print('Plotting reference dimer curves (to be fixed still)')
        for ax, dimer in zip(axes_main, dimers):
            if dimer!='OO':
                make_ref_plot(dimer, ax, calc_type=ref_name)

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

    isolated_atoms = read(isolated_atoms_fname, ':')
    for ax, dimer in zip(axes_main, dimers):
        ax.legend(loc='upper right')
        ax.set_title(dimer)
        ax.set_ylabel('energy (eV)')
        ax.grid(color='lightgrey')

        e_shift = 0
        for sym in dimer:
            for iso_at in isolated_atoms:
                if sym in iso_at.symbols:
                    e_shift += iso_at.info['dft_energy']
        ylimits = (e_shift-ylim[0], e_shift+ylim[1])
        ax.set_ylim(ylimits)



    print('Plotting distance histogram')
    for ax, dimer in zip(axes_hist, dimers):
        data = distances_dict[dimer]
        ax.hist(data, bins=np.arange(min(data), max(data)+0.1, 0.1))
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


def desymbolise_force_dict(my_dict):
    '''from dict['forces'][sym]:values makes dict['forces']:[values1, values2...]'''
    force_dict = OrderedDict()
    for sym, sym_dict in my_dict.items():
        for config_type, values in sym_dict.items():
            try:
                force_dict[config_type] = np.append(force_dict[config_type], values)
            except KeyError:
                force_dict[config_type] = np.array([])
                force_dict[config_type] = np.append(force_dict[config_type], values)

    return force_dict


def get_rmse_dict(obs, dft_data, gap_data):
    rmses = dict()
    rmses['Comb'] = util.get_rmse(dict_to_vals(dft_data[obs]), dict_to_vals(gap_data[obs]))
    for dft_config, gap_config in zip(dft_data[obs].keys(), gap_data[obs].keys()):
        if dft_config!=gap_config:
            raise ValueError('gap and dft config_types did not match')
        rmses[gap_config] = util.get_rmse(dft_data[obs][dft_config], gap_data[obs][gap_config])
    return rmses


def plot_heatmap(data_dict, ax, obs):
    df = pd.DataFrame.from_dict(data_dict)
    hmap = ax.pcolormesh(df, vmin=0)
    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels(df.index)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels(df.columns, rotation=90)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            color = 'red'
            text = ax.text(j + 0.5, i + 0.5, round(df.iat[i, j], 3), ha='center', color=color)
    cbar = plt.colorbar(hmap, ax=ax)
    if obs == 'Energy':
        units = 'eV'
    elif obs == 'Force':
        units = 'eV/Å '
    cbar.ax.set_ylabel(f'{obs} RMSE, {units}', rotation=90, labelpad=6)
    ax.set_title(f'{obs} RMSE', fontsize=14 )


def get_last_bunch(full_data, bunch=20):
    new_keys = list(full_data['energy'].keys())[-bunch:]
    new_data = dict()
    new_data['energy'] = OrderedDict()
    new_data['forces'] = OrderedDict()
    for key in new_keys:
        new_data['energy'][key] = full_data['energy'][key]
        new_data['forces'][key] = full_data['forces'][key]
    return new_data


def rmse_plots(train_fname, gaps_dir, output_dir=None, prefix=None):

    train_ats = read(train_fname, index=':')
    dft_data = get_E_F_dict(train_ats, calc_type='dft')
    dft_data['forces'] = desymbolise_force_dict(dft_data['forces'])

    # TODO think this through better
    max_gap_dset_no = 15
    if len(dft_data['energy'].keys()) > max_gap_dset_no:
        print(f'more than {max_gap_dset_no} config types, taking the last {max_gap_dset_no}')
        dft_data = get_last_bunch(dft_data, bunch=max_gap_dset_no)

    E_rmses = dict()
    F_rmses = dict()

    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = util.natural_sort(gap_fnames)

    if len(gap_fnames) > max_gap_dset_no:
        print(f'more than {max_gap_dset_no} gaps, taking the last {max_gap_dset_no}')
        gap_fnames = gap_fnames[-max_gap_dset_no:]


    for gap_fname in tqdm(gap_fnames):
        gap_title = os.path.splitext(gap_fname)[0]
        gap_fname = os.path.join(gaps_dir, gap_fname)
        gap_data = get_E_F_dict(train_ats, calc_type='gap', param_fname=gap_fname)
        gap_data['forces'] = desymbolise_force_dict(gap_data['forces'])

        # if more than 20, take the last 20 only
        if len(gap_data['energy'].keys()) > max_gap_dset_no:
            gap_data = get_last_bunch(gap_data, bunch=max_gap_dset_no)

        E_rmses[gap_title] = get_rmse_dict(obs='energy', dft_data=dft_data, gap_data=gap_data)
        F_rmses[gap_title] = get_rmse_dict(obs='forces', dft_data=dft_data, gap_data=gap_data)

    # Make plot
    # TODO save the pandas dataframe somewhere somehow

    N = len(gap_fnames)
    width = (N * 0.6 + 1.2) * 2
    height = N * 0.6
    plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(1, 2)
    all_ax = [plt.subplot(g) for g in gs]

    plot_heatmap(E_rmses, all_ax[0], 'Energy')
    plot_heatmap(F_rmses, all_ax[1], 'Force')
    plt.tight_layout()

    if not prefix:
        prefix = 'summary'
    picture_fname = f'{prefix}_RMSE_heatmap.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.savefig(picture_fname, dpi=300)


def dimer_summary_plot(gaps_dir, train_fname, output_dir=None, prefix=None, glue_fname=None, plot_2b_contribution=True,
                plot_ref_curve=True, isolated_atoms_fname=None, ref_name='dft', dimer_scatter=None, ylim=(15, 15)):

    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = [os.path.join(gaps_dir, name) for name in gap_fnames]
    gap_fnames = util.natural_sort(gap_fnames)

    max_gap_dset_no =10
    if len(gap_fnames) > max_gap_dset_no:
        gap_fnames = gap_fnames[-max_gap_dset_no:]

    if prefix is None:
        prefix='summary'

    make_dimer_curves(gap_fnames, train_fname=train_fname, output_dir=output_dir, prefix=prefix,
                                glue_fname=glue_fname, plot_2b_contribution=plot_2b_contribution,  \
                                plot_ref_curve=plot_ref_curve, isolated_atoms_fname=isolated_atoms_fname, \
                                ref_name=ref_name, dimer_scatter=dimer_scatter, ylim=ylim)


def summary_scatter(gaps_dir, output_dir=None, prefix=None, ref_name='dft'):
    # not worked out and not really worth it.
    # gap fnames
    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = [os.path.join(gaps_dir, name) for name in gap_fnames]
    gap_fnames = util.natural_sort(gap_fnames)

    # get how many elements are present and make up axis of the right shape
    # assumes that last gap file has all of them
    tmp_ats = atoms_from_gap(gap_fnames[-1])[0]
    counts = get_counts(tmp_ats)
    no_unique_elements = len(counts.keys())
    width = 10
    height = width * 0.6
    height *= no_unique_elements
    plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(no_unique_elements + 1, 2)
    ax = [plt.subplot(g) for g in gs]

    for param_fname in gap_fnames:
        train_ats = atoms_from_gap(param_fname)
        scatter_plot(param_fname, train_ats, ax)


def get_train_test_rmse_dicts(gap_idx, dft_data, gap_data):
    no_dsets = len(dft_data['energy'].keys())

    training_rmses = dict()
    test_rmses = dict()

    e_train_dft = {}
    e_test_dft = {}
    e_train_gap = {}
    e_test_gap = {}

    for key1, key2 in zip(dft_data['energy'].keys(), gap_data['energy'].keys()):
        if key1 != key2:
            raise Exception(f'observations do not match. dft: {key1}, gap: {key2}')
        dset_no = int(re.search(r'\d+', key1).group())
        if dset_no <= gap_idx:
            e_train_dft[key1] = dft_data['energy'][key1]
            e_train_gap[key2] = gap_data['energy'][key2]
        else:
            e_test_dft[key1] = dft_data['energy'][key1]
            e_test_gap[key2] = gap_data['energy'][key2]

    training_rmses['energy'] = util.get_rmse(dict_to_vals(e_train_dft), dict_to_vals(e_train_gap))
    if gap_idx != no_dsets:
        test_rmses['energy'] = util.get_rmse(dict_to_vals(e_test_dft), dict_to_vals(e_test_gap))

    training_rmses['forces'] = {}
    test_rmses['forces'] = {}
    for sym1, sym2 in zip(dft_data['forces'].keys(), gap_data['forces'].keys()):
        if sym1 != sym2:
            raise Exception(f'observations do not match. dft: {sym1}, gap: {sym2}')

        f_train_dft = {}
        f_test_dft = {}
        f_train_gap = {}
        f_test_gap = {}
        for key1, key2 in zip(dft_data['forces'][sym1].keys(), gap_data['forces'][sym2].keys()):
            if key1 != key2:
                raise Exception(f'observations do not match. dft: {key1}, gap: {key2}')
            dset_no = int(re.search(r'\d+', key1).group())
            if dset_no <= gap_idx:
                f_train_dft[key1] = dft_data['forces'][sym1][key1]
                f_train_gap[key2] = gap_data['forces'][sym2][key2]
            else:
                f_test_dft[key1] = dft_data['forces'][sym1][key1]
                f_test_gap[key2] = gap_data['forces'][sym2][key2]
        training_rmses['forces'][sym1] = util.get_rmse(dict_to_vals(f_train_dft), dict_to_vals(f_train_gap))
        if gap_idx != no_dsets:
            test_rmses['forces'][sym2] = util.get_rmse(dict_to_vals(f_test_dft), dict_to_vals(f_test_gap))

    return (training_rmses, test_rmses)


def rmse_line_plots(gaps_dir, train_fname, output_dir=None, prefix=None):
    train_ats = read(train_fname, index=':')
    dft_data = get_E_F_dict(train_ats, calc_type='dft')

    all_train_rmses = {}
    all_test_rmses = {}

    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = util.natural_sort(gap_fnames)

    for idx, gap_fname in enumerate(tqdm(gap_fnames)):
        gap_title = os.path.splitext(gap_fname)[0]
        gap_fname = os.path.join(gaps_dir, gap_fname)
        gap_data = get_E_F_dict(train_ats, calc_type='gap', param_fname=gap_fname)

        train_rmses, test_rmses = get_train_test_rmse_dicts(gap_idx=idx + 1, dft_data=dft_data, gap_data=gap_data)

        all_train_rmses[gap_title] = train_rmses
        all_test_rmses[gap_title] = test_rmses

    plt.figure(figsize=(10, 6))

    ax2 = plt.gca()
    syms = all_train_rmses[gap_title]['forces'].keys()
    cmap = mpl.cm.get_cmap('viridis')
    colors = np.linspace(0, 0.9, len(syms))
    for idx, sym in enumerate(syms):
        color = cmap(colors[idx])
        forces_train = [value['forces'][sym] for key, value in all_train_rmses.items()]
        ax2.plot(range(len(forces_train)), forces_train, linestyle='-', color=color, marker='x', \
                 label=f'on {sym}, training set')

        forces_test = [value['forces'][sym] for key, value in all_test_rmses.items()]
        ax2.plot(range(len(forces_test)), forces_test, linestyle='-.', color=color, marker='x', \
                label=f'on {sym}, testing set')

    ax1 = ax2.twinx()
    energies_train = [value['energy'] for key, value in all_train_rmses.items()]
    ax1.plot(range(len(energies_train)), energies_train, linestyle='-', color='tab:red', marker='x', label='training set')

    energies_test = [value['energy'] for key, value in all_test_rmses.items() if 'energy' in value.keys()]
    ax1.plot(range(len(energies_test)), energies_test, linestyle='-.', color='tab:red', marker='x', label='testing set')

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    ax1.legend(title='Energy')
    ax2.legend(title='Force', loc='best', bbox_to_anchor=(0, 0, 1, 0.85))

    ax1.set_ylabel('Energy RMSE, eV')
    ax2.set_ylabel('Force component RMSE, eV/Å')

    ax2.set_xlabel('Iteration')
    plt.title('RMSEs for GAP_i')

    if not prefix:
        prefix = 'summary'
    picture_fname = f'{prefix}_RMSEs.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.tight_layout()
    plt.savefig(picture_fname, dpi=300)


def kpca_plot(xyz_fname, pic_fname, output_dir):
    atoms = read(xyz_fname, ':')

    # training set points
    xs_train = [at.info['pca_coord'][0] for at in atoms if 'iter' in at.info['config_type']]
    ys_train = [at.info['pca_coord'][1] for at in atoms if 'iter' in at.info['config_type']]
    colors_train_names = [at.info['config_type'] for at in atoms if 'iter' in at.info['config_type']]
    cmap = mpl.cm.get_cmap('tab20')
    # TODO: smartify this
    color_mapping_train = {'iter_1': cmap(0.05), 'iter_2': cmap(0.15), 'iter_3': cmap(0.25), 'iter_4': cmap(0.35), \
                           'iter_5': cmap(0.45), 'iter_6': cmap(0.55)}
    colors_train = [color_mapping_train[c] for c in colors_train_names]

    # optg_points
    color_map_opt = {'first_guess': cmap(0), 'opt_1': cmap(0.1), 'opt_2': cmap(0.2), 'opt_3': cmap(0.3),
                     'opt_4': cmap(0.4), \
                     'opt_5': cmap(0.5), 'dft_optg': 'k'}

    optg_pts = []
    optg_ats = [at for at in atoms if 'iter' not in at.info['config_type']]
    for at in optg_ats:
        entry = {}
        label = at.info['config_type']
        entry['label'] = label
        entry['x'] = at.info['pca_coord'][0]
        entry['y'] = at.info['pca_coord'][1]
        entry['color'] = color_map_opt[label]
        optg_pts.append(entry)

    xs_optg = [at.info['pca_coord'][0] for at in atoms if 'iter' not in at.info['config_type']]
    ys_optg = [at.info['pca_coord'][1] for at in atoms if 'iter' not in at.info['config_type']]
    colors_opt = [at.info['config_type'] for at in atoms if 'iter' not in at.info['config_type']]
    colors_opt = [color_map_opt[c] for c in colors_opt]

    plt.figure(figsize=(8, 4))

    plt.scatter(xs_train, ys_train, color=colors_train, label='training sets')

    for pt in optg_pts:
        plt.scatter(pt['x'], pt['y'], color=pt['color'], label=pt['label'], marker='X', linewidth=0.5, s=80, \
                    linewidths=10, edgecolors='k')

    plt.legend()
    which=''
    if 'default' in pic_fname:
        which = 'default soap'
    elif 'my_soap' in pic_fname:
        which = 'my soap'

    plt.title(f'kPCA ({which}) on GAP_i training sets and GAP_i-optimised structures')
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')

    if output_dir is not None:
        pic_fname = os.path.join(output_dir, pic_fname)

    plt.tight_layout()
    plt.savefig(pic_fname, dpi=300)
    plt.show()


def make_kpca_dset(training_set, all_opt_ats, first_guess, dft_optg, xyz_fname, calc):
    ## main training points
    kpca_ats = read(training_set, ':')
    kpca_ats = [at for at in kpca_ats if len(at) != 1]

    ## all gap_i optimised atoms. expect dft_energy to be correct...
    gap_opt_ats = read(all_opt_ats, ':')
    if 'config_type' not in gap_opt_ats[0].info.keys():
        for i, at in enumerate(gap_opt_ats):
            at.info['config_type'] = f'opt_{i}'
    elif gap_opt_ats[0].info['config_type'] != 'opt_1':
        print("WARNING: GAP-optimised atoms have config, types, but they don't follow convention of opt_{i}, renaming")
        for i, at in enumerate(gap_opt_ats):
            at.info['config_type'] = f'opt_{i}'

    ##label first_guess
    first_guess = read(first_guess)
    first_guess.info['config_type'] = 'first_guess'
    if 'dft_energy' not in first_guess.info.keys():
        if calc is None:
            raise Exception('First guess: Calculator not given, nor is dft_energy')
        print('Calculating first_guess energy')
        first_guess.set_calculator(calc)
        first_guess.info['dft_energy'] = first_guess.get_potential_energy()

    ## dft-optimized structure
    dft_optg = read(dft_optg)
    dft_optg.info['config_type'] = 'dft_optg'
    if 'dft_energy' not in dft_optg.info.keys():
        if calc is None:
            raise Exception('dft_optg: Calculator not given, nor is dft_energy')
        print('Calculating first_guess energy')
        dft_optg.set_calculator(calc)
        dft_optg.info['dft_energy'] = dft_optg.get_potential_energy()

    write(xyz_fname, kpca_ats + gap_opt_ats + [first_guess, dft_optg], 'extxyz', write_results=False)


def make_kpca_plots(training_set, all_opt_ats='xyzs/opt_all.xyz', first_guess='xyzs/first_guess.xyz', \
                    dft_optg='molpro_optg/optimized.xyz', xyz_fname='xyzs/for_kpca.xyz', calc=None, \
                    param_fname=None, output_dir='pictures'):
    '''
    arguments:
        training set    last training set that has all the appropriate config_types 'dset_{i}'
        all_opt_ats     optimised atoms from all iterations with config types 'opt_{i}'
        first_guess     first_guess that was used to get the first dataset
        dft_optg        structure optimised with dft
        param_fname     one of the gaps' filename with the appropriate command for training gap, optional
    '''

    # set up a dataset for kpca
    print('Making up dataset for kpca')
    make_kpca_dset(training_set, all_opt_ats, first_guess, dft_optg, xyz_fname, calc)

    print('Kpca with default setings')
    xyz_title = os.path.splitext(os.path.basename(xyz_fname))[0]
    kpca_default_xyz_fname = f'xyzs/{xyz_title}_default.xyz'

    kpca.main(xyz_fname, pbc=False, output_filename=kpca_default_xyz_fname)

    pic_fnames = ['kpca_default.png']
    kpca_xyzs = [kpca_default_xyz_fname]

    if param_fname:
        print('Kpca with my soap settings')
        kpca_my_soap_xyz_fname = f'xyzs/{xyz_title}_my_soap.xyz'
        s = ugap.get_soap_params(param_fname)
        kpca.main(xyz_fname, pbc=False, output_filename=kpca_my_soap_xyz_fname, \
                  cutoff=s['cutoff'], n_max=s['n_max'], l_max=s['l_max'], zeta=s['zeta'], \
                  atom_sigma=s['atom_gaussian_width'])

        kpca_xyzs.append(kpca_my_soap_xyz_fname)
        pic_fnames.append('kpca_my_soap.png')

    print('plotting plots')
    for xyz_fname, pic_fname in zip(kpca_xyzs, pic_fnames):
        kpca_plot(xyz_fname, pic_fname, output_dir)





