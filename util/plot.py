
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
from util.vib import Vibrations
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
from ase.optimize.precon import PreconLBFGS
from ase.units import Ha
from util import dict_to_vals

''' Interesting things:
        make_scatter_plots - scatter for given gap and train/test set
        make_dimer_curves  - dimers given list of param_fnames
        rmse_heatmap       - heatmap of gap rmses evaluated on all config types in given dataset
        dimer_summary_plot - all gaps in dir evaluated on dimers
        rmse_line_plots    - energy and f on atom gap rmse train/test plots vs iteration 
        summary_scatter    - actually not interesting. scatter plot of gap_i on its training set
        kpca_plot          - kpce of by config_type in training set, optimised structures and dft-optg structure
        evec_plot          - eigenvector heatmap for given gap
        opt_summary_plots  - fmax and E error wrt dft optg vs iteration 
        eval_plot          - eigenvalue plot for all gaps
'''






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


def make_scatter_plots_from_file(param_fname, test_fname=None, output_dir=None, prefix=None, \
                                 by_config_type=False, ref_name='dft'):

    # train_ats = read(train_fname, index=':')
    test_ats = None
    if test_fname:
        test_ats = read(test_fname, index=':')

    make_scatter_plots(param_fname=param_fname,  test_ats=test_ats, output_dir=output_dir,\
                       prefix=prefix, by_config_type=by_config_type, ref_name=ref_name)


def scatter_plot(param_fname, train_ats, ax, test_ats=None, by_config_type=False, ref_name='dft'):

    '''ax - axes list'''
    test_set = False
    if test_ats:
        test_set = True

    train_ref_data = util.get_E_F_dict(train_ats, calc_type=ref_name)
    train_pred_data = util.get_E_F_dict(train_ats, calc_type='gap', param_fname=param_fname)

    if test_set:
        test_ref_data = util.get_E_F_dict(test_ats, calc_type=ref_name)
        test_pred_data = util.get_E_F_dict(test_ats, calc_type='gap', param_fname=param_fname)

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
    this_ax.set_xlabel(f'{ref_name.upper()} energy / eV/atom')
    this_ax.set_ylabel(f'GAP energy / eV/atom')
    this_ax.set_title('Energies')
    lgd = this_ax.legend(title='Set: RMSE $\pm$ STD, eV/atom', bbox_to_anchor=(2.9, 1.05))

    this_ax = ax[1]
    do_plot(train_ref_es, error_dict(train_pred_es, train_ref_es), this_ax, 'Training', by_config_type)
    if test_set:
        # do_plot(test_ref_es, test_pred_es - test_ref_es, this_ax, 'Test:      ')
        do_plot(test_ref_es, error_dict(test_pred_es, test_ref_es), this_ax, 'Test', by_config_type)
    this_ax.set_xlabel(f'{ref_name.upper()} energy / eV/atom')
    this_ax.set_ylabel(f'E$_{{GAP}}$ - E$_{{{ref_name.upper()}}}$ / eV/atom')
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


def make_scatter_plots(param_fname, test_ats=None, output_dir='pictures', prefix=None, by_config_type=False, ref_name='dft', close=True):

    train_ats = ugap.atoms_from_gap(param_fname)

    counts = util.get_counts(train_ats[0])
    no_unique_elements = len(counts.keys())
    width = 10
    height = width * 0.6
    height *= no_unique_elements

    fig = plt.figure(figsize=(width, height))
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
    # plt.tight_layout()
    plt.savefig(picture_fname, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.savefig(picture_fname, dpi=300)
    if close:
        plt.close(fig)
    else:
        plt.show()


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
    if label is None:
        label = 'GAP: only 2b'
    if color is None:
        color='tab:orange'
    ax.plot(distances, es, label=label, color=color)


def make_dimer_plot(dimer_name, ax, calc, label, color=None, isolated_atoms_fname=None):
    init_dimer = Atoms(dimer_name, positions=[(0, 0, 0), (0, 0, 2)])

    distances = np.linspace(0.2, 9, 50)

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
    ref_data = util.get_E_F_dict(dimer, calc_type=calc_type)
    ax.plot(distances, dict_to_vals(ref_data['energy'])*2, label='(RKS) reference', linestyle='--', color='k')


def make_dimer_curves(param_fnames, output_dir='pictures', prefix=None, glue_fname=None, plot_2b_contribution=True, \
                      plot_ref_curve=True, isolated_atoms_fname=None, ref_name='dft', dimer_scatter=None, ylim=(15, 25), close=False):
    # param_fname - list of param_fnames, most often one

    # train_ats = read(train_fname, index=':')
    train_ats = ugap.atoms_from_gap(param_fnames[-1])

    distances_dict = util.distances_dict(train_ats)
    dimers = [key for key, item in distances_dict.items() if len(item)!=0]

    # get number of subplots
    if len(dimers)%2==0:
        no_vert = int(len(dimers)/2)
    else:
        no_vert = int((len(dimers)+1)/2)

    fig = plt.figure(figsize=(12, no_vert*5))
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
                label = os.path.basename(param_fname)
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
        ylimits_default = ax.get_ylim()
        # ax.set_ylim(bottom=min(ylimits[0], ylimits_default[0]), top=max(ylimits_default[1], ylimits[1]))
        # ax.set_ylim(bottom=min(ylimits[0], ylimits_default[0]), top=ylimits[1])
        ax.set_xlim(0.03, 9)

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
    if close:
        plt.close(fig)
    else:
        plt.show()




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
        units = 'eV/atom'
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


def rmse_heatmap(train_fname, gaps_dir='gaps', output_dir='pictures', prefix=None):

    train_ats = read(train_fname, index=':')
    dft_data = util.get_E_F_dict(train_ats, calc_type='dft')
    dft_data['forces'] = util.desymbolise_force_dict(dft_data['forces'])

    # TODO think this through better
    # max_gap_dset_no = 15
    # if len(dft_data['energy'].keys()) > max_gap_dset_no:
    #     print(f'more than {max_gap_dset_no} config types, taking the last {max_gap_dset_no}')
    #     dft_data = get_last_bunch(dft_data, bunch=max_gap_dset_no)

    E_rmses = dict()
    F_rmses = dict()

    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = util.natural_sort(gap_fnames)

    # if len(gap_fnames) > max_gap_dset_no:
    #     print(f'more than {max_gap_dset_no} gaps, taking the last {max_gap_dset_no}')
    #     gap_fnames = gap_fnames[-max_gap_dset_no:]


    for gap_fname in tqdm(gap_fnames):
        gap_title = os.path.splitext(gap_fname)[0]
        gap_fname = os.path.join(gaps_dir, gap_fname)
        gap_data = util.get_E_F_dict(train_ats, calc_type='gap', param_fname=gap_fname)
        gap_data['forces'] = util.desymbolise_force_dict(gap_data['forces'])

        # if more than 20, take the last 20 only
        # if len(gap_data['energy'].keys()) > max_gap_dset_no:
        #     gap_data = get_last_bunch(gap_data, bunch=max_gap_dset_no)

        E_rmses[gap_title] = get_rmse_dict(obs='energy', dft_data=dft_data, gap_data=gap_data)
        F_rmses[gap_title] = get_rmse_dict(obs='forces', dft_data=dft_data, gap_data=gap_data)

    # Make plot
    # TODO save the pandas dataframe somewhere somehow

    N = len(gap_fnames)
    width = (N * 0.6 + 1.2) * 2
    height = N * 0.6
    fig = plt.figure(figsize=(width, height))
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
    plt.close(fig)


def dimer_summary_plot(gaps_dir='gaps', output_dir='pictures', prefix=None, glue_fname=None, plot_2b_contribution=True,
                plot_ref_curve=True, isolated_atoms_fname=None, ref_name='dft', dimer_scatter=None, ylim=(15, 15)):

    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = [os.path.join(gaps_dir, name) for name in gap_fnames]
    gap_fnames = util.natural_sort(gap_fnames)

    # max_gap_dset_no =10
    # if len(gap_fnames) > max_gap_dset_no:
    #     gap_fnames = gap_fnames[-max_gap_dset_no:]

    for idx, gap_fname_group in enumerate(util.grouper(gap_fnames, 10)):
        gap_fname_group = [x for x in gap_fname_group if x is not None]

        if prefix is None:
            prefix_new=f'summary'
        prefix_new += f'_{idx+1}'

        make_dimer_curves(gap_fname_group, output_dir=output_dir, prefix=prefix_new,
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

def rmse_line_plots(train_fname, gaps_dir='gaps', output_dir='pictures', prefix=None):
    train_ats = read(train_fname, index=':')
    dft_data = util.get_E_F_dict(train_ats, calc_type='dft')

    all_train_rmses = {}
    all_test_rmses = {}

    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = util.natural_sort(gap_fnames)

    for idx, gap_fname in enumerate(tqdm(gap_fnames)):
        gap_title = os.path.splitext(gap_fname)[0]
        gap_fname = os.path.join(gaps_dir, gap_fname)
        gap_data = util.get_E_F_dict(train_ats, calc_type='gap', param_fname=gap_fname)

        train_rmses, test_rmses = get_train_test_rmse_dicts(gap_idx=idx + 1, dft_data=dft_data, gap_data=gap_data)

        all_train_rmses[gap_title] = train_rmses
        all_test_rmses[gap_title] = test_rmses

    fig = plt.figure(figsize=(12, 8))

    ax2 = plt.gca()
    syms = all_train_rmses[gap_title]['forces'].keys()
    cmap = mpl.cm.get_cmap('viridis')
    colors = np.linspace(0, 0.9, len(syms))
    for idx, sym in enumerate(syms):
        color = cmap(colors[idx])
        forces_train = [value['forces'][sym] for key, value in all_train_rmses.items()]
        ax2.plot(range(1, len(forces_train)+1), forces_train, linestyle='-', color=color, marker='x', \
                 label=f'on {sym}, training set')

        # plt.gca().annotate(int(no), xy=(pt['x'], pt['y']))

        forces_test = [value['forces'][sym] for key, value in all_test_rmses.items()]
        ax2.plot(range(1, len(forces_test)+1), forces_test, linestyle=':', color=color, marker='x', \
                label=f'on {sym}, testing set')

        ax2.annotate(forces_test[-1], xy=(len(forces_test)+1, forces_test[-1]))
        ax2.annotate(forces_train[-1], xy=(len(forces_train) + 1, forces_train[-1]))


    ax1 = ax2.twinx()
    energies_train = [value['energy'] for key, value in all_train_rmses.items()]
    ax1.plot(range(1, len(energies_train)+1), energies_train, linestyle='-', color='tab:red', marker='x', label='training set')

    energies_test = [value['energy'] for key, value in all_test_rmses.items() if 'energy' in value.keys()]
    ax1.plot(range(1, len(energies_test)+1), energies_test, linestyle=':', color='tab:red', marker='x', label='testing set')

    ax1.annotate(energies_test[-1], xy=(len(energies_test)+1, energies_test[-1]))
    ax1.annotate(energies_train[-1], xy=(len(energies_train)+1, energies_train[-1]))

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        formatter = mpl.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.4))
        ax.get_yaxis().set_minor_formatter(formatter)


    ax1.legend(title='Energy')
    ax2.legend(title='Force', loc='best', bbox_to_anchor=(0, 0, 1, 0.85))

    ax1.set_ylabel('Energy RMSE, eV/atom')
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
    plt.close(fig)


def kpca_plot(xyz_fname, pic_name, output_dir, colour_by_energy=False):
    atoms = read(xyz_fname, ':')
    cmap = mpl.cm.get_cmap('tab20')

    fig = plt.figure(figsize=(14, 8))


    # training set points
    xs_train = [at.info['pca_coord'][0] for at in atoms if 'iter' in at.info['config_type']]
    ys_train = [at.info['pca_coord'][1] for at in atoms if 'iter' in at.info['config_type']]

    if colour_by_energy:
        energies = [at.info['dft_energy']/len(at) for at in atoms if 'iter' in at.info['config_type']]
        plt.scatter(xs_train, ys_train, c=energies, cmap='inferno', label='training points')
        cb = plt.colorbar()
        cb.set_label('DFT energy, eV/atom')

    else:
        colors_train_names = [at.info['config_type'] for at in atoms if 'iter' in at.info['config_type']]

        # TODO: smartify this
        color_mapping_train = {'iter_1': cmap(0.05), 'iter_2': cmap(0.15), 'iter_3': cmap(0.25), 'iter_4': cmap(0.35), \
                               'iter_5': cmap(0.45), 'iter_6': cmap(0.55), 'iter_7':cmap(0.65), 'iter_8':cmap(0.75), 'iter_9':cmap(0.85), 'iter_10':cmap(0.95), \
                               'iter_11': cmap(0.05), 'iter_12': cmap(0.15), 'iter_13': cmap(0.25), 'iter_14': cmap(0.35), \
                               'iter_15': cmap(0.45), 'iter_16': cmap(0.55), 'iter_17': cmap(0.65), 'iter_18': cmap(0.75),\
                               'iter_19': cmap(0.85), 'iter_20': cmap(0.95), 'iter_21':cmap(0.05)}

        colors_train = [color_mapping_train[c] for c in colors_train_names]
        plt.scatter(xs_train, ys_train, color=colors_train, label='training points')



    # optg_points
    color_map_opt = {'first_guess': cmap(0), 'opt_1': cmap(0.1), 'opt_2': cmap(0.2), 'opt_3': cmap(0.3),\
                     'opt_4': cmap(0.4), 'opt_5': cmap(0.5), 'opt_6':cmap(0.6), 'opt_7':cmap(0.7), 'opt_8':cmap(0.8), 'opt_9':cmap(0.9), 'opt_10':cmap(1), 'dft_optg': 'white',\
                                             'opt_11': cmap(0.1), 'opt_12': cmap(0.2), 'opt_13': cmap(0.3),\
                     'opt_14': cmap(0.4), 'opt_15': cmap(0.5), 'opt_16': cmap(0.6), 'opt_17': cmap(0.7), 'opt_18': cmap(0.8),\
                     'opt_19': cmap(0.9), 'opt_20': cmap(1), 'opt_21':cmap(0.1)}

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


    for idx, pt in enumerate(optg_pts):

        # if idx != 0 and idx != 1 and idx != len(optg_pts)-1:
        #     label=None
        # else:
        #     label = pt['label']
        label = pt['label']

        plt.scatter(pt['x'], pt['y'], color=pt['color'], label=label, marker='X', linewidth=0.5, s=80, \
                    linewidths=10, edgecolors='k')
        no = re.findall(r'\d+', pt['label'])
        if len(no)>0:
            plt.gca().annotate(int(no[0]), xy=(pt['x'], pt['y']))


    plt.legend()
    which=''
    if 'default' in pic_name:
        which = 'default soap'
    elif 'my_soap' in pic_name:
        which = 'my soap'

    plt.title(f'kPCA ({which}) on GAP_i training sets and GAP_i-optimised structures')
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')

    if output_dir is not None:
        pic_name = os.path.join(output_dir, pic_name)

    plt.tight_layout()
    plt.savefig(f'{pic_name}.png', dpi=300)
    plt.close(fig)


def make_kpca_dset(training_set, all_opt_ats, first_guess, dft_optg, xyz_fname):
    ## main training points
    kpca_ats = read(training_set, ':')
    kpca_ats = [at for at in kpca_ats if len(at) != 1]

    ## all gap_i optimised atoms. expect dft_energy to be correct...
    gap_opt_ats = read(all_opt_ats, ':')
    if 'config_type' not in gap_opt_ats[0].info.keys():
        for i, at in enumerate(gap_opt_ats):
            at.info['config_type'] = f'opt_{i+1}'
    elif gap_opt_ats[0].info['config_type'] != 'opt_1':
        print("WARNING: GAP-optimised atoms have config, types, but they don't follow convention of opt_{i}, renaming")
        for i, at in enumerate(gap_opt_ats):
            at.info['config_type'] = f'opt_{i}'

    ##label first_guess
    first_guess = read(first_guess)
    first_guess.info['config_type'] = 'first_guess'

    ## dft-optimized structure
    dft_optg = read(dft_optg)
    dft_optg.info['config_type'] = 'dft_optg'

    write(xyz_fname, kpca_ats + [first_guess] + gap_opt_ats + [dft_optg], 'extxyz', write_results=False)


def make_kpca_plots(training_set, all_opt_ats='xyzs/opt_all.xyz', first_guess='xyzs/first_guess.xyz', \
                    dft_optg='molpro_optg/optimized.xyz', xyz_fname='xyzs/for_kpca.xyz',  \
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
    make_kpca_dset(training_set, all_opt_ats, first_guess, dft_optg, xyz_fname)

    print('Kpca with default setings')
    xyz_title = os.path.splitext(xyz_fname)[0]
    kpca_default_xyz_fname = f'{xyz_title}_default.xyz'

    kpca.main(xyz_fname, pbc=False, output_filename=kpca_default_xyz_fname)

    pic_names = ['kpca_default']
    kpca_xyzs = [kpca_default_xyz_fname]

    if param_fname:
        print('Kpca with my soap settings')
        kpca_my_soap_xyz_fname = f'{xyz_title}_my_soap.xyz'
        s = ugap.get_soap_params(param_fname)
        kpca.main(xyz_fname, pbc=False, output_filename=kpca_my_soap_xyz_fname, \
                  cutoff=s['cutoff'], n_max=s['n_max'], l_max=s['l_max'], zeta=s['zeta'], \
                  atom_sigma=s['atom_gaussian_width'])

        kpca_xyzs.append(kpca_my_soap_xyz_fname)
        pic_names.append('kpca_my_soap')

    print('plotting plots')
    for colour_by_energy in [True, False]:
        for xyz_fname, pic_name in zip(kpca_xyzs, pic_names):
            if colour_by_energy:
                pic_name +='_by_energy'
            kpca_plot(xyz_fname, pic_name, output_dir, colour_by_energy=colour_by_energy)


def do_evec_plot(evals_dft, evecs_dft, evals_pred, evecs_pred, name, output_dir='pictures/'):
    mx = dict()
    for i, ev_dft in enumerate(evecs_dft):
        mx[f'dft_{i}'] = dict()
        for j, ev_pr in enumerate(evecs_pred):
            mx[f'dft_{i}'][f'pred_{j}'] = np.dot(ev_dft, ev_pr)

    df = pd.DataFrame(mx)
    N = len(evals_dft)
    if N > 30:
        figsize = (0.25 * N, 0.21 * N)
    else:
        figsize = (10, 8)
    fig, ax = plt.subplots(figsize=figsize)
    hmap = ax.pcolormesh(df, vmin=-1, vmax=1, cmap='bwr', edgecolors='lightgrey', linewidth=0.01)
    cbar = plt.colorbar(hmap)
    plt.yticks(np.arange(0.5, len(evals_pred), 1), [round(x, 3) for x in evals_pred])
    plt.xticks(np.arange(0.5, len(evals_dft), 1), [round(x, 3) for x in evals_dft], rotation=90)
    plt.xlabel('DFT eigenvalues')
    plt.ylabel('Predicted eigenvalues')
    plt.title(f'Dot products between DFT and {name} eigenvectors', fontsize=14)
    plt.tight_layout()

    name = f'{name}_evecs.png'
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        name = os.path.join(output_dir, name)

    plt.savefig(name, dpi=300)
    plt.close(fig)

def evec_plot(param_fname, first_guess='xyzs/first_guess.xyz', dft_optg='molpro_optg/optimized.xyz', fmax=1e-2, steps=1000, output_dir='pictures'):
    gap_title = os.path.splitext(os.path.basename(param_fname))[0]
    gap = Potential(param_filename=param_fname)
    gap_no = int(re.findall(r'\d+', gap_title)[0])

    gap_optg_name = f'xyzs/{gap_title}_optg_for_NM.xyz'
    # might have optimised stuff with this gap for eval plot, check.
    if not os.path.isfile(f'{gap_title}_optg.all.pckl'):

        if not os.path.isfile(f'xyzs/opt_at_{gap_no}.xyz'):
            print(f'\n---Optimised structure xyzs/opt_at_{gap_no}.xyz not found, optimising first_guess with {gap_title} and getting Normal Modes\n')
            atoms = read(first_guess)
            atoms.set_calculator(gap)

            # optimize
            opt = PreconLBFGS(atoms, trajectory=f'xyzs/{gap_title}_optg_for_NM.traj')
            opt.run(fmax=fmax, steps=steps)
            write(gap_optg_name, atoms, 'extxyz', write_results=False)

        else:
            print(f'\n---Loading structure xyzs/opt_at_{gap_no}.xyz optimised with {gap_title} previously')
            atoms = read(f'xyzs/opt_at_{gap_no}.xyz')
            atoms.set_calculator(gap)

        # get NM
        vib_gap = Vibrations(atoms, name=f'{gap_title}_optg')
        atoms_gap = vib_gap.run()
        vib_gap.summary()
    else:
        print(f'Found .all.pckl for {gap_title}, loading stuff')
        try:
            atoms = read(f'xyzs/opt_at_{gap_no}.xyz')
        except FileNotFoundError:
            atoms = read(gap_optg_name)
        vib_gap = Vibrations(atoms, name=f'{gap_title}_optg')
        vib_gap.summary()

    # dft NM
    dft_atoms = read(dft_optg)
    vib_dft = Vibrations(dft_atoms, name='dft_optg')
    vib_dft.summary()

    do_evec_plot(vib_dft.evals, vib_dft.evecs, vib_gap.evals, vib_gap.evecs, gap_title, output_dir=output_dir)


def opt_summary_plots(opt_all='xyzs/opt_all.xyz', dft_optg='molpro_optg/optimized.xyz', gaps_dir='gaps', \
                      output_dir='pictures'):
    dft_optg = read(dft_optg)
    dft_min = dft_optg.info['dft_energy']

    gap_fnames_all = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames_all = util.natural_sort(gap_fnames_all)

    atoms_all = read(opt_all, ':')

    group_size = 10

    for super_idx, (atoms, gap_fnames) in enumerate(zip(util.grouper(atoms_all, group_size), util.grouper(gap_fnames_all, group_size))):

        # remove 'None' padding from grouper
        atoms = [at for at in atoms if at is not None]
        gap_fnames = [gap_fname for gap_fname in gap_fnames if gap_fname is not None]

        dft_energies = [at.info['dft_energy']/len(at) - dft_min/len(at) for at in atoms]
        dft_fmaxs = [max(at.arrays['dft_forces'].flatten()) for at in atoms]

        fig1 = plt.figure(figsize=(12, 8))
        ax1 = plt.gca()

        fig2 = plt.figure(figsize=(12, 8))
        ax2 = plt.gca()

        N = len(atoms)
        cmap = mpl.cm.get_cmap('tab10')
        colors = np.linspace(0, 1, 10)


        for idx, gap_fname in enumerate(gap_fnames):

            absolute_idx = group_size*super_idx + idx

            gap_title = os.path.splitext(gap_fname)[0]
            gap_fname = os.path.join(gaps_dir, gap_fname)
            gap = Potential(param_filename=gap_fname)

            gap_energies_shifted = []
            gap_fmaxes = []
            for aa in atoms:
                at = aa.copy()
                at.set_calculator(gap)
                gap_energies_shifted.append((at.get_potential_energy()-dft_min)/len(at))
                gap_fmaxes.append(max(at.get_forces().flatten()))
            # gap_energies_shifted = [e - dft_min for e in gap_energies]

            c = cmap(colors[idx%10])

            E_label = f'GAP {absolute_idx + 1}'
            F_label = f'GAP {absolute_idx + 1}'


            if idx != 0:
                ax1.plot(range(group_size*super_idx+1, absolute_idx + 2), gap_fmaxes[:idx + 1], marker='x', label=F_label, color=c)
                ax2.plot(range(group_size*super_idx+1, absolute_idx + 2), np.absolute(gap_energies_shifted[:idx + 1]), marker='x', markersize=10, linestyle='-', label=E_label, color=c)

            if idx != N - 1:
                if idx != 0:
                    E_label = None
                    F_label = None


                ax1.plot(range(absolute_idx + 1, group_size*super_idx + len(gap_fmaxes) + 1), gap_fmaxes[idx:], marker='x', label=F_label, linestyle=':', color=c, alpha=0.7)
                ax2.plot(range(absolute_idx + 1, group_size*super_idx + len(gap_energies_shifted) + 1), np.absolute(gap_energies_shifted[idx:]), marker='x',
                         markersize=10, linestyle=':',  label=E_label, color=c, alpha=0.7)

            ax1.annotate(f'{gap_fmaxes[idx]:.4f}', xy=(absolute_idx + 1, gap_fmaxes[idx]))
            ax2.annotate(f'{np.absolute(gap_energies_shifted[idx]):.4f}', xy=(absolute_idx+1, np.absolute(gap_energies_shifted[idx])))

        ax1.plot(range(group_size*super_idx+1, group_size*super_idx + len(dft_fmaxs) + 1), dft_fmaxs, marker='+', markersize=10, label=f'DFT', color='k', linestyle='--')
        ax2.plot(range(group_size*super_idx+1, group_size*super_idx + len(dft_energies) + 1), np.absolute(dft_energies), marker='+', markersize=10, label=f'DFT', color='k', linestyle='--')

        ax1.annotate(f'{dft_fmaxs[idx]:.4f}', xy=(absolute_idx + 1, dft_fmaxs[idx]))
        ax2.annotate(f'{np.absolute(dft_energies[idx]):.4f}', xy=(absolute_idx + 1, np.absolute(dft_energies[idx])))

        for ax in [ax1, ax2]:
            ax.set_xlabel('iteration')
            ax.grid(which='both', c='lightgrey')
            ax.set_yscale('log')
            ax.legend(title='Evaluated with:', loc='upper left')
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        ax2.set_title(f'Energy error wrt DFT-OPTG structure on GAP_i-optimised structures {super_idx+1}')
        ax2.set_ylabel('|E - E$_{DFT\ OPTG}$|, eV/atom', fontsize=12)
        # fig2.tight_layout()

        ax1.set_title(f'Maximum force component on GAP_i-optimised structures {super_idx+1}')
        ax1.set_ylabel('Fmax, eV/Å', fontsize=12)
        # fig1.tight_layout()

    all_fig_nos = plt.get_fignums()
    fmax_fig_nos = all_fig_nos[0::2]
    e_fig_nos = all_fig_nos[1::2]

    for name_prfx, nos in zip(['opt_fmax_vs_iter', 'opt_energy_vs_iter'], [fmax_fig_nos, e_fig_nos]):

        all_ylim_upper = []
        all_ylim_lower = []
        for idx in nos:
            fig = plt.figure(idx)
            ylim = fig.get_axes()[0].get_ylim()
            all_ylim_upper.append(ylim[1])
            all_ylim_lower.append(ylim[0])

        lower_ylim = min(all_ylim_lower)
        upper_ylim = max(all_ylim_upper)

        for i, idx in enumerate(nos):
            fig = plt.figure(idx)
            fig.get_axes()[0].set_ylim(lower_ylim, upper_ylim)
            fig.tight_layout()

            name = f'{name_prfx}_{i+1}.png'
            if output_dir is not None:
                name = os.path.join(output_dir, name)

            plt.savefig(name, dpi=300)
            plt.close(fig)



        # fig2_name = f'opt_energy_vs_iter_{super_idx+1}.png'
        # fig1_name = f'opt_fmax_vs_iter_{super_idx+1}.png'
        # if output_dir:
        #     fig2_name = os.path.join(output_dir, fig2_name)
        #     fig1_name = os.path.join(output_dir, fig1_name)
        # fig2.savefig(fig2_name, dpi=300)
        # fig1.savefig(fig1_name, dpi=300)


def eval_plot(gaps_dir='gaps', first_guess='xyzs/first_guess.xyz', dft_optg='molpro_optg/optimized.xyz',
              dft_vib_name='dft_optg', \
              fmax=1e-3, steps=1000, output_dir='pictures'):


    print('\n---DFT vib modes\n')
    dft_atoms = read(dft_optg)
    vib_dft = Vibrations(dft_atoms, name=dft_vib_name)
    vib_dft.summary()

    gap_fnames_all = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames_all = util.natural_sort(gap_fnames_all)

    group_size=10
    # group_size=3

    for super_idx, gap_fnames in enumerate(util.grouper(gap_fnames_all, group_size)):
        # if super_idx == 2:
        #     break

        gap_fnames = [gap_fname for gap_fname in gap_fnames if gap_fname is not None]

        plt.figure(figsize=(8, 5))
        for idx, gap_fname in enumerate(gap_fnames):

            absolute_idx = group_size * super_idx + idx

            gap_title = os.path.splitext(gap_fname)[0]
            gap_fname = os.path.join(gaps_dir, gap_fname)
            gap = Potential(param_filename=gap_fname)
            gap_no = int(re.findall(r'\d+', gap_title)[0])

            gap_optg_name = f'xyzs/{gap_title}_optg_for_NM.xyz'
            # might have done this for eval plots, check
            if not os.path.isfile(f'{gap_title}_optg.all.pckl'):
                if not os.path.isfile(f'xyzs/opt_at_{gap_no}.xyz'):
                    print(f'\n--- optimised structure (xyzs/opt_at_{gap_no}.xyz) not found, Optimising first_guess with {gap_title}\n')
                    atoms = read(first_guess)
                    atoms.set_calculator(gap)

                    # optimize
                    opt = PreconLBFGS(atoms, trajectory=f'xyzs/{gap_title}_optg_for_NM.traj')
                    opt.run(fmax=fmax, steps=steps)
                    write(gap_optg_name, atoms, 'extxyz', write_results=False)

                else:
                    print(f'\n---Loading structure optimised with {gap_title} previously')
                    atoms = read(f'xyzs/opt_at_{gap_no}.xyz')
                    atoms.set_calculator(gap)

                # get NM
                print(f'\n{gap_title} Normal Modes\n')
                vib_gap = Vibrations(atoms, name=f'{gap_title}_optg')
                atoms_gap = vib_gap.run()
                vib_gap.summary()
            else:
                print(f'\n---Found .all.pckl for {gap_title}, loading stuff\n')
                # either optimised now and saved in optg_for_NM.xyz or load previously optimised stuff
                try:
                    atoms = read(f'xyzs/opt_at_{gap_no}.xyz')
                except FileNotFoundError:
                    atoms = read(gap_optg_name)
                vib_gap = Vibrations(atoms, name=f'{gap_title}_optg')
                vib_gap.summary()

            evals = vib_gap.evals
            rmse = util.get_rmse(evals, vib_dft.evals)
            # plt.plot(range(len(evals)), evals, label=f'{gap_title}, RMSE: {rmse:.4f} eV$^2$')
            plt.scatter(vib_dft.evals, evals-vib_dft.evals, marker='x', label=f'{gap_title}, RMSE: {rmse:.4f} eV$^2$')

        # plt.plot(range(len(vib_dft.evals)), vib_dft.evals, label='DFT', linewidth=0.8, linestyle='--', color='k')

        ax = plt.gca()
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        plt.legend(loc='upper left')
        plt.grid(color='lightgrey')
        # plt.xlabel('#')
        # plt.ylabel('eigenvalue, eV$^2$')
        plt.xlabel('DFT eigenvalue, eV$^2$')
        plt.ylabel('GAP_i eval - DFT eval, eV$^2$')
        plt.title('Ordered Eigenvalues')
        plt.tight_layout()


    all_ylim_upper = []
    all_ylim_lower = []
    for idx in plt.get_fignums():
        fig = plt.figure(idx)
        ylim = fig.get_axes()[0].get_ylim()
        all_ylim_upper.append(ylim[1])
        all_ylim_lower.append(ylim[0])


    lower_ylim = min(all_ylim_lower)
    upper_ylim = max(all_ylim_upper)

    for idx in plt.get_fignums():
        fig = plt.figure(idx)
        fig.get_axes()[0].set_ylim(lower_ylim, upper_ylim)

        name = f'eval_plot_{idx}.png'
        if output_dir is not None:
            name = os.path.join(output_dir, name)

        plt.savefig(name, dpi=300)
        plt.close(fig)

def rmsd_plot(opt_all='xyzs/opt_all.xyz', dft_optg='molpro_optg/optimized.xyz', output_dir='pictures'):
    dft_optg_at = read(dft_optg)
    atoms = read(opt_all, ':')
    rmsd = [util.get_rmse(dft_optg_at.positions, at.positions) for at in atoms]

    fig = plt.figure(figsize=(8,5))
    plt.plot(range(1, len(rmsd)+1), rmsd, marker='x', markersize=10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.xlabel('Iteration')
    plt.ylabel('RMSD, Å')
    plt.grid(color='lightgrey', which='both')
    plt.title('Geometry optimisation, GAP i vs DFT')
    plt.yscale('log')
    plt.tight_layout()
    name = 'rmsd.png'
    if output_dir is not None:
        name = os.path.join(output_dir, name)
    plt.savefig(name, dpi=300)
    plt.close(fig)


