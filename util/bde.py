from wfl.configset import ConfigSet_in, ConfigSet_out
import re
import numpy as np
from ase.io import read, write
from ase import Atoms
import util
from util import iter_tools as it
import pandas as pd
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
from wfl.utils.parallel import construct_calculator_picklesafe
import matplotlib.ticker as mticker



def dirs_to_fnames(dft_dir, gap_dir=None, start_dir=None, exclude=None):
    dft_basenames = util.natural_sort(os.listdir(dft_dir))

    if exclude is not None:
        if isinstance(exclude, str):
            exclude = exclude.split(' ')
        dft_basenames = [bn for bn in dft_basenames if bn not in exclude]

    dft_fnames = [os.path.join(dft_dir, fname) for fname in dft_basenames]
    if gap_dir is not None:
        gap_fnames = [
            os.path.join(gap_dir, basename.replace('optimised', 'gap_optimised'))
            for basename in dft_basenames]

    if start_dir is not None:
        start_fnames = [os.path.join(start_dir, basename.replace('optimised',
                                                                 'non_optimised'))
                        for
                        basename in dft_basenames]
    else:
        start_fnames = [None for _ in dft_fnames]

    return dft_fnames, gap_fnames, start_fnames


def multi_bde_summaries(dft_dir, gap_dir=None, calculator=None, start_dir=None,
                         precision=3, printing=True, dft_prefix='dft_',
                        exclude=None):

    """BDEs for all files in the directory"""
    if gap_dir is not None:
        if not os.path.isdir(gap_dir):
            os.makedirs(gap_dir)

    dft_basenames = util.natural_sort(os.listdir(dft_dir))
    if exclude is not None:
        exclude = exclude.split(' ')
        dft_basenames = [bn for bn in dft_basenames if bn not in exclude]

    dft_fnames = [os.path.join(dft_dir, fname) for fname in dft_basenames]
    if gap_dir is not None :
        gap_fnames = [os.path.join(gap_dir, basename.replace('optimised',
                                                             'gap_optimised'))
                      for
                      basename in dft_basenames]
        if start_dir is not None:
            start_fnames = [basename.replace('optimised', 'non_optimised')
                            for basename in dft_basenames]
            start_fnames = [os.path.join(start_dir, start_fname)
                            for start_fname in start_fnames]
        else:
            start_fnames = [None for _ in dft_basenames]

        if calculator:
            gap_optimise(start_fnames, gap_fnames, calculator)
            calculator=None

    else:
        gap_fnames = [None for _ in dft_fnames]
        start_fnames = [None for _ in dft_fnames]

    all_bdes = []
    for dft_fname, gap_fname, start_fname in zip(dft_fnames, gap_fnames,
                                                 start_fnames):
        bdes = bde_summary(dft_fname, gap_fname, calculator, start_fname,
                        precision, printing, dft_prefix)
        all_bdes.append(bdes)

    return all_bdes


def bde_summary(dft_fname, gap_fname=None, calculator=None, start_fname=None,
                precision=3, printing=True, dft_prefix='dft_'):

    dft_ats = read(dft_fname, ':')
    dft_h = Atoms('H', positions=[(0, 0, 0)])
    dft_h.info['config_type'] = 'H'
    dft_h.info[f'{dft_prefix}energy'] = -13.547449778462548
    dft_ats = [dft_h] + dft_ats


    if gap_fname is not None and os.path.isfile(gap_fname):
        gap_ats = read(gap_fname, ':')
    elif gap_fname is not None and not os.path.isfile(gap_fname):
        gap_optimise(start_fname, gap_fname, calculator)
        gap_ats = read(gap_fname, ':')
    else:
        gap_ats = None

    bdes = get_bdes(dft_ats, gap_ats, dft_prefix)

    if printing:
        print('-' * 30)
        print(os.path.basename(os.path.splitext(dft_fname)[0]))
        print('-' * 30)

        headers = [' ', "eV\nDFT E", "eV\nDFT BDE"]
        if gap_ats is not None:
            headers += ["eV\nGAP E", "eV\nGAP BDE", "meV\nBDE abs error",
                        "Å\nRMSD", "SOAP dist", 'meV\nGAP E abs error']

        print(tabulate(bdes, headers=headers, floatfmt=f".{precision}f"))

    return pd.DataFrame(bdes)


def gap_optimise(start_fnames, gap_fnames, calculator):
    if start_fnames is None or start_fnames[0] is None:
        raise RuntimeError('Don\'t have a start file to optimise')

    if isinstance(start_fnames, str) and isinstance(gap_fnames, str):
        start_fnames = [start_fnames]
        gap_fnames = [gap_fnames]

    gap_tmp_fnames = [os.path.splitext(gap_fname)[0] + '_tmp.xyz' for gap_fname
                      in gap_fnames]

    dir = os.path.dirname(gap_fnames[0])
    opt_dir = os.path.join(dir, 'opt_trajectories')
    if not os.path.isdir(opt_dir):
        os.makedirs(opt_dir)

    base_names = [os.path.basename(os.path.splitext(gap_fname)[0]) for
                  gap_fname in gap_fnames]
    traj_fnames = [os.path.join(opt_dir, base_name + '_traj.xyz') for base_name
                   in base_names]

    # log_fname = os.path.join(opt_dir, 'opt_log.txt') 

    gap_h = Atoms('H', positions=[(0, 0, 0)])
    gap_h.info['config_type'] = 'H'


    for start_fname, gap_tmp_fname in zip(start_fnames, gap_tmp_fnames):
        starts = [gap_h.copy()] + read(start_fname, ':')
        write(gap_tmp_fname, starts)

    output_dict = {}
    for in_fname, out_fname in zip(gap_tmp_fnames, traj_fnames):
        output_dict[in_fname] = out_fname

    inputs = ConfigSet_in(input_files=gap_tmp_fnames)
    outputs = ConfigSet_out(output_files=output_dict)


    it.run_opt(inputs, outputs, calculator, fmax=1e-2, return_traj=True,
               logfile=None, chunksize=20)

    calculator = construct_calculator_picklesafe(calculator)

    for traj_fname, gap_fname, gap_tmp_fname in zip(traj_fnames, gap_fnames,
                                                    gap_tmp_fnames):
        optimised_atoms = read(traj_fname, ':')
        opt_ats = [at for at in optimised_atoms if 'minim_config_type' in
                   at.info.keys() and 'converged' in at.info[
                       'minim_config_type']]

        for at in opt_ats:
            at.set_calculator(calculator),
            at.info['gap_energy'] = at.get_potential_energy()
            at.arrays['gap_forces'] = at.get_forces()

        write(gap_fname, opt_ats)
        os.remove(gap_tmp_fname)


def get_bdes(dft_ats, gap_ats=None, dft_prefix='dft_'):
    label_pattern = re.compile(r"rad_?\d+$|mol$|H$")

    dft_h = dft_ats[0]
    dft_mol = dft_ats[1]

    assert 'H' == label_pattern.search(dft_h.info['config_type']).group()
    assert 'mol' == label_pattern.search(dft_mol.info['config_type']).group()

    dft_h_energy = dft_h.info[f'{dft_prefix}energy']
    dft_mol_energy = dft_mol.info[f'{dft_prefix}energy']

    h_data = ['H', dft_h_energy, np.nan]
    mol_data = ['mol', dft_mol_energy, np.nan]

    if gap_ats is not None:

        gap_h = gap_ats[0]
        gap_mol = gap_ats[1]

        assert 'H' == label_pattern.search(gap_h.info['config_type']).group()
        assert 'mol' == label_pattern.search(
            gap_mol.info['config_type']).group()

        try:
            gap_h_energy = gap_h.info['gap_energy']
            gap_mol_energy = gap_mol.info['gap_energy']

            dft_e_of_gap_mol = gap_mol.info['dft_energy']
        except KeyError:
            print(f'info: {gap_h.info}, {gap_mol.info}')
            raise

        h_error = abs(dft_h_energy - gap_h_energy) * 1e3
        mol_error = abs(dft_mol_energy - gap_mol_energy) * 1e3
        mol_rmsd = util.get_rmse(dft_mol.positions, gap_mol.positions)
        mol_soap_dist = util.soap_dist(dft_mol, gap_mol)
        gap_abs_e_error = abs(dft_e_of_gap_mol - gap_mol_energy) * 1e3

        h_data += [gap_h_energy, np.nan, h_error, np.nan, np.nan, np.nan]
        mol_data += [gap_mol_energy, np.nan, mol_error, mol_rmsd, mol_soap_dist,
                     gap_abs_e_error]
    else:
        gap_ats = [None for _ in dft_ats]

    data = []
    data.append(h_data)
    data.append(mol_data)

    bde_errors = []
    rmsds = []
    soap_dists = []
    gap_e_errors = []
    for dft_at, gap_at in zip(dft_ats[2:], gap_ats[2:]):

        label = label_pattern.search(dft_at.info['config_type']).group()

        dft_rad_e = dft_at.info[f'{dft_prefix}energy']
        dft_bde = dft_rad_e + dft_h_energy - dft_mol_energy

        data_line = [label, dft_rad_e, dft_bde]

        if gap_at is not None:
            gap_rad_e = gap_at.info['gap_energy']
            gap_bde = gap_rad_e + gap_h_energy - gap_mol_energy
            bde_error = abs(dft_bde - gap_bde) * 1e3
            bde_errors.append(bde_error)

            rmsd = util.get_rmse(dft_at.positions, gap_at.positions)
            rmsds.append(rmsd)

            soap_dist = util.soap_dist(dft_at, gap_at)
            soap_dists.append(soap_dist)

            dft_e_of_gap_rad = gap_at.info['dft_energy']
            gap_e_error = abs(gap_rad_e - dft_e_of_gap_rad) * 1e3
            gap_e_errors.append(gap_e_error)

            data_line += [gap_rad_e, gap_bde, bde_error, rmsd, soap_dist, gap_e_error]

        data.append(data_line)

    if gap_ats[0] is not None:
        data.append(
            ['mean', np.nan, np.nan, np.nan, np.nan, np.mean(bde_errors),
             np.mean(rmsds), np.mean(soap_dists), np.mean(gap_e_errors)])

    return data


def bde_bar_plot(gap_fnames, dft_fnames, plot_title='bde_bar_plot',
                 start_fnames=None,
                 calculator=None,
                 output_dir='pictures'):
    if start_fnames is None:
        start_fnames = [None for _ in gap_fnames]

    all_titles = []
    all_dft_bdes = []
    all_gap_bdes = []
    for gap_fname, dft_fname, start_fname in zip(gap_fnames, dft_fnames,
                                                 start_fnames):
        #         print(gap_fname)

        title = os.path.basename(os.path.splitext(dft_fname)[0]).replace(
            '_optimised', '')
        if '_bde_train_set' in title:
            title = title.replace('_bde_train_set', '')
        all_titles.append(title)

        bdes = bde_summary(dft_fname=dft_fname, gap_fname=gap_fname,
                           start_fname=start_fname,
                           calculator=calculator,
                           printing=False)

        dft_bdes = np.array(bdes[2][2:-1])
        gap_bdes = np.array(bdes[4][2:-1])

        all_dft_bdes.append(dft_bdes)
        all_gap_bdes.append(gap_bdes)

    bar_categories = all_titles
    bars_dft = [np.mean(dft_bdes) for dft_bdes in all_dft_bdes]
    errors_dft = [np.std(dft_bdes) for dft_bdes in all_dft_bdes]
    bars_gap = [np.mean(gap_bdes) for gap_bdes in all_gap_bdes]
    errors_gap = [np.std(gap_bdes) for gap_bdes in all_gap_bdes]
    width = 0.4

    plt.figure()
    plt.grid(color='lightgrey')
    plt.bar(bar_categories, bars_dft, yerr=errors_dft, width=-width,
            align='edge', color='tab:red',
            zorder=2, label='DFT')
    plt.bar(bar_categories, bars_gap, yerr=errors_gap, width=width,
            align='edge', color='tab:blue',
            zorder=2, label='GAP')
    plt.ylabel('Mean BDE / eV')
    plt.title(plot_title)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xticks(rotation=90)
    # plt.show()
    plt.tight_layout()
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = ''

    plt.savefig(os.path.join(output_dir, plot_title + '.png'), dpi=300)


def get_data(dft_fnames, gap_fnames, selection=None, start_fnames=None,
             calculator=None, which_data='bde'):
    '''which_data = 'bde', 'rmsd' or 'soap'. if 'rmsd' is selected,
    all_values[title][set]['dft'] corresponds to the DFT bdes and
    all_values[title][set]['gap'] = to rmsd or soap'''

    dft_val_idx = 2
    if which_data == 'bde':
        gap_val_idx = 4
    elif which_data == 'rmsd':
        gap_val_idx = 6
    elif which_data == 'soap_dist':
        gap_val_idx = 7
    elif which_data == 'gap_e_error':
        gap_val_idx = 8

    if selection is None:
        selection = {}

    if start_fnames is None:
        start_fnames = [None for _ in dft_fnames]

    all_values = {}

    for gap_fname, dft_fname, start_fname, in zip(gap_fnames, dft_fnames,
                                                  start_fnames):

        print(gap_fname)

        title = os.path.basename(os.path.splitext(gap_fname)[0]).replace(
            '_gap_optimised.xyz', '')
        if '_bde_train_set' in title:
            title = title.replace('_bde_train_set', '')

        all_values[title] = {'train': {'gap': [], 'dft': []},
                           'test': {'gap': [], 'dft': []}}

        bde_table = bde_summary(dft_fname=dft_fname, gap_fname=gap_fname,
                           start_fname=start_fname, calculator=calculator,
                           printing=False)

        bde_table = bde_table.drop(bde_table.index[[0, 1, -1]])

        selected_h_list = []
        for key, vals in selection.items():
            if key in title:
                selected_h_list = vals

        if selected_h_list == 'all':
            all_values[title]['test']['dft'] = bde_table[dft_val_idx]
            all_values[title]['test']['gap'] = bde_table[gap_val_idx]
        elif len(selected_h_list) == 0:
            all_values[title]['train']['dft'] = bde_table[dft_val_idx]
            all_values[title]['train']['gap'] = bde_table[gap_val_idx]
        else:
            for idx, row in bde_table.iterrows():
                for rad_no in selected_h_list:
                    if str(rad_no) in row[0]:
                        all_values[title]['test']['dft'].append(row[dft_val_idx])
                        all_values[title]['test']['gap'].append(row[gap_val_idx])
                        break
                else:
                    all_values[title]['train']['dft'].append(row[dft_val_idx])
                    all_values[title]['train']['gap'].append(row[gap_val_idx])

        for set_name in ['train', 'test']:
            for method_name in ['dft', 'gap']:
                all_values[title][set_name][method_name] = np.array(
                    all_values[title][set_name][method_name])

    return all_values


def scatter_plot(all_data,
                 plot_title=None,
                 output_dir='pictures',
                 which_data = 'bde'):

    if isinstance(all_data, dict):
        all_data = [all_data]

    if which_data == 'bde':
        shade = 50
        if plot_title is None:
            plot_title = 'bde_scatter'
        fill_label = f'<50 meV'
        hline_label = '50 meV'
        ylabel = '|DFT BDE - GAP BDE| / meV'

    elif which_data == 'rmsd':
        shade = 0.1
        if plot_title is None:
            plot_title = 'rmsd_scatter'
        fill_label = f'< 0.1 Å'
        hline_label = '0.1 Å'
        ylabel = 'RMSD / Å'
    elif which_data == 'soap_dist':
        shade =None
        if plot_title is None:
            plot_title = 'soap_distance_scatter'
        fill_label=f'< {shade}'
        hline_label = f'{shade}'
        ylabel = 'SOAP distance'
    elif which_data == 'gap_e_error':
        shade = 50
        if plot_title is None:
            plot_title = 'soap_energy_error_scatter'
        fill_label = f'< {shade} meV'
        hline_label = f'{shade} meV'
        ylabel = 'absolute GAP vs DFT error, meV'

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    # plt.axhline(shade, linewidth=0.8, color='k', zorder=2,
    # label=hline_label)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.arange(10)]

    ref_data = all_data[0]

    for idx, label in enumerate(ref_data.keys()):

        print_label = label.replace('_gap_optimised', '')
        phrase_2 = '_bde_train_set'
        if phrase_2 in print_label:
            print_label = print_label.replace(phrase_2, '')

        for set_name, m in zip(ref_data[label].keys(), ['.', 'x']):

            if len(ref_data[label][set_name]['dft']) == 0:
                continue

            if set_name == 'test':
                label += 'test'

            color = colors[idx % 10]

            if which_data in ['rmsd', 'soap_dist', 'gap_e_error']:
                all_ys = []
                all_xs = []
                for data in all_data:
                    y_data = data[label][set_name]['gap']
                    all_ys.append(y_data)
                    all_xs.append(data[label][set_name]['dft'])

            else:
                all_ys = []
                all_xs = []
                for data in all_data:
                    dft_bdes = data[label][set_name]['dft']
                    gap_bdes = data[label][set_name]['gap']
                    all_ys.append(np.abs(dft_bdes - gap_bdes) * 1000)
                    all_xs.append(data[label][set_name]['dft'])

            all_xs = np.array(all_xs).T
            stds = np.array([np.std(vals) for vals in all_xs])
            xs = np.array([np.mean(vals) for vals in all_xs])
            if np.all(stds != 0):
                raise RuntimeError("All DFT BDEs aren't the same")

            all_ys = np.array(all_ys)
            all_ys = all_ys.T

            if len(all_data) == 1:
                ys = all_ys
                yerr = None
                if m == '.':
                    m = 'o'
                fmt = m
                legend_title = None

            elif len(all_data) == 2:
                lower = np.array([np.min(vals) for vals in all_ys])
                upper = np.array([np.max(vals) for vals in all_ys])
                ys = upper
                yerr = np.array([lower, upper])
                fmt = 'none'
                legend_title = 'Err bars: 2 BDEs - min & max'

            elif len(all_data) == 3:
                lower = np.array([np.min(vals) for vals in all_ys])
                upper = np.array([np.max(vals) for vals in all_ys])
                median = np.array([np.median(vals) for vals in all_ys])

                ys = median
                yerr = np.array([lower, upper])
                fmt = m

                legend_title = 'Err bars: 3 BDEs - min, median & max'

            else:

                means = np.array([np.mean(vals) for vals in all_ys])
                stds = np.array([np.std(vals) for vals in all_ys])

                ys = means
                yerr = stds
                fmt = m
                legend_title = 'Err bars: 3+ BDES - mean +- STD'

            plt.errorbar(xs, ys, yerr=yerr, zorder=3, fmt=fmt, color=color,
                         label=label, capsize=2)

    if shade is not None:
        xmin, xmax = ax.get_xlim()
        plt.fill_between([xmin, xmax], 0, shade, color='lightgrey',
                         label=fill_label, alpha=0.5)
        ax.set_xlim(xmin, xmax)

    plt.yscale('log')
    plt.grid(color='grey', ls=':', which='both')
    plt.ylabel(ylabel)
    plt.xlabel('DFT BDE / eV')
    plt.title(plot_title)
    plt.legend(title=legend_title, bbox_to_anchor=(1, 1))
    plt.tight_layout()


    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = ''

    plt.savefig(os.path.join(output_dir, plot_title + '.png'), dpi=300)

def iter_plot(all_data,
              plot_title=None,
              output_dir='pictures',
              which_data = 'bde',
              include=None):

    if which_data == 'bde':
        shade = 50
        if plot_title is None:
            plot_title = 'bde_scatter'
        fill_label = f'<50 meV'
        hline_label = '50 meV'
        ylabel = '|DFT BDE - GAP BDE| / meV'

    elif which_data == 'rmsd':
        shade = 0.1
        if plot_title is None:
            plot_title = 'rmsd_scatter'
        fill_label = f'< 0.1 Å'
        hline_label = '0.1 Å'
        ylabel = 'RMSD / Å'

    elif which_data == 'soap_dist':
        shade = 0.01
        if plot_title is None:
            plot_title = 'soap_distance_scatter'
        fill_label = f'< {shade}'
        hline_label = '{shade}'
        ylabel = 'SOAP distance'

    elif which_data == 'gap_e_error':
        shade = 50
        if plot_title is None:
            plot_title = 'soap_energy_error_scatter'
        fill_label = f'< {shade} meV'
        hline_label = f'{shade} meV'
        ylabel = 'absolute GAP vs DFT error, meV'


    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.arange(10)]

    ref_data = all_data[0]

    for idx, label in enumerate(ref_data.keys()):

        print_label = label.replace('_gap_optimised', '')
        phrase_2 = '_bde_train_set'
        if phrase_2 in print_label:
            print_label = print_label.replace(phrase_2, '')

        if include is not None:
            if label not in include and print_label not in include:
                continue

        for set_name, m in zip(ref_data[label].keys(), ['.', 'x']):

            if len(ref_data[label][set_name]['dft']) == 0:
                continue

            if set_name == 'test':
                label += 'test'

            color = colors[idx % 10]

            if which_data in ['rmsd', 'soap_dist', 'gap_e_error']:
                all_ys = []
                for data in all_data:
                    # corresponds to RMSD
                    all_ys.append(data[label][set_name]['gap'])


            elif which_data == 'bde':
                all_ys = []
                for data in all_data:
                    dft_bdes = data[label][set_name]['dft']
                    gap_bdes = data[label][set_name]['gap']
                    all_ys.append(np.abs(dft_bdes - gap_bdes) * 1000)

            xs = np.arange(len(all_data))

            all_ys = np.array(all_ys)

            for ys in all_ys.T:
                plt.plot(xs, ys, zorder=3, marker='x', color=color,
                         label=print_label)
                if print_label is not None:
                    print_label = None

    xmin, xmax = ax.get_xlim()
    plt.fill_between([xmin, xmax], 0, shade, color='lightgrey',
                     label=fill_label, alpha=0.5)
    ax.set_xlim(xmin, xmax)

    plt.yscale('log')
    plt.grid(color='grey', ls=':', which='both')
    plt.ylabel(ylabel)
    plt.xlabel('Iteration')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title(plot_title)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = ''

    plt.savefig(os.path.join(output_dir, plot_title + '.png'), dpi=300)
