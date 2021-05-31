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


def opt_to_bde_files(gap_ats_in, dft_ats_in,
                     iso_h_fname, output_dir='bdes_from_optimisations',
                     max_no_files=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dft_dir = os.path.join(output_dir, 'dft')
    gap_dir = os.path.join(output_dir, 'gap')
    if not os.path.exists(dft_dir):
        os.makedirs(dft_dir)
    if not os.path.exists(gap_dir):
        os.makedirs(gap_dir)

    gap_atoms = read(gap_ats_in, ':')
    dft_atoms = read(dft_ats_in, ':')
    hydrogen = read(iso_h_fname)

    assert len(gap_atoms) == len(dft_atoms)

    compounds1 = [at.info['compound'] for at in gap_atoms]
    compounds2 = [at.info['compound'] for at in dft_atoms]
    mol_or_rad1 = [at.info['mol_or_rad'] for at in gap_atoms]
    mol_or_rad2 = [at.info['mol_or_rad'] for at in dft_atoms]

    assert np.all(compounds1 == compounds2)
    assert np.all(mol_or_rad1 == mol_or_rad2)


    mol_indices = np.nonzero(np.array(mol_or_rad1) == 'mol')[0]

    for file_idx, (start, end) in enumerate(zip(mol_indices[0:-1], mol_indices[1:])):

        dft_bde_atoms = dft_atoms[start:end]
        gap_bde_atoms = [hydrogen] + gap_atoms[start:end]

        fname_prefix = gap_atoms[start].info['compound']

        for idx in range(max_no_files):
            gap_fname = os.path.join(gap_dir,
                                     f'{fname_prefix}_{idx}_gap_optimised.xyz')
            dft_fname = os.path.join(dft_dir,
                                     f'{fname_prefix}_{idx}_optimised.xyz')

            if os.path.exists(gap_fname) or os.path.exists(dft_fname):
                continue
            else:
                break
        else:
            raise RuntimeError(f'Havve {max_no_files} bde '
                               f'files for {fname_prefix} already')

        # print(f'{file_idx}. {gap_fname}')
        # print(f'   {dft_fname}')
        write(gap_fname, gap_bde_atoms)
        write(dft_fname, dft_bde_atoms)


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
                precision=3, printing=True, dft_prefix='dft_', gap_prefix='gap_'):

    dft_ats = read(dft_fname, ':')
    dft_h = Atoms('H', positions=[(0, 0, 0)])
    dft_h.info['config_type'] = 'H'
    dft_h.info[f'{dft_prefix}energy'] = -13.547449778462548
    dft_ats = [dft_h] + dft_ats


    if gap_fname is not None and os.path.isfile(gap_fname):
        gap_ats = read(gap_fname, ':')
    elif gap_fname is not None and not os.path.isfile(gap_fname):
        print(gap_fname)
        try:
            gap_optimise(start_fname, gap_fname, calculator)
        except RuntimeError as e:
            print(f'start_fname: {start_fname}\ngap_fname: {gap_fname}\ndft_fname: {dft_fname}')
            raise e
        gap_ats = read(gap_fname, ':')
    else:
        gap_ats = None

    bdes = get_bdes(dft_ats, gap_ats, dft_prefix, gap_prefix)

    if printing:
        print('-' * 30)
        print(os.path.basename(os.path.splitext(dft_fname)[0]))
        print('-' * 30)

        headers = [' ', "eV\nDFT BDE"]
        if gap_ats is not None:
            headers += ["eV\nGAP BDE", "BDE meV\nabs error",
                        "Å\nRMSD", "\nSOAP dist", 'GAP E meV\nabs error',
                        'GAP F meV/Å\nRMSE', 'GAP F meV/Å\nmax error']

        print(tabulate(bdes, headers=headers, floatfmt=f".{precision}f"))

    return pd.DataFrame(bdes)


def gap_optimise(start_fnames, gap_fnames, calculator):
    print('hiiii')
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


def get_bdes(dft_ats, gap_ats=None, dft_prefix='dft_', gap_prefix='gap_'):
    label_pattern = re.compile(r"rad_?\d+$|mol$|H$")

    dft_h = dft_ats[0]
    dft_mol = dft_ats[1]

    assert 'H' == label_pattern.search(dft_h.info['config_type']).group()
    assert 'mol' == label_pattern.search(dft_mol.info['config_type']).group()

    dft_h_energy = dft_h.info[f'{dft_prefix}energy']
    dft_mol_energy = dft_mol.info[f'{dft_prefix}energy']

    h_data = ['H', np.nan]
    mol_data = ['mol', np.nan]

    if gap_ats is not None:

        gap_h = gap_ats[0]
        gap_mol = gap_ats[1]

        assert 'H' == label_pattern.search(gap_h.info['config_type']).group()
        assert 'mol' == label_pattern.search(
            gap_mol.info['config_type']).group()

        have_dft = False
        if f'{dft_prefix}energy' in gap_mol.info.keys():
            have_dft=True

        try:
            gap_h_energy = gap_h.info[f'{gap_prefix}energy']
            gap_mol_energy = gap_mol.info[f'{gap_prefix}energy']
            if f'{gap_prefix}forces' in gap_mol.arrays.keys():
                gap_mol_forces = gap_mol.arrays[f'{gap_prefix}forces']
            else:
                gap_mol_forces=None

            if have_dft:
                dft_e_of_gap_mol = gap_mol.info['dft_energy']
                dft_f_of_gap_mol = gap_mol.arrays['dft_forces']
        except KeyError:
            print(f'info: {gap_h.info}, {gap_mol.info}')
            raise

        h_error = abs(dft_h_energy - gap_h_energy) * 1e3
        mol_error = abs(dft_mol_energy - gap_mol_energy) * 1e3
        mol_rmsd = util.get_rmse(dft_mol.positions, gap_mol.positions)
        mol_soap_dist = util.soap_dist(dft_mol, gap_mol)
        if have_dft:
            gap_abs_e_error = abs(dft_e_of_gap_mol - gap_mol_energy) * 1e3
            gap_f_rmse = util.get_rmse(gap_mol_forces, dft_f_of_gap_mol) * 1e3
            gap_f_max_err = np.max(np.abs(gap_mol_forces - dft_f_of_gap_mol)) * 1e3

        h_data += [ np.nan, h_error, np.nan, np.nan]
        mol_data += [np.nan, mol_error, mol_rmsd, mol_soap_dist]

        if have_dft:
            h_data += [np.nan, np.nan, np.nan]
            mol_data += [gap_abs_e_error, gap_f_rmse, gap_f_max_err]
    else:
        gap_ats = [None for _ in dft_ats]

    data = []
    data.append(h_data)
    data.append(mol_data)

    bde_errors = []
    rmsds = []
    soap_dists = []
    gap_e_errors = []
    gap_f_errors = []
    for dft_at, gap_at in zip(dft_ats[2:], gap_ats[2:]):

        label = label_pattern.search(dft_at.info['config_type']).group()

        dft_rad_e = dft_at.info[f'{dft_prefix}energy']
        dft_bde = dft_rad_e + dft_h_energy - dft_mol_energy

        data_line = [label, dft_bde]

        if gap_at is not None:
            gap_rad_e = gap_at.info['gap_energy']
            gap_bde = gap_rad_e + gap_h_energy - gap_mol_energy
            bde_error = abs(dft_bde - gap_bde) * 1e3
            bde_errors.append(bde_error)

            rmsd = util.get_rmse(dft_at.positions, gap_at.positions)
            rmsds.append(rmsd)

            soap_dist = util.soap_dist(dft_at, gap_at)
            soap_dists.append(soap_dist)

            if have_dft:
                dft_e_of_gap_rad = gap_at.info['dft_energy']
                gap_e_error = abs(gap_rad_e - dft_e_of_gap_rad) * 1e3
                gap_e_errors.append(gap_e_error)

                dft_f_of_gap_rad = gap_at.arrays['dft_forces']
                gap_rad_f = gap_at.arrays[f'{gap_prefix}forces']
                rad_f_rmse = util.get_rmse(dft_f_of_gap_rad, gap_rad_f) * 1e3
                gap_f_errors.append(list(np.ravel(dft_f_of_gap_rad - gap_rad_f)))

                gap_max_f_error = np.max(np.abs(dft_f_of_gap_rad - gap_rad_f)) * 1e3

            data_line += [gap_bde, bde_error, rmsd, soap_dist]

            if have_dft:
                data_line += [gap_e_error, rad_f_rmse, gap_max_f_error]

        data.append(data_line)

    if gap_ats[0] is not None:
        data.append(
            ['mean',  np.nan, np.nan, np.mean(bde_errors),
             np.mean(rmsds), np.mean(soap_dists), np.mean(gap_e_errors),
            np.sqrt(np.mean(np.array(gap_f_errors) ** 2))*1e3,
             np.nan])

    return data


def bde_bar_plot(gap_fnames, dft_fnames, plot_title='bde_bar_plot',
                 start_fnames=None,
                 calculator=None,
                 output_dir='pictures', which_data='bde'):
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

        dft_bdes = np.array(bdes[get_dft_idx_in_table()][2:-1])
        gap_bdes = np.array(bdes[property_to_table_idx(which_data)][2:-1])

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


def property_to_table_idx(which_data):
    if which_data == 'bde' or which_data == 'bde_correlation':
        return  2 # was 4
    elif which_data == 'rmsd':
        return  4  # was 6
    elif which_data == 'soap_dist':
        return  5  # was 7
    elif which_data == 'gap_e_error':
        return  6 # was 8
    elif which_data == 'gap_f_rmse':
        return 7
    elif which_data == 'gap_f_max':
        return 8

def get_dft_idx_in_table():
    return 1

def get_data(dft_fnames, gap_fnames, selection=None, start_fnames=None,
             calculator=None, which_data='bde', gap_prefix='gap_'):
    '''which_data = 'bde', 'rmsd' or 'soap'. if 'rmsd' is selected,
    all_values[title][set]['dft'] corresponds to the DFT bdes and
    all_values[title][set]['gap'] = to rmsd or soap'''

    drop_idx = [0, -1]
    if which_data in ['bde', 'bde_correlation']:
        drop_idx = [0, 1,  -1]

    dft_val_idx = get_dft_idx_in_table()
    gap_val_idx = property_to_table_idx(which_data)

    if selection is None:
        selection = {}

    if start_fnames is None:
        start_fnames = [None for _ in dft_fnames]

    all_values = {}

    for gap_fname, dft_fname, start_fname, in zip(gap_fnames, dft_fnames,
                                                  start_fnames):


        title = os.path.basename(os.path.splitext(gap_fname)[0]).replace(
            '_gap_optimised.xyz', '')
        if '_bde_train_set' in title:
            title = title.replace('_bde_train_set', '')

        all_values[title] = {'train': {'gap': [], 'dft': []},
                           'test': {'gap': [], 'dft': []}}

        bde_table = bde_summary(dft_fname=dft_fname, gap_fname=gap_fname,
                           start_fname=start_fname, calculator=calculator,
                           printing=False, gap_prefix=gap_prefix)

        bde_table = bde_table.drop(bde_table.index[drop_idx])

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
            plot_title = 'energy_error_scatter'
        fill_label = f'< {shade} meV'
        hline_label = f'{shade} meV'
        ylabel = 'absolute GAP vs DFT error, meV'
    elif which_data == 'gap_f_rmse':
        shade = 100
        if plot_title is None:
            plot_title = 'force_forces_scatter'
        fill_label = f'< {shade} meV/Å'
        hline_label = f'{shade} meV/Å'
        ylabel = 'force RMSE per molecule, meV/Å'
    elif which_data == 'gap_f_max':
        shade = 100
        if plot_title is None:
            plot_title = 'max_f_error_scatter'
        fill_label = f'< {shade} meV/Å'
        hline_label = f'{shade} meV/Å'
        ylabel = 'maximum force component error per molecule, meV/Å'
    elif which_data == 'bde_correlation':
        shade = 50
        if plot_title is None:
            plot_title = 'bde_scatter'
        fill_label = f'$\pm$ 50 meV'
        ylabel='GAP BDE / eV'

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # plt.axhline(shade, linewidth=0.8, color='k', zorder=2,
    # label=hline_label)

    n_colors = len(all_data[0].keys())
    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.linspace(0, 1, 10)]
    done_labels = []

    ref_data = all_data[0]
    color_idx = -1

    for idx, label in enumerate(ref_data.keys()):

        print_label = label.replace('_gap_optimised', '')
        # print(print_label)
        phrase_2 = '_bde_train_set'
        if phrase_2 in print_label:
            print_label = print_label.replace(phrase_2, '')
            # print(print_label)
        print_label = ' '.join(print_label.split('_')[:-1])
        # print(print_label)
        if print_label in done_labels:
            print_label = None
        else:
            color_idx += 1
            done_labels.append(print_label)

        color = colors[color_idx]

        for set_name, m in zip(ref_data[label].keys(), ['.', 'x']):

            if len(ref_data[label][set_name]['dft']) == 0:
                continue

            # color = colors[idx % 10]

            if which_data in ['rmsd', 'soap_dist', 'gap_e_error', 'gap_f_rmse', 
                              'gap_f_max']:
                all_ys = []
                all_xs = []
                for data in all_data:
                    y_data = data[label][set_name]['gap']
                    all_ys.append(y_data)
                    all_xs.append(data[label][set_name]['dft'])

            elif which_data == 'bde':
                all_ys = []
                all_xs = []
                for data in all_data:
                    dft_bdes = data[label][set_name]['dft']
                    gap_bdes = data[label][set_name]['gap']
                    all_ys.append(np.abs(dft_bdes - gap_bdes) * 1000)
                    all_xs.append(data[label][set_name]['dft'])

            elif which_data == 'bde_correlation':
                all_ys = []
                all_xs = []
                for data in all_data:
                    dft_bdes = data[label][set_name]['dft']
                    gap_bdes = data[label][set_name]['gap']
                    all_ys.append(gap_bdes)
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

            # plt.errorbar(xs, ys, yerr=yerr, zorder=3, fmt=fmt, color=color,
            #              label=label, capsize=2)
            marker = 'o'
            if set_name == 'test':
                print_label += ' test'
                marker = 'x'

            plt.scatter(xs, ys, zorder=3, label=print_label,  color=color,  marker=marker)

    if shade is not None:
        xmin, xmax = ax.get_xlim()
        if which_data == 'bde_correlation':
            extend_axis = 0.1
            xmin -=extend_axis
            xmax += extend_axis

            shade /= 1e+3
            plt.plot([xmin, xmax], [xmin, xmax],
                     color='k', lw=0.5)
            plt.fill_between([xmin, xmax], [xmin - shade, xmax - shade],
                            [xmin+ shade, xmax + shade],
                              color='lightgrey', alpha=0.5)



        else:
            plt.fill_between([xmin, xmax], 0, shade, color='lightgrey',
                             label=fill_label, alpha=0.5)
            ax.set_xlim(xmin, xmax)

    if which_data != 'bde_correlation':
        plt.yscale('log')
        plt.legend(title=legend_title, bbox_to_anchor=(1, 1))
    else:
        if len(ref_data.keys()) <11:
            plt.legend(title=legend_title)



    plt.grid(color='grey', ls=':', which='both')
    plt.ylabel(ylabel)
    plt.xlabel('DFT BDE / eV')
    plt.title(plot_title)
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
              include=None,
              means=False):

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
            plot_title = 'gap_energy_error_scatter'
        fill_label = f'< {shade} meV'
        hline_label = f'{shade} meV'
        ylabel = 'absolute GAP vs DFT error, meV'
    elif which_data == 'gap_f_rmse':
        shade = 100
        if plot_title is None:
            plot_title = 'gap_force_rmse_scatter'
        fill_label = f'< {shade} meV/Å'
        hline_label = f'{shade} meV/Å'
        ylabel = 'GAP f RMSE per molecule / meV/Å'
    elif which_data == 'gap_f_max':
        shade = 100
        if plot_title is None:
            plot_title = 'gap_max_f_error_scatter'
        fill_label = f'< {shade} meV/Å'
        ylabel='GAP max f error per molecule/ meV/Å'

    figsize=(10, 5)
    if means:
        figsize=(8, 5)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.arange(10)]

    ref_data = all_data[0]

    means_data = None
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

            if which_data in ['rmsd', 'soap_dist', 'gap_e_error', 'gap_f_rmse',
                              'gap_f_max']:
                all_ys = []
                for data in all_data:
                    all_ys.append(data[label][set_name]['gap'])


            elif which_data == 'bde':
                all_ys = []
                for data in all_data:
                    dft_bdes = data[label][set_name]['dft']
                    gap_bdes = data[label][set_name]['gap']
                    all_ys.append(np.abs(dft_bdes - gap_bdes) * 1000)

            xs = np.arange(len(all_data))

            all_ys = np.array(all_ys)

            if not means:
                for ys in all_ys.T:
                    plt.plot(xs, ys, zorder=3, marker='x', color=color,
                             label=print_label)
                    if print_label is not None:
                        print_label = None
            else:
                # print(f'{idx}. {print_label}')
                # print(f'all_ys:, {all_ys.shape}')
                # print(all_ys)
                # if means_data is not None:
                    # print(f'means_data: {means_data.shape}')
                if means_data is None:
                    means_data = np.array(all_ys.T)
                else:
                    means_data =  np.concatenate((means_data, all_ys.T))


    if means:
        mean_ys = np.mean(means_data, axis=0)
        for data in means_data:
            plt.scatter(xs, data, alpha=0.5, edgecolor='none')
        plt.plot(xs, mean_ys, color='k', marker='x', ms=10, label='mean')

    xmin, xmax = ax.get_xlim()
    plt.fill_between([xmin, xmax], 0, shade, color='lightgrey',
                     label=fill_label, alpha=0.5, zorder=0)
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
