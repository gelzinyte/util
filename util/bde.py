from quippy.potential import Potential
from wfl.configset import ConfigSet_in, ConfigSet_out
import re
import shutil
import numpy as np
from ase.io import read, write
from ase import Atoms
import click
import util
from util import iter_tools as it
import pandas as pd
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
from wfl.utils.parallel import construct_calculator_picklesafe


def multi_bde_summaries(dft_dir, gap_dir=None, calculator=None, start_dir=None, dft_only=False,
                        precision=3):
    """BDEs for all files in the directory"""
    if gap_dir is not None:
        if not os.path.isdir(gap_dir):
            os.makedirs(gap_dir)

    dft_basenames = util.natural_sort(os.listdir(dft_dir))
    dft_fnames = [os.path.join(dft_dir, fname) for fname in dft_basenames]
    if not dft_only:
        gap_fnames = [os.path.join(gap_dir, basename.replace('optimised', 'gap_optimised')) for
                      basename in dft_basenames]
        start_fnames = [os.path.join(start_dir, basename.replace('optimised', 'non_optimised')) for
                        basename in dft_basenames]
    else:
        gap_fnames = [None for _ in dft_fnames]
        start_fnames = [None for _ in dft_fnames]

    for dft_fname, gap_fname, start_fname in zip(dft_fnames, gap_fnames, start_fnames):
        _ = bde_summary(dft_fname, gap_fname, calculator, start_fname, precision)


def bde_summary(dft_fname, gap_fname=None, calculator=None, start_fname=None, precision=3,
                printing=True):
    dft_ats = read(dft_fname, ':')
    dft_h = Atoms('H', positions=[(0, 0, 0)])
    dft_h.info['config_type'] = 'H'
    dft_h.info['dft_energy'] = -13.547449778462548
    dft_ats = [dft_h] + dft_ats

    if gap_fname is not None and os.path.isfile(gap_fname):
        gap_ats = read(gap_fname, ':')
    elif gap_fname is not None and not os.path.isfile(gap_fname):
        gap_optimise(start_fname, gap_fname, calculator)
        gap_ats = read(gap_fname, ':')
    else:
        gap_ats = None

    bdes = get_bdes(dft_ats, gap_ats)

    if printing:
        print('-' * 30)
        print(os.path.basename(os.path.splitext(dft_fname)[0]))
        print('-' * 30)

        headers = [' ', "eV\nDFT E", "eV\nDFT BDE"]
        if gap_ats is not None:
            headers += ["eV\nGAP E", "eV\nGAP BDE", "meV\nabs error", "Å\nRMSD"]

        print(tabulate(bdes, headers=headers, floatfmt=f".{precision}f"))

    return pd.DataFrame(bdes)

def gap_optimise(start_fname, gap_fname, calculator):

    gap_tmp_fname = os.path.splitext(gap_fname)[0] + '_tmp.xyz'

    if start_fname is None:
        raise RuntimeError('Don\'t have a start file to optimise')

    dir = os.path.dirname(gap_fname)
    opt_dir = os.path.join(dir, 'opt_trajectories')
    if not os.path.isdir(opt_dir):
        os.makedirs(opt_dir)

    base_name =  os.path.basename(os.path.splitext(gap_fname)[0])
    traj_fname = os.path.join(opt_dir, base_name + '_traj.xyz')
    log_fname = os.path.join(opt_dir, base_name + '_opt.log')


    gap_h = Atoms('H', positions=[(0, 0, 0)])
    gap_h.info['config_type'] = 'H'
    starts = [gap_h] + read(start_fname, ':')

    #tmp non-optimised structures to be overwritten
    write(gap_tmp_fname, starts)


    inputs = ConfigSet_in(input_files=gap_tmp_fname)
    outputs = ConfigSet_out(output_files=traj_fname, force=True)

    print(f'optimising: {gap_tmp_fname}')

    it.run_opt(inputs, outputs, calculator, fmax=1e-2, return_traj=True, logfile=log_fname, chunksize=1)

    optimised_atoms = read(traj_fname, ':')
    opt_ats = [at for at in optimised_atoms if 'minim_config_type' in
               at.info.keys() and 'converged' in at.info['minim_config_type']]

    calculator = construct_calculator_picklesafe(calculator)
    for at in opt_ats:
        at.set_calculator(calculator),
        at.info['gap_energy'] = at.get_potential_energy()
        at.arrays['gap_forces'] = at.get_forces()

    write(gap_fname, opt_ats)
    os.remove(gap_tmp_fname)


def get_bdes(dft_ats, gap_ats=None):
    label_pattern = re.compile(r"rad_?\d+$|mol$|H$")


    dft_h = dft_ats[0]
    dft_mol = dft_ats[1]

    assert 'H' == label_pattern.search(dft_h.info['config_type']).group()
    assert 'mol' == label_pattern.search(dft_mol.info['config_type']).group()

    dft_h_energy = dft_h.info['dft_energy']
    dft_mol_energy = dft_mol.info['dft_energy']

    mol_data = ['H', dft_h_energy, np.nan]
    h_data = ['mol', dft_mol_energy, np.nan]

    if gap_ats is not None:

        gap_h = gap_ats[0]
        gap_mol = gap_ats[1]

        assert 'H' == label_pattern.search(gap_h.info['config_type']).group()
        assert 'mol' == label_pattern.search(gap_mol.info['config_type']).group()

        try:
            gap_h_energy = gap_h.info['gap_energy']
            gap_mol_energy = gap_mol.info['gap_energy']
        except KeyError:
            print(f'info: {gap_h.info}, {gap_mol.info}')
            raise

        h_error = abs(dft_h_energy - gap_h_energy) * 1e3
        mol_error = abs(dft_mol_energy - gap_mol_energy) * 1e3
        mol_rmsd = util.get_rmse(dft_mol.positions, gap_mol.positions)

        h_data += [gap_h_energy, np.nan, h_error, np.nan]
        mol_data += [gap_mol_energy, np.nan, mol_error, mol_rmsd]
    else:
        gap_ats = [None for _ in dft_ats]

    data = []
    data.append(h_data)
    data.append(mol_data)

    bde_errors = []
    rmsds = []
    for dft_at, gap_at in zip(dft_ats[2:], gap_ats[2:]):

        label = label_pattern.search(dft_at.info['config_type']).group()

        dft_rad_e = dft_at.info['dft_energy']
        dft_bde = dft_rad_e + dft_h_energy - dft_mol_energy

        data_line = [label, dft_rad_e, dft_bde]

        if gap_at is not None:
            gap_rad_e = gap_at.info['gap_energy']
            gap_bde = gap_rad_e + gap_h_energy - gap_mol_energy
            bde_error = abs(dft_bde - gap_bde) * 1e3
            bde_errors.append(bde_error)
            rmsd = util.get_rmse(dft_at.positions, gap_at.positions)
            rmsds.append(rmsd)

            data_line += [gap_rad_e, gap_bde, bde_error, rmsd]

        data.append(data_line)

    if gap_ats[0] is not None:
        data.append(['mean', np.nan, np.nan, np.nan, np.nan, np.mean(bde_errors), np.mean(rmsds)])

    return data


def bde_bar_plot(gap_fnames, dft_fnames, plot_title='bde_bar_plot', start_fnames=None, calculator=None,
                 output_dir='pictures'):

    if start_fnames is None:
        start_fnames = [None for _ in gap_fnames]

    all_titles = []
    all_dft_bdes = []
    all_gap_bdes = []
    for gap_fname, dft_fname, start_fname in zip(gap_fnames, dft_fnames, start_fnames):
        #         print(gap_fname)

        title = os.path.basename(os.path.splitext(dft_fname)[0]).replace('_optimised', '')
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

    fig = plt.figure()
    plt.grid(color='lightgrey')
    plt.bar(bar_categories, bars_dft, yerr=errors_dft, width=-width, align='edge', color='tab:red',
            zorder=2, label='DFT')
    plt.bar(bar_categories, bars_gap, yerr=errors_gap, width=width, align='edge', color='tab:blue',
            zorder=2, label='GAP')
    plt.ylabel('Mean BDE / eV')
    plt.title(plot_title)
    plt.legend()
    plt.xticks(rotation=90)
    # plt.show()
    plt.tight_layout()
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir=''

    plt.savefig(os.path.join(output_dir, plot_title+'.png'), dpi=300)


def get_data(gap_fnames, dft_fnames, selection, start_fnames=None, calculator=None):
    all_bdes = {}

    for gap_fname, dft_fname, start_fname, in zip(gap_fnames, dft_fnames, start_fnames):

        title = os.path.basename(os.path.splitext(gap_fname)[0]).replace('_gap_optimised.xyz', '')
        if '_bde_train_set' in title:
            title = title.replace('_bde_train_set', '')


        all_bdes[title] = {'train': {'gap': [], 'dft': []}, 'test': {'gap': [], 'dft': []}}

        bdes = bde_summary(dft_fname=dft_fname, gap_fname=gap_fname,
                                           start_fname=start_fname, calculator=calculator,
                                           printing=False)

        selected_h_list = []
        for key, vals in selection.items():
            if key in title:
                selected_h_list = vals

        if selected_h_list == 'all':
            all_bdes[title]['test']['dft'] = bdes[2]
            all_bdes[title]['test']['gap'] = bdes[4]
        elif len(selected_h_list) == 0:
            all_bdes[title]['train']['dft'] = bdes[2]
            all_bdes[title]['train']['gap'] = bdes[4]
        else:
            for idx, row in bdes.iterrows():
                for rad_no in selected_h_list:
                    if str(rad_no) in row[0]:
                        all_bdes[title]['test']['dft'].append(row[2])
                        all_bdes[title]['test']['gap'].append(row[4])
                        break
                else:
                    all_bdes[title]['train']['dft'].append(row[2])
                    all_bdes[title]['train']['gap'].append(row[4])

        for set_name in ['train', 'test']:
            for method_name in ['dft', 'gap']:
                all_bdes[title][set_name][method_name] = np.array(
                    all_bdes[title][set_name][method_name])

    return all_bdes


def scatter_plot(gap_fnames, dft_fnames, selection=None, plot_title='bde_scatter',
                     start_fnames=None, calculator=None, output_dir='pictures'):

    if start_fnames is None:
        start_fnames = [None for _ in gap_fnames]

    if selection is None:
        selection = {}


    data = get_data(gap_fnames, dft_fnames, selection, start_fnames, calculator)

    shade = 50
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plt.axhline(0, linewidth=0.8, color='k', zorder=2)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.arange(10)]

    for idx, (label, comp_vals) in enumerate(data.items()):
        for (set_name, set_vals), m in zip(comp_vals.items(), ['o', 'x']):
            if len(set_vals['dft']) == 0:
                continue
            if set_name == 'test':
                label += ' test'

            color = colors[idx % 10]

            plt.scatter(set_vals['dft'], (set_vals['dft'] - set_vals['gap']) * 1000,
                        label=label, zorder=3, marker=m, color=color)

    xmin, xmax = ax.get_xlim()
    plt.fill_between([xmin, xmax], -shade, shade, color='lightgrey', label=f'$\pm$ 50 meV')
    ax.set_xlim(xmin, xmax)
    plt.grid(color='lightgrey')
    plt.ylabel('DFT BDE - GAP BDE / meV')
    plt.xlabel('DFT BDE / eV')
    plt.title(plot_title)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = ''

    plt.savefig(os.path.join(output_dir, plot_title+'.png'), dpi=300)


"""

@click.command()
@click.option('--bde_start_dir', type=click.Path(), help='which files to gap-optimise')
@click.option('--dft_bde_dir', type=click.Path(), help='which files to compare to')
@click.option('--gap_fname')
@click.option('--dft_bde_fname', type=click.Path(), help='single name with dft bde info')
@click.option('--bde_start_fname', type=click.Path(),
              help='single bde start fname to optimise with GAP')
@click.option('--dft_only', is_flag=True, help='Prints only the dft summary')
@click.option('--bde_out_dir', type=click.Path(), help='Directory where to save all opt_files')
@click.option('--gap_bde_out_fname', type=click.Path(),
              help='fname to save to/load from gap bde energies')
@click.option('--precision', '-p', type=int, default=3)
def cli_gap_bde_summary(bde_start_dir=None, dft_bde_dir=None, gap_fname=None, bde_out_dir=None,
                        dft_bde_fname=None,
                        bde_start_fname=None, gap_bde_out_fname=None, dft_only=False, precision=3):
    gap_bde_summary(bde_start_dir=bde_start_dir, dft_bde_dir=dft_bde_dir, gap_fname=gap_fname,
                    bde_out_dir=bde_out_dir, dft_bde_fname=dft_bde_fname,
                    bde_start_fname=bde_start_fname, gap_bde_out_fname=gap_bde_out_fname,
                    dft_only=dft_only, precision=precision)


def gap_bde_summary(bde_start_dir=None, dft_bde_dir=None, gap_fname=None, bde_out_dir=None,
                    dft_bde_fname=None,
                    bde_start_fname=None, gap_bde_out_fname=None, dft_only=False, precision=3,
                    printing=False):
    if bde_out_dir:
        if not os.path.exists(bde_out_dir):
            os.makedirs(bde_out_dir)
        os.chdir(bde_out_dir)

    if not dft_only:
        if bde_start_dir is not None and bde_start_fname is not None:
            raise ValueError(
                f" only one of bde_start_dir ({bde_start_dir}) and bde_start_fname ("
                f"{bde_start_fname}) must be given")
        elif bde_start_dir is not None:
            start_fnames = [os.path.join(bde_start_dir, name) for name in os.listdir(bde_start_dir)]
            start_fnames = util.natural_sort(start_fnames)
        elif bde_start_fname is not None:
            start_fnames = [bde_start_fname]
        else:
            start_fnames = None

        # if bde_out_dir is not None:
        #     bde_out_fnames =  util.natural_sort(os.listdir('.'))

    else:
        if dft_bde_dir is not None and dft_bde_fname is not None:
            raise ValueError(f'only one of dft_bde_dir and dft_bde_fname can be given')
        elif dft_bde_dir:
            dft_fnames = [os.path.join(dft_bde_dir, name) for name in os.listdir(dft_bde_dir)]
            dft_fnames = util.natural_sort(dft_fnames)
        else:
            dft_fnames = [dft_bde_fname]

    gap = None
    if gap_fname is not None and not dft_only:
        dir = '.'
        if bde_out_dir:
            dir = '..'
        gap = Potential(param_filename=os.path.join(dir, gap_fname))

    if not dft_only:
        # if have start names, cycle through start names
        if start_fnames is not None:
            for start_fname in start_fnames:
                basename = os.path.basename(start_fname)
                gap_bde_out_fname = basename.replace('_non-optimised.xyz', '_gap_optimised.xyz')
                if not dft_bde_fname:
                    dft_fname = os.path.join(dft_bde_dir, basename.replace('_non-optimised.xyz',
                                                                           '_optimised.xyz'))
                else:
                    dft_fname = dft_bde_fname
                gap_ats = get_optimised_gap_atoms(gap_bde_out_fname, start_fname, gap)
                # title = basename.replace('_non-optimised.xyz', '')
                title = gap_bde_out_fname
                table(title, dft_fname, gap_ats, precision, printing=printing)
        # elif bde_out_fnames is not None:
        #     for gap_bde_out_fname in bde_out_fnames:
        #         basename = os.path.basename(gap_bde_out_fname)
        #         dft_fname = os.path.join(dft_bde_dir, basename.replace('gap_optimised.xyz',
        #         'optimised.xyz'))
        #         gap_ats = read(gap_bde_out_fname, ':')
        #         title = basename.replace('_gap_optimised.xyz', '')
        #         table(title, dft_fname, gap_ats, precision)
        elif gap_bde_out_fname is not None and dft_bde_fname is not None:
            gap_ats = get_optimised_gap_atoms(gap_bde_out_fname, dft_bde_fname, gap)
            title = os.path.splitext(gap_bde_out_fname)[0]
            bdes = table(title, dft_bde_fname, gap_ats, precision, printing=printing)
            return bdes


    else:
        for dft_fname in dft_fnames:
            title = os.path.basename(os.path.splitext(dft_fname)[0])
            bdes = table(title, dft_fname, gap_ats=None, precision=precision)
            return bdes


def table(title, dft_fname, gap_ats, precision, printing=True):
    # print BDE summary while we're at it
    if printing:
        print('-' * 30)
        print(title)
        print('-' * 30)
    if os.path.exists(dft_fname):
        dft_ats = read(dft_fname, ':')

        dft_mol = dft_ats[0]
        dft_h = dft_ats[1]
        dft_rads = dft_ats[2:]

        headers = [' ', "eV\nDFT E", "eV\nDFT BDE"]
        data = []

        dft_mol_e = dft_mol.info["dft_energy"]
        mol_data = ['mol', dft_mol_e, np.nan]

        dft_h_e = dft_h.info["dft_energy"]
        h_data = ['H', dft_h_e, np.nan]

        if gap_ats:
            headers += ["eV\nGAP E", "eV\nGAP BDE", "meV\nabs error", "Å\nRMSD"]

            gap_mol = gap_ats[0]
            gap_h = gap_ats[1]
            gap_rads = gap_ats[2:]

            gap_mol_e = gap_mol.info["gap_energy"]
            error = abs(dft_mol_e - gap_mol_e) * 1e3
            rmsd = util.get_rmse(dft_mol.positions, gap_mol.positions)
            mol_data += [gap_mol_e, np.nan, error, rmsd]

            gap_h_e = gap_h.info["gap_energy"]
            error = abs(dft_h_e - gap_h_e) * 1e3
            h_data += [gap_h_e, np.nan, error, np.nan]

        else:
            gap_rads = [None for _ in range(len(dft_rads))]

        # data.append(headers)
        data.append(mol_data)
        data.append(h_data)

        bde_errors = []
        rmsds = []
        dft_bdes = []
        for idx, (dft_rad, gap_rad) in enumerate(zip(dft_rads, gap_rads)):

            dft_rad_e = dft_rad.info['dft_energy']
            dft_bde = - dft_mol_e + dft_rad_e + dft_h_e
            dft_bdes.append(dft_bde)

            if "config_type" in dft_rad.info.keys():
                label = dft_rad.info["config_type"]
            else:
                label = f'rad {idx}'

            if gap_rad:
                gap_rad_e = gap_rad.info['gap_energy']
                gap_bde = - gap_mol_e + gap_rad_e + gap_h_e

                error = abs(dft_bde - gap_bde) * 1e3
                rmsd = util.get_rmse(dft_rad.positions, gap_rad.positions)
                bde_errors.append(error)
                rmsds.append(rmsd)

                if 'config_type' in gap_rad.info.keys():
                    label = gap_rad.info['config_type']
                else:
                    label = f'rad {idx}'

                data_line = [label, dft_rad_e, dft_bde, gap_rad_e, gap_bde, error, rmsd]
            else:
                data_line = [label, dft_rad_e, dft_bde]

            data.append(data_line)

        if gap_ats:
            data.append(
                ['mean', np.nan, np.nan, np.nan, np.nan, np.mean(bde_errors), np.mean(rmsds)])

        if printing:
            print(tabulate(data, headers=headers, floatfmt=f".{precision}f"))
        table = pd.DataFrame(data)
        return table
    else:
        print(f"\n\nCouldn't find {dft_fname} to print BDE summary")


def get_optimised_gap_atoms(gap_bde_out_fname, start_fname, gap):
    # optimise structures
    if not os.path.exists(
            gap_bde_out_fname) and start_fname and gap is not None:
        print(f"Didn't find {gap_bde_out_fname}, optimising bde starts")
        bde_starts = read(start_fname, ':')
        gap_ats = []
        for at in bde_starts:
            relaxed = util.relax(at, gap)
            relaxed.info['gap_energy'] = relaxed.get_potential_energy()
            gap_ats.append(relaxed)
        write(gap_bde_out_fname, gap_ats, 'extxyz')
    elif os.path.exists(gap_bde_out_fname):
        gap_ats = read(gap_bde_out_fname, ':')
    else:
        raise FileNotFoundError(
            "Need either bde start file to optimise or gap_bde_out_fname "
            "with optimised structures")
    return gap_ats
    
"""

