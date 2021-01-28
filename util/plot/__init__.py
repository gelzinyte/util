
# Standard imports
import os, subprocess, re, sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
from collections import OrderedDict
from os.path import join as pj


# My imports
import util
from util import ugap
# from util.vibrations import Vibrations
# from util import urdkit

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
from quippy.descriptors import Descriptor




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
        compare_gopt
'''

def make_2b_only_plot(dimer_name, ax, param_fname, label=None, color=None):

    corr_desc = ugap.get_gap_2b_dict(param_fname)

    tmp_dimer_in_name = 'tmp_dimer_in.xyz'
    tmp_dimer_out_name = 'tmp_dimer_out.xyz'
    distances = np.linspace(0.05, 6, 50)
    atoms = [Atoms(dimer_name, positions=[(0, 0, 0), (0, 0, d)]) for d in distances]
    write(tmp_dimer_in_name, atoms, 'extxyz')

    which_quip = '/home/eg475/programs/QUIPwo0/build/linux_x86_64_gfortran_openmp/quip'

    command = f"{which_quip} E=T F=T atoms_filename={tmp_dimer_in_name} param_filename={param_fname} calc_args={{only_descriptor={corr_desc[dimer_name]}}} \
                         | grep AT | sed 's/AT//' > {tmp_dimer_out_name}"

    subprocess.run(command, shell=True)
    atoms = read(tmp_dimer_out_name, index=':')
    os.remove(tmp_dimer_out_name)
    os.remove(tmp_dimer_in_name)
    es = [at.info['energy'] for at in atoms]
    if label is None:
        label = 'GAP: only 2b'
    if color is None:
        color='tab:orange'
    ax.plot(distances, es, label=label, color=color)


def make_dimer_plot(dimer_name, ax, calc, label, color=None, isolated_atoms_fname=None):
    init_dimer = Atoms(dimer_name, positions=[(0, 0, 0), (0, 0, 2)])

    distances = np.linspace(0.05, 6, 50)

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
                        # print(f'dimer: {dimer_name}, symbol: {sym}, adding: {iso_at.info["dft_energy"]}')
                        e_shift +=iso_at.info['dft_energy']
            energies = [e + e_shift for e in energies]
        color='tab:green'

    # print(f'curve: {label}, dimer: {dimer_name}, last energy as plotted: {energies[-1]}')

    ax.plot(distances, energies, label=label, color=color)


def make_ref_plot(dimer_name, ax, calc_type='dft'):
    atoms_fname=f'/home/eg475/scripts/data/dft_{dimer_name}_dimer.xyz'
    dimer = read(atoms_fname, ':')
    distances = [at.get_distance(0, 1) for at in dimer]
    ref_data = util.get_E_F_dict(dimer, calc_type=calc_type)
    ax.plot(distances, dict_to_vals(ref_data['energy'])*2, label='(RKS) reference', linestyle='--', color='k')


def make_dimer_curves(param_fnames, output_dir='pictures', prefix=None, glue_fname=None, plot_2b_contribution=True, \
                      plot_ref_curve=True, isolated_atoms_fname=None, ref_name='dft', dimer_scatter=None, close=True):
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

        # ylimits_default = ax.get_ylim()
        # bottom_diff = e_shift - ylimits_default[0]
        # top_diff = ylimits_default[1] - e_shift
        # print('bottom difference:', bottom_diff)
        # print('top difference:', top_diff)
        # print(f'current_lims: {ylimits_default}')
        # top_new =  min(ylimits_default[1], e_shift + bottom_diff * 2.5)
        # print(top_new)

        limits_dict = {'CC':(-2100, -1900), 'CH': (-1055, -900), 'HH': (-50, 10)}
        ax.set_ylim(limits_dict[dimer])

        ax.set_xlim(0.03, 6)

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



def get_last_bunch(full_data, bunch=20):
    new_keys = list(full_data['energy'].keys())[-bunch:]
    new_data = dict()
    new_data['energy'] = OrderedDict()
    new_data['forces'] = OrderedDict()
    for key in new_keys:
        new_data['energy'][key] = full_data['energy'][key]
        new_data['forces'][key] = full_data['forces'][key]
    return new_data


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

def evec_plot(param_fname,  dft_eq_fname, smiles_to_opt=None, at_fname_to_opt=None, fmax=1e-2, steps=1000, output_dir='pictures'):


    db_path = '/home/eg475/scripts/gopt_test/'

    if not os.path.exists('xyzs'):
        os.makedirs('xyzs')

    dft_eq_fname = os.path.join(db_path, 'dft_minima', dft_eq_fname)
    if at_fname_to_opt=='dft':
        at_fname_to_opt = dft_eq_fname

    gap_name = os.path.basename(os.path.splitext(param_fname)[0])
    gap = Potential(param_filename=param_fname)

    if smiles_to_opt is not None and at_fname_to_opt is None:
        atoms = urdkit.smi_to_xyz(smiles_to_opt, useBasicKnowledge=False, useExpTorsionAnglePrefs=False)
    elif smiles_to_opt is None and at_fname_to_opt is not None:
        atoms = read(at_fname_to_opt)
    else:
        raise RuntimeError('Give one olny of smiles or xyz to be optimised')

    atoms.set_calculator(gap)

    # optimize
    if not os.path.isfile(f'{gap_name}.all.pckl'):
        opt = PreconLBFGS(atoms, trajectory=f'xyzs/{gap_name}_opt_for_NM.traj')
        opt.run(fmax=fmax, steps=steps)

        # get NM
    vib_gap = Vibrations(atoms, name=gap_name)
    atoms_gap = vib_gap.run()
    vib_gap.summary()


    dft_atoms = read(dft_eq_fname)
    dft_name = dft_atoms.info['name']
    shutil.copy(os.path.join(db_path, f'dft_minima/normal_modes/{dft_name}.all.pckl'), '.')
    vib_dft = Vibrations(dft_atoms, name=dft_name)
    vib_dft.summary()

    do_evec_plot(vib_dft.evals, vib_dft.evecs, vib_gap.evals, vib_gap.evecs, gap_name, output_dir=output_dir)




