#!/usr/bin/env python3

# Standard imports
import sys, os, subprocess, pdb
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
from collections import OrderedDict
from tqdm import tqdm

# My imports
sys.path.append("/home/eg475/reactions")
sys.path.append('/home/eg475/programs/my_scripts')
import util
import gap_plots

# Specific imports
import click

# Atomistic imports
from quippy.potential import Potential
from ase.io import read, write


def desymbolise_force_dict(my_dict):

    force_dict = OrderedDict()
    for sym, sym_dict in my_dict.items():
        for config_type, values in sym_dict.items():
            try:
                force_dict[config_type].append(values)
            except KeyError:
                force_dict[config_type] = []
                force_dict[config_type].append(values)

    # TODO check how are you array-ing thie stuff
    for config_type, values in force_dict.items():
        force_dict[config_type] = np.concatenate(values)

    return force_dict

def get_rmse_dict(obs, dft_data, gap_data):
    rmses = dict()
    rmses['Comb'] = util.get_rmse(util.dict_to_vals(dft_data[obs]), util.dict_to_vals(gap_data[obs]))
    for dft_config, gap_config in zip(dft_data[obs].keys(), gap_data[obs].keys()):
        if dft_config!=gap_config:
            raise ValueError('gap and dft config_types did not match')
        rmses[gap_config] = util.get_rmse(dft_data[obs][dft_config], gap_data[obs][gap_config])
    return rmses


def plot_heatmap(data_dict, ax, obs):
    df = pd.DataFrame.from_dict(data_dict)
    hmap = ax.pcolormesh(df)
    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels(df.index)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels(df.columns, rotation=90)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            color = 'red'
            text = ax.text(j + 0.5, i + 0.5, round(df.iat[i, j], 2), ha='center', color=color)
    cbar = plt.colorbar(hmap, ax=ax)
    if obs == 'Energy':
        units = 'eV'
    elif obs == 'Force':
        units = 'eV/Å '
    cbar.ax.set_ylabel(f'{obs} RMSE, {units}', rotation=90, labelpad=6)
    ax.set_title(f'{obs} RMSE', fontsize=14 )

def get_dimer_data(gap):
    pass


def rmse_plots(train_filename, gaps_dir, output_dir=None, prefix=None):

    train_ats = read(train_filename, index=':')
    dft_data = gap_plots.get_E_F_dict(train_ats, calc_type='dft')
    dft_data['forces'] = desymbolise_force_dict(dft_data['forces'])

    E_rmses = dict()
    F_rmses = dict()

    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = util.natural_sort(gap_fnames)

    for gap_fname in tqdm(gap_fnames):
        gap_title = os.path.splitext(gap_fname)[0]
        gap_fname = os.path.join(gaps_dir, gap_fname)
        gap_data = gap_plots.get_E_F_dict(train_ats, calc_type='gap', param_filename=gap_fname)
        gap_data['forces'] = desymbolise_force_dict(gap_data['forces'])

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
        prefix='multiple_gaps'
    picture_fname = f'{prefix}_RMSEs.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.savefig(picture_fname, dpi=300)


def make_dimer_plot(dimer_name, ax, param_filename, color='tab:red'):
    # which pair potential corresponds to which descriptor
    corr_desc = {'HH': 1, 'CH': 2, 'HO': 3, 'CC': 4, 'CO': 5}
    dimer = read(f'/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz', index=':')
    distances = [at.get_distance(0, 1) for at in dimer]

    if param_filename == 'dft':
        data = gap_plots.get_E_F_dict(dimer, calc_type='dft')
        es = util.dict_to_vals(data['energy'])
        label='dft'
        kwargs = {'color':'k', 'linestyle':'--'}
    elif 'gap' in param_filename and 'xml' in param_filename:
        command = f"quip E=T F=T atoms_filename=/home/eg475/programs/my_scripts/data/dft_{dimer_name}_dimer.xyz \
                    param_filename={param_filename} calc_args={{only_descriptor={corr_desc[dimer_name]}}} \
                        | grep AT | sed 's/AT//' > ./tmp_atoms.xyz"

        subprocess.run(command, shell=True)
        atoms = read('./tmp_atoms.xyz', index=':')
        os.remove('./tmp_atoms.xyz')
        es = np.array([at.info['energy'] for at in atoms])
        label = os.path.basename(param_filename)
        label = os.path.splitext(label)[0]
        kwargs = {'color':color}
    else:
        raise KeyError('either giva a gap name or "dft" key for which data to calculate')

    ax.plot(distances, es, label=label, **kwargs)



def dimer_plots(gaps_dir, output_dir=None, prefix=None):
    dimers = ['CC', 'CH', 'CO', 'HH', 'HO']
    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = util.natural_sort(gap_fnames)

    cmap = mpl.cm.get_cmap('Blues')
    colors = np.linspace(0.2, 1, len(gap_fnames))

    plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(3, 2)
    axes = [plt.subplot(g) for g in gs]

    for ax, dimer in zip(axes, dimers):
        for gap_fname, color in zip(tqdm(gap_fnames), colors):
            gap_fname = os.path.join(gaps_dir, gap_fname)
            make_dimer_plot(dimer, ax, gap_fname, color=cmap(color))

        make_dimer_plot(dimer, ax, 'dft')

        ax.set_title(dimer)
        ax.set_xlabel('distance (Å)')
        ax.set_ylabel('energy (eV)')
        # Potentially sort out the legend
        ax.legend()

    plt.tight_layout()

    if not prefix:
        prefix = 'multiple_gaps'
    plt.suptitle(prefix)
    picture_fname = f'{prefix}_dimers.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)
    plt.savefig(picture_fname, dpi=300)



@click.command()
@click.option('--train_filename',  type=click.Path(exists=True), required=True, \
              help='.xyz file with multiple config_types to evaluate against')
@click.option('--gaps_dir',  type=click.Path(exists=True), help='Directory that stores all GAP .xml files to evaluate')
@click.option('--output_dir', type=click.Path(), help='directory for figures. Create if not-existent')
@click.option('--prefix', help='prefix to label plots')
def make_plots(train_filename, gaps_dir=None, output_dir=None, prefix=None):
    """Makes evaluates """

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if gaps_dir == None:
        gaps_dir = os.getcwd()

    rmse_plots(train_filename=train_filename, gaps_dir=gaps_dir, output_dir=output_dir, prefix=prefix)
    dimer_plots(gaps_dir=gaps_dir, output_dir=output_dir, prefix=prefix)


if __name__=='__main__':
    make_plots()
    print('\n\n-----------------CONGRATS YOU MADE IT')

