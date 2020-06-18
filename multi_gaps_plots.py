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
    hmap = ax.pcolormesh(df, vmin=0, vmax=20)
    # ax.set_yticks(np.arange(0.5, len(df.index), 1), df.index)
    # ax.set_xticks(np.arange(0.5, len(df.columns), 1), df.columns)

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
    cbar.ax.set_ylabel(f'{obs} RMSE, {units}', rotation=90, labelpad=12)
    ax.title(f'{obs} RMSE', fontsize=14 )

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
        prefix = os.path.basename(train_filename)
        prefix = os.path.splitext(prefix)[0]
    picture_fname = f'{prefix}_heatmaps.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.savefig(picture_fname, dpi=300)


def dimer_plots(gaps_dir, output_dir=None, prefix=None):
    pass



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
    # dimer_plots(gaps_dir=gaps_dir, output_dir=output_dir, prefix=prefix)


if __name__=='__main__':
    make_plots()
    print('\n\n-----------------CONGRATS YOU MADE IT')

