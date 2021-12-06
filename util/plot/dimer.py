import matplotlib.pyplot as plt
import util
import numpy as np
from ase.io import read
import click
import os
from ase import Atoms
from matplotlib import gridspec


def setup_axes(dimers):
    # set up axes
    ## get number of subplots
    if len(dimers)%2==0:
        no_vert = int(len(dimers)/2)
    else:
        no_vert = int((len(dimers)+1)/2)

    fig = plt.figure(figsize=(12, no_vert*5))
    gs1 = gridspec.GridSpec(no_vert, 2)
    axes_dimer = {}
    axes_hist = {}
    for dimer, gs in zip(dimers, gs1):
        gs2 = gs.subgridspec(2, 1, height_ratios=[2,1])
        ax1 = plt.subplot(gs2[0])
        axes_dimer[dimer] = ax1
        axes_hist[dimer] = plt.subplot(gs2[1], sharex=ax1)

    return axes_dimer, axes_hist

def plot_histograms(axes_hist, distances_dict):

    for dimer, data in distances_dict.items():
        ax = axes_hist[dimer]
        ax.hist(data, bins=np.arange(min(data), max(data) + 0.1, 0.1), density=True)
        ax.set_xlabel("Distance, Å")
        ax.set_ylabel("density")

def plot_dimers(axes_dimer, calc, dimers, pred_label):
    for dimer in dimers:
        plot_dimer(axes_dimer[dimer], calc, dimer, pred_label)

def plot_dimer(ax, calc, dimer_name, pred_label):
    distances = np.linspace(0.005, 6, 50)
    ats = [Atoms(dimer_name, positions=[(0, 0, 0), (0, 0, d)]) for d in distances]
    for at in ats: at.cell = [50, 50, 50]
    for at in ats:
        calc.reset()
        at.calc = calc
        at.info[f"{pred_label}energy"] = at.get_potential_energy()

    energies = [at.info[f'{pred_label}energy'] for at in ats]

    ax.plot(distances, energies, label='evaluated_on_dimer')
    ax.set_title(dimer_name)


def plot_isolated_at_energy(axes_dimer, ref_isolated_ats, ref_prefix):
    isolated_at_data = get_isolated_atom_data(ref_isolated_ats, ref_prefix) 
    
    for dimer, ax in axes_dimer.items():
        energy = np.sum([isolated_at_data[symbol] for symbol in dimer])
        ax.axhline(energy, ls='--', lw=0.8, color='k', label=f'{ref_prefix} isolated atom energy')


def get_isolated_atom_data(isolated_atoms, prop_prefix):    
    isolated_at_data = {}
    for at in isolated_atoms:
        isolated_at_data[list(at.symbols)[0]] = at.info[f'{prop_prefix}energy']
        
    return isolated_at_data

def set_limits(axes_dimers):

    limits_dict = {'CC':(-2100, -1950), 'CH': (-1055, -980), 'HH': (-35, 10),
                       'CO':(-3100, -3000), 'HO':(-2060, -2020),
                        'OO':(-4100, -4000)}

    for dimer, ax in axes_dimers.items():
        ax.set_xlim(0.005, 6)
        ax.set_ylim(limits_dict[dimer])


def calc_on_dimer(calc, train_ats, ref_isolated_ats, ref_prefix='dft_',
                  pred_label=None, title='dimer_curve'):

    distances_dict = util.distances_dict(train_ats)
    dimers = [key for key, item in distances_dict.items() if len(item)!=0]
    axes_dimer, axes_hist = setup_axes(dimers)
    plot_histograms(axes_hist, distances_dict)
    plot_dimers(axes_dimer, calc, dimers, pred_label)
    plot_isolated_at_energy(axes_dimer, ref_isolated_ats, ref_prefix)
    set_limits(axes_dimer)

    for ax in axes_dimer.values():
        if pred_label is None: pred_label = ''
        ax.set_ylabel(f"{pred_label}energy, eV")
        # ax.set_xlabel("distance, Å")
        ax.legend()

    for ax_dict in [axes_dimer, axes_hist]:
        for ax in ax_dict.values():
            ax.grid(color='lightgrey', ls=':')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(title + '.pdf', bbox_inches='tight')
    plt.close()











def dimer_generic(dimer_fnames, prefix, energy_names):

    plt.figure()
    all_ends = []
    all_mins = []

    for energy_name, dimer_fname in zip(energy_names, dimer_fnames):
        atoms = read(dimer_fname, ':')
        energies = np.array([at.info[energy_name] for at in atoms if energy_name in at.info.keys()])

        if len(energies)< 2:
            continue

        if len(atoms)>2:
            distances = [at.get_distance(0, 1) for at in atoms if energy_name in at.info.keys()]
            plt.plot(distances, energies, marker='x', label=f'{os.path.splitext(dimer_fname)[0]} = {energies[-1]:4f}')
            plt.scatter(distances[-1], energies[-1])
            all_ends.append(energies[-1])
            all_mins.append(min(energies))
        elif len(atoms)==2:
            plt.axhline(y=energies.sum(), lw='0.8', label=f'{os.path.splitext(dimer_fname)[0]} = {energies.sum():4f}')
        else:
            raise RuntimeError('too few atoms')


    plt.legend()
    plt.ylabel('energy, eV')
    plt.xlabel('distance, Å')
    plt.grid(color='lightgrey')

    lim_top=max(all_ends) + 3 * (min(all_ends) - min(all_mins))
    lim_bottom=min(all_mins) - 1 * (min(all_ends) - min(all_mins))
    if lim_top != lim_bottom:
        plt.ylim(top=lim_top, bottom=lim_bottom)

    plt.savefig(f'{prefix}.png', dpi=300)


