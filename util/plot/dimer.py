import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
import click
import os

def dimer(dimer_fnames, prefix):

    plt.figure()
    all_ends = []
    all_mins = []

    for dimer_fname in dimer_fnames:
        atoms = read(dimer_fname, ':')
        energies = np.array([at.info['energy'] for at in atoms if 'energy' in at.info.keys()])
        if len(atoms)>2:
            distances = [at.get_distance(0, 1) for at in atoms if 'energy' in at.info.keys()]
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


