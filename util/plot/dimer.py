import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
import click
import os

@click.command()
@click.argument('dimer_fnames', nargs=-1)
@click.option('--isolated_fname', help='xyz with isolated atom energies')
@click.option('--prefix', help='prefix for the png')
def dimer(dimer_fnames, isolated_fname, prefix):

    if isolated_fname:
        isolated_ats = read(isolated_fname, ':')
        isolated_es = np.array([ at.info['energy'] for at in isolated_ats])

    plt.figure()
    all_ends = []
    all_mins = []

    for dimer_fname in dimer_fnames:
        atoms = read(dimer_fname, ':')
        energies = [at.info['energy'] for at in atoms]
        distances = [at.get_distance(0, 1) for at in atoms]
        plt.plot(distances, energies, label=f'{os.path.splitext(dimer_fname)[0]} = {energies[-1]:4f}')
        plt.scatter(distances[-1], energies[-1])
        all_ends.append(energies[-1])
        all_mins.append(min(energies))

    if isolated_fname:
        plt.axhline(y=isolated_es.sum(), c='k', lw='0.8', label=f'{os.path.splitext(isolated_fname)[0]} = {isolated_es.sum():4f}')
    plt.legend()
    plt.ylabel('energy, eV')
    plt.xlabel('distance, Å')
    plt.grid(color='lightgrey')

    lim_top=max(all_ends) + 3 * (min(all_ends) - min(all_mins))
    lim_bottom=min(all_mins) - 1 * (min(all_ends) - min(all_mins))
    if lim_top != lim_bottom:
        plt.ylim(top=lim_top, bottom=lim_bottom)

    plt.savefig(f'{prefix}.png', dpi=300)


if __name__=='__main__':
    dimer()
