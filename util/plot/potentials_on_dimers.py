
import click
import os
from ase import Atoms
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from quippy.potential import Potential
import util

@click.command()
@click.option('--dimers', help='list of dimers to evaluate potentials on')
@click.option('--prefix')
@click.argument('potentials', nargs=-1)
def potentials_on_dimers(dimers, potentials, prefix='dimer'):

    dimers = util.str_to_list(dimers, type=str)
    distances = np.arange(0.05, 4, 0.005)

    print(f'dimers: {dimers}')

    if len(dimers)%2==0:
        no_vert = int(len(dimers)/2)
    else:
        no_vert = int((len(dimers)+1)/2)

    fig = plt.figure(figsize=(12, no_vert*5))
    gs1 = gridspec.GridSpec(no_vert, 2)
    axes = [plt.subplot(gs) for gs in gs1]

    for ax, dimer in zip(axes, dimers):

        for potential in potentials:

            if 'GLUE' in potential.upper():
                calc = Potential('IP GLUE', param_filename=potential)
            else:
                calc = Potential(param_filename=potential)

            energies = []
            for d in distances:
                at = Atoms(dimer, positions=[(0, 0, 0), (0, 0, d)])
                at.set_calculator(calc)
                energies.append(at.get_potential_energy())

            ax.plot(distances, energies, label=f'{os.path.splitext(potential)[0]}')

        ax.set_xlabel('distance, Å')
        ax.set_ylabel('energy, eV')
        ax.axhline(y=0, c='k', lw=0.8)
        ax.grid(color='lightgrey')
        ax.set_title(dimer)
        ax.set_ylim((-20, 50))
        ax.legend()

    plt.savefig(f'{prefix}.png', dpi=300)

if __name__=='__main__':
    potentials_on_dimers()



