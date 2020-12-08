from quippy.potential import Potential
import numpy as np
from ase.io import read, write
import click
import util

@click.command()
@click.command('--bde_start_fname')
@click.option('--dft_bde_fname')
@click.option('--gap_fname')
def gap_bde_summary(bde_start_fname, dft_bde_fname, gap_fname):

    dft_ats = read(dft_bde_fname, ':')
    bde_starts = read(bde_start_fname, ':')
    gap = Potential(pram_fnme=gap_fname)


    gap_ats = []
    for at in bde_starts:
        relaxed = util.relax(at, gap)
        relaxed.info['gap_energy'] = relaxed.get_potential_energy()
        gap_ats.append(relaxed)



    gap_mol = gap_ats[0]
    gap_h = gap_ats[1]
    gap_rads = gap_ats[3:]

    dft_mol = dft_ats[0]
    dft_h = dft_ats[1]
    dft_rads = dft_ats[2:]

    print(f'{"Structure":<12} {"DFT energy":<12} {"GAP energy":<12} {"abs. error":<12}')

    dft_mol_e = dft_mol.info["dft_energy"]
    gap_mol_e = gap_mol.info["gap_energy"]
    error = abs(dft_mol_e - gap_mol_e)
    print(f'{"molecule":<10} {dft_mol_e:<12} {gap_mol_e:<12} {error:<12}')

    dft_h_e = dft_h.info["dft_energy"]
    gap_h_e = gap_h.info["gap_energy"]
    error = abs(dft_h_e - gap_h_e)
    print(f'{"isolated H":<12} {dft_h_e:<12} {gap_h_e:<12} {error:<12}')

    bde_errors = []
    for idx, (dft_rad, gap_rad) in enumerate(zip(dft_rads, gap_rads)):

        dft_rad_e = dft_rad.info['dft_energy']
        dft_bde = dft_mol_e - dft_rad_e - dft_h_e

        gap_rad_e = gap_rad.info['gap_energy']
        gap_bde = gap_mol_e - gap_rad_e - gap_h_e

        error = abs(dft_bde - gap_bde)

        label = f'radical {idx}'
        print(f'{label:<12} {dft_bde:<12} {gap_bde:<12} {error:<12}')

    print('-' * 12*4 + 3)
    print(f'{"average abs. BDE error":<38} {np.average:<12}')


