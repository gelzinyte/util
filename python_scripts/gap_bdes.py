from quippy.potential import Potential
import numpy as np
from ase.io import read, write
import click
import util
import os
from tabulate import tabulate

@click.command()
@click.option('--bde_start_fname')
@click.option('--dft_bde_fname')
@click.option('--gap_fname')
@click.option('--gap_bde_fname', help='.xyz with gap_energy for gap-optimised structures, either to save into or load')
def gap_bde_summary(bde_start_fname, dft_bde_fname, gap_fname, gap_bde_fname):

    dft_ats = read(dft_bde_fname, ':')

    if not os.path.exists(gap_bde_fname) and bde_start_fname:
        print(f"Didn't find {gap_bde_fname}, optimising bde starts")
        gap = Potential(param_filename=gap_fname)
        bde_starts = read(bde_start_fname, ':')
        gap_ats = []
        for at in bde_starts:
            relaxed = util.relax(at, gap)
            relaxed.info['gap_energy'] = relaxed.get_potential_energy()
            gap_ats.append(relaxed)
        write(gap_bde_fname, gap_ats, 'extxyz')
    elif os.path.exists(gap_bde_fname):
        gap_ats = read(gap_bde_fname, ':')
    else:
        raise FileNotFoundError("Need either bde start file to optimise or gap_bde_fname with optimised structures")


    gap_mol = gap_ats[0]
    gap_h = gap_ats[1]
    gap_rads = gap_ats[3:]

    dft_mol = dft_ats[0]
    dft_h = dft_ats[1]
    dft_rads = dft_ats[2:]

    headers = [' ', "eV\nDFT E", "eV\nDFT BDE", "eV\nGAP E", "eV\nGAP BDE", "meV\nabs error"]
    data = []

    dft_mol_e = dft_mol.info["dft_energy"]
    gap_mol_e = gap_mol.info["gap_energy"]
    error = abs(dft_mol_e - gap_mol_e)*1e3
    data.append(['mol', dft_mol_e, np.nan, gap_mol_e, np.nan, error])

    dft_h_e = dft_h.info["dft_energy"]
    gap_h_e = gap_h.info["gap_energy"]
    error = abs(dft_h_e - gap_h_e)*1e3
    data.append(['H', dft_h_e, np.nan, gap_h_e, np.nan, error])

    bde_errors = []
    for idx, (dft_rad, gap_rad) in enumerate(zip(dft_rads, gap_rads)):

        dft_rad_e = dft_rad.info['dft_energy']
        dft_bde = - dft_mol_e + dft_rad_e + dft_h_e

        gap_rad_e = gap_rad.info['gap_energy']
        gap_bde = - gap_mol_e + gap_rad_e + gap_h_e

        error = abs(dft_bde - gap_bde)*1e3
        bde_errors.append(error)

        if "config_type" in gap_rad.info.keys():
            label = gap_rad.info["config_type"] + ' ' + str(idx)
        else:
            label = f'rad {idx}'
        data.append([label, dft_rad_e, dft_bde, gap_rad_e, gap_bde, error])

    data.append(['mean BDE\nerror', np.nan, np.nan, np.nan, np.nan, np.mean(bde_errors)])

    print(tabulate(data, headers=headers,  floatfmt=".3f" ))


if __name__ == '__main__':
    gap_bde_summary()