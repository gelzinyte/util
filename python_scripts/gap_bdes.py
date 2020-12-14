from quippy.potential import Potential
import numpy as np
from ase.io import read, write
import click
import util
import os
from tabulate import tabulate

@click.command()
@click.option('--bde_start_dir', type=click.Path(), default='/home/eg475/data/bde_files/starts', help='which files to gap-optimise')
@click.option('--dft_bde_dir', type=click.Path(), default='/home/eg475/data/bde_files/dft', help='which files to compare to')
@click.option('--gap_fname')
@click.option('--bde_out_dir', type=click.Path(), default='GAP_BDE', help='Directory where to save all opt_files')
def gap_bde_summary(bde_start_dir, dft_bde_dir, gap_fname, bde_out_dir):

    if not os.path.exists(bde_out_dir):
        os.makedirs(bde_out_dir)
    os.chdir(bde_out_dir)

    # dft_fnames = [os.path.join(dft_bde_dir, name) for name in os.listdir(dft_bde_dir)]
    # dft_fnames = util.natural_sort(dft_fnames)

    start_fnames = [os.path.join(bde_start_dir, name) for name in os.listdir(bde_start_dir)]
    start_fnames = util.natural_sort(start_fnames)

    if gap_fname is not None:
        gap = Potential(param_filename=os.path.join('..', gap_fname))

    for start_fname in start_fnames:
        basename = os.path.basename(start_fname)
        gap_bde_fname = basename.replace('_non-optimised.xyz', '_gap_optimised.xyz')
        dft_fname = os.path.join(dft_bde_dir, basename.replace('_non-optimised.xyz', '_optimised.xyz'))

        # optimise structures
        if not os.path.exists(gap_bde_fname) and start_fname and gap_fname is not None:
            print(f"Didn't find {gap_bde_fname}, optimising bde starts")
            bde_starts = read(start_fname, ':')
            gap_ats = []
            for at in bde_starts:
                relaxed = util.relax(at, gap)
                relaxed.info['gap_energy'] = relaxed.get_potential_energy()
                gap_ats.append(relaxed)
            write(gap_bde_fname, gap_ats, 'extxyz')
        elif os.path.exists(gap_bde_fname):
            gap_ats = read(gap_bde_fname, ':')
        else:
            raise FileNotFoundError(
                "Need either bde start file to optimise or gap_bde_fname "
                "with optimised structures")

        # print BDE summary while we're at it
        print('\n\n', '-' * 30)
        print(basename.replace('_non-optimised.xyz', ''))
        print('-' * 30)
        if os.path.exists(dft_fname):
            dft_ats = read(dft_fname, ':')

            gap_mol = gap_ats[0]
            gap_h = gap_ats[1]
            gap_rads = gap_ats[2:]

            dft_mol = dft_ats[0]
            dft_h = dft_ats[1]
            dft_rads = dft_ats[2:]

            headers = [' ', "Å\nRMSD", "eV\nDFT E", "eV\nDFT BDE", "eV\nGAP E", "eV\nGAP BDE", "meV\nabs error"]
            data = []

            dft_mol_e = dft_mol.info["dft_energy"]
            gap_mol_e = gap_mol.info["gap_energy"]
            error = abs(dft_mol_e - gap_mol_e)*1e3
            rmsd = util.get_rmse(dft_mol.positions, gap_mol.positions)
            data.append(['mol', rmsd, dft_mol_e, np.nan, gap_mol_e, np.nan, error])

            dft_h_e = dft_h.info["dft_energy"]
            gap_h_e = gap_h.info["gap_energy"]
            error = abs(dft_h_e - gap_h_e)*1e3
            data.append(['H', np.nan,  dft_h_e, np.nan, gap_h_e, np.nan, error])

            bde_errors = []
            rmsds = []
            for idx, (dft_rad, gap_rad) in enumerate(zip(dft_rads, gap_rads)):

                dft_rad_e = dft_rad.info['dft_energy']
                dft_bde = - dft_mol_e + dft_rad_e + dft_h_e

                gap_rad_e = gap_rad.info['gap_energy']
                gap_bde = - gap_mol_e + gap_rad_e + gap_h_e

                error = abs(dft_bde - gap_bde)*1e3
                rmsd = util.get_rmse(dft_rad.positions, gap_rad.positions)
                bde_errors.append(error)
                rmsds.append(rmsd)

                if "config_type" in gap_rad.info.keys():
                    label = gap_rad.info["config_type"]
                else:
                    label = f'rad {idx}'
                data.append([label, rmsd, dft_rad_e, dft_bde, gap_rad_e, gap_bde, error])

            data.append(['mean', np.mean(rmsds), np.nan, np.nan, np.nan, np.nan, np.mean(bde_errors)])


            print(tabulate(data, headers=headers,  floatfmt=".3f" ))
        else:
            print(f"\n\nCouldn't find {dft_fname} to print BDE summary")


if __name__ == '__main__':
    gap_bde_summary()