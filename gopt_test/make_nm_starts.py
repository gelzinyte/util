import os
import click
from util.vibrations import Vibrations
from ase.io import read, write
from os.path import join as pj
import shutil
import util


@click.command()
@click.option('--dft_min_fname', type=click.Path(), help='dft minima to generate structures for. Assumes a pickle file somewhere')
@click.option('--temperatures', type=str, help='list of temperatures for normal mode displacements')
def generate_starts(dft_min_fname, temperatures):

    temperatures =  util.str_to_list(temperatures)
    temperatures = [int(t) for t in temperatures]

    dft_ats = read(dft_min_fname, ':')
    db_path = '/home/eg475/programs/my_scripts/gopt_test'

    for temp in temperatures:
        for at in dft_ats:
            dft_name = at.info['name']
            starts_name = pj(db_path, f'starts/NM_starts_{dft_name}_{temp}K.xyz')

            if not os.path.exists(starts_name):
                pckl_name = f'/home/eg475/programs/my_scripts/gopt_test/dft_minima/normal_modes/{dft_name}.all.pckl'
                if not os.path.isfile(f'{dft_name}.all.pckl'):
                    shutil.copy(pckl_name, '.')
                vib = Vibrations(at, name=dft_name)
                starts = vib.displace_all_nms(temp)

                for idx, at in enumerate(starts):
                    at.info['config_type'] = f'start_{idx}'

                print(f'writing {starts_name}')
                write(starts_name, starts, 'extxyz', write_results=False)

    for at in dft_ats:
        pckl_name = f'{at.info["name"]}.all.pckl'
        if os.path.exists(pckl_name):
            os.remove(pckl_name)


if __name__=='__main__':
    generate_starts()
