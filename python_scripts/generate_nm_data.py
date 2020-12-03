from ase.io import read, write
import shutil
import click
import util
from util.vibrations import Vibrations
from util import itools
from util import plot
from util import ugap
from os.path import join as pj
from copy import deepcopy
import os
from ase.calculators.orca import ORCA
import datetime


@click.command()
@click.option('--dft_min_fname', type=click.Path(), help='Name of xyz with named dft minima')
@click.option('--pckl_fname', type=click.Path(), help='Name of the pickle for vibration stuff')
@click.option('--no_dpoints', type=int, help=' no of samples to generate from each dft min at for each standard deviation')
@click.option('--temps', type=str, help='list of temperatures for normal modedisplacements')
@click.option('--prefix', type=str, help='prefix for the file to save structures to')
@click.option('--config_type', type=str, help='config_type to add to (all) of the atoms')
@click.option('--append_isolated_ats', type=bool, default=True, show_default=True, help = 'whether to append isolated atoms to the dataset for training')
def gen_nm_data(dft_min_fname, pckl_fname, no_dpoints, temps, prefix, append_isolated_ats, config_type):

    print(f'Time: {datetime.datetime.now()}')
    temperatures = util.str_to_list(temps)

    smearing = 2000
    n_wfn_hop = 1
    task = 'gradient'
    orcasimpleinput = 'UKS B3LYP def2-SV(P) def2/J D3BJ'
    orcablocks =  f"%scf Convergence tight \n SmearTemp {smearing}  end \n"

    print('orca simple input:')
    print(orcasimpleinput)

    scratch_dir = '/scratch/eg475'
    no_cores = os.environ['OMP_NUM_THREADS']

    n_run_glob_op = 1
    kw_orca = f'smearing={smearing}'


    db_path = '/home/eg475/scripts/'
    isolated_at_fname = pj(db_path, 'source_files/isolated_atoms_orca.xyz')

    #################
    # generate data
    ###############################################

    dft_min_fname = pj(db_path, 'dft_minima', dft_min_fname)
    dft_atoms = read(dft_min_fname, ':')

    for temp in  temperatures:
        for dft_at in dft_atoms:
            dft_name = dft_at.info['name']

            output_fname = f'{prefix}_{dft_name}_{int(temp)}K.xyz'
            print(f'getting data for: {output_fname}')


            orca_xyz = output_fname
            wfl_command = f'wfl -v ref-method orca-eval --output-file {orca_xyz} -tmp ' \
                  f'{scratch_dir} --keep-files True --base-rundir orca_outputs ' \
                  f'-nr {n_run_glob_op} -nh {n_wfn_hop} --kw "{kw_orca}" ' \
                  f'--orca-simple-input "{orcasimpleinput}"'



            if not os.path.isfile(output_fname):
                pckl_fname = pj(db_path, 'dft_minima/normal_modes', f'{dft_name}.all.pckl')
                print(f'copying pckl file: {pckl_fname}')
                shutil.copy(pckl_fname, '.')
                vib = Vibrations(dft_at, name=dft_name)
                all_ats_to_fit = vib.multi_at_nm_displace(temp=temp, n_samples=no_dpoints)

                dset_1 = itools.orca_par_data(atoms_in=all_ats_to_fit,
                                              out_fname=output_fname,
                                              wfl_command=wfl_command, config_type=config_type)

                if append_isolated_ats:
                    isolated_atoms = read(isolated_at_fname, ':')
                    write(output_fname, dset_1 + isolated_atoms, 'extxyz', write_results=False)

                if no_dpoints != len(dset_1):
                    print(f' asked for: {no_dpoints}, got: {len(dset_1)}')
                    raise RuntimeError('Did not get enough data points')

                # os.remove(orca_xyz)
                # os.remove(f'{dft_name}.all.pckl')
            else:
                print(f'found {output_fname}')




if __name__=='__main__':
    gen_nm_data()
