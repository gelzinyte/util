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
@click.option('--temp', type=float, help='temperature for normal modedisplacements')
@click.option('--output_fname', type=click.Path(), help='name of the file to save structures to')
@click.option('--append_isolated_ats', type=bool, default=True, show_default=True, help = 'whether to append isolated atoms to the dataset for training')
def fit_gap_from_dft_minima(dft_min_fname, pckl_fname, no_dpoints, temp, output_fname, append_isolated_ats):

    print(f'Time: {datetime.datetime.now()}')

    print(f'temperature: {temp} and no_dpoints: {no_dpoints}')

    db_path = '/home/eg475/programs/my_scripts/'
    dft_min_fname = pj(db_path, 'gopt_test/dft_minima', dft_min_fname)
    isolated_at_fname = pj(db_path, 'source_files/isolated_atoms_orca.xyz')
    dft_at = read(dft_min_fname)
    dft_name = dft_at.info['name']
    if not pckl_fname:
       pckl_fname = pj(db_path, 'gopt_test/dft_minima/normal_modes', f'{dft_name}.all.pckl')

    scratch_dir = '/scratch/eg475'

    no_cores = os.environ['OMP_NUM_THREADS']

    smearing = 2000
    maxiter = 200
    n_wfn_hop = 1
    task = 'gradient'
    orcasimpleinput = 'UKS B3LYP def2-SV(P) def2/J D3BJ'
    orcablocks =  f"%scf Convergence tight \n SmearTemp {smearing} \n maxiter {maxiter} end \n"

    print('orca simple input:')
    print(orcasimpleinput)

    n_run_glob_op = 1
    kw_orca = f'n_hop={n_wfn_hop} smearing={smearing} maxiter={maxiter}'

    # orca_xyz = os.path.splitext(output_fname)[0] + '_orca_tmp_out.xyz'
    orca_xyz = output_fname
    wfl_command = f'wfl -v ref-method orca-eval -o {orca_xyz} -tmp ' \
                  f'{scratch_dir} ' \
                  f'-n {n_run_glob_op} -p {no_cores} --kw "{kw_orca}" ' \
                  f'--orca-simple-input "{orcasimpleinput}"'

    #################
    # generate data
    ###############################################


    if not os.path.isfile(output_fname):
        shutil.copy(pckl_fname, '.')
        vib = Vibrations(dft_at, name=dft_name)
        all_ats_to_fit = vib.multi_at_nm_displace(temp=temp, n_samples=no_dpoints)

        dset_1 = itools.orca_par_data(atoms_in=all_ats_to_fit,
                                      out_fname=output_fname,
                                      wfl_command=wfl_command, config_type='none')

        if append_isolated_ats:
            isolated_atoms = read(isolated_at_fname, ':')
            write(output_fname, dset_1 + isolated_atoms, 'extxyz', write_results=False)

        # os.remove(orca_xyz)
        # os.remove(f'{dft_name}.all.pckl')




if __name__=='__main__':
    fit_gap_from_dft_minima()
