from ase.io import read, write
import click
import util
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
@click.option('--no_dpoints', type=str, help='list of no of samples to generate from each dft min at for each standard deviation')
@click.option('--stdev', type=str, help='list of standard deviations for normal distribution from which cartesian displacements are sampled')
@click.option('--n_rattle_atoms', type=str, required=True, help='list of Number of atoms to rattle in the structure')
@click.option('--n_sparse', type=int, help='GAP n_sparse')
@click.option('--data_only', type=bool, default=False, help='if True, generates data, but doesnt fit GAP')
def fit_gap_from_dft_minima(dft_min_fname, no_dpoints, stdev, n_rattle_atoms, n_sparse, data_only):

    print(f'Time: {datetime.datetime.now()}')

    stdev = util.str_to_list(stdev)
    no_dpoints = [int(n) for n in util.str_to_list(no_dpoints)]
    n_rattle_atoms = [int(n) for n in util.str_to_list(n_rattle_atoms)]

    print(f'standard deviations: {stdev} and corresponding no_dpoints: {no_dpoints}')


    db_path = '/home/eg475/programs/my_scripts/'
    dft_min_fname = pj(db_path, 'gopt_test/dft_minima', dft_min_fname)
    glue_fname = pj(db_path, 'source_files/glue_orca.xml')
    isolated_at_fname = pj(db_path, 'source_files/isolated_atoms_orca.xyz')

    scratch_dir = '/scratch/eg475'

    gap_fit_path = '/home/eg475/programs/QUIPwo0/build' \
                   '/linux_x86_64_gfortran_openmp/gap_fit'

    no_cores = os.environ['OMP_NUM_THREADS']

    # if not os.path.isdir('gaps'):
    #     os.makedirs('gaps')

    if not os.path.isdir('xyzs'):
        os.makedirs('xyzs')

    if not os.path.isdir('pictures'):
        os.makedirs('pictures')

    ########################################################
    ## set up GAP
    ######################################################

    default_sigma = [0.005, 0.025, 0.0, 0.0]

    config_type_sigma = {'isolated_atom': [0.0001, 0.0, 0.0, 0.0]}

    descriptors = dict()
    descriptors['soap'] = {'name': 'soap',
                           'l_max': '4',
                           'n_max': '8',
                           'cutoff': '3.0',
                           'delta': '0.5',
                           'covariance_type': 'dot_product',
                           'zeta': '4.0',
                           'n_sparse': str(n_sparse),
                           'sparse_method': 'cur_points',
                           'atom_gaussian_width': '0.3',
                           'add_species': 'True'}


    ########################################################
    ## set up calculator
    ########################################################

    smearing = 2000
    maxiter = 200
    n_wfn_hop = 1
    task = 'gradient'
    orcasimpleinput = 'UKS B3LYP def2-SV(P) def2/J D3BJ'
    orcablocks =  f"%scf Convergence tight \n SmearTemp {smearing} \n maxiter {maxiter} end \n"
                  # f'%pal nprocs {no_cores} end'

    print(orcasimpleinput)

    mult=1
    charge=0

    orca_command = '/home/eg475/programs/orca/orca_4_2_1_linux_x86' \
                   '-64_openmpi314/orca'

    # for small non-parallel calculations
    calc = ORCA(label="ORCA/orca",
                orca_command=orca_command,
                charge=charge,
                mult=mult,
                task=task,
                orcasimpleinput=orcasimpleinput,
                orcablocks=orcablocks
                )

    # for calculating dataset energies
    n_run_glob_op = 1
    kw_orca = f'n_hop={n_wfn_hop} smearing={smearing} maxiter={maxiter}'

    orca_xyz = f'xyzs/orca_out.xyz'
    wfl_command = f'wfl -v ref-method orca-eval -o {orca_xyz} -tmp ' \
                  f'{scratch_dir} ' \
                  f'-n {n_run_glob_op} -p {no_cores} --kw "{kw_orca}" ' \
                  f'--orca-simple-input "{orcasimpleinput}"'

    #################
    # generate data
    ###############################################

    dset_name = 'xyzs/gap_dset.xyz'

    if not os.path.isfile(dset_name):
        dft_atoms = read(dft_min_fname, ':')
        all_ats_to_fit = []
        for std, no_str, n_rtl in zip(stdev, no_dpoints, n_rattle_atoms):
            ats_to_fit = itools.get_structures(dft_atoms, n_dpoints=int(no_str), stdev=std, n_rattle_atoms=n_rtl)
            all_ats_to_fit += ats_to_fit

        dset_1 = itools.orca_par_data(atoms_in=all_ats_to_fit,
                                      out_fname=orca_xyz,
                                      wfl_command=wfl_command, config_type='none')
        isolated_atoms = read(isolated_at_fname, ':')
        write(dset_name, dset_1 + isolated_atoms, 'extxyz', write_results=False)


    if not data_only:
        ###############
        # fit gap

        gap_name = 'gap.xml'
        if not os.path.isfile(gap_name):
            command = ugap.make_gap_command(gap_filename=gap_name, training_filename=dset_name, descriptors_dict=deepcopy(descriptors),
                                          default_sigma=default_sigma, output_filename='out_gap.txt', glue_fname=glue_fname,
                                            config_type_sigma=config_type_sigma, gap_fit_path=gap_fit_path)

            print(f'----GAP command\n {command}')
            stdout, stderr  = util.shell_stdouterr(command)

            print(f'---gap stdout:\n {stdout}')
            print(f'---gap stderr:\n {stderr}')

        #########
        # plt some plots

        plot.make_scatter_plots_from_file(gap_name, output_dir='pictures/',
                                          by_config_type=True)
        plot.make_dimer_curves([gap_name],
                               output_dir='pictures/', \
                               plot_2b_contribution=False,
                               glue_fname=glue_fname,
                               isolated_atoms_fname=isolated_at_fname)

        # plot.make_kpca_plots(training_set=dset_name)


if __name__=='__main__':
    fit_gap_from_dft_minima()
