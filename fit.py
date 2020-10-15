import random
import sys
import os
import subprocess

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from ase.io import read, write
from ase.optimize.precon import PreconLBFGS
from ase.build import molecule
from quippy.potential import Potential

sys.path.append('/home/eg475/molpro_stuff/driver')
import util
from util.vib import Vibrations
from util import itools
from util import urdkit
from util import plot
from molpro import Molpro
from ase.calculators.orca import ORCA
import time
import click
import datetime

@click.command()
@click.option('--n_iterations', type=int, required=True, help='number of fit-optimise-collect_data to execute')
@click.option('--n_dpoints', type=int, required=True, help='Number of structures to derive from a base structure by rattling its atoms')
@click.option('--n_rattle_atoms', type=int, required=True, help='Number of atoms to rattle in the structure')
@click.option('--first_n_dpoints', type=int, default=30, show_default=True, help='Number of data points for first GAP')
@click.option('--stdev', type=float, default=0.1, show_default=True, help='standard deviation for normal distribution from which cartesian displacements are sampled')
@click.option('--fmax', type=float, default=0.01, show_default=True, help='Target maximum force componenet for optimiser')
@click.option('--opt_steps', type=int, default=500, show_default=True, help='Maximum number of steps for the optimiser')
@click.option('--stride', type=int, default=5, show_default=True, help='Stride for sampling base structures from optimisation trajectory')
@click.option('--up_e_lim', type=float, help='energy, eV/atom above which the structures are not included in the training set')
@click.argument('smiles', nargs=-1, required=True)
def fit(n_iterations, stdev, n_dpoints, n_rattle_atoms, first_n_dpoints, fmax, opt_steps, stride, up_e_lim, smiles):

    print(f'Time: {datetime.datetime.now()}')

    ######################################################
    ## parameters for the run
    ########################################################


    scratch_dir = '/scratch/eg475'

    gap_fit_path = '/home/eg475/programs/QUIPwo0/build/linux_x86_64_gfortran_openmp/gap_fit'

    print('smiles to work with:', smiles)

    run_rnd_seed = int(time.time())%(2**8-1)
    print(f'random seed for the run: {run_rnd_seed}')

    no_cores = os.environ['OMP_NUM_THREADS']
    print(f'Number of cores: {no_cores}')
    print(f'Parameters:\nn_iterations \t {n_iterations}\nn_dpoints \t {n_dpoints}'
          f'\nn_rattle_atoms \t {n_rattle_atoms}\nstdev \t\t {stdev}\nfmax \t\t {fmax}'
          f'\nopt_steps \t {opt_steps}\nstride \t\t {stride}\nup_e_lim \t {up_e_lim}')



    if not os.path.isdir('gaps'):
        os.makedirs('gaps')

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
                           'n_sparse': '300',
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
    orcasimpleinput = 'UKS BLYP 6-31G slowconv'
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

    ######################################################
    ## generate first dataset
    ######################################################

    all_first_structures = []
    for smi in smiles:
        first_structure = itools.get_structure_to_optimise(smi, seed=None)
        first_structure.set_calculator(calc)
        first_structure.info['dft_energy'] = first_structure.get_potential_energy()
        first_structure.arrays['dft_forces'] = first_structure.get_forces()
        first_structure.info['config_type'] = 'opt_0'
        all_first_structures.append(first_structure)
    write('xyzs/base_structs_for_dset_1.xyz', all_first_structures, 'extxyz', write_results=False)


    if up_e_lim is not None:
        mean_first_guess_energy_per_at = np.mean([at.info['dft_energy']/len(at) for at in all_first_structures])
        print(f"Mean first guesses' energy per atom to use as a reference for upper limit: {mean_first_guess_energy_per_at} eV/atom")
        upper_energy_cutoff = mean_first_guess_energy_per_at + up_e_lim # eV/atom
        print(f"upper energy cutoff used to exclude structures above: {upper_energy_cutoff} eV/atom")
    else:
        upper_energy_cutoff = None


    if not os.path.isfile(f'xyzs/dset_1.xyz'):
        print('Getting dataset 1')
        dset_name = 'xyzs/dset_1.xyz'

        isolated_atoms = read(f'isolated_atoms_orca.xyz', index=':')

        dset_1_to_compute = itools.get_structures(all_first_structures, n_dpoints=first_n_dpoints,
                            n_rattle_atoms=n_rattle_atoms, stdev=stdev)

        orca_tmp_fname = f'xyzs/orca_out_dset_1.xyz'
        wfl_command = f'wfl -v ref-method orca-eval -o {orca_tmp_fname} -tmp {scratch_dir} ' \
                      f'-n {n_run_glob_op} -p {no_cores} --kw "{kw_orca}" ' \
                      f'--orca-simple-input "{orcasimpleinput}"'

        # 'wfl -v ref-method orca-eval -o dset_4_out2.xyz  -tmp /scratch/eg475 -n 1 -p 2 ' \
        # '--kw "n_hop=1 smearing=200 maxiter=200 " --orca-simple-input ' \
        # '"UHF B3LYP def2-SV(P) def2/J D3BJ slowconv" dset_4.xyz'

        dset_1 = itools.orca_par_data(atoms_in=dset_1_to_compute, out_fname=orca_tmp_fname,
                                       wfl_command=wfl_command, iter_no=1)

        if upper_energy_cutoff is not None:
            dset_1 = itools.remove_high_e_structs(dset_1, upper_energy_cutoff)

        write(dset_name, dset_1+isolated_atoms, 'extxyz', write_results=False)
    else:
        print('Dataset 1 is present')


    ######################################################
    ## fit-optimise-get data cycle
    ########################################################


    for iter_no in range(1, n_iterations+1):
        print('---------------Iteration %d-----------------' % iter_no)

        # Fit GAP
        gap_name = f'gaps/gap_{iter_no}.xml'
        if not os.path.isfile(gap_name):
            print(f'Iteration {iter_no}, fitting gap')
            itools.fit_gap(iter_no, descriptors=descriptors, default_sigma=default_sigma, gap_fit_path=gap_fit_path, config_type_sigma=config_type_sigma)

            plot.make_scatter_plots_from_file(gap_name, output_dir='pictures/', by_config_type=True)
            plot.make_dimer_curves([f'gaps/gap_{iter_no}.xml'], output_dir='pictures/',\
                                        plot_2b_contribution=False, glue_fname='glue_orca.xml', isolated_atoms_fname=f'isolated_atoms_orca.xyz')
        else:
            print(f'Found GAP {iter_no}, not fitting')


        # Optimise structure with new gap
        opt_str_name = f'xyzs/opt_at_{iter_no}.xyz'
        if not os.path.isfile(opt_str_name):
            print('optimising structure, name:', opt_str_name)

            # if seed is not None, the structure will be perturbed
            seed = run_rnd_seed + iter_no
            structs_to_opt_list = []
            for smi in smiles:
                structure_to_optimise = itools.get_structure_to_optimise(smi, seed=seed)
                structs_to_opt_list.append(structure_to_optimise)

            write(f'xyzs/ats_to_opt_{iter_no}.xyz', structs_to_opt_list)


            opt_atoms_list = []
            for structure_to_optimise in structs_to_opt_list:
                opt_atoms = itools.optimise_structure(iter_no, atoms=structure_to_optimise, fmax=fmax, steps=opt_steps)
                opt_atoms.set_calculator(calc)
                opt_atoms.info['dft_energy'] = opt_atoms.get_potential_energy()
                opt_atoms.arrays['dft_forces'] = opt_atoms.get_forces()
                opt_traj = read(f'xyzs/optimisation_{iter_no}.traj', ':')
                write(f'xyzs/optimisation_{iter_no}_{opt_atoms.info["smiles"]}.xyz', opt_traj, 'extxyz', write_results=False)
                opt_atoms_list.append(opt_atoms)
            write(opt_str_name, opt_atoms_list, 'extxyz', write_results=False)
        else:
            print('found optimised_structure, reading', opt_str_name)
            opt_atoms_list = read(opt_str_name)


        # Extend dataset with jiggles of structure optimised with new gap
        next_dset_name = f'xyzs/dset_{iter_no+1}.xyz'
        if not os.path.isfile(next_dset_name):
            print(f'Iteration {iter_no}, extending dataset')
            # TODO do extend_dset with correct calculator

            orca_tmp_fname = f'xyzs/orca_tmp_dset_{iter_no+1}.xyz'
            wfl_command =  f'wfl -v ref-method orca-eval -o {orca_tmp_fname} -tmp {scratch_dir} ' \
                      f'-n {n_run_glob_op} -p {no_cores} --kw "{kw_orca}" ' \
                      f'--orca-simple-input "{orcasimpleinput}"'


            itools.extend_dset(iter_no, smiles=smiles, n_dpoints=n_dpoints, n_rattle_atoms=n_rattle_atoms,
                               stdev=stdev, stride=stride, wfl_command=wfl_command, orca_tmp_fname=orca_tmp_fname,
                               upper_energy_cutoff=upper_energy_cutoff)

        else:
            print(f'Found dset_{iter_no+1}.xyz, not generating more data')

        plt.close('all')
        print(f'Iteration {iter_no} done!')


    ########################################################
    ## post-processing
    ########################################################

    # get all optimised atoms in one file
    all_opt_atoms = []
    atoms =  read('xyzs/base_structs_for_dset_1.xyz', ':')
    for at in atoms:
        at.info['config_type'] = 'opt_0'
        all_opt_atoms.append(at)

    for i in range(1, n_iterations+1):
        atoms = read(f'xyzs/opt_at_{i}.xyz', ":")
        for at in atoms:
            at.info['config_type'] = f'opt_{i}'
            all_opt_atoms.append(at)

    write(f'xyzs/opt_all.xyz', all_opt_atoms, 'extxyz', write_results=False)


    # all structures that were optimised
    structs_to_opt_all = []
    for i in range(1, n_iterations+1):
        at = read(f'xyzs/ats_to_opt_{i}.xyz')
        at.info['config_type'] = f'non_opt_{i}'
        structs_to_opt_all.append(at)
    write(f'xyzs/to_opt_all.xyz', structs_to_opt_all, 'extxyz', write_results=False)



    # summary plots
    plot.dimer_summary_plot(glue_fname='glue_orca.xml', plot_2b_contribution=False,
                           plot_ref_curve=False, isolated_atoms_fname='isolated_atoms_orca.xyz')

    plot.rmse_heatmap(train_fname=f'xyzs/dset_{iter_no}.xyz')


    plot.make_kpca_plots(training_set=f'xyzs/dset_{iter_no}.xyz')


    print('---FINITO---')


if __name__=='__main__':
    fit()