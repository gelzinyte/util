import os
from ase.io import read, write
from quippy.potential import Potential
from wfl.calculators import orca, generic
from util import ugap
from wfl.configset import ConfigSet_in, ConfigSet_out
from util import iter_tools as it
import util
from util import bde

# will be changed
from wfl.plotting import error_table

no_opt = 5  # number of optimisations per compound (rad or mol) per cycle
no_cycles = 3  # number of GAP - fit cycles
# -> 5 x 5 x 20 (?) compounds = 500 extra

# files and other definitions
first_train_set_fname = 'train.xyz'
test_fname = 'test.xyz'
smiles_csv = 'small_cho.csv'

scratch_dir = '/tmp/eg475'
gap_fit_path = '/home/eg475/programs/QUIPwo0/build/linux_x86_64_gfortran_openmp/gap_fit'
glue_fname = '/home/eg475/scripts/source_files/glue_repulsive_fitted.xml'
bde_root = '/home/eg475/data/bde_files'

required_dirs = ['gaps', 'xyzs', 'bdes', 'xyzs/opt_trajs']
for dir_name in required_dirs:
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

test_set = ConfigSet_in(input_files=test_fname)
calculator = None

##########################
## Orca params
########################

output_prefix = 'dft_'
orca_kwargs = {'smearing': 2000,
               'n_run': 1,
               'n_hop': 1,
               'scratch_path': scratch_dir,
               'orca_simple_input': "UKS B3LYP def2-SV(P) def2/J D3BJ"}

########################################################
## set up GAP
######################################################

e_sigma = 0.001
f_sigma = 0.05

l_max = 6
n_max = 12
delta = 1
zeta = 4
n_sparse = 600

cutoff_soap_1 = 3
atom_gaussian_width_1 = 0.3
cutoff_transition_width_1 = 0.5

cutoff_soap_2 = 6
atom_gaussian_width_2 = 0.6
cutoff_transition_width_2 = 1

default_sigma = [e_sigma, f_sigma, 0.0, 0.0]
config_type_sigma = {'isolated_atom': [0.0001, 0.0, 0.0, 0.0]}

descriptors = dict()
descriptors['soap_1'] = {'name': 'soap',
                         'l_max': l_max,
                         'n_max': n_max,
                         'cutoff': cutoff_soap_1,
                         'delta': delta,
                         'covariance_type': 'dot_product',
                         'zeta': zeta,
                         'n_sparse': n_sparse,
                         'sparse_method': 'cur_points',
                         'atom_gaussian_width': atom_gaussian_width_1,
                         'cutoff_transition_width': cutoff_transition_width_1,
                         'add_species': 'True'}

descriptors['soap_2'] = {'name': 'soap',
                         'l_max': l_max,
                         'n_max': n_max,
                         'cutoff': cutoff_soap_2,
                         'delta': delta,
                         'covariance_type': 'dot_product',
                         'zeta': zeta,
                         'n_sparse': n_sparse,
                         'sparse_method': 'cur_points',
                         'atom_gaussian_width': atom_gaussian_width_2,
                         'cutoff_transition_width': cutoff_transition_width_2,
                         'add_species': 'True'}

## set up first dset
dset = read(first_train_set_fname, ':')
for at in dset:
    if len(at) == 1:
        continue
    if 'iter_no' not in at.info.keys():
        at.info['iter_no'] = '0'

dset_0_fname = 'xyzs/train_0.xyz'
write(dset_0_fname, dset, write_results=False)

## Fit first GAP
gap_fname = 'gaps/gap_0.xml'
if not os.path.exists(gap_fname):
    gap_command = ugap.make_gap_command(gap_filename=gap_fname, training_filename=dset_0_fname,
                                        descriptors_dict=descriptors, default_sigma=default_sigma,
                                        gap_fit_path=gap_fit_path, output_filename='gaps/out_0.txt',
                                        glue_fname=glue_fname, config_type_sigma=config_type_sigma)

    print(f'gap 0 command:\n{gap_command}')
    stdout, stderr = util.shell_stdouterr(gap_command)
    print(f'stdout: {stdout}')
    print(f'stderr: {stderr}')


train_set = ConfigSet_in(input_files=dset_0_fname)
for dataset, name in zip([train_set, test_set], ['train', 'test']):
    output_fname = f'xyzs/gap_0_on_{name}.xyz'
    if not os.path.isfile(output_fname):
        calculator = Potential(param_filename=gap_fname)
        print('-'*30, f'\n{name}\n', '-'*30)
        error_table.plot(data=dataset, ref_prefix='dft_', pred_prefix='gap0_',
                         calculator=calculator,
                         output_fname=output_fname, chunksize=200)


# for set_name in ['train', 'test']:
#     gap_name = 'gap0'
#     output_dir = f'bdes/{gap_name}/{set_name}'
#     bde_start_dir = os.path.join(bde_root, set_name, 'starts')
#     bde_dft_dir = os.path.join(bde_root, set_name, 'dft')
#     calculator = (Potential, [], {'param_filename':gap_fname})
#     print('making bde files')
#     bde.multi_bde_summaries(dft_dir=bde_dft_dir, gap_dir=output_dir, calculator=calculator,
#                             start_dir=bde_start_dir)
    # it.make_bde_files(bde_start_dir, output_dir, calculator, 'gap0')

###############################
## iterations
###############################


for cycle_idx in range(1, no_cycles + 1):


    print(f'-' * 30, f'\nCycle {cycle_idx} \n', '-' * 30)


    non_opt_fname = f'xyzs/non_opt_mols_rads_{cycle_idx}.xyz'
    opt_fname = f'xyzs/opt_mols_rads_{cycle_idx}.xyz'
    opt_fname_w_dft = f'xyzs/opt_mols_rads_dft_{cycle_idx}.xyz'
    opt_traj_fname = f'xyzs/opt_trajs/opt_{cycle_idx}.xyz'
    train_set_fname = f'xyzs/train_{cycle_idx}.xyz'

    # if os.path.isfile(new_gap_fname):
    #     print(f'Found {new_gap_fname}, continuing with the cycles')
    #     gap_fname = new_gap_fname
    #     continue

    if not os.path.exists(train_set_fname):
        # generate structures for optimisation
        if not os.path.exists(opt_fname):
            calculator = (Potential, [], {'param_filename':gap_fname})
            optimised_mols_rads = it.make_structures(smiles_csv, cycle_idx, calculator, no_opt,
                                                     non_opt_fname=non_opt_fname, opt_fname=opt_fname,
                                                     opt_traj_fname=opt_traj_fname,
                                                     logfile=f'xyzs/opt_trajs/opt_log_{cycle_idx}.txt')
        else:
            optimised_mols_rads = ConfigSet_in(input_files=opt_fname)

        tmp_at = read(opt_fname)
        if 'dft_energy' not in tmp_at.info.keys():
            print(f'Calculating dft energies')
            dft_evaled_opt_mols_rads = ConfigSet_out(output_files=opt_fname_w_dft)
            dft_evaled_opt_mols_rads = orca.evaluate(inputs=optimised_mols_rads,
                                                     outputs=dft_evaled_opt_mols_rads,
                                                     orca_kwargs=orca_kwargs, output_prefix='dft_')

            if calculator is None:
                calculator = Potential(param_filename=gap_fname)

            print('-' * 15, f'Optimisation {cycle_idx} errors:')
            error_table.plot(data=dft_evaled_opt_mols_rads,
                             ref_prefix='dft_', pred_prefix=f'gap{cycle_idx - 1}_',
                             calculator=calculator, output_fname=opt_fname, chunksize=200)

        previous_dset = read(f'xyzs/train_{cycle_idx - 1}.xyz', ':')
        new_atoms = read(opt_fname, ':')
        write(train_set_fname, previous_dset + new_atoms, write_results=False)
    else:
        print(f'Found {train_set_fname}, not re-generating')


    # refit gap
    gap_fname = f'gaps/gap_{cycle_idx}.xml'
    out_fname = f'gaps/out_{cycle_idx}.xml'

    if not os.path.exists(gap_fname):
        gap_command = ugap.make_gap_command(gap_filename=gap_fname, training_filename=train_set_fname,
                                            descriptors_dict=descriptors, default_sigma=default_sigma,
                                            gap_fit_path=gap_fit_path, output_filename=out_fname,
                                            glue_fname=glue_fname, config_type_sigma=config_type_sigma)
        print(f'-' * 15, f'gap {cycle_idx} command: {gap_command}')
        stdout, stderr = util.shell_stdouterr(gap_command)
        print(f'stdout: {stdout}')
        print(f'stderr: {stderr}')


    # test/evaluate gap
    train_set = ConfigSet_in(input_files=train_set_fname)
    for dataset, name in zip([train_set, test_set], ['train', 'test']):
        output_fname = f'xyzs/gap_{cycle_idx}_on_{name}.xyz'
        if not os.path.isfile(output_fname):
            if calculator is None:
                calculator = Potential(param_filename=gap_fname)
            print(f'gap fname: {gap_fname}')
            error_table.plot(data=dataset, ref_prefix='dft_', pred_prefix=f'gap{cycle_idx}_',
                             calculator=calculator,
                             output_fname=output_fname, chunksize=200)

    # for set_name in ['train', 'test']:
    #     gap_name = f'gap{cycle_idx}'
    #     output_dir = f'bdes/{gap_name}/{set_name}'
    #     bde_start_dir = os.path.join(bde_root, set_name, 'starts')
    #     bde_dft_dir = os.path.join(bde_root, set_name, 'dft')
    #
    #     print('making bde files')
    #     bde.multi_bde_summaries(dft_dir=bde_dft_dir, gap_dir=output_dir,
    #                             calculator=(Potential, [], {'param_filename':gap_fname}),
    #                             start_dir=bde_start_dir)

##################
# Analyse stuff
#########################

# learning curves: rmse & bdes + final GAP's energy/bde by structure

print('Finised iterations')
