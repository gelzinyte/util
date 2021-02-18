import os
import yaml
from ase.io import read, write
from quippy.potential import Potential
from wfl.calculators import orca, generic
from util import ugap
from wfl.configset import ConfigSet_in, ConfigSet_out
from util import iter_tools as it
import util
from util import bde
from util.config import Config
from wfl.generate_configs import vib

# will be changed
# from wfl.plotting import error_table

def fit(no_cycles, test_fname='test.xyz',
        first_train_fname='train.xyz', e_sigma=0.0005, f_sigma=0.02,
        n_sparse=600,
        smiles_csv=None, num_smiles_opt=None, 
        opt_starts_fname=None,  num_nm_displacements=None):
    """ iteratively fits and optimises stuff
    
    Parameters
    ---------
  
    no_cycles: int
        number of gap-fit and optimise cycles to do
    test_fname: str, default='test.xyz'
        test set fname
    first_train_fname: str, default='train.xyz'
        fname for fitting the first GAP
    e_sigma: float, default=0.0005
        gap_fit e_sigma
    f_sigma: float, default=0.02
        gap_fit f_sigma
    smiles_csv: str, default None
        CSV of idx, name, smiles to generate structures from
    num_smiles_opt: int, default None
        for smiles, how many to generate-optimise structures
    opt_starts_fname: str, default=None
        xyz with structures to optimise every cycle
    num_nm_displacements: int, default=None
        how many normal modes to sample from GAP-optimised structures

    """

    assert smiles_csv is None or opt_starts_fname is None
    
    if smiles_csv is not None:
        print(f'Optimising structures from {smiles_csv}, {num_smiles_opt} times')
    elif opt_starts_fname is not None:
        print(f'Optimising structures from {opt_starts_fname}')
    else:
        raise RuntimeError('Need either smiles csv or opt strart file for optimisations every cycle')

    bde_root = '/home/eg475/data/bde_files'

    cfg = Config.load()
    gap_fit_path =  cfg['programs']['gap_fit']
    scratch_dir = cfg['other_paths']['tmp']
    glue_fname = cfg['gap_files']['glue_fname']


    required_dirs = ['gaps', 'xyzs', 'bdes', 'xyzs/opt_trajs']
    for dir_name in required_dirs:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    test_set = ConfigSet_in(input_files=test_fname)
    calculator = None
    
    nm_temp = 300 

    ##########################
    ## Orca params
    ########################
    
    default_kw = Config.from_yaml(os.path.join(cfg['util_root'], 'default_kwargs.yml'))

    output_prefix = 'dft_'
    orca_kwargs = default_kw['orca']

    orca_kwargs['smearing'] = 2000

    print(f'orca_kwargs: {orca_kwargs}')

    ########################################################
    ## set up GAP
    ######################################################

    # e_sigma = 0.0005
    # f_sigma = 0.02

    l_max = 6
    n_max = 12
    delta = 1
    zeta = 4
    n_sparse = n_sparse

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

    ########################################################
    ## set up first dset
    ########################################################

    dset_0_fname = 'xyzs/train_0.xyz'
    if not os.path.exists(dset_0_fname):
        print('generating 0-th dset')
        dset = read(first_train_fname, ':')
        for at in dset:
            if 'iter_no' not in at.info.keys():
                at.info['iter_no'] = '0'
        write(dset_0_fname, dset, write_results=False)


    ########################################################
    ## Fit first GAP
    ########################################################

    gap_fname = 'gaps/gap_0.xml'
    if not os.path.exists(gap_fname):
        print('fitting 0-th gap')
        gap_command = ugap.make_gap_command(gap_filename=gap_fname, training_filename=dset_0_fname,
                                            descriptors_dict=descriptors, default_sigma=default_sigma,
                                            gap_fit_path=gap_fit_path, output_filename='gaps/out_0.txt',
                                            glue_fname=glue_fname, config_type_sigma=config_type_sigma)

        print(f'gap 0 command:\n{gap_command}')
        stdout, stderr = util.shell_stdouterr(gap_command)
        print(f'stdout: {stdout}')
        print(f'stderr: {stderr}')


    # train_set = ConfigSet_in(input_files=dset_0_fname)
    # for dataset, name in zip([train_set, test_set], ['train', 'test']):
    #     output_fname = f'xyzs/gap_0_on_{name}.xyz'
    #     if not os.path.isfile(output_fname):
    #         print('error table')
    #         if calculator is None:
    #             calculator = Potential(param_filename=gap_fname)
    #         print('-'*30, f'\n{name}\n', '-'*30)
    #         error_table.plot(data=dataset, ref_prefix='dft_', pred_prefix='gap0_',
    #                          calculator=calculator,
    #                          output_fname=output_fname, chunksize=200)


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

        #################
        # define filenames
        #######################

        if  smiles_csv is not None:
            opt_starts_fname = f'xyzs/non_opt_mols_rads_{cycle_idx}.xyz'

        nm_ref_fname = None
        nm_sample_fname = None
        if num_nm_displacements is not None:
            nm_ref_fname = f'xyzs/normal_modes_reference_{cycle_idx}.xyz'
            nm_sample_fname = f'xyzs/normal_modes_sample_{cycle_idx}.xyz'

        opt_fname = f'xyzs/opt_mols_rads_{cycle_idx}.xyz'
        extra_data_with_dft = f'xyzs/opt_mols_rads_dft_{cycle_idx}.xyz'
        opt_traj_fname = f'xyzs/opt_trajs/opt_{cycle_idx}.xyz'
        train_set_fname = f'xyzs/train_{cycle_idx}.xyz'
        opt_logfile = f'xyzs/opt_trajs/opt_log_{cycle_idx}.txt'


        calculator = (Potential, [], {'param_filename':gap_fname})
        

        if not os.path.exists(train_set_fname):
            
            # generate structures for optimisation
            if not os.path.exists(opt_starts_fname):
                print(f'making structures to optimise at {opt_starts_fname}')
                
                it.make_structures(smiles_csv, iter_no=cycle_idx,
                                   num_smi_repeat=num_smiles_opt,
                                   opt_starts_fname=opt_starts_fname)

            # optimise
            if not os.path.exists(opt_fname):
                print(f'gap-optimising {opt_starts_fname} to {opt_fname}')
                
                it.optimise(calculator=calculator, opt_starts_fname=opt_starts_fname,
                            opt_fname=opt_fname, opt_traj_fname=opt_traj_fname,
                            logfile=opt_logfile)

            file_for_dft = opt_fname
            
            if num_nm_displacements is not None:

                # generate normal mode reference
                if not os.path.exists(nm_ref_fname):
                    inputs = ConfigSet_in(input_files=opt_fname)
                    outputs = ConfigSet_out(output_files=nm_ref_fname)
                    vib.generate_normal_modes_parallel_atoms(inputs=inputs,
                                         outputs=outputs, 
                                         calculator=calculator,
                                         prop_prefix='gap_')


                # sample normal modes
                if not os.path.exists(nm_sample_fname):
                    inputs = ConfigSet_in(input_files=nm_ref_fname)
                    outputs = ConfigSet_out(output_files=nm_sample_fname)
                    info_to_keep = ['config_type', 'iter_no', 'minim_n_steps']
                    
                    vib.sample_normal_modes(inputs=inputs,
                                             outputs=outputs, 
                                             temp=nm_temp, 
                                             sample_size=num_nm_displacements, 
                                             info_to_keep=info_to_keep,
                                            prop_prefix='gap_')

                    ats = read(nm_sample_fname, ":")
                    for at in ats:
                        at.cell = [40, 40, 40]
                        at.info['iter_no'] = cycle_idx
                    write(nm_sample_fname, ats)

                # re-define file to be re-calculated with dft
                file_for_dft = nm_sample_fname



            if not os.path.exists(extra_data_with_dft):
                print(f'Calculating dft energies')
                dft_evaled_opt_mols_rads = ConfigSet_out(output_files=extra_data_with_dft)
                inputs = ConfigSet_in(input_files=file_for_dft)
                dft_evaled_opt_mols_rads = orca.evaluate(inputs=inputs,
                                                         outputs=dft_evaled_opt_mols_rads,
                                                         orca_kwargs=orca_kwargs, output_prefix='dft_',
                                                         keep_files=True, base_rundir=f'orca_outputs_{cycle_idx}'
                                                         )

                # if calculator is None:
                #     calculator = Potential(param_filename=gap_fname)

                # print('-' * 15, f'Optimisation {cycle_idx} errors:')
                # error_table.plot(data=dft_evaled_opt_mols_rads,
                #                  ref_prefix='dft_', pred_prefix=f'gap{cycle_idx - 1}_',
                #                  calculator=calculator, output_fname=opt_fname)

            previous_dset = read(f'xyzs/train_{cycle_idx - 1}.xyz', ':')
            new_atoms = read(extra_data_with_dft, ':')
            write(train_set_fname, previous_dset + new_atoms, write_results=False)
        else:
            print(f'Found {train_set_fname}, not re-generating')


        # refit gap
        gap_fname = f'gaps/gap_{cycle_idx}.xml'
        out_fname = f'gaps/out_{cycle_idx}.xml'

        if not os.path.exists(gap_fname):
            print('fitting gap')
            gap_command = ugap.make_gap_command(gap_filename=gap_fname, training_filename=train_set_fname,
                                                descriptors_dict=descriptors, default_sigma=default_sigma,
                                                gap_fit_path=gap_fit_path, output_filename=out_fname,
                                                glue_fname=glue_fname, config_type_sigma=config_type_sigma)
            print(f'-' * 15, f'gap {cycle_idx} command: {gap_command}')
            stdout, stderr = util.shell_stdouterr(gap_command)
            print(f'stdout: {stdout}')
            print(f'stderr: {stderr}')


        # test/evaluate gap
        # train_set = ConfigSet_in(input_files=train_set_fname)
        # for dataset, name in zip([train_set, test_set], ['train', 'test']):
        #     output_fname = f'xyzs/gap_{cycle_idx}_on_{name}.xyz'
        #     if not os.path.isfile(output_fname):
        #         if calculator is None:
        #             calculator = Potential(param_filename=gap_fname)
        #         print(f'gap fname: {gap_fname}')
        #         error_table.plot(data=dataset, ref_prefix='dft_', pred_prefix=f'gap{cycle_idx}_',
        #                          calculator=calculator,
        #                          output_fname=output_fname, chunksize=200)

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
