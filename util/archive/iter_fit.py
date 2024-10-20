import os
import yaml
import logging
import numpy as np
from ase.io import read, write
try:
    from quippy.potential import Potential
except ModuleNotFoundError:
    pass
from wfl.calculators import orca, generic
from util import ugap
from wfl.configset import ConfigSet, ConfigSet_out
from util import iter_tools as it
import util
from util import bde
from util.util_config import Config
from util import configs
from wfl.generate_configs import vib


logger = logging.getLogger(__name__)


def fit(no_cycles,
        first_train_fname='train.xyz', e_sigma=0.0005, f_sigma=0.02,
        gap_descriptor_filename=None,
        smiles_csv=None, num_smiles_opt=None, 
        opt_starts_fname=None,  num_nm_displacements_per_temp=None,
        num_nm_temps=None, energy_filter_threshold=0.05,
        max_force_energy_filter_threshold=0.5):
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
    gap_descriptor_filename: str, default None
        .yml with gap descriptors
    smiles_csv: str, default None
        CSV of idx, name, smiles to generate structures from
    num_smiles_opt: int, default None
        for smiles, how many to generate-optimise structures
    opt_starts_fname: str, default=None
        xyz with structures to optimise every cycle
    num_nm_displacements_per_temp: int, default=None
        how many normal modes to sample from GAP-optimised structures for each
        temperature
    num_nm_temps: int, default=None
        how many temperatures to draw nm samples from
    energy_filter_threshold: float, default 0.05
        structures with total energy error above will be included in next training set
    max_force_energy_filter_threshold: float, default=0.5
        structures with largest force component error above this will be
        included in the next training set

    """

    assert smiles_csv is None or opt_starts_fname is None

    if smiles_csv is not None:
        logger.info(f'Optimising structures from {smiles_csv}, {num_smiles_opt} times')
    elif opt_starts_fname is not None:
        logger.info(f'Optimising structures from {opt_starts_fname}')
    else:
        raise RuntimeError('Need either smiles csv or opt strart file for optimisations every cycle')

    bde_root = '/home/eg475/data/bde_files'


    cfg = Config.load()
    gap_fit_path =  cfg['programs']['gap_fit']
    scratch_dir = cfg['other_paths']['tmp']
    glue_fname = cfg['gap_files']['glue_fname']


    required_dirs = ['gaps', 'xyzs', 'xyzs/opt_trajs']
    for dir_name in required_dirs:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    calculator = None
    
    nm_temp = 300 

    ##########################
    ## Orca params
    ########################
    
    default_kw = Config.from_yaml(os.path.join(cfg['util_root'], 'default_kwargs.yml'))

    output_prefix = 'dft_'
    orca_kwargs = default_kw['orca']


    logger.info(f'orca_kwargs: {orca_kwargs}')

    ########################################################
    ## set up GAP
    ######################################################

    descriptors = Config.from_yaml(gap_descriptor_filename)['descriptors']

    default_sigma = [e_sigma, f_sigma, 0.0, 0.0]
    config_type_sigma = {'isolated_atom': [0.0001, 0.0, 0.0, 0.0]}


    ########################################################
    ## set up first dset
    ########################################################

    dset_0_fname = 'xyzs/train_0.xyz'
    if not os.path.exists(dset_0_fname):
        logger.info('generating 0-th dset')
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
        logger.info('fitting 0-th gap')
        gap_command = ugap.make_gap_command(gap_filename=gap_fname, training_filename=dset_0_fname,
                                            descriptors_dict=descriptors, default_sigma=default_sigma,
                                            gap_fit_path=gap_fit_path, output_filename='gaps/out_0.txt',
                                            glue_fname=glue_fname, config_type_sigma=config_type_sigma)

        logger.info(f'gap 0 command:\n{gap_command}')

        orig_omp_n = os.environ.get('OMP_NUM_THREADS', None)
        if 'GAP_FIT_OMP_NUM_THREADS' in os.environ:
            os.environ['OMP_NUM_THREADS'] = os.environ['GAP_FIT_OMP_NUM_THREADS']

        stdout, stderr = util.shell_stdouterr(gap_command)

        if orig_omp_n is not None:
            os.environ['OMP_NUM_THREADS'] = str(orig_omp_n)

        logger.info(f'stdout: {stdout}')
        logger.info(f'stderr: {stderr}')


    ###############################
    ## iterations
    ###############################


    for cycle_idx in range(1, no_cycles + 1):

        logger.info(f'Cycle {cycle_idx}' + '-' * 20)

        #################
        # define filenames
        #######################

        if smiles_csv is not None:
            opt_starts_fname = f'xyzs/non_opt_mols_rads_{cycle_idx}.xyz'

        nm_ref_fname = None
        nm_sample_fname = None
        if num_nm_displacements_per_temp is not None:
            nm_ref_fname = f'xyzs/normal_modes_reference_{cycle_idx}.xyz'
            nm_sample_fname = f'xyzs/normal_modes_sample_{cycle_idx}.xyz'
            nm_sample_fname_for_test = f'xyzs/normal_modes_test_sample_{cycle_idx}.xyz'
            nm_sample_fname_for_test_w_dft = f'xyzs/normal_modes_test_sample_dft_{cycle_idx}.xyz'

        opt_fname = f'xyzs/opt_mols_rads_{cycle_idx}.xyz'
        opt_fname_w_dft = f'xyzs/opt_mols_rads_dft_{cycle_idx}.xyz'
        opt_fname_w_dft_and_gap = f'xyzs/opt_mols_rads_dft_gap_{cycle_idx}.xyz'
        structures_to_derive_normal_modes = f'xyzs/opt_mols_for_nms_{cycle_idx}.xyz'

        extra_data_with_dft = f'xyzs/all_extra_data_w_dft_{cycle_idx}.xyz'
        extra_data_with_dft_and_gap = f'xyzs/all_extra_data_w_dft_and_gap_{cycle_idx}.xyz'
        additional_data = f'xyzs/data_to_add_{cycle_idx}.xyz'
        opt_traj_fname = f'xyzs/opt_trajs/opt_{cycle_idx}.xyz'
        train_set_fname = f'xyzs/train_{cycle_idx}.xyz'
        opt_logfile = f'xyzs/opt_trajs/opt_log_{cycle_idx}.txt'
        bad_geometries_file = f'xyzs/filtered_out_geometries_{cycle_idx}.xyz'


        calculator = (Potential, [], {'param_filename':gap_fname})
        

        if not os.path.exists(train_set_fname):
            
            # generate structures for optimisation
            if not os.path.exists(opt_starts_fname):
                logger.info(f'making structures to optimise at {opt_starts_fname}')
                
                it.make_structures(smiles_csv, iter_no=cycle_idx,
                                   num_smi_repeat=num_smiles_opt,
                                   output_fname=opt_starts_fname)

            raise RuntimeError("just stopping it here")

            # optimise
            if not os.path.exists(opt_fname):
                logger.info(f'gap-optimising {opt_starts_fname} to {opt_fname}')

                opt_inputs = ConfigSet(input_files=opt_starts_fname)
                opt_outputs = ConfigSet_out(output_files=opt_traj_fname)
                opt_wdir='xyzs'

                opt_atoms = ugap.optimise(inputs=opt_inputs,
                                          outputs=opt_outputs,
                                          calculator=calculator,
                                          wdir=opt_wdir)

                write(opt_fname, opt_atoms)


            # evaluate with dft
            if not os.path.exists(opt_fname_w_dft):
                logger.info(f'Calculating dft energies')
                dft_evaled_opt_mols_rads = ConfigSet_out(output_files=opt_fname_w_dft)
                inputs = ConfigSet(input_files=opt_fname)
                dft_evaled_opt_mols_rads = orca.evaluate(inputs=inputs,
                                                         outputs=dft_evaled_opt_mols_rads,
                                                         orca_kwargs=orca_kwargs, output_prefix='dft_',
                                                         keep_files='default', base_rundir=f'orca_outputs_{cycle_idx}'
                                                         )

            # evaluate optimised with gap
            if not os.path.exists(opt_fname_w_dft_and_gap):
                logger.info('Reevaluating extra data with gap')
                inputs = ConfigSet(input_files=opt_fname_w_dft)
                outputs = ConfigSet_out(output_files=opt_fname_w_dft_and_gap)

                no_cores = int(os.environ['WFL_AUTOPARA_NPOOL'])
                no_compounds = len(read(opt_fname_w_dft, ':'))

                chunksize = int(no_compounds/no_cores) + 1

                generic.run(inputs=inputs, outputs=outputs, calculator=calculator,
                            properties=['energy', 'forces'], chunksize=chunksize)

                atoms = read(opt_fname_w_dft_and_gap, ':')
                for at in atoms:
                    at.info[f'gap_{cycle_idx-1}_energy'] = at.info['Potential_energy']
                    at.arrays[f'gap_{cycle_idx-1}_forces'] = at.arrays['Potential_forces']

                write(opt_fname_w_dft_and_gap, atoms)

            # filter by energy
            if not os.path.exists(structures_to_derive_normal_modes):
                atoms = read(opt_fname_w_dft_and_gap, ':')
                atoms = configs.filter_insane_geometries(atoms, mult=1,
                                                         bad_structures_fname=bad_geometries_file)
                atoms = it.filter_by_error(atoms, gap_prefix=f'gap_{cycle_idx-1}_',
                                            e_threshold=energy_filter_threshold,
                                           f_threshold=max_force_energy_filter_threshold)
                write(structures_to_derive_normal_modes, atoms)


            if num_nm_displacements_per_temp is not None:

                # generate normal mode reference
                if not os.path.exists(nm_ref_fname):
                    inputs = ConfigSet(input_files=structures_to_derive_normal_modes)
                    outputs = ConfigSet_out(output_files=nm_ref_fname)
                    vib.generate_normal_modes_parallel_atoms(inputs=inputs,
                                         outputs=outputs, 
                                         calculator=calculator,
                                         prop_prefix='gap_')


                # sample normal modes
                if not os.path.exists(nm_sample_fname):
                    nm_temperatures = np.random.randint(1, 500, num_nm_temps)
                    sampled_configs_train = []
                    sampled_configs_test = []
                    for nm_temp in nm_temperatures:
                        inputs = ConfigSet(input_files=nm_ref_fname)
                        outputs = ConfigSet_out()
                        info_to_keep = ['config_type', 'iter_no', 'minim_n_steps']


                        configs.sample_downweighted_normal_modes(inputs=inputs,
                                                outputs=outputs,
                                                temp=nm_temp,
                                                sample_size=num_nm_displacements_per_temp * 2,
                                                info_to_keep=info_to_keep,
                                                prop_prefix='gap_')
                        sampled_configs_train += outputs.output_configs[0::2]
                        sampled_configs_test += outputs.output_configs[1::2]



                    for ats, fname in zip([sampled_configs_train, sampled_configs_test],
                                          [nm_sample_fname, nm_sample_fname_for_test ]):
                        for at in ats:
                            at.cell = [40, 40, 40]
                            at.info['iter_no'] = cycle_idx
                        write(fname, ats)

                # re-define file to be re-calculated with dft
                file_for_dft = nm_sample_fname


            # training set dft
            if not os.path.exists(extra_data_with_dft):
                logger.info(f'Calculating train set dft energies')
                dft_evaled_opt_mols_rads = ConfigSet_out(output_files=extra_data_with_dft)
                inputs = ConfigSet(input_files=file_for_dft)
                dft_evaled_opt_mols_rads = orca.evaluate(inputs=inputs,
                                                         outputs=dft_evaled_opt_mols_rads,
                                                         orca_kwargs=orca_kwargs, output_prefix='dft_',
                                                         keep_files='default', base_rundir=f'orca_outputs_{cycle_idx}'
                                                         )

            # test set dft
            # if not os.path.exists(nm_sample_fname_for_test_w_dft):
            #     logger.info(f'Calculating test set dft energies')
            #     dft_evaled_opt_mols_rads = ConfigSet_out(output_files=nm_sample_fname_for_test_w_dft)
            #     inputs = ConfigSet(input_files=file_for_dft)
            #     dft_evaled_opt_mols_rads = orca.evaluate(inputs=inputs,
            #                                              outputs=dft_evaled_opt_mols_rads,
            #                                              orca_kwargs=orca_kwargs,
            #                                              output_prefix='dft_',
            #                                              keep_files='default',
            #                                              base_rundir=f'orca_outputs_{cycle_idx}'
            #                                              )


            if not os.path.exists(extra_data_with_dft_and_gap):
                logger.info('Reevaluating extra data with gap')
                inputs = ConfigSet(input_files=extra_data_with_dft)
                outputs = ConfigSet_out(output_files=extra_data_with_dft_and_gap)

                no_cores = int(os.environ['WFL_AUTOPARA_NPOOL'])
                no_compounds = len(read(extra_data_with_dft, ':'))

                chunksize = int(no_compounds/no_cores) + 1

                generic.run(inputs=inputs, outputs=outputs, calculator=calculator,
                            properties=['energy', 'forces'], chunksize=chunksize)

                atoms = read(extra_data_with_dft_and_gap, ':')
                for at in atoms:
                    at.info[f'gap_{cycle_idx-1}_energy'] = at.info['Potential_energy']
                    at.arrays[f'gap_{cycle_idx-1}_forces'] = at.arrays['Potential_forces']

                write(extra_data_with_dft_and_gap, atoms)

            if not os.path.exists(additional_data):
                atoms = read(extra_data_with_dft_and_gap, ':')
                # atoms = it.filter_by_error(atoms, gap_prefix=f'gap_{cycle_idx-1}_',
                #                            f_threshold=None)
                write(additional_data, atoms)



            previous_dset = read(f'xyzs/train_{cycle_idx - 1}.xyz', ':')
            new_atoms = read(additional_data, ':')
            write(train_set_fname, previous_dset + new_atoms, write_results=False)
        else:
            logger.info(f'Found {train_set_fname}, not re-generating')


        # refit gap
        gap_fname = f'gaps/gap_{cycle_idx}.xml'
        out_fname = f'gaps/out_{cycle_idx}.xml'

        if not os.path.exists(gap_fname):
            logger.info('fitting gap')

            gap_command = ugap.make_gap_command(gap_filename=gap_fname, training_filename=train_set_fname,
                                                descriptors_dict=descriptors, default_sigma=default_sigma,
                                                gap_fit_path=gap_fit_path, output_filename=out_fname,
                                                glue_fname=glue_fname, config_type_sigma=config_type_sigma)
            logger.info(f'gap {cycle_idx} command: {gap_command}')

            orig_omp_n = os.environ.get('OMP_NUM_THREADS', None)
            if 'GAP_FIT_OMP_NUM_THREADS' in os.environ:
                os.environ['OMP_NUM_THREADS'] = os.environ[
                    'GAP_FIT_OMP_NUM_THREADS']

            stdout, stderr = util.shell_stdouterr(gap_command)

            if orig_omp_n is not None:
                os.environ['OMP_NUM_THREADS'] = str(orig_omp_n)

            logger.info(f'stdout: {stdout}')
            logger.info(f'stderr: {stderr}')


    ##################
    # Analyse stuff
    #########################

    # learning curves: rmse & bdes + final GAP's energy/bde by structure

    logger.info('Finised iterations')
