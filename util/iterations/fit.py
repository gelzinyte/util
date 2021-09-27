import os
import shutil
from copy import deepcopy
import yaml
import logging
import numpy as np
from pathlib import Path

from ase.io import read, write

try:
    from quippy.potential import Potential
except ModuleNotFoundError:
    pass

from wfl.calculators import orca, generic
from wfl.configset import ConfigSet_in, ConfigSet_out
import wfl.fit.gap_simple
from wfl.generate_configs import vib

from util import opt
from util import normal_modes as nm
from util.util_config import Config
from util.iterations import tools as it
from util import configs

logger = logging.getLogger(__name__)


def fit(no_cycles,
        first_train_fname='train.xyz',
        gap_param_filename=None,
        smiles_csv=None, num_smiles_opt=None,
        num_nm_displacements_per_temp=None,
        num_nm_temps=None, energy_filter_threshold=0.05,
        max_force_filter_threshold=0.1):
    """ iteratively fits and optimises stuff

    Parameters
    ---------

    no_cycles: int
        number of gap-fit and optimise cycles to do
    first_train_fname: str, default='train.xyz'
        fname for fitting the first GAP
    gap_param_filename: str, default None
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
        structures with total energy error above will be included in next
        training set
    max_force_filter_threshold: float, default=0.1
        structures with largest force component error above this will be
        included in the next training set

    """


    logger.info(f'Optimising structures from {smiles_csv}, {num_smiles_opt} '
                f'times. Displacing each at {num_nm_temps} different'
                f'temperatures between 1 and 800 K '
                f'{num_nm_displacements_per_temp} times each. GAP '
                f'parameters from {gap_param_filename}.')



    cfg = Config.load()
    gap_fit_path =  cfg['gap_fit_path']
    scratch_dir = cfg['scratch_path']
    glue_fname = cfg['glue_fname']

    logger.info(f'Using:\n\tgap_fit {gap_fit_path}\n\tscratch_dir '
                f'{scratch_dir}\n\tglue {glue_fname}')

    it.make_dirs(['gaps', 'xyzs'])

    # will need to add `atoms_filename` and `gap_file` when fitting
    with open(gap_param_filename) as yaml_file:
        gap_fit_base_params = yaml.safe_load(yaml_file)

    # setup orca parameters
    default_kw = Config.from_yaml(os.path.join(cfg['util_root'],
                                               'default_kwargs.yml'))
    output_prefix = 'dft_'
    orca_kwargs = default_kw['orca']
    logger.info(f'orca_kwargs: {orca_kwargs}')


    # prepare 0th dataset
    initial_train_fname = 'xyzs/train_for_gap_0.xyz'
    if not os.path.exists(initial_train_fname):
        dset = read(first_train_fname, ':')
        for at in dset:
            if 'iter_no' not in at.info.keys():
                at.info['iter_no'] = '0'
        write(initial_train_fname, dset, write_results=False)


    for cycle_idx in range(0, no_cycles+1):

        train_set_fname = f'xyzs/{cycle_idx-1}_train_for_gap' \
                          f'_{cycle_idx}.xyz'
        if cycle_idx == 0:
            train_set_fname = initial_train_fname

        opt_starts_fname = f'xyzs/{cycle_idx}.2_non_opt_mols_rads.xyz'
        opt_fname = f'xyzs/{cycle_idx}.3.0_gap_opt_mols_rads.xyz'
        opt_filtered_fname =  f'xyzs/' \
                          f'{cycle_idx}.3.1.0_gap_opt_mols_rads_filtered.xyz'
        opt_fname_with_gap = f'xyzs/' \
                  f'{cycle_idx}.3.2_gap_opt_mols_rads.filtered.gap.xyz'
        opt_fname_w_dft = f'xyzs/' \
                f'{cycle_idx}.3.3_gap_opt_mols_rads.filtered.gap.dft.xyz'

        configs_with_large_errors = f'xyzs/' \
                            f'{cycle_idx}.4.1_opt_mols_w_large_errors.xyz'
        bad_structures_fname = f'xyzs/' \
                              f'{cycle_idx}.3.1.1_filtered_out_geometries.xyz'

        nm_ref_fname = f'xyzs/{cycle_idx}.5_normal_modes_reference.xyz'

        nm_sample_fname_for_train = f'xyzs/' \
                        f'{cycle_idx}.6.1_normal_modes_train_sample.xyz'
        nm_sample_fname_for_test = f'xyzs/' \
                     f'{cycle_idx}.6.2_normal_modes_test_sample.xyz'


        gap_fname = f'gaps/gap_{cycle_idx}.xml'
        gap_out_fname = f'gaps/gap_{cycle_idx}.out'
        gap_prop_prefix = f'gap{cycle_idx}_'

        if 'WFL_AUTOPARA_REMOTEINFO_TEMPLATE' in os.environ:
            it.prepare_remoteinfo(gap_fname)


        # Check for the final training set from this iteration and skip if
        # found.
        next_training_set_fname =f'xyzs/{cycle_idx}_train_for_gap' \
                                      f'_{cycle_idx+1}.xyz'

        if os.path.exists(next_training_set_fname):
            logger.info(f'Found {next_training_set_fname}, skipping iteration '
                        f'{cycle_idx}')
            continue
        

        # 1. fit GAP
        if not os.path.exists(gap_fname):
            logger.info(f'fitting gap {gap_fname} on {train_set_fname}')
            gap_params = deepcopy(gap_fit_base_params)
            gap_params['gap_file'] = gap_fname
            wfl.fit.gap_simple.run_gap_fit(fitting_configs=train_set_fname,
                                             fitting_dict=gap_params,
                                           stdout_file=gap_out_fname,
                                           gap_fit_exec=gap_fit_path)

        print(f'Full gap name: {Path(gap_fname).resolve()}')
        calculator = (Potential, [],
                      {'param_filename':str(Path(gap_fname).resolve())})

        # calculator = (Potential, [], {'param_filename':gap_fname})

        # 2. generate structures for optimisation
        logger.info('generating structures to optimise')
        outputs = ConfigSet_out(output_files=opt_starts_fname,
                                force=True, all_or_none=True,
                                verbose=False)
        inputs = it.make_structures(smiles_csv, iter_no=cycle_idx,
                           num_smi_repeat=num_smiles_opt,
                           outputs=outputs)


        # 3 optimise structures with current GAP and re-evaluate them
        # with GAP and DFT
        logger.info("optimising structures with GAP")
        outputs = ConfigSet_out(output_files=opt_fname, force=True,
                                all_or_none=True)
        inputs = opt.optimise(inputs=inputs, outputs=outputs,
                               calculator=calculator)


        # filter out insane geometries
        outputs = ConfigSet_out(output_files=opt_filtered_fname,
                                force=True, all_or_none=True)
        inputs = it.filter_configs_by_geometry(inputs=inputs,
                          bad_structures_fname=bad_structures_fname,
                                outputs=outputs)


        # evaluate GAP
        logger.info("evaluating gap on optimised structures")
        outputs = ConfigSet_out(output_files=opt_fname_with_gap,
                                force=True, all_or_none=True)
        inputs = generic.run(inputs=inputs, outputs=outputs,
                             calculator=calculator,
                             properties=['energy', 'forces'],
                             output_prefix=gap_prop_prefix, chunksize=50)

        # evaluate DFT
        logger.info('evaluatig dft on optimised structures')
        outputs = ConfigSet_out(output_files=opt_fname_w_dft, force=True,
                                all_or_none=True)
        inputs = orca.evaluate(inputs=inputs, outputs=outputs,
                           orca_kwargs=orca_kwargs,
                           output_prefix=output_prefix,
                           keep_files=False,
                           base_rundir=f'xyzs/wdir/'
                                       f'i{cycle_idx}_orca_outputs')


        # 4 filter by energy and force error
        outputs = ConfigSet_out(output_files=configs_with_large_errors,
                                force=True, all_or_none=True)
        inputs = it.filter_configs(inputs=inputs, outputs=outputs,
                             gap_prefix=gap_prop_prefix,
                             e_threshold=energy_filter_threshold,
                             f_threshold=max_force_filter_threshold)
        logger.info(f'# opt structures: {len(read(opt_fname, ":"))}; # '
                    f'of selected structures: '
                    f'{len(read(configs_with_large_errors, ":"))}')


        # 5. derive normal modes
        outputs = ConfigSet_out(output_files=nm_ref_fname,
                                force=True, all_or_none=True)
        vib.generate_normal_modes_parallel_atoms(inputs=inputs,
                                             outputs=outputs,
                                             calculator=calculator,
                                             prop_prefix=gap_prop_prefix)


        # 6. sample normal modes and get DFT energies and forces
        outputs_train = ConfigSet_out(output_files=nm_sample_fname_for_train,
                                      force=True, all_or_none=True)
        outputs_test = ConfigSet_out(output_files=nm_sample_fname_for_test,
                                     force=True, all_or_none=True)

        info_to_keep = ['config_type', 'iter_no', 'minim_n_steps',
                        'compound_type', 'mol_or_rad', 'smiles']
        nm_temperatures = np.random.randint(1, 800, num_nm_temps)
        logger.info(f'sampling {nm_ref_fname} at temperatures '
                    f'{nm_temperatures} K')
        for temp in nm_temperatures:
            inputs = ConfigSet_in(input_files=nm_ref_fname)
            outputs = ConfigSet_out()
            nm.sample_downweighted_normal_modes(inputs=inputs,
                            outputs=outputs, temp=temp,
                            sample_size=num_nm_displacements_per_temp*2,
                            prop_prefix=gap_prop_prefix,
                            info_to_keep=info_to_keep)

            for idx, at in enumerate(outputs.to_ConfigSet_in()):
                at.cell = [50, 50, 50]
                at.info['normal_modes_temp'] = f'{temp:.2f}'
                if idx % 2 == 0:
                    outputs_train.write(at)
                else:
                    outputs_test.write(at)

        outputs_train.end_write()
        outputs_test.end_write()

        # evaluate DFT
        outputs = ConfigSet_out(output_files=nm_sample_fname_for_train,
                                force=True, all_or_none=True)
        orca.evaluate(inputs=outputs_train.to_ConfigSet_in(),
                           outputs=outputs,
                           orca_kwargs=orca_kwargs,
                           output_prefix=output_prefix,
                           keep_files=False,
                           base_rundir=f'xyzs/wdir/'
                                       f'i{cycle_idx}_orca_outputs')


        # 7. Combine data
        if not os.path.exists(next_training_set_fname):
            previous_dataset = read(train_set_fname, ':')
            additional_data = read(nm_sample_fname_for_train, ':')
            write(next_training_set_fname, previous_dataset + additional_data)


    logger.info('Finished iterations')


