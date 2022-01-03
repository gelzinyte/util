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
from util.calculators import xtb2_plus_gap

logger = logging.getLogger(__name__)


def fit(num_cycles,
        base_train_fname='train.xyz',
        fit_param_fname=None,
        additional_smiles_csv=None,
        md_temp=500, energy_filter_threshold=0.05):
    """ iteratively fits potetnials

    Parameters
    ---------

    num_cycles: int
        number of fit-optimise cycles to do
    base_train_fname: str, default='train.xyz'
        fname for fitting the first IP
    fit_param_fname: str, default None
        template ace_fit.jl
    additional_smiles_csv: str, default None
        CSV of id and smiles to generate structures from
    md_temp: float, default 500
        temperature to run MD at
    energy_filter_threshold: float, default 0.05
        structures with total energy error above will be sampled and included in next
        training set
    ref_type: str, default "dft"
        "dft" or "dft-xtb2" for the type of reference energies/forces to
        fit to.
    """

    assert ref_type in ['dft', 'dft-xtb2']


    logger.info(f'Optimising structures from {additional_smiles_csv}. '
                f'Displacing each at {num_nm_temps} different'
                f'temperatures between 1 and 800 K '
                f'{num_nm_displacements_per_temp} times each. GAP '
                f'parameters from {fit_param_fname}. Fitting to '
                f'{ref_type} energies and forces.')



    cfg = Config.load()
    scratch_dir = cfg['scratch_path']

    logger.info(f'Using:scratch_dir: {scratch_dir}')

    it.make_dirs(['fits', 'xyzs'])

    # setup orca parameters
    default_kw = Config.from_yaml(os.path.join(cfg['util_root'],
                                               'default_kwargs.yml'))

    dft_prop_prefix = 'dft_'
    orca_kwargs = default_kw['orca']
    logger.info(f'orca_kwargs: {orca_kwargs}')
    logger.info(f'orca in environment: {shutil.which("orca")}')

    if ref_type == 'dft':
        fit_to_prop_prefix = dft_prop_prefix
        calc_predicted_prop_prefix = 'ace_'
        xtb2_prop_prefix=None
    if ref_type == 'dft-xtb2':
        fit_to_prop_prefix = 'dft_minus_xtb2_'
        calc_predicted_prop_prefix = 'gap_plus_xtb2_'
        xtb2_prop_prefix = 'xtb2_'


    # prepare 0th dataset
    initial_train_fname = 'xyzs/train_for_fit_0.xyz'
    if not os.path.isfile(initial_train_fname):
        logger.info('preparing initial dataset')
        ci = ConfigSet_in(input_files=base_train_fname)
        co = ConfigSet_out(output_files=initial_train_fname,
                           force=True, all_or_none=True)
        fit_inputs = it.prepare_0th_dataset(ci,cox)


    for cycle_idx in range(0, num_cycles+1):

        train_set_fname = f'xyzs/{cycle_idx-1}.train_for_fit{cycle_idx}.xyz'
        if cycle_idx == 0:
            train_set_fname = initial_train_fname
        # 
        # opt_starts_fname = f'xyzs/{cycle_idx}.2_non_opt_mols_rads.xyz'
        # opt_fname = f'xyzs/{cycle_idx}.3.0_gap_opt_mols_rads.xyz'
        # 
        # opt_filtered_fname =  f'xyzs/{cycle_idx}.3.1.0_gap_opt_mols_rads_filtered.xyz'
        # bad_structures_fname = f'xyzs/{cycle_idx}.3.1.1_filtered_out_geometries.xyz'
        # opt_fname_with_gap = f'xyzs/{cycle_idx}.3.2_gap_opt_mols_rads.filtered.gap.xyz'
        # opt_fname_w_dft = f'xyzs/{cycle_idx}.3.3_gap_opt_mols_rads.filtered.gap.dft.xyz'
        # 
        # configs_with_large_errors = f'xyzs/{cycle_idx}.4.1_opt_mols_w_large_errors.xyz'
        # energy_force_accurate_fname = f'xyzs/{cycle_idx}.4.2_opt_mols_w_small_errors.xyz'
        # 
        # nm_ref_fname = f'xyzs/{cycle_idx}.5_normal_modes_reference.xyz'
        # 
        # nm_sample_fname_for_train = f'xyzs/{cycle_idx}.6.1_normal_modes_train_sample.xyz'
        # nm_sample_fname_for_test = f'xyzs/{cycle_idx}.6.2_normal_modes_test_sample.xyz'
        # 
        # nm_sample_fname_for_train_with_dft = f'xyzs/{cycle_idx}.7_normal_modes_train_sample.dft.xyz'


        fit_fname = f'gaps/gap_{cycle_idx}.xml'
        gap_out_fname = f'gaps/gap_{cycle_idx}.out'
        if ref_type == 'dft':
            calc_predicted_prop_prefix = f'gap{cycle_idx}_'
        elif ref_type == 'dft-xtb2':
            calc_predicted_prop_prefix = f'gap{cycle_idx}_plus_xtb2_'


        # Check for the final training set from this iteration and skip if
        # found.
        next_training_set_fname =f'xyzs/{cycle_idx}_train_for_gap' \
                                      f'_{cycle_idx+1}.xyz'

        if os.path.exists(next_training_set_fname):
            logger.info(f'Found {next_training_set_fname}, skipping iteration '
                        f'{cycle_idx}')
            continue
        

        # 1. fit GAP
        if not os.path.exists(fit_fname):
            logger.info(f'fitting gap {fit_fname} on {train_set_fname}')
            gap_params = deepcopy(gap_fit_base_params)
            gap_params['gap_file'] = fit_fname
            if cycle_idx > 0:
                fit_inputs = ConfigSet_in(input_files=train_set_fname)

            if "energy_parameter_name" in gap_params and \
                gap_params["energy_parameter_name"] != \
                    f'{fit_to_prop_prefix}energy':
                logger.warn(f'Overwriting '
                            f'{gap_params["energy_parameter_name"]} found '
                            f'in gap_params with "{fit_to_prop_prefix}energy"')
                gap_params["energy_parameter_name"] =  \
                                              f'{fit_to_prop_prefix}energy'
            if "force_parameter_name" in gap_params and \
                    gap_params["force_parameter_name"] != \
                    f'{fit_to_prop_prefix}forces':
                logger.warn(f'Overwriting '
                            f'{gap_params["force_parameter_name"]} found '
                            f'in gap_params with "'
                            f'{fit_to_prop_prefix}forces"')
                gap_params["force_parameter_name"] = \
                    f'{fit_to_prop_prefix}forces'

            wfl.fit.gap_simple.run_gap_fit(fitting_configs=fit_inputs,
                                             fitting_dict=gap_params,
                                           stdout_file=gap_out_fname,
                                           gap_fit_exec=gap_fit_path)

        full_fit_fname = str(Path(fit_fname).resolve())
        if ref_type == 'dft':
            calculator = (Potential, [],
                         {'param_filename':full_fit_fname})
        elif ref_type == 'dft-xtb2':
            calculator = (xtb2_plus_gap, [],
                          {'gap_filename': full_fit_fname})


        # 2. generate structures for optimisation
        if not os.path.isfile(opt_starts_fname):
            logger.info('generating structures to optimise')
            outputs = ConfigSet_out(output_files=opt_starts_fname,
                                    force=True, all_or_none=True,
                                    verbose=False)
            inputs = it.make_structures(additional_smiles_csv, iter_no=cycle_idx,
                               num_smi_repeat=1,
                               outputs=outputs)
        else:
            inputs = ConfigSet_in(input_files=opt_starts_fname)


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
                             output_prefix=calc_predicted_prop_prefix, chunksize=50)

        # evaluate DFT
        logger.info('evaluatig dft on optimised structures')
        outputs = ConfigSet_out(output_files=opt_fname_w_dft, force=True,
                                all_or_none=True)
        inputs = orca.evaluate(inputs=inputs, outputs=outputs,
                           orca_kwargs=orca_kwargs,
                           output_prefix=dft_prop_prefix,
                           keep_files=False,
                           base_rundir=f'xyzs/wdir/'
                                       f'i{cycle_idx}_orca_outputs')


        # 4 filter by energy and force error
        if not os.path.exists(configs_with_large_errors):
            logger.info('Filtering by energy and force errors')
            outputs = ConfigSet_out(output_files=configs_with_large_errors,
                                    force=True, all_or_none=True)
            outputs_accurate_structures = ConfigSet_out(
                                output_files=energy_force_accurate_fname,
                                force=True, all_or_none=True)
            inputs = it.filter_configs(inputs=inputs, outputs=outputs,
                                 gap_prefix=calc_predicted_prop_prefix,
                                 e_threshold=energy_filter_threshold,
                                 f_threshold=max_force_filter_threshold,
                     outputs_accurate_structures=outputs_accurate_structures)
            logger.info(f'# opt structures: {len(read(opt_fname, ":"))}; # '
                        f'of selected structures: '
                        f'{len(read(configs_with_large_errors, ":"))}')
        else:
            logger.info('found file with structures with high errors')


        # 5. derive normal modes
        logger.info('generating normal modes')
        outputs = ConfigSet_out(output_files=nm_ref_fname,
                                force=True, all_or_none=True)
        vib.generate_normal_modes_parallel_atoms(inputs=inputs,
                                             outputs=outputs,
                                             calculator=calculator,
                                             prop_prefix=calc_predicted_prop_prefix)


        # 6. sample normal modes and get DFT energies and forces
        if not os.path.exists(nm_sample_fname_for_train):
            outputs_train = ConfigSet_out(output_files=nm_sample_fname_for_train,
                                          force=True, all_or_none=True)
            outputs_test = ConfigSet_out(output_files=nm_sample_fname_for_test,
                                         force=True, all_or_none=True)

            info_to_keep = ['config_type', 'iter_no', 'minim_n_steps',
                            'compound_type', 'mol_or_rad', 'smiles']
            nm_temperatures = np.random.randint(1, 800, num_nm_temps)
            logger.info(f'sampling {nm_ref_fname} at temperatures '
                        f'{nm_temperatures} K')
            inputs = ConfigSet_in(input_files=nm_ref_fname)
            for temp in nm_temperatures:
                outputs = ConfigSet_out()
                nm.sample_downweighted_normal_modes(inputs=inputs,
                                outputs=outputs, temp=temp,
                                sample_size=num_nm_displacements_per_temp*2,
                                prop_prefix=calc_predicted_prop_prefix,
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
            inputs = outputs_train.to_ConfigSet_in()
        else:
            inputs = ConfigSet_in(input_files=nm_sample_fname_for_train)

        # evaluate DFT
        if not os.path.exists(nm_sample_fname_for_train_with_dft):
            logger.info('evaluating dft on new training set')
            outputs = ConfigSet_out(
                output_files=nm_sample_fname_for_train_with_dft,
                                    force=True, all_or_none=True)
            orca.evaluate(inputs=inputs,
                           outputs=outputs,
                           orca_kwargs=orca_kwargs,
                           output_prefix=dft_prop_prefix,
                           keep_files=False,
                           base_rundir=f'xyzs/wdir/'
                                       f'i{cycle_idx}_orca_outputs')


        # 7. Combine data
        if not os.path.exists(next_training_set_fname):
            logger.info('combining new dataset')
            previous_dataset = read(train_set_fname, ':')
            additional_data = read(nm_sample_fname_for_train_with_dft, ':')
            write(next_training_set_fname, previous_dataset + additional_data)


    logger.info('Finished iterations')


