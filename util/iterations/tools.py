import os
import logging

import pandas as pd
import numpy as np

import ace
from quippy.potentials import Potential

from wfl.calculators import generic
from wfl.configset import ConfigSet_in, ConfigSet_out

import util
from util import radicals
from util import configs

logger = logging.getLogger(__name__)

def prepare_remoteinfo(gap_fname):
    output_filename = os.environ["WFL_AUTOPARA_REMOTEINFO"]
    template = os.environ["WFL_AUTOPARA_REMOTEINFO_TEMPLATE"]
    with open(template, 'r') as f:
        file = f.read()
    file = file.replace("<gap_filename>", gap_fname)
    with open(output_filename, 'w') as f:
        f.write(file)


def make_dirs(dir_names):
    for dir_name in dir_names:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

def prepare_0th_dataset(ci, co):

    if co.is_done():
        logger.info("initial dataset is prepared")
        return co.to_ConfigSet_in()

    
    logger.info('preparing initial dataset')

    for at in ci:
        if 'iter_no' not in at.info.keys():
            at.info['iter_no'] = '0'
        if at.cell is None:
            at.cell = [50, 50, 50]

        co.write(at)
    co.end_write()
    return co.to_ConfigSet_in()

def make_structures(smiles_csv, iter_no, num_smi_repeat, outputs,
                    num_rads_per_mol, smiles_col='smiles',
                    name_col='zinc_id'):

    atoms_out = []

    logger.info(f"writing to {outputs.output_files}")

    # generate molecules
    df = pd.read_csv(smiles_csv, delim_whitespace=True)
    for smi, name in zip(df[smiles_col], df[name_col]):
        for _ in range(num_smi_repeat):
            try:
                mol_and_rads = radicals.rad_conformers_from_smi(smi=smi,
                                compound=name, num_radicals=num_rads_per_mol)
            except RuntimeError:
                continue
            atoms_out += mol_and_rads

    logger.info(f'length of output atoms: {len(atoms_out)}')

    for at in atoms_out:
        at.cell = [50, 50, 50]
        if iter_no is not None:
            at.info['iter_no'] = iter_no
        outputs.write(at)

    outputs.end_write()
    return outputs.to_ConfigSet_in()

def filter_configs_by_geometry(inputs, bad_structures_fname, outputs):
    atoms = configs.filter_insane_geometries(inputs, mult=1,
                             bad_structures_fname=bad_structures_fname)
    outputs.write(atoms)
    outputs.end_write()
    return outputs.to_ConfigSet_in()


def filter_configs(inputs, outputs,
                   gap_prefix,
                   e_threshold,
                   f_threshold,
                   outputs_accurate_structures,
                   dft_prefix='dft_'):

    for at in inputs:
        e_error = at.info[f'{gap_prefix}energy'] - \
                  at.info[f'{dft_prefix}energy']
        if np.abs(e_error) > e_threshold:
            outputs.write(at)
            continue

        if f_threshold is not None:
            f_error = at.arrays[f'{gap_prefix}forces'] - \
                      at.arrays[f'{dft_prefix}forces']
            if np.max(np.abs(f_error.flatten())) > f_threshold:
                outputs.write(at)
        else:
            outputs_accurate_structures.write(at)

    outputs_accurate_structures.end_write()
    outputs.end_write()
    return outputs.to_ConfigSet_in()



def do_gap_fit(fit_dir, idx, ref_type, train_set_fname, 
               fit_params_base, gap_fit_path):

        fit_fname = fit_dir / f'gap_{idx}.xml' 
        gap_out_fname = fit_dir / f'gap_{idx}.out'

        if not fit_fname.exists():
            logger.info(f'fitting gap {fit_fname} on {train_set_fname}')

            gap_params = deepcopy(fit_params_base)
            gap_params['gap_file'] = fit_fname

            fit_inputs = ConfigSet_in(input_files=train_set_fname)

            wfl.fit.gap_simple.run_gap_fit(fitting_configs=fit_inputs,
                                           fitting_dict=gap_params,
                                           stdout_file=gap_out_fname,
                                           gap_fit_exec=gap_fit_path)

        full_fit_fname = str(fit_fname.resolve())
        if ref_type == 'dft':
            return (Potential, [], {'param_filename':full_fit_fname})
        elif ref_type == 'dft-xtb2':
            return (xtb2_plus_gap, [], {'gap_filename': full_fit_fname})


def fix_fit_params(fit_base_params, fit_to_prop_prefix):
    # for gap only for now

    if "energy_parameter_name" in fit_base_params and \
        fit_params_base["energy_parameter_name"] !=  f'{fit_to_prop_prefix}energy':

        logger.warn(f'Overwriting {fit_params_base["energy_parameter_name"]} found '
                    f'in fit_params_base with "{fit_to_prop_prefix}energy"')
        fit_params_base["energy_parameter_name"] =  f'{fit_to_prop_prefix}energy'

    if "force_parameter_name" in fit_params_base and \
            fit_params_base["force_parameter_name"] != f'{fit_to_prop_prefix}forces':

        logger.warn(f'Overwriting {fit_params_base["force_parameter_name"]} found '
                    f'in fit_params_base with "{fit_to_prop_prefix}forces"')
        fit_params_base["force_parameter_name"] = f'{fit_to_prop_prefix}forces'

        return fit_params_base



def do_ace_fit(fit_dir, idx, ref_type, train_set_fname, 
                fit_params_base, fit_to_prop_prefix, ace_fit_exec):

    assert ref_type == 'dft'

    # TODO:
    # * set correct labels to fit to/check if they're set properly
    # * modify fit_params_base

    if "NSLOTS" in os.environ.keys():
        ncores = os.environ["NSLOTS"] 
        os.environ["ACE_FIT_JULIA_THREADS"] = str(ncores)
        os.environ["ACE_FIT_BLAS_THREADS"] = str(ncores)

    fit_inputs = ConfigSet_in(input_files=train_set_fname)

    ace_name = f'ace_{idx}'
    ace_fname = fit_dir / (ace_name + '.json')

    ace_file_base = wfl.fit.ace.fit(fitting_configs=fit_inputs, 
                                    ACE_name=ace_name,
                                    params=fit_params_base, 
                                    ref_property_prefix=fit_to_prop_prefix, 
                                    skip_if_present=True, 
                                    run_dir=fit_dir, 
                                    formats=['.json'], 
                                    ace_fit_exec=ace_fit_exec, 
                                    verbose=True, 
                                    remote_info=None, 
                                    wait_for_results=True)

    assert ace_file_base + '.json' == ace_fname

    return (ace.ACECalculator, [], {"jsonpath":ace_fname})
    


        