import os
import logging
import random
import subprocess
from copy import deepcopy
from pathlib import Path
import pytest
import yaml

import pandas as pd
import numpy as np

from PyPDF4 import PdfFileMerger

from ase.io import read, write

try:
    import ace
except ModuleNotFoundError:
    pass

# from pyjulip import ACE1
from util.calculators import pyjulip_ace


from quippy.potential import Potential

from wfl.configset import ConfigSet, OutputSpec
import wfl.fit.gap.simple
import wfl.fit.ace
from wfl.calculators import generic
from wfl.calculators import orca
from wfl.generate import md
from wfl.autoparallelize import autoparainfo
from wfl.descriptors import quippy

import util
from util import radicals
from util import configs
from util.calculators import xtb2_plus_gap
from util.bde import generate
from util.plot import dataset
from util.plot import rmse_scatter_evaled, multiple_error_files
from util.util_config import Config
from util.md.stopper import BadGeometry
from util.configs import cur

logger = logging.getLogger(__name__)


def get_filenames(no, wdir, ip, train_set_dir, val_fn):   

    # cycle dir
    cd = wdir / f"iteration_{no:02d}"
    cd.mkdir(exist_ok=True)

    # fit dir
    fd = cd / "fit_dir"
    fd.mkdir(exist_ok=True)

    # tests dir
    td = cd / "tests"
    td.mkdir(exist_ok=True)


    if ip == "ace":
        model = 'ace.json'
    elif ip == 'gap':
        moel = 'gap.xml'

    fns = {}

    fns["tests_dir"] = td
    fns["cycle_dir"] = cd
    fns["fit_dir"] = fd

    fns["model"] = {}
    fns["model"]["fname"]= fd / model 
    fns['model']['params'] = fd / f'{ip}_params.yaml'

    fns["val"] = val_fn 

    fns["this_train"] = train_set_dir / f"{no-1:02d}.train_for_{ip}_{no:02d}.xyz" 
    fns["next_train"] = train_set_dir / f"{no:02d}.train_for_{ip}_{no+1:02d}.xyz" 

    fns["smiles"] = cd / "01.extra_smiles.csv"

    fns["md_starts"] = {}
    fns["md_starts"]["all"] = cd / "02.0.rdkit_md_starts.all.xyz"
    fns["md_starts"]["successful"] = cd / "02.1.rdkit_md_starts.successful.xyz"
    fns["md_starts"]["failed"] = cd / "02.2.rdkit_md_starts.failed.xyz"

    fns["md_traj"] = {}
    fns["md_traj"]["all"] = cd / f"03.0.{ip}.md_traj.xyz"
    fns["md_traj"]["failed"] = cd / f"03.1.{ip}.md_traj.failed.xyz"
    fns["md_traj"]["successful"] = {}
    fns["md_traj"]["successful"]["plain"]= cd / f"03.2.0.{ip}.md_traj.successful.xyz"
    fns["md_traj"]["successful"]["soap"] = cd / f"03.2.1.{ip}.md_traj.successful.soap.xyz"
    fns["md_traj"]["successful"]["cur"] = cd / f"03.2.2.{ip}.md_traj.successful.soap.cur.xyz"

    fns["extra"] = {}
    fns["extra"]["train"] = {}
    fns["extra"]["validation"] = {}

    fns["extra"]["validation"]["via_soap"] = cd / f"04.0.{ip}.md_traj.extra_test.soap_cur.xyz"
    fns["extra"]["validation"]["dft"] = cd / f"04.1.{ip}.md_traj.extra_test.soap_cur.dft.xyz"

    fns["extra"]["train"]["via_soap"] = cd / f"05.0.{ip}.md_traj.extra_train.soap_cur.xyz" 
    fns["extra"]["train"]["from_failed"] = cd / f"05.1.{ip}.md_traj.extra_train.from_failed.xyz"
    fns["extra"]["train"]["all_dft"] = cd / f"05.2.{ip}.md_traj.extra_train.dft.xyz"

    fns["tests"] = {}
    fns["tests"]["mlip_on_train"] = td / f"{ip}_on_{fns['this_train'].name}"
    fns["tests"]["mlip_on_val"] = td / f"{ip}_on_{fns['val'].name}"

    fns["orca_wdir"] = cd / "orca_wdir_extra_data"

    return fns


def cleanup(inputs, cycle_idx):
    remove_all_calc_results(inputs)
    outspec = OutputSpec()
    inputs = prepare_dataset(
        cs=inputs, 
        outspec=outspec, 
        cycle_idx=cycle_idx+1,
        arrays_to_delete=["small_soap"])
    return inputs


def organise_md_trajs(all_md_trajs, fns):

    traj_success = OutputSpec(
        output_files=fns["md_traj"]["successful"]["plain"], 
        set_tags={"md_traj_outcome":"success"})
    traj_failed = OutputSpec(
        output_files=fns["md_traj"]["failed"],
        set_tags={"md_traj_outcome":"fail"})
    start_success = OutputSpec(
        output_files=fns["md_starts"]["successful"],
        set_tags={"md_traj_outcome":"success"})
    start_failed = OutputSpec(
        output_files=fns["md_starts"]["failed"],
        set_tags={"md_traj_outcome":"fail"})

    all_os = [traj_success, traj_failed, start_success, start_failed]

    doneness = np.sum([outs.is_done() for outs in all_os])
    if doneness in [2, 4]: # in case of no failure
        logger.info("skipping organising md trajectories, since all outputs are done")
        return
    elif doneness == 0:
        pass
    else:
        raise RuntimeError("Some of the organised trajectories are done, some not")

    trajs = configs.into_dict_of_labels(all_md_trajs, info_label="md_start_hash")

    for traj in trajs.values():

        if traj[-1].info["md_geometry_check"]:
            # successful trajectory
            traj_success.write(traj)
            start_success.write(traj[0])

        else:
            traj_failed.write(traj)
            start_failed.write(traj[0])
    
    for outspec in all_os:
        outspec.end_write()


def sample_with_cur_soap(fns, soap_params, num_cur_environments, cycle_no):

    inputs = ConfigSet(input_files=fns["md_traj"]["successful"]["plain"])

    os_soap = OutputSpec(output_files=fns["md_traj"]["successful"]["soap"])
    os_cur = OutputSpec(output_files=fns["md_traj"]["successful"]["cur"])

    os_extra_train = OutputSpec(output_files=fns["extra"]["train"]["via_soap"])
    os_extra_test = OutputSpec(output_files=fns["extra"]["validation"]["via_soap"])

    doneness = int(os_extra_train.is_done()) + int(os_extra_test.is_done())
    if doneness == 2:
        logger.info("sampling with cur is done, not doing anything")
        return os_extra_test.to_ConfigSet(), os_extra_train.to_ConfigSet()
    elif doneness == 0:
        pass
    else:
        raise RuntimeError("only some of the cur outputs are done. ")


    if not os_soap.is_done():
        logger.info("Calculating SOAP descriptor")
        inputs = quippy.calc(
            inputs=inputs,
            outputs=os_soap,
            descs=soap_params,
            key="small_soap",  # where to store the descriptor
            local=True)  
    else:
        inputs = os_soap.to_ConfigSet()


    logger.info("Sampling with CUR")
    inputs = cur.per_environment(
        inputs=inputs,
        outputs=os_cur,
        num=num_cur_environments,
        at_descs_key="small_soap",
        kernel_exp=4,
        leverage_score_key="cur_leverage_score",
        write_all_configs=False)

    inputs = list(inputs)
    random.shuffle(inputs)

    os_extra_test.write(inputs[0::2])
    os_extra_train.write(inputs[1::2])
    os_extra_test.end_write()
    os_extra_train.end_write()

    return os_extra_test.to_ConfigSet(), os_extra_train.to_ConfigSet()
            

def sample_failed(fns, pred_prop_prefix):

    if not fns["md_traj"]["failed"].exists():
        return

    inputs = ConfigSet(input_files=fns["md_traj"]["failed"])
    outputs = OutputSpec(output_files=fns["extra"]["train"]["from_failed"])
    if outputs.is_done():
        return outputs.to_ConfigSet()

    trajs = configs.into_dict_of_labels(inputs, info_label="md_start_hash")
    for traj in trajs.values():
        accuracies = [check_accuracy(at, pred_prop_prefix) for at in traj]
        first_failed = accuracies.index(False)
        outputs.write(traj[first_failed - 1])
    outputs.end_write()
    return outputs.to_ConfigSet()


def remove_all_calc_results(atoms):
    for at in atoms:
        util.remove_energy_force_containing_entries(at)
 

def prepare_dataset(cs, outspec, cycle_idx, arrays_to_delete=None, info_to_delete=None):

    if arrays_to_delete is None:
        arrays_to_delete = []

    if info_to_delete is None:
        info_to_delete = []

    if outspec.is_done():
        logger.info(f"dataset is preapared {outspec}")
        return outspec.to_ConfigSet()

    logger.info(f"preparing dataset {outspec.output_files}")

    for at in cs:
        if "iter_no" not in at.info.keys():
            at.info["iter_no"] = cycle_idx 
        for array in arrays_to_delete:
            if array in at.arrays:
                del at.arrays[array]
        for info in info_to_delete:
            if info in at.info:
                del at.info[info]
        if at.cell is None:
            at.cell = [50, 50, 50]
        outspec.write(at)
    outspec.end_write()
    return outspec.to_ConfigSet()


def make_structures(
    smiles_csv,
    num_smi_repeat,
    outputs,
    num_rads_per_mol,
    smiles_col="smiles",
    name_col="zinc_id",
    ):

    if outputs.is_done():
        logger.info(f"outputs ({outputs} from {smiles_csv.name}) are done, returning")
        return outputs.to_ConfigSet()

    atoms_out = []
    logger.info(f"writing new structures from {smiles_csv} to {outputs.output_files}")

    # generate molecules
    df = pd.read_csv(smiles_csv, delim_whitespace=True)
    for smi, name in zip(df[smiles_col], df[name_col]):
        for _ in range(num_smi_repeat):
            try:
                mol_and_rads = radicals.rad_conformers_from_smi(
                    smi=smi, compound=name, num_radicals=num_rads_per_mol
                )
                atoms_out += mol_and_rads
            except RuntimeError:
                logger.info(f"could not generate structure from {smi}")
            except IndexError:
                logger.info(f'could not generate structure from {smi}')

    logger.info(f"length of output atoms: {len(atoms_out)}")

    for at in atoms_out:
        at.cell = [50, 50, 50]
        at.info["md_start_hash"] = configs.hash_atoms(at)
        outputs.write(at)

    outputs.end_write()
    return outputs.to_ConfigSet()


def do_gap_fit(fit_dir, idx, ref_type, train_set_fname, fit_params_base, gap_fit_path):

    fit_dir.mkdir(exist_ok=True)
    fit_fname = fit_dir / f"gap_{idx}.xml"
    gap_out_fname = fit_dir / f"gap_{idx}.out"

    if not fit_fname.exists():
        logger.info(f"fitting gap {fit_fname} on {train_set_fname}")

        gap_params = deepcopy(fit_params_base)
        gap_params["gap_file"] = fit_fname

        fit_inputs = ConfigSet(input_files=train_set_fname)

        wfl.fit.gap.simple.run_gap_fit(
            fitting_configs=fit_inputs,
            fitting_dict=gap_params,
            stdout_file=gap_out_fname,
            gap_fit_exec=gap_fit_path,
        )

    full_fit_fname = str(fit_fname.resolve())
    if ref_type == "dft":
        return (Potential, [], {"param_filename": full_fit_fname})
    elif ref_type == "dft-xtb2":
        return (xtb2_plus_gap, [], {"gap_filename": full_fit_fname})


def update_fit_params(fit_params_base, fit_to_prop_prefix):
    # for gap only for now

    if (
        "energy_parameter_name" in fit_params_base
        and fit_params_base["energy_parameter_name"] != f"{fit_to_prop_prefix}energy"
    ):

        logger.warning(
            f'Overwriting {fit_params_base["energy_parameter_name"]} found '
            f'in fit_params_base with "{fit_to_prop_prefix}energy"'
        )
        fit_params_base["energy_parameter_name"] = f"{fit_to_prop_prefix}energy"

    if (
        "force_parameter_name" in fit_params_base
        and fit_params_base["force_parameter_name"] != f"{fit_to_prop_prefix}forces"
    ):

        logger.warning(
            f'Overwriting {fit_params_base["force_parameter_name"]} found '
            f'in fit_params_base with "{fit_to_prop_prefix}forces"'
        )
        fit_params_base["force_parameter_name"] = f"{fit_to_prop_prefix}forces"

    return fit_params_base


def do_ace_fit(
    fns,
    idx,
    ref_type,
    fit_params_base,
    fit_to_prop_prefix,
    ace_fit_exec,
    ):

    assert ref_type == "dft"

    if "NSLOTS" in os.environ.keys():
        ncores = os.environ["NSLOTS"]
        os.environ["ACE_FIT_JULIA_THREADS"] = str(ncores)
        os.environ["ACE_FIT_BLAS_THREADS"] = str(ncores)

    fit_inputs = read(fns["this_train"], ":")

    params = update_ace_params(fit_params_base, fit_inputs)

    ace_name = f"ace"
    ace_fname = fns["model"]["fname"]
    params["ACE_fname"] = str(ace_fname)

    with open(fns["model"]["params"], "w") as f:
        yaml.dump(params, f)

    ace_fname = wfl.fit.ace.run_ace_fit(
        fitting_configs=fit_inputs, 
        ace_fit_params=params,
        run_dir=fns["fit_dir"], 
        ace_fit_command=ace_fit_exec,
        skip_if_present=True)

    # return (ace.ACECalculator, [], {"jsonpath": str(ace_fname), 'ACE_version':1})
    # return (ACE1, [str(ace_fname)], {})
    return (pyjulip_ace, [str(ace_fname)], {})


def update_ace_params(base_params, fit_inputs):
    """ for now just select inner cutoff"""
    params = deepcopy(base_params)
    dists = util.distances_dict(fit_inputs)
    # cutoffs_mb = params["cutoffs_mb"]
    cutoffs_mb = params["basis"]["ace_basis"]["transform"]["cutoffs"]
    for key, dists in dists.items():
        if len(dists) == 0:
            logger.warning(f"did not find any pairs between elements {key}")
            continue
        update_cutoffs(cutoffs_mb, key, np.min(dists))

    return params


def update_cutoffs(cutoffs_mb, symbols, min_dist):
    # assumes keys are always alphabetical :/
    logger.info(f"old cutoffs: {cutoffs_mb}")
    key = f'({symbols[0]}, {symbols[1]})'
    vals = cutoffs_mb[key]
    vals = [float(val) for val in vals.strip("()").split(',')]
    cutoffs_mb[key] = f"({min_dist:.2f}, {vals[1]})"


def parse_cutoffs(key, cutoffs_mb):
    vals = cutoffs_mb[key]
    return [float(val) for val in vals.strip("()").split(',')]


def check_dft(train_set_fname, dft_prop_prefix, dft_calc, tests_wdir):

    tests_wdir.mkdir(exist_ok=True)

    all_ats = read(train_set_fname, ':')
    cs = ConfigSet(input_configs=random.choices(all_ats, k=2))
    os = OutputSpec(output_files=tests_wdir/"all_dft_check.xyz")
    inputs = generic.run(
        inputs=cs,
        outputs=os,
        properties=["energy", "forces"],
        output_prefix='dft_recalc_',
        calculator=dft_calc
    )

    for at in inputs:
        energy_ok = pytest.approx(at.info[f'{dft_prop_prefix}energy']) == at.info['dft_recalc_energy']
        forces_ok = np.all(pytest.approx(at.arrays[f'{dft_prop_prefix}forces'], abs=1e-4) == at.arrays['dft_recalc_forces'])
        logger.info(f"forces_ok: {forces_ok}")
        if not (energy_ok and forces_ok):
            logger.info(f'energy ok: {energy_ok}, forces_ok: {forces_ok}')
            fname = tests_wdir/'failed_dft_check.xyz' 
            if not fname.exists():
                write(fname, at)
            else:
                # pass
                raise RuntimeError("Failed fname esists, not overwriting!")
            logger.warn("failed dft check")
    # logger.info("dft check is done")


def select_extra_smiles(all_extra_smiles_csv, smiles_selection_csv, num_extra_smiles=10):

    df = pd.read_csv(all_extra_smiles_csv, delim_whitespace=True)

    free = df[~df["has_been_used"]]

    selection = free[:num_extra_smiles]
    del selection["has_been_used"]

    selection.to_csv(smiles_selection_csv, sep=" ")

    for idx in selection.index:
        df.at[idx, "has_been_used"] = True

    df.to_csv(all_extra_smiles_csv, sep=" ")


def manipulate_smiles_csv(all_extra_smiles, wdir):

    df = pd.read_csv(all_extra_smiles, delim_whitespace=True)
    if "has_been_used" not in df.columns:
        df["has_been_used"] = [False] * len(df)

    df.to_csv(wdir / all_extra_smiles.name, sep=" ")
    

def check_accuracy(at, pred_prop_prefix, dft_prop_prefix=None):

    pred_forces = at.arrays[f'{pred_prop_prefix}forces']
    max_pred_f = np.max(np.abs(pred_forces))

    if max_pred_f > 10:
        return False

    if dft_prop_prefix is not None:
        dft_forces = at.arrays[f'{dft_prop_prefix}forces']
        max_dft_f = np.max(np.abs(dft_forces))

        if max_dft_f > 10:
            return False

        ratios = np.divide(dft_forces, pred_forces)

        ## only care about non-minute forces
        # check for dft forces > 1 eV/A and only take 
        # relevant predicted forces elements to check further
        relevant_pred_forces = np.where(np.abs(dft_forces) > 1, pred_forces, np.nan)
        # check predicted forces to be 1 eV/A
        # and only return the relevant ratios
        relevant_ratios = np.where(np.abs(relevant_pred_forces) > 1, ratios, np.nan)

        # print(f"max_dft: {max_dft_f}, max_pred: {max_pred_f}, ratios: min: {np.min(relevant_ratios)} max: {np.max(relevant_ratios)}")

        if np.any(relevant_ratios < 0.25) or np.any(relevant_ratios > 4):
            return False

    return True

def run_md(calculator, inputs, outputs, md_params):

    md_stopper = BadGeometry(info_label="md_geometry_check") 
    autopara_info = autoparainfo.AutoparaInfo(num_inputs_per_python_subprocess=1)

    inputs = md.sample(
        inputs=inputs, 
        outputs=outputs,
        calculator=calculator, 
        update_config_type=False,
        abort_check=md_stopper,
        **md_params)
    
    return inputs
