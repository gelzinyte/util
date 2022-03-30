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

from pyjulip import ACE1


from quippy.potential import Potential

from wfl.configset import ConfigSet_in, ConfigSet_out
import wfl.fit.gap_simple
import wfl.fit.ace
from wfl.calculators import generic
from wfl.calculators import orca

import util
from util import radicals
from util import configs
from util.calculators import xtb2_plus_gap
from util.bde import generate
from util.plot import dataset
from util.plot import rmse_scatter_evaled, multiple_error_files
from util.util_config import Config

logger = logging.getLogger(__name__)


def prepare_remoteinfo(gap_fname):
    output_filename = os.environ["WFL_AUTOPARA_REMOTEINFO"]
    template = os.environ["WFL_AUTOPARA_REMOTEINFO_TEMPLATE"]
    with open(template, "r") as f:
        file = f.read()
    file = file.replace("<gap_filename>", gap_fname)
    with open(output_filename, "w") as f:
        f.write(file)


def make_dirs(dir_names):
    for dir_name in dir_names:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)


def prepare_0th_dataset(ci, co):

    if co.is_done():
        logger.info(f"initial dataset is preapared {co.output_files[0].name}")
        return co.to_ConfigSet_in()

    logger.info(f"preparing initial dataset {co.output_files[0].name}")

    for at in ci:
        if "iter_no" not in at.info.keys():
            at.info["iter_no"] = 0
        if at.cell is None:
            at.cell = [50, 50, 50]

        co.write(at)
    co.end_write()
    return co.to_ConfigSet_in()


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
        return outputs.to_ConfigSet_in()

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

    logger.info(f"length of output atoms: {len(atoms_out)}")

    for at in atoms_out:
        at.cell = [50, 50, 50]
        outputs.write(at)

    outputs.end_write()
    return outputs.to_ConfigSet_in()


def filter_configs_by_geometry(inputs, outputs_good, outputs_bad):

    if not outputs_bad.is_done() and not outputs_good.is_done():

        all_configs = configs.filter_insane_geometries(inputs, mult=1)
        outputs_bad.write(all_configs["bad_geometries"])
        outputs_bad.end_write()

        outputs_good.write(all_configs["good_geometries"])
        outputs_good.end_write()
    elif outputs_bad.is_done() and outputs_good.is_done():
        logger.info(
            "not_writing to `outputs_good` or `outputs_bad`, because they are done."
        )
    else:
        raise RuntimeError("one output is done, but not the other. ")

    return outputs_good.to_ConfigSet_in()


def filter_configs(
    inputs,
    outputs_large_error,
    outputs_small_error,
    pred_prop_prefix,
    e_threshold_total=None,
    e_threshold_per_atom=None,
    max_f_comp_threshold=None,
    dft_prefix="dft_",
):
    have_small_errors = False
    have_large_errors = False

    if outputs_large_error.is_done() and outputs_small_error.is_done():
        logger.info("both outputs are done, not filtering configs by energy/force error")
        return outputs_large_error.to_ConfigSet_in()

    elif not outputs_large_error.is_done() and not outputs_small_error.is_done():
        pass
    else:
        raise RuntimeError("one of the outputs is done, but not the other")

    assert e_threshold_total is not None or e_threshold_per_atom is not None
    if e_threshold_per_atom is not None and e_threshold_total is not None:
        raise RuntimeError("give either energy per atom or total energy threshold")

    for at in inputs:
        e_error = at.info[f"{pred_prop_prefix}energy"] - at.info[f"{dft_prefix}energy"]

        if e_threshold_total is not None and np.abs(e_error) > e_threshold_total:
            outputs_large_error.write(at)
            have_large_errors = True
            continue
        elif (
            e_threshold_per_atom is not None
            and np.abs(e_error) / len(at) > e_threshold_per_atom
        ):
            outputs_large_error.write(at)
            have_large_errors = True
            continue

        if max_f_comp_threshold is not None:
            f_error = (
                at.arrays[f"{pred_prop_prefix}forces"]
                - at.arrays[f"{dft_prefix}forces"]
            )
            if np.max(np.abs(f_error.flatten())) > max_f_comp_threshold:
                outputs_large_error.write(at)
                have_large_errors = True
                continue

        have_small_errors = True
        outputs_small_error.write(at)

    if not have_small_errors:
        outputs_small_error.write([])
    if not have_large_errors:
        outputs_large_error.write([])

    outputs_small_error.end_write()
    outputs_large_error.end_write()

    if not have_large_errors:
        return None
    else:
        return outputs_large_error.to_ConfigSet_in()


def do_gap_fit(fit_dir, idx, ref_type, train_set_fname, fit_params_base, gap_fit_path):

    fit_dir.mkdir(exist_ok=True)
    fit_fname = fit_dir / f"gap_{idx}.xml"
    gap_out_fname = fit_dir / f"gap_{idx}.out"

    if not fit_fname.exists():
        logger.info(f"fitting gap {fit_fname} on {train_set_fname}")

        gap_params = deepcopy(fit_params_base)
        gap_params["gap_file"] = fit_fname

        fit_inputs = ConfigSet_in(input_files=train_set_fname)

        wfl.fit.gap_simple.run_gap_fit(
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
    fit_dir,
    idx,
    ref_type,
    train_set_fname,
    fit_params_base,
    fit_to_prop_prefix,
    ace_fit_exec,
):

    assert ref_type == "dft"

    if "NSLOTS" in os.environ.keys():
        ncores = os.environ["NSLOTS"]
        os.environ["ACE_FIT_JULIA_THREADS"] = str(ncores)
        os.environ["ACE_FIT_BLAS_THREADS"] = str(ncores)

    fit_inputs = read(train_set_fname, ":")

    params = update_ace_params(fit_params_base, fit_inputs)

    ace_name = f"ace_{idx}"
    ace_fname = fit_dir / (ace_name + ".json")
    params["ACE_fname"] = str(ace_fname)

    with open(fit_dir / f"ace_{idx}_params.yaml", "w") as f:
        yaml.dump(params, f)

    ace_fname = wfl.fit.ace.run_ace_fit(
        fitting_configs=fit_inputs, 
        ace_fit_params=params,
        run_dir=fit_dir, 
        skip_if_present=True)

    # return (ace.ACECalculator, [], {"jsonpath": str(ace_fname), 'ACE_version':1})
    return  (ACE1, [str(ace_fname)], {})


def update_ace_params(base_params, fit_inputs):
    """ for now just select inner cutoff"""
    params = deepcopy(base_params)
    dists = util.distances_dict(fit_inputs)
    # cutoffs_mb = params["cutoffs_mb"]
    cutoffs_mb = params["basis"]["rpi_basis"]["transform"]["cutoffs"]
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


def check_dft(train_set_fname, dft_prop_prefix, orca_kwargs, tests_wdir):

    tests_wdir.mkdir(exist_ok=True)

    all_ats = read(train_set_fname, ':')
    ci = ConfigSet_in(input_configs=random.choices(all_ats, k=2))
    co = ConfigSet_out()
    inputs = orca.evaluate(
        inputs=ci,
        outputs=co,
        orca_kwargs=orca_kwargs,
        output_prefix='dft_recalc_',
        keep_files='default',
        base_rundir=tests_wdir / "orca_wdir")

    write(tests_wdir/"all_dft_check.xyz", [at for at in inputs])

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
                pass
                # raise RuntimeError("Failed fname esists, not overwriting!")
            logger.warn("failed dft check")
    # logger.info("dft cyeck is done")


def select_extra_smiles(all_extra_smiles_csv, smiles_selection_csv, chunksize=10):

    df = pd.read_csv(all_extra_smiles_csv, delim_whitespace=True)

    free = df[~df["has_been_used"]]

    selection = free[:chunksize]
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
    


def process_trajs(traj_ci, good_traj_configs_co, bad_traj_bad_cfg_co, bad_traj_good_cfg_co, traj_sample_rule):
    """
    traj_sample_rule - how many/which configs to return from the good trajectory
    "last" - only last is returned
    otherwise that's the interval to sample at

    """

    doneness = sum([good_traj_configs_co.is_done(), bad_traj_bad_cfg_co.is_done(), bad_traj_good_cfg_co.is_done()])
    if doneness == 3:
        logger.info("outputs_are_done, not re-splitting the full trajectories ")
        return
    elif doneness != 0:
        raise RuntimeError("some outputs done, but not all!")

    # divide into graphs 
    trajs = configs.into_dict_of_labels(traj_ci, "graph_name")
    
    count_good_traj = 0
    count_bad_traj_good_cfg = 0
    count_bad_traj_bad_cfg = 0

    for graph_name, traj in trajs.items():

        co_good = ConfigSet_out() 
        co_bad = ConfigSet_out()
        filter_configs_by_geometry(inputs=traj, outputs_good=co_good, outputs_bad=co_bad)
        if len(cotl(co_bad)) == 0:
            #  trajectory is reasonable, only need to return part of it
            cfgs_good = cotl(co_good)
            if traj_sample_rule == "last":
                good_traj_configs_co.write(cfgs_good[-1])
            else:
                good_traj_configs_co.write(cfgs_good[::traj_sample_rule])
            count_good_traj += 1
        else:
            # some of the trajectory is unreasonable
            good_cfgs = [at for at in co_good.to_ConfigSet_in()]
            bad_cfgs = [at for at in co_bad.to_ConfigSet_in()]
            bad_traj_bad_cfg_co.write(bad_cfgs)
            bad_traj_good_cfg_co.write(good_cfgs)
            count_bad_traj_good_cfg += len(good_cfgs)
            count_bad_traj_bad_cfg += len(bad_cfgs)

    if count_good_traj == 0:
        good_traj_configs_co.write([])
    if count_bad_traj_good_cfg == 0:
        bad_traj_good_cfg_co.write([])
    if count_bad_traj_bad_cfg == 0:
        bad_traj_bad_cfg_co.write([])
    
    good_traj_configs_co.end_write()
    bad_traj_bad_cfg_co.end_write()
    bad_traj_good_cfg_co.end_write()


def cotl(co):
    """configset_out to configset in to list"""
    return [at for at in co.to_ConfigSet_in()]


def sample_failed_trajectory(ci, co, orca_kwargs, dft_prop_prefix, cycle_dir, pred_prop_prefix):

    if co.is_done():
        logger.info("Sub-sampling is already done, returning")
        return co.to_ConfigSet_in()

    dft_dir = cycle_dir / "sample_failed_traj_DFT"
    dft_dir.mkdir(exist_ok=True)

    from util import configs

    trajs = configs.into_dict_of_labels(ci, "graph_name")

    logger.info(f"Number of failed trajectories: {len(trajs)}")

    for label, traj in trajs.items():
        found_good = False
        logger.info(f"{label}: checking first couple of configs from trajectory")
        dft_sample_ci = ConfigSet_in(input_configs=[at.copy() for at in traj[0:10]])
        dft_sample_co = ConfigSet_out(output_files= dft_dir / f"{label}.dft.xyz")
        orca.evaluate(inputs=dft_sample_ci, outputs=dft_sample_co, 
                      orca_kwargs=orca_kwargs, output_prefix=dft_prop_prefix, 
                      keep_files=False, base_rundir=dft_dir/"orca_wdir")        

        cfgs = [at for at in dft_sample_co.to_ConfigSet_in()]
        for idx, at in enumerate(cfgs):
            if found_good:
                break
            is_accurate = check_accuracy(at, dft_prop_prefix, pred_prop_prefix)
            if not is_accurate:
                found_good = True
                picked_at = cfgs[idx-1]
                logger.info(f"Picked this one!: {picked_at.info}")
                co.write(picked_at)
                break

        if not found_good:
            logger.info(f"{label}: going through the trajectory in reverse")
        # iterate in reverse until found first good one
        if "NSLOTS" in os.environ:
            chunksize = int(os.environ["NSLOTS"])
        else:
            chunksize=8
        for group_idx, at_group in enumerate(util.grouper(reversed(traj), chunksize)):
            if found_good:
                break
            at_group = [at for at in at_group if at is not None]
            dft_sample_ci = ConfigSet_in(input_configs=at_group)
            dft_sample_co = ConfigSet_out(output_files= dft_dir / f"{label}.{group_idx}.dft.xyz")
            orca.evaluate(inputs=dft_sample_ci, outputs=dft_sample_co, 
                        orca_kwargs=orca_kwargs, output_prefix=dft_prop_prefix, 
                        keep_files=False, base_rundir=dft_dir/"orca_wdir")        
                
            for at in dft_sample_co.to_ConfigSet_in():
                is_accurate = check_accuracy(at, dft_prop_prefix, pred_prop_prefix)
                if is_accurate:
                    logger.info(f"Picked this one!: {at.info}")
                    found_good = True
                    co.write(at)
                    break
        else:
            logger.info(f"found no suitable configs for {at.info}")
    
    co.end_write()

    return co.to_ConfigSet_in()


def check_accuracy(at, dft_prop_prefix, pred_prop_prefix, no_dft=False):

    pred_forces = at.arrays[f'{pred_prop_prefix}forces']
    max_pred_f = np.max(np.abs(pred_forces))

    if max_pred_f > 15:
        return False

    if not no_dft:
        dft_forces = at.arrays[f'{dft_prop_prefix}forces']
        max_dft_f = np.max(np.abs(dft_forces))

        if max_dft_f > 15:
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

def md_subselector_function(traj):
    all_configs = configs.filter_insane_geometries(inputs, mult=1, skin=0)

    if len(all_configs["bad_geometries"]) == 0:
        return []

    for at in all_configs["good_geometries"]:
        at.info["config_type"] = "bad_md_good_geometry"

    return all_configs["good_geometries"]

def launch_analyse_md(inputs, pred_prop_prefix, outputs_to_fit, outputs_traj, outputs_rerun, calculator, md_params):
    """
    1. run MD
    2. on-the-fly parse whether trajectory is ok or not
    3.1 return [None] if trajectory was ok
    3.2 return good part of the trajectory if it wasn't
    4. parse the trajectory:
        * pick first config to restart next iteration's trajectory from
        * pick single config to include into training 
        * ~save the rest of the trajectory~ 
          -> actually that will have been saved. 
    """

    doneness = sum([outputs_to_fit.is_done(), outputs_traj.is_done(), outputs_rerun.is_done()])
    if doneness == 3:
        logger.info("outputs_are_done, not re-splitting the full trajectories ")
        return outputs_to_fit.to_ConfigSet_in()
    elif doneness != 0:
        raise RuntimeError("some outputs done, but not all!")

    # 1. run md 
    outputs = ConfigSet_out()
    inputs = md.sample(
        inputs=inputs, 
        outputs=outputs, 
        calculator=calculator, 
        selector_function=md_subselector_function,
        **md_params)

    # 2. reevaluate ace
    inputs = generic.run(inputs=inputs, 
                         outputs=outputs_traj,
                         calculator=calculator,
                         properties=["energy", "forces"],
                         output_prefix=pred_prop_prefix)

    # 3. select configs we need
    dict_of_trajs = configs.into_dict_of_labels([at for at in inputs], "graph_name")
    for label, traj in dict_of_trajs:
        outputs_rerun.write(traj[0])
        outputs_to_fit.write(select_at_from_failed_md(traj))

    outputs_rerun.end_write()
    outputs_to_fit.end_write()
    return outputs_to_fit.to_ConfigSet_in()


def select_at_from_failed_md(traj):
    
    for at in reversed(traj):
        is_accurate = check_accuracy(at, None, pred_prop_prefix, no_dft=True)
        if is_accurate:
            return at

         






