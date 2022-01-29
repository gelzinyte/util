import os
import logging
import random
import subprocess
from copy import deepcopy
from pathlib import Path
import pytest

import pandas as pd
import numpy as np

from PyPDF4 import PdfFileMerger

from ase.io import read, write

import ace
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
            except RuntimeError:
                logger.info(f"could not generate structure from {smi}")
            atoms_out += mol_and_rads

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

    fit_inputs = ConfigSet_in(input_files=train_set_fname)

    params = update_ace_params(fit_params_base, fit_inputs)

    ace_name = f"ace_{idx}"
    ace_fname = fit_dir / (ace_name + ".json")

    ace_file_base = wfl.fit.ace.fit(
        fitting_configs=fit_inputs,
        ACE_name=ace_name,
        params=params,
        ref_property_prefix=fit_to_prop_prefix,
        skip_if_present=True,
        run_dir=fit_dir,
        formats=[".json"],
        ace_fit_exec=ace_fit_exec,
        verbose=True,
        remote_info=None,
        wait_for_results=True,
    )

    ace_fname = str(ace_fname)

    assert str(ace_file_base) + ".json" == ace_fname

    return (ace.ACECalculator, [], {"jsonpath": ace_fname, 'ACE_version':2})


def update_ace_params(base_params, fit_inputs):
    """ for now just select inner cutoff"""
    params = deepcopy(base_params)
    dists = util.distances_dict(fit_inputs)
    cutoffs_mb = params["cutoffs_mb"]
    for key, dists in dists.items():
        if len(dists) == 0:
            logger.warning(f"did not find any pairs between elements {key}")
            continue
        update_cutoffs(cutoffs_mb, key, np.min(dists))

    return params


def update_cutoffs(cutoffs_mb, symbols, min_dist):
    # assumes keys are always alphabetical :/
    logger.info(f"old cutoffs: {cutoffs_mb}")
    key = f'(:{symbols[0]}, :{symbols[1]})'
    vals = cutoffs_mb[key]
    vals = [float(val) for val in vals.strip("()").split(',')]
    cutoffs_mb[key] = f"({min_dist:.2f}, {vals[1]})"


def run_tests(
    calculator,
    pred_prop_prefix,
    dft_prop_prefix,
    train_set_fname,
    test_set_fname,
    tests_wdir,
    bde_test_fname,
    orca_kwargs,
):

    tests_wdir.mkdir(exist_ok=True)

    train_evaled = tests_wdir / f"{pred_prop_prefix}on_{train_set_fname.name}"
    test_evaled = tests_wdir / f"{pred_prop_prefix}on_{test_set_fname.name}"

    # evaluate on training and test sets
    ci = ConfigSet_in(input_files=[train_set_fname, test_set_fname])
    co = ConfigSet_out(output_files={train_set_fname: train_evaled, test_set_fname: test_evaled},
                       force=True, all_or_none=True)

    generic.run(
        inputs=ci,
        outputs=co,
        calculator=calculator,
        properties=["energy", "forces"],
        output_prefix=pred_prop_prefix,
        chunksize=20,
    )

    # check the offset is not there
    check_for_offset(train_evaled, pred_prop_prefix, dft_prop_prefix)

    # re-evaluate a couple of DFTs
    check_dft(train_evaled, dft_prop_prefix, orca_kwargs, tests_wdir)

    # bde test - final file of bde_test_fname.stem + ip_prop_prefix + bde.xyz
    bde_ci = generate.everything(
        calculator=calculator,
        dft_bde_filename=bde_test_fname,
        dft_prop_prefix=dft_prop_prefix,
        ip_prop_prefix=pred_prop_prefix,
        wdir=tests_wdir / "bde_wdir",
    )

    # dft vs ip bde correlation
    rmse_scatter_evaled.scatter_plot(
        ref_energy_name=f"{dft_prop_prefix}opt_{dft_prop_prefix}bde_energy",
        pred_energy_name=f"{pred_prop_prefix}opt_{pred_prop_prefix}bde_energy",
        ref_force_name=None,
        pred_force_name=None,
        all_atoms=bde_ci,
        output_dir=tests_wdir,
        prefix=f"{dft_prop_prefix}bde_vs_{pred_prop_prefix}bde",
        color_info_name="bde_type",
        isolated_atoms=None,
        energy_type="total_energy",
        error_type='rmse',
        skip_if_prop_not_present=True,
    )

    # ip bde absolute error vs ip energy absolute error
    co = ConfigSet_out(
        output_files=tests_wdir / f"{pred_prop_prefix}bde_file_with_errors.xyz",
        force=True,
        all_or_none=True,
    )
    if not co.is_done():
        for at in bde_ci:
            if at.info["mol_or_rad"] == "mol":
                continue

            at.info[f"{pred_prop_prefix}bde_absolute_error"] = np.abs(
                at.info[f"{dft_prop_prefix}opt_{dft_prop_prefix}bde_energy"]
                - at.info[f"{pred_prop_prefix}opt_{pred_prop_prefix}bde_energy"]
            )

            at.info[
                f"{pred_prop_prefix}absolute_error_on_{pred_prop_prefix}opt"
            ] = np.abs(
                at.info[f"{pred_prop_prefix}opt_{dft_prop_prefix}energy"]
                - at.info[f"{pred_prop_prefix}opt_{pred_prop_prefix}energy"]
            )
            at.info[
                f"{pred_prop_prefix}absolute_error_on_{dft_prop_prefix}opt"
            ] = np.abs(
                at.info[f"{dft_prop_prefix}opt_{dft_prop_prefix}energy"]
                - at.info[f"{dft_prop_prefix}opt_{pred_prop_prefix}energy"]
            )

            co.write(at)
        co.end_write()
    else:
        logger.info("Not re-assigning bde errors, because ConfigSet_out is done")

    rmse_scatter_evaled.scatter_plot(
        ref_energy_name=f"{pred_prop_prefix}absolute_error_on_{pred_prop_prefix}opt",
        pred_energy_name=f"{pred_prop_prefix}bde_absolute_error",
        ref_force_name=None,
        pred_force_name=None,
        all_atoms=co.to_ConfigSet_in(),
        output_dir=tests_wdir,
        prefix=f"{pred_prop_prefix}error_on_{pred_prop_prefix}opt_vs_{pred_prop_prefix}bde_error",
        color_info_name="bde_type",
        isolated_atoms=None,
        energy_type="total_energy",
        skip_if_prop_not_present=True,
    )

    rmse_scatter_evaled.scatter_plot(
        ref_energy_name=f"{pred_prop_prefix}absolute_error_on_{dft_prop_prefix}opt",
        pred_energy_name=f"{pred_prop_prefix}bde_absolute_error",
        ref_force_name=None,
        pred_force_name=None,
        all_atoms=co.to_ConfigSet_in(),
        output_dir=tests_wdir,
        prefix=f"{pred_prop_prefix}error_on_{dft_prop_prefix}opt_vs_{pred_prop_prefix}bde_error",
        color_info_name="bde_type",
        isolated_atoms=None,
        energy_type="total_energy",
        skip_if_prop_not_present=True,
    )

    dimer_2b(calculator, tests_wdir)

    # other tests are coming sometime
    # CH dissociation curve
    # dimer curves (2b, total)



def dimer_2b(calculator, tests_wdir):

    ace_fname = calculator[2]["jsonpath"]
    fname = tests_wdir / "ace_2b.pdf"

    cfg = Config.load()
    ace_2b_script_path = cfg["julia_2b_script"] 

    subprocess.run(f"julia {ace_2b_script_path} --param-fname {ace_fname} --fname {fname}", shell=True)


def check_for_offset(train_evaled, pred_prop_prefix, dft_prop_prefix):

    ats = read(train_evaled, ":")
    is_at = [at for at in ats if len(at) == 1]
    ats = [at for at in ats if len(at) != 1]

    ref_e = [util.get_binding_energy_per_at(at, is_at, f'{dft_prop_prefix}energy') for at in ats]
    pred_e = [util.get_binding_energy_per_at(at, is_at, f'{pred_prop_prefix}energy') for at in ats]

    errors = np.array([(ref - pred) * 1e3 for ref, pred in zip(ref_e, pred_e)])
    mean_error = np.mean(errors)

    shifted_errors = errors - mean_error
    non_shifter_rmse = np.sqrt(np.mean(errors ** 2))
    shifted_rmse = np.sqrt(np.mean(shifted_errors ** 2))

    difference = (non_shifter_rmse - shifted_rmse) / shifted_rmse

    logger.info(
        f"non-shifted rmse: {non_shifter_rmse:.3f}, "
        f"shifted rmse: {shifted_rmse:.3f}, difference: {difference * 100:.1f}%"
    )

    assert (
        difference < 0.01
    ), f"Offset in training set ({difference * 100:.1f}) 1%, smelling foul!"


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

    for at in inputs:
        energy_ok =  pytest.approx(at.info[f'{dft_prop_prefix}energy']) == at.info['dft_recalc_energy']
        forces_ok = np.all(pytest.approx(at.arrays[f'{dft_prop_prefix}forces']) == at.arrays['dft_recalc_forces'])
        if not (energy_ok and forces_ok):
            print(f'energy ok: {energy_ok}, forces_ok: {forces_ok}')
            write(tests_wdir/'failed_dft_check.xyz', at)
            raise RuntimeError("failed dft check")
    logger.info("dft cyeck is ok")


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
    

def summary_plots(
    cycle_idx,
    pred_prop_prefix,
    dft_prop_prefix,
    train_fname,
    test_fname,
    bde_fname,
    ip_optimised_fname,
    train_extra_fname,
    test_extra_fname,
    tests_wdir,
):

    # set up correct fnames
    train_fname = tests_wdir / f"{pred_prop_prefix}on_{train_fname.name}"
    test_fname = tests_wdir / f"{pred_prop_prefix}on_{test_fname.name}"

    bde_dft_opt = tests_wdir / "bde_wdir" / (Path(bde_fname).stem + "." + pred_prop_prefix[:-1] + ".xyz")
    
    bde_ip_reopt = tests_wdir / "bde_wdir" / (Path(bde_fname).stem + "." + pred_prop_prefix + "reoptimised.dft.xyz")

    all_outputs = tests_wdir / "all_configs_for_plot.xyz"

    isolated_atoms = [at for at in read(train_fname, ":") if len(at) == 1]

    # plot the dataset summary plots
    atoms = read(train_fname, ":") + read(train_extra_fname, ":")
    title = f"{cycle_idx:02d}_training_set_for_{pred_prop_prefix}{cycle_idx+1:02d}"
    dataset.energy_by_idx(
        atoms=atoms,
        title=title + '_energy',
        isolated_atoms=isolated_atoms,
        info_label="iter_no",
        prop_prefix=dft_prop_prefix,
        dir=tests_wdir,
    )
    dataset.forces_by_idx(
        atoms=atoms, title=title + '_forces', info_label="iter_no", prop_prefix=dft_prop_prefix,
        dir=tests_wdir
    )

    # combine all data together
    expected_dataset_types = [
        "next_addition_from_md",
        "next_addition_from_md",
        "test",
        "train",
        f"bde_{dft_prop_prefix}optimised",
        f"bde_{pred_prop_prefix}reoptimised",
        f"next_rdkit_{pred_prop_prefix}optimised",
    ]
    all_fnames = [
        train_extra_fname,
        test_extra_fname,
        test_fname,
        train_fname,
        bde_dft_opt,
        bde_ip_reopt,
        ip_optimised_fname,
    ]

    for fname, expected_dset_type in zip(all_fnames, expected_dataset_types):
        ats = read(fname, ":")
        at = ats[0]
        assert at.info["dataset_type"] == expected_dset_type
        assert pred_prop_prefix + "energy" in at.info.keys()
        assert dft_prop_prefix + "energy" in at.info.keys()

        write(all_outputs, ats, append=True)

    rmse_scatter_evaled.scatter_plot(
        ref_energy_name=dft_prop_prefix + "energy",
        pred_energy_name=pred_prop_prefix + "energy",
        ref_force_name=dft_prop_prefix + "forces",
        pred_force_name=pred_prop_prefix + "forces",
        all_atoms=read(all_outputs, ":"),
        output_dir=tests_wdir,
        prefix=f"{cycle_idx:02d}_ef_correlation",
        color_info_name="dataset_type",
        isolated_atoms=None,
        energy_type="binding_energy",
    )

def combine_plots(pred_prop_prefix, dft_prop_prefix, tests_wdir, cycle_idx, figs_dir, wdir):
    """
    binding_energies
    dft_bde_vs_ace_bde
    energy_training_set
    forces_training_set
    ace_error_on_dft_opt_vs_bde_error
    ace_error_on_ace_opt_vs_bde_error 
    """

    combined_out = figs_dir / f"{cycle_idx:02d}_summary.pdf"

    fnames = [
        f"{cycle_idx:02d}_ef_correlation_by_dataset_type_scatter.pdf",
        f"ace_2b.pdf",
        f"{dft_prop_prefix}bde_vs_{pred_prop_prefix}bde_by_bde_type_scatter.pdf",
        f"{cycle_idx:02d}_training_set_for_{pred_prop_prefix}{cycle_idx+1:02d}_energy.pdf",
        f"{cycle_idx:02d}_training_set_for_{pred_prop_prefix}{cycle_idx+1:02d}_forces.pdf",
        f"{pred_prop_prefix}error_on_{dft_prop_prefix}opt_vs_{pred_prop_prefix}bde_error_by_bde_type_scatter.pdf",
        f"{pred_prop_prefix}error_on_{pred_prop_prefix}opt_vs_{pred_prop_prefix}bde_error_by_bde_type_scatter.pdf",
    ]


    merger = PdfFileMerger(strict=False)
    for fn in fnames:
        merger.append(fileobj=open(tests_wdir / fn, 'rb'), pages=(0,1))        

    merger.write(fileobj=open(combined_out, 'wb'))
    merger.close()


    update_tracker_plot(pred_prop_prefix=pred_prop_prefix,
                        dft_prop_prefix=dft_prop_prefix,
                        cycle_idx=cycle_idx, 
                        figs_dir=figs_dir,
                        wdir=wdir)

def update_tracker_plot(pred_prop_prefix, dft_prop_prefix, cycle_idx, figs_dir, wdir):


    atoms_filenames = [wdir / f"iteration_{idx:02d}/tests/all_configs_for_plot.xyz" \
        for idx in range(cycle_idx+1)]

    multiple_error_files.main(ref_energy_name=f'{dft_prop_prefix}energy',
                              pred_energy_name=f'{pred_prop_prefix}energy',
                              ref_force_name=f'{dft_prop_prefix}forces',
                              pred_force_name=f'{pred_prop_prefix}forces',
                              atoms_filenames=atoms_filenames,
                              output_dir=figs_dir,
                              prefix=f"up_to_{cycle_idx}",
                              color_info_name="dataset_type",
                              xvals=range(cycle_idx+1),
                              xlabel="iteration")










