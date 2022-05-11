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
from quippy.potential import Potential

from wfl.configset import ConfigSet, ConfigSet_out
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
from util.plot import rmse_scatter_evaled, multiple_error_files, julia_plots
from util.util_config import Config
from util.iterations import tools as it

logger = logging.getLogger(__name__)

def run_tests(
    calculator,
    pred_prop_prefix,
    dft_prop_prefix,
    train_set_fname,
    tests_wdir,
    # bde_test_fname,
    orca_kwargs,
    output_dir,
    validation_fname,
    quick = True,
    fit_params=None):

    tests_wdir.mkdir(exist_ok=True)

    train_evaled = tests_wdir / f"{pred_prop_prefix}on_{train_set_fname.name}"
    val_evaled = tests_wdir / f"{pred_prop_prefix}on_{validation_fname.name}"

    # evaluate on training and test sets
    ci = ConfigSet(input_files=[train_set_fname, validation_fname])
    co = ConfigSet_out(output_files={train_set_fname: train_evaled, validation_fname: val_evaled},
                       force=True, all_or_none=True)

    generic.run(
        inputs=ci,
        outputs=co,
        calculator=calculator,
        properties=["energy", "forces"],
        output_prefix=pred_prop_prefix,
        chunksize=200)

    # check the offset is not there
    check_for_offset(train_evaled, pred_prop_prefix, dft_prop_prefix)

    dimer_2b(calculator, tests_wdir, fit_params)

    # training & validation set scatter plots
    ats_train = read(train_evaled, ":")
    ats_val = read(val_evaled, ":")
    rmse_scatter_evaled.scatter_plot(
        ref_energy_name=f"{dft_prop_prefix}energy",
        pred_energy_name=f"{pred_prop_prefix}energy",
        ref_force_name=f"{dft_prop_prefix}forces",
        pred_force_name=f"{pred_prop_prefix}forces",
        all_atoms=ats_train + ats_val,
        output_dir=output_dir, 
        prefix=None, 
        color_info_name="dataset_type", 
        isolated_atoms=None,
        energy_type="binding_energy", 
        energy_shift=False,
        no_legend=False,
        error_type='mae', 
        skip_if_prop_not_present=False)

    # re-evaluate a couple of DFTs
    # check_dft(train_evaled, dft_prop_prefix, orca_kwargs, tests_wdir)

    if quick:
        return

    
    # bde test - final file of bde_test_fname.stem + ip_prop_prefix + bde.xyz
    bde_ci = generate.everything(
        calculator=calculator,
        dft_bde_filename=bde_test_fname,
        dft_prop_prefix=dft_prop_prefix,
        ip_prop_prefix=pred_prop_prefix,
        wdir=tests_wdir / "bde_wdir",
        output_dir = output_dir,
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
        all_atoms=co.to_ConfigSet(),
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
        all_atoms=co.to_ConfigSet(),
        output_dir=tests_wdir,
        prefix=f"{pred_prop_prefix}error_on_{dft_prop_prefix}opt_vs_{pred_prop_prefix}bde_error",
        color_info_name="bde_type",
        isolated_atoms=None,
        energy_type="total_energy",
        skip_if_prop_not_present=True,
    )


    # other tests are coming sometime
    # CH dissociation curve
    # dimer curves (2b, total)

def dimer_2b(calculator, tests_wdir, fit_params=None):

    if fit_params is None:
        cc_in = None, 
        ch_in = None, 
        hh_in = None,
    else:
        cutoffs_mb = fit_params["basis"]["rpi_basis"]["transform"]["cutoffs"]        
        cc_in = it.parse_cutoffs(f'(C, C)', cutoffs_mb)[0] 
        ch_in = it.parse_cutoffs(f'(C, H)', cutoffs_mb)[0]
        hh_in = it.parse_cutoffs(f'(H, H)', cutoffs_mb)[0]

    ace_fname = calculator[1][0]

    for plot_type in ["2b", "full"]:
        julia_plots.plot_ace_2b(ace_fname, plot_type, cc_in=cc_in, ch_in=ch_in, hh_in=hh_in)



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
    tests_wdir):

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
    all_fnames = [
        test_fname,
        train_fname,
        bde_dft_opt,
        bde_ip_reopt,
        ip_optimised_fname,
        train_extra_fname,
        test_extra_fname,
    ]
    expected_dataset_types = [
        "test",
        "train",
        f"bde_{dft_prop_prefix}optimised",
        f"bde_{pred_prop_prefix}reoptimised",
        f"next_rdkit_{pred_prop_prefix}opt",
        "next_train",
        "next_test",
    ]

    # import pytest; pytest.set_trace()
    for fname, expected_dset_type in zip(all_fnames, expected_dataset_types):
        ats = read(fname, ":")
        at = ats[0]
        print(at.info["dataset_type"]) 
        print(expected_dset_type)
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
        # f"ace_2b.pdf",
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

    xvals = [len(read(wdir / f"training_sets/{idx:02d}.train_for_{pred_prop_prefix}{idx+1:02d}.xyz", ':')) for idx in range(cycle_idx)]
    xvals = [len(read(wdir / f"training_sets/train_for_fit_0.xyz", ":"))] + xvals

    multiple_error_files.main(ref_energy_name=f'{dft_prop_prefix}energy',
                              pred_energy_name=f'{pred_prop_prefix}energy',
                              ref_force_name=f'{dft_prop_prefix}forces',
                              pred_force_name=f'{pred_prop_prefix}forces',
                              atoms_filenames=atoms_filenames,
                              output_dir=figs_dir,
                              prefix=f"up_to_{cycle_idx}",
                              color_info_name="dataset_type",
                              xvals=xvals,
                              xlabel="training_set_size")


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

    if difference > 0.05:
        logger.warn(f"Offset in training set ({difference * 100:.1f}%) > 5%, smelling foul!")
