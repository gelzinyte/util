import os
import shutil
import yaml
import logging
import random

from pathlib import Path

from ase.io import read, write

from wfl.calculators import orca, generic
from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.generate_configs import md
import wfl.calc_descriptor

from util import remove_energy_force_containing_entries
from util import opt
from util.util_config import Config
from util.iterations import tools as it
from util.iterations import plots as ip
from util.configs import cur
from util import error_table
import util

logger = logging.getLogger(__name__)


def fit(
    num_cycles,
    base_train_fname="train.xyz",
    fit_param_fname=None,
    all_extra_smiles_csv=None,
    md_temp=500,
    wdir="runs",
    ref_type="dft",
    ip_type="ace",
    # bde_test_fname="dft_bde.xyz",
    num_extra_smiles_per_cycle=10,
    num_rads_per_mol=0, 
    validation_fname = 'validation.xyz',
    md_steps = 2000):
    """ iteratively fits potetnials

    No files are created or modified in the original directory, everything's done in "wdir"

    * Each iteration is run in a directory (e.g."wdir"/"iteration_03").
    * An iteration starts with a fit to a training set file from the previous iteration
        ("02_train_for_ACE_03.xyz").
    * Then run tests with this potential:
        - Accuracy on train set and new compounds' validation set
        - dimer curves
        - offset in training/testing configs
    * Generate initial (rdkit) configs
        - slice a slice from all of the extra csv
        - generate 3D structures
    * Run MD at a given temperature
        - check if any geometry is unreasonable
            - if all good: 
                - continue
            - if not:
                - select one config for training
                - save this structure for next iteration
    * Get dft and combine datasets.

    Info keys:

    * dataset_type - train, test, next_configs_ip_optimised, next_configs_selected,
                     bde_dft_opt, bde_ip_reopt
                   - benchmarks to test the potential
    * config_type - how has the config been generated
                  - rdkit, ip-optimised, md

    TODO:
    * Generate heuristics for fitting params. For now:
        - select inner cutoff where data ends
        - Start with overly-large basis and cut down with ARD.
    * do configs have unique identifier?
    * is parallelisation done correctly

    Parameters
    ---------

    num_cycles: int
        number of fit-optimise cycles to do
    base_train_fname: str, default='train.xyz'
        fname for fitting the first IP
    fit_param_fname: str, default None
        template ace_fit.jl
    ip_type: str, default "ace"
        "ace" or "gap"
    all_extra_smiles_csv: str, default None
        CSV of id and smiles to generate structures from
    md_temp: float, default 500
        temperature to run MD at
    energy_filter_threshold: float, default 0.05
        structures with total energy error above will be sampled and included in next
        training set
    ref_type: str, default "dft"
        "dft" or "dft-xtb2" for the type of reference energies/forces to
        fit to.
    wdir: wehere to base all the runs in.
    """

    assert ref_type in ["dft", "dft-xtb2"]
    assert ip_type in ["gap", "ace"]

    if ref_type == "dft-xtb2":
        raise NotImplementedError("haven't redone dft-xtb2 fitting")

    logger.info(
        f"Generating structures from {all_extra_smiles_csv}. "
        f"fit parameters from {fit_param_fname}. Fitting to "
        f"{ref_type} energies and forces."
    )

    cfg = Config.load()
    scratch_dir = cfg["scratch_path"]
    logger.info(f"Using: scratch_dir: {scratch_dir}")

    if ip_type == "gap":
        fit_exec_path = cfg["gap_fit_path"]
    elif ip_type == "ace":
        fit_exec_path = Path(wfl.__file__).parent.resolve() / 'scripts/ace_fit.jl'
    logger.info(f"Fit exec: {fit_exec_path}")

    wdir = Path(wdir)
    train_set_dir = wdir / "training_sets"
    train_set_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = wdir / "figures"
    figs_dir.mkdir(exist_ok=True)

    all_extra_smiles_csv = Path(all_extra_smiles_csv)
    if not (wdir / all_extra_smiles_csv.name).exists():
        it.manipulate_smiles_csv(all_extra_smiles_csv, wdir)
    all_extra_smiles_csv = wdir / all_extra_smiles_csv.name

    # setup orca parameters
    default_kw = Config.from_yaml(Path(util.__file__).parent /  "default_kwargs.yml")
    dft_prop_prefix = "dft_"
    orca_kwargs = default_kw["orca"]
    orca_kwargs["orca_command"] = cfg["orca_path"]
    orca_kwargs["scratch_path"] = scratch_dir
    logger.info(f"orca_kwargs: {orca_kwargs}")

    if ref_type == "dft":
        fit_to_prop_prefix = dft_prop_prefix
        pred_prop_prefix = ip_type + "_"
        # xtb2_prop_prefix = None
    if ref_type == "dft-xtb2":
        fit_to_prop_prefix = "dft_minus_xtb2_"
        pred_prop_prefix = "gap_plus_xtb2_"
        # xtb2_prop_prefix = "xtb2_"

    with open(fit_param_fname) as yaml_file:
        fit_params_base = yaml.safe_load(yaml_file)
    fit_params_base = it.update_fit_params(fit_params_base, fit_to_prop_prefix)
    logger.info(f"fit_params_base: {fit_params_base}")

    # md params
    # TODO revisit
    md_params = {
        "steps": md_steps,
        "dt": 0.5,  # fs
        "temperature": md_temp,  # K
        "temperature_tau": 200,  # fs, somewhat quicker than recommended (???)
        "traj_step_interval": 20,
        "results_prefix": pred_prop_prefix,
    }
    logger.info(f"MD params: {md_params}")

 
    initial_train_fname = train_set_dir / "train_for_fit_0.xyz"
    # if not initial_train_fname.exists():
    #     it.check_dft(base_train_fname, dft_prop_prefix=dft_prop_prefix, orca_kwargs=orca_kwargs, tests_wdir=wdir/"dft_check_wdir")

    # prepare 0th dataset
    ci = ConfigSet_in(input_files=base_train_fname)
    co = ConfigSet_out(output_files=initial_train_fname, force=True, all_or_none=True,
                       set_tags={"dataset_type": "train"})
    it.prepare_0th_dataset(ci, co)

    for cycle_idx in range(0, num_cycles + 1):

        logger.info("-" * 50)
        logger.info(f"ITERATION {cycle_idx}")

        # Check for the final training set from this iteration and skip if found.
        next_train_set_fname = (train_set_dir / f"{cycle_idx:02d}.train_for_{ip_type}_{cycle_idx+1:02d}.xyz")
        if os.path.exists(next_train_set_fname):
            logger.info(f"Found {next_train_set_fname}, skipping iteration {cycle_idx}")
            continue

        # define all the filenames
        cycle_dir = wdir / f"iteration_{cycle_idx:02d}"
        cycle_dir.mkdir(exist_ok=True)

        train_set_fname = (train_set_dir / f"{cycle_idx - 1:02d}.train_for_{ip_type}_{cycle_idx:02d}.xyz")

        if cycle_idx == 0:
            train_set_fname = initial_train_fname

        extra_smiles_for_this_cycle_csv = cycle_dir / "01.extra_smiles.csv"
        extra_md_starts_fname = cycle_dir / "02.rdkit_mols_rads.xyz"
        combined_md_starts_fname = cycle_dir / "03.md_starts.xyz"
        md_traj_fn = cycle_dir / f"04.{pred_prop_prefix[:-1]}.md_traj.xyz"
        selected_for_train_fn = cycle_dir / f"05.{pred_prop_prefix[:-1]}.md_traj.train_subselect.xyz"
        bad_mds_to_rerun_fn_name =  f"06.0.rdkit_mols_to_next_restart_configs.xyz"
        bad_mds_to_rerun_fn = cycle_dir / bad_mds_to_rerun_fn_name
        good_mds_starts_fn = cycle_dir / f"06.1.rdkit_mols_ok_mds.xyz"
        selected_for_train_fn_dft = cycle_dir / f"07.{pred_prop_prefix[:-1]}.md_traj.train_subselect.dft.xyz"

        # 1. Fit ace/gap
        # TODO: redo to ACE1pack

        fit_dir = cycle_dir / "fit_dir"
        fit_dir.mkdir(exist_ok=True)

        if ip_type == "gap":
            calculator = it.do_gap_fit(
                fit_dir=fit_dir,
                idx=cycle_idx,
                ref_type=ref_type,
                train_set_fname=train_set_fname,
                fit_params_base=fit_params_base,
                gap_fit_path=fit_exec_path,
            )

        elif ip_type == "ace":
            calculator = it.do_ace_fit(
                fit_dir=fit_dir,
                idx=cycle_idx,
                ref_type=ref_type,
                train_set_fname=train_set_fname,
                fit_params_base=fit_params_base,
                fit_to_prop_prefix=fit_to_prop_prefix,
                ace_fit_exec=fit_exec_path,
            )

        if ref_type == "dft":
            # pred_prop_prefix = f"{ip_type}{cycle_idx}_"
            pred_prop_prefix = f"{ip_type}_"
        elif ref_type == "dft-xtb2":
            # pred_prop_prefix = f"{ip_type}{cycle_idx}_plus_xtb2_"
            pred_prop_prefix = f"{ip_type}_plus_xtb2_"

        # 2. Run tests
        tests_wdir = cycle_dir / "tests"
        if not (tests_wdir / f"{pred_prop_prefix}on_{train_set_fname.name}").exists():
            it.check_dft(train_set_fname, "dft_", orca_kwargs, tests_wdir)
            logger.info("running_tests")
            with open(fit_dir / f"ace_{cycle_idx}_params.yaml") as f:
                fit_params = yaml.safe_load(f)
            ip.run_tests(
                calculator=calculator,
                pred_prop_prefix=pred_prop_prefix,
                dft_prop_prefix=dft_prop_prefix,
                train_set_fname=train_set_fname,
                # test_set_fname=test_set_fname,
                tests_wdir=tests_wdir,
                # bde_test_fname=bde_test_fname,
                orca_kwargs=orca_kwargs,
                output_dir = cycle_dir,
                validation_fname=validation_fname, 
                quick = True, 
                fit_params = fit_params)
        

        # 3. Select some smiles from the initial smiles csv
        if not extra_smiles_for_this_cycle_csv.exists():
            it.select_extra_smiles(
                all_extra_smiles_csv=all_extra_smiles_csv,
                smiles_selection_csv=extra_smiles_for_this_cycle_csv,
                chunksize=num_extra_smiles_per_cycle)

        # 4. Generate actual structures for md 
        logger.info("generating structures to work with")
        outputs = ConfigSet_out(
            output_files=extra_md_starts_fname,
            force=True,
            all_or_none=True,
            verbose=False,
            set_tags={"iter_no": cycle_idx, "config_type": "rdkit"})

        inputs = it.make_structures(
            extra_smiles_for_this_cycle_csv,
            num_smi_repeat=1,
            outputs=outputs,
            num_rads_per_mol=num_rads_per_mol)

        # 5. add structures from previous cycle
        if not combined_md_starts_fname.exists():
            if cycle_idx == 0: 
                structs_to_rerun = []
                pass
            else:
                previous_bad_md_to_rerun_fname = wdir / f"iteration_{cycle_idx-1:02d}" / bad_mds_to_rerun_fn_name
                if os.stat(previous_bad_md_to_rerun_fname).st_size != 0:
                    structs_to_rerun = read(previous_bad_md_to_rerun_fname, ":")
                else:
                    structs_to_rerun = []

            extra_md_starts = read(extra_md_starts_fname, ":")
            write(combined_md_starts_fname, structs_to_rerun + extra_md_starts)
    
        # 6. Run MD and select needed configs 
        outputs_to_fit = ConfigSet_out(output_files=selected_for_train_fn,
                                force=True, 
                                all_or_none=True,
                                set_tags={"fit_config_type": f"selected_from_{pred_prop_prefix}md"})
        outputs_traj = ConfigSet_out(output_files=md_traj_fn, 
                                     force=True, 
                                     all_or_none=True,
                                     set_tags={"fit_config_type":f"{pred_prop_prefix}md"})
        outputs_rerun = ConfigSet_out(output_files=bad_mds_to_rerun_fn, 
                                     force=True, 
                                     all_or_none=True,
                                     set_tags={"fit_config_type":f"md_to_restart"})
        outputs_good_md = ConfigSet_out(output_files=good_mds_starts_fn,
                                        force=True,
                                        all_or_none=True,
                                        set_tags={"fit_config_type":f"ok_md"})

        inputs = it.launch_analyse_md(
            inputs=inputs,
            pred_prop_prefix=pred_prop_prefix,
            outputs_to_fit=outputs_to_fit, 
            outputs_traj=outputs_traj,
            outputs_rerun=outputs_rerun,
            outputs_good_md=outputs_good_md,
            calculator=calculator, 
            md_params=md_params)
    

        outputs = ConfigSet_out(output_files=selected_for_train_fn_dft,
            force=True, all_or_none=True, set_tags={"dataset_type": "train", "iter_no": cycle_idx + 1})

        inputs = orca.evaluate(
            inputs=inputs,
            outputs=outputs,
            orca_kwargs=orca_kwargs,
            output_prefix=dft_prop_prefix,
            keep_files=False,
            base_rundir=cycle_dir / "orca_wdir_on_extra_train",
        )

        # 14. do summary plots
        # it.summary_plots(
        #     cycle_idx,
        #     pred_prop_prefix=pred_prop_prefix,
        #     dft_prop_prefix=dft_prop_prefix,
        #     train_fname=train_set_fname,
        #     test_fname=test_set_fname,
        #     bde_fname=bde_test_fname,
        #     ip_optimised_fname=good_opt_for_md_w_dft_fn,
        #     train_extra_fname=selected_for_train_fn_dft,
        #     test_extra_fname=test_extra_fname_dft,
        #     tests_wdir=tests_wdir,
        # )

        # it.combine_plots(pred_prop_prefix=pred_prop_prefix, dft_prop_prefix=dft_prop_prefix, tests_wdir=tests_wdir, cycle_idx=cycle_idx, figs_dir=figs_dir, wdir=wdir)

        # 15. Combine datasets
        if not next_train_set_fname.exists():
            logger.info("combining old and extra data")

            previous_train = read(train_set_fname, ":")
            extra_train = read(selected_for_train_fn_dft, ":")
            write(next_train_set_fname, previous_train + extra_train)

        logger.info(f"cycle {cycle_idx} done. ")


    logger.info("Finished iterations")

    # it.summary_plots(
    #     cycle_idx,
    #     pred_prop_prefix=pred_prop_prefix,
    #     dft_prop_prefix=dft_prop_prefix,
    #     train_fname=train_set_fname,
    #     test_fname=test_set_fname,
    #     bde_fname=bde_test_fname,
    #     ip_optimised_fname=good_opt_for_md_w_dft_fn,
    #     train_extra_fname=selected_for_train_fn_dft,
    #     test_extra_fname=test_extra_fname_dft,
    #     tests_wdir=tests_wdir)
