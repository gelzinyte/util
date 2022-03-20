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
from util.configs import cur
from util import error_table

logger = logging.getLogger(__name__)


def fit(
    num_cycles,
    base_train_fname="train.xyz",
    base_test_fname="test.xyz",
    fit_param_fname=None,
    all_extra_smiles_csv=None,
    md_temp=500,
    wdir="runs",
    ref_type="dft",
    ip_type="ace",
    bde_test_fname="dft_bde.xyz",
    soap_params_for_cur_fname="soap_params_for_cur.xyz",
    num_train_environments_per_cycle=10,
    num_test_environments_per_cycle=10,
    num_extra_smiles_per_cycle=10,
    num_rads_per_mol=0
):
    """ iteratively fits potetnials

    No files are created or modified in the original directory, everything's done in "wdir"

    * Each iteration is run in a directory (e.g."wdir"/"iteration_03").
    * An iteration starts with a fit to a training set file from the previous iteration
        ("02_train_for_ACE_03.xyz").
    * Then run tests with this potential:
        - Accuracy on train set and similarly generated test set
        - BDE test on tracked configs
        - (dimer curves)?
        - offset in training/testing configs
        - removal of hydrogen
        - error on the configs vs bde error
    * Generate initial (rdkit) configs
        - slice a slice from all of the extra csv
        - generate 3D structures
    * Potential-optimise
    * Select configs with high error
    * Run MD at a given temperature
        - look out for unreasonable structures
            - filter geometry
            - (check for weird energy drifts)
    * CUR-select new configs
        - BASED ON SOAP?!
    * Accuracy summary on things that don't need extra dft evaluations
        - train set
        - test set
        - potential - optimised structures
        - structures selected from md
        - dft-optimised bde structures
        - potential-reoptimised bde structures
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
    * select configs based on ACE basis?
    * make sure dataset and config types are correctly assinged
    * check that wherever I am using configset out, outputs are skipped if done.
    * will md run remotely?
    * what happens if ConfigSet_out is done, so action isn't performed and then I try to access co.to_ConfigSet_in()
    * do configs have unique identifier?
    * 2b plot 

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
        f"Optimising structures from {all_extra_smiles_csv}. "
        f"fit parameters from {fit_param_fname}. Fitting to "
        f"{ref_type} energies and forces."
    )

    cfg = Config.load()
    scratch_dir = cfg["scratch_path"]
    logger.info(f"Using:scratch_dir: {scratch_dir}")

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
    default_kw = Config.from_yaml(os.path.join(cfg["util_root"], "default_kwargs.yml"))
    dft_prop_prefix = "dft_"
    orca_kwargs = default_kw["orca"]
    orca_kwargs["orca_command"] = cfg["orca_path"]
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
        "steps": 2000,
        "dt": 0.5,  # fs
        "temperature": md_temp,  # K
        "temperature_tau": 200,  # fs, somewhat quicker than recommended (???)
        "traj_step_interval": 2,
        "results_prefix": pred_prop_prefix,
    }
    logger.info(f"MD params: {md_params}")

    # soap descriptor for cur params
    # with open(soap_params_for_cur_fname, "r") as f:
    #     soap_params_for_cur = yaml.safe_load(f)

    initial_train_fname = train_set_dir / "train_for_fit_0.xyz"
    # if not initial_train_fname.exists():
    #     it.check_dft(base_train_fname, dft_prop_prefix=dft_prop_prefix, orca_kwargs=orca_kwargs, tests_wdir=wdir/"dft_check_wdir")

    # prepare 0th dataset
    ci = ConfigSet_in(input_files=base_train_fname)
    co = ConfigSet_out(output_files=initial_train_fname, force=True, all_or_none=True,
                       set_tags={"dataset_type": "train"})
    it.prepare_0th_dataset(ci, co)

    # if base_test_fname is not None:
    #     initial_test_fname = train_set_dir / "test_for_fit_0.xyz"
    #     ci = ConfigSet_in(input_files=base_test_fname)
    #     co = ConfigSet_out(output_files=initial_test_fname, force=True, all_or_none=True,
    #                        set_tags={"dataset_type": "test"})
    #     it.prepare_0th_dataset(ci, co)
    # else:
    #     initial_test_fname = None

    for cycle_idx in range(0, num_cycles + 1):

        logger.info("-" * 50)
        logger.info(f"ITERATION {cycle_idx}")

        # Check for the final training set from this iteration and skip if found.
        next_train_set_fname = (train_set_dir / f"{cycle_idx:02d}.train_for_{ip_type}_{cycle_idx+1:02d}.xyz")
        # next_test_set_fname = (train_set_dir / f"{cycle_idx:02d}.test_for_{ip_type}_{cycle_idx+1:02d}.xyz")
        if os.path.exists(next_train_set_fname):
            logger.info(f"Found {next_train_set_fname}, skipping iteration {cycle_idx}")
            continue

        # define all the filenames
        cycle_dir = wdir / f"iteration_{cycle_idx:02d}"
        cycle_dir.mkdir(exist_ok=True)

        train_set_fname = (train_set_dir / f"{cycle_idx - 1:02d}.train_for_{ip_type}_{cycle_idx:02d}.xyz")
        # test_set_fname = (train_set_dir / f"{cycle_idx - 1:02d}.test_for_{ip_type}_{cycle_idx:02d}.xyz")

        if cycle_idx == 0:
            train_set_fname = initial_train_fname
            # test_set_fname = initial_test_fname

        extra_smiles_for_this_cycle_csv = cycle_dir / "01.extra_smiles.csv"
        md_starts_fname = cycle_dir / "02.rdkit_mols_rads.xyz"
        md_traj_fn = cycle_dir / f"03.0.{pred_prop_prefix}optimised.md_traj.xyz"
        md_traj_evaled_fn = cycle_dir / f"03.1.{pred_prop_prefix}optimised.md_traj.{pred_prop_prefix[:-1]}.xyz"
        good_md_to_sample_fn = cycle_dir / f"04.0.{pred_prop_prefix}optimised.md.xyz"
        bad_md_bad_configs_fn = cycle_dir / f"04.1.{pred_prop_prefix}bad_md_traj_bad_configs.xyz"
        bad_md_good_configs_fn = cycle_dir / f"04.2.{pred_prop_prefix}bad_md_traj_good_configs.xyz"
        bad_md_good_configs_sample_train_fn = cycle_dir / f"05.0.{pred_prop_prefix}bad_md_traj_good_configs.sample_for_train.xyz"
        md_with_soap_fname = cycle_dir / f"06.{pred_prop_prefix}optimised.md.good_geometries.soap.xyz"
        test_md_selection_fname = cycle_dir / f"07.1.{pred_prop_prefix}optimised.md.test_sample.xyz"
        train_md_selection_fname = cycle_dir / f"07.2.{pred_prop_prefix}optimised.md.train_sample.xyz"
        test_extra_fname_dft = cycle_dir / f"08.1.{pred_prop_prefix}optimised.md.test_sample.dft.xyz"
        train_extra_fname_dft = cycle_dir / f"08.2.{pred_prop_prefix}optimised.md.extra_train_configs.dft.xyz"

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
        # it.check_dft(train_set_fname, "dft_", orca_kwargs, tests_wdir)
        # if not (tests_wdir / f"{pred_prop_prefix}bde_file_with_errors.xyz").exists():
        #     logger.info("running_tests")
        #     it.run_tests(
        #         calculator=calculator,
        #         pred_prop_prefix=pred_prop_prefix,
        #         dft_prop_prefix=dft_prop_prefix,
        #         train_set_fname=train_set_fname,
        #         test_set_fname=test_set_fname,
        #         tests_wdir=tests_wdir,
        #         bde_test_fname=bde_test_fname,
        #         orca_kwargs=orca_kwargs,
        #         output_dir = cycle_dir,
        #     )
        

        # 3. Select some smiles from the initial smiles csv
        if not extra_smiles_for_this_cycle_csv.exists():
            it.select_extra_smiles(
                all_extra_smiles_csv=all_extra_smiles_csv,
                smiles_selection_csv=extra_smiles_for_this_cycle_csv,
                chunksize=num_extra_smiles_per_cycle,
            )

        # 4. Generate actual structures for optimisation
        logger.info("generating structures to optimise")
        outputs = ConfigSet_out(
            output_files=md_starts_fname,
            force=True,
            all_or_none=True,
            verbose=False,
            set_tags={"iter_no": cycle_idx, "config_type": "rdkit"},
        )

        inputs = it.make_structures(
            extra_smiles_for_this_cycle_csv,
            num_smi_repeat=1,
            outputs=outputs,
            num_rads_per_mol=num_rads_per_mol,
        )

  
        # TODO 
        if not tests_wdir:
            raise RuntimeError("need to run MD")

        # # process optimisation trajectories: check for bad geometry and process accordingly
        # good_traj_configs_co = ConfigSet_out(output_files=good_md_to_sample_fn,
        #                            force=True, all_or_none=True,
        #                            set_tags={"dataset_type": f"next_rdkit_{pred_prop_prefix}_md", "config_type": "md"})
        # bad_traj_bad_cfg_co = ConfigSet_out(output_files=bad_md_bad_configs_fn,
        #                                   force=True, all_or_none=True,
        #                                   set_tags={"config_type": "bad_opt_bad_config"})
        # bad_traj_good_cfg_co = ConfigSet_out(output_files=bad_md_good_configs_fn,
        #                                   force=True, all_or_none=True,
        #                                   set_tags={"config_type": "bad_md_good_config"})
        # it.process_trajs(
        #     traj_ci=inputs, 
        #     good_traj_configs_co=good_traj_configs_co,
        #     bad_traj_bad_cfg_co=bad_traj_bad_cfg_co,
        #     bad_traj_good_cfg_co=bad_traj_good_cfg_co, 
        #     traj_sample_rule=100) 

        # 13. evaluate DFT
        logger.info("evaluatig dft structures selected for next training set")
        


        # next: all extra for training set
        input_files = [train_md_selection_fname]

        # if bad_opt_traj_good_configs_sample_for_train.exists() and \
            # os.stat(bad_opt_traj_good_configs_sample_for_train).st_size != 0:
            # input_files.append(bad_opt_traj_good_configs_sample_for_train)

        if bad_md_good_configs_sample_train_fn.exists() and \
            os.stat(bad_md_good_configs_sample_train_fn).st_size != 0:
            input_files.append(bad_md_good_configs_sample_train_fn)

        inputs = ConfigSet_in(input_files=input_files)
        outputs = ConfigSet_out(output_files=train_extra_fname_dft,
            force=True, all_or_none=True, set_tags={"dataset_type": "next_train"})

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
        #     train_extra_fname=train_extra_fname_dft,
        #     test_extra_fname=test_extra_fname_dft,
        #     tests_wdir=tests_wdir,
        # )

        # it.combine_plots(pred_prop_prefix=pred_prop_prefix, dft_prop_prefix=dft_prop_prefix, tests_wdir=tests_wdir, cycle_idx=cycle_idx, figs_dir=figs_dir, wdir=wdir)

        # 15. Combine datasets
        if not next_train_set_fname.exists():
            logger.info("combining old and extra data")

            previous_train = read(train_set_fname, ":")
            # if test_set_fname is not None:
            #     previous_test = read(test_set_fname, ":")
            # else: 
            #     previous_test = []

            extra_train = read(train_extra_fname_dft, ":")
            # extra_test = read(test_extra_fname_dft, ":")

            for at in extra_train:
                at.info["dataset_type"] = "train"
                at.info["iter_no"] = cycle_idx + 1
            # for at in extra_test:
            #     at.info["dataset_type"] = "test"
            #     at.info["iter_no"] = cycle_idx + 1

            write(next_train_set_fname, previous_train + extra_train)
            # write(next_test_set_fname, previous_test + extra_test)

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
    #     train_extra_fname=train_extra_fname_dft,
    #     test_extra_fname=test_extra_fname_dft,
    #     tests_wdir=tests_wdir)
