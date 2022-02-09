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
    energy_error_per_atom_threshold=0.05,
    energy_error_total_threshold=None,
    max_f_comp_error_threshold=None,
    wdir="runs",
    ref_type="dft",
    ip_type="ace",
    bde_test_fname="dft_bde.xyz",
    soap_params_for_cur_fname="soap_params_for_cur.xyz",
    num_train_environments_per_cycle=10,
    num_test_environments_per_cycle=10,
    num_extra_smiles_per_cycle=10,
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
    with open(soap_params_for_cur_fname, "r") as f:
        soap_params_for_cur = yaml.safe_load(f)

    initial_train_fname = train_set_dir / "train_for_fit_0.xyz"
    if not initial_train_fname.exists():
        it.check_dft(base_train_fname, dft_prop_prefix=dft_prop_prefix, orca_kwargs=orca_kwargs, tests_wdir=wdir/"dft_check_wdir")

    # prepare 0th dataset
    ci = ConfigSet_in(input_files=base_train_fname)
    co = ConfigSet_out(output_files=initial_train_fname, force=True, all_or_none=True,
                       set_tags={"dataset_type":"train"})
    it.prepare_0th_dataset(ci, co)

    initial_test_fname = train_set_dir / "test_for_fit_0.xyz"
    ci = ConfigSet_in(input_files=base_test_fname)
    co = ConfigSet_out(output_files=initial_test_fname, force=True, all_or_none=True,
                       set_tags={"dataset_type":"test"})
    it.prepare_0th_dataset(ci, co)



    for cycle_idx in range(0, num_cycles + 1):

        logger.info("-"*50)
        logger.info(f"ITERATION {cycle_idx}")

        # Check for the final training set from this iteration and skip if found.
        next_train_set_fname = (train_set_dir / f"{cycle_idx:02d}.train_for_{ip_type}_{cycle_idx+1:02d}.xyz")
        next_test_set_fname = (train_set_dir / f"{cycle_idx:02d}.test_for_{ip_type}_{cycle_idx+1:02d}.xyz")
        if os.path.exists(next_train_set_fname):
            logger.info(f"Found {next_train_set_fname}, skipping iteration {cycle_idx}")
            continue

        # define all the filenames
        cycle_dir = wdir / f"iteration_{cycle_idx:02d}"
        cycle_dir.mkdir(exist_ok=True)

        train_set_fname = (train_set_dir / f"{cycle_idx - 1:02d}.train_for_{ip_type}_{cycle_idx:02d}.xyz")
        test_set_fname = (train_set_dir / f"{cycle_idx - 1:02d}.test_for_{ip_type}_{cycle_idx:02d}.xyz")

        if cycle_idx == 0:
            train_set_fname = initial_train_fname
            test_set_fname = initial_test_fname

        extra_smiles_for_this_cycle_csv = cycle_dir / "02.extra_smiles.csv"
        opt_starts_fname = cycle_dir / "03.rdkit_mols_rads.xyz"
        opt_traj_fn = cycle_dir / f"04.0.rdkit.{pred_prop_prefix}opt_traj.xyz"
        opt_traj_evaled_fname = cycle_dir / f"04.1.rdkit.{pred_prop_prefix}opt_traj.{pred_prop_prefix[:-1]}.xyz"
        good_opt_for_md_fname = cycle_dir / f"05.0.rdkit.{pred_prop_prefix}optimised.xyz"
        bad_opt_traj_bad_configs_fname = cycle_dir / f"05.1.rdkit.{pred_prop_prefix}bad_opt_traj_bad_configs.xyz"
        bad_opt_traj_good_configs_fn = cycle_dir / f"05.2.rdkit.{pred_prop_prefix}bad_opt_traj_good_configs.xyz"
        bad_opt_traj_good_configs_sample_for_train = cycle_dir / f"06.0.rdkit.{pred_prop_prefix}bad_opt_traj_good_configs.sample_for_train.xyz"
        # bad_opt_traj_good_configs_sample_for_train_dft = cycle_dir / f"06.1.rdkit.{pred_prop_prefix}bad_opt_traj_good_configs.sample_for_train.dft.xyz"
        # opt_filtered_fname = cycle_dir / f"05.1.rdkit.{pred_prop_prefix}optimised.good_geometries.xyz"
        # bad_structures_fname = cycle_dir / f"05.2.rdkit.{pred_prop_prefix}optimised.bad_geometries.xyz"
        good_opt_for_md_w_dft_fn = cycle_dir / f"07.rdkit.{pred_prop_prefix}optimised.good_geometries.dft.xyz"
        large_error_configs = cycle_dir / f"08.1.rdkit.{pred_prop_prefix}optimised.good_geometries.dft.large_error.xyz"
        small_error_configs = cycle_dir / f"08.2.rdkit.{pred_prop_prefix}optimised.good_geometries.dft.small_error.xyz"
        full_md_fname_sanitised = cycle_dir / f"09.{pred_prop_prefix}optimised.sanitised.xyz"
        md_traj_fn = cycle_dir / f"10.0.{pred_prop_prefix}optimised.md_traj.xyz"
        md_traj_evaled_fn = cycle_dir / f"10.1.{pred_prop_prefix}optimised.md_traj.{pred_prop_prefix[:-1]}.xyz"
        good_md_to_sample_fn = cycle_dir / f"11.0.{pred_prop_prefix}optimised.md.xyz"
        bad_md_bad_configs_fn = cycle_dir / f"11.1.{pred_prop_prefix}bad_md_traj_bad_configs.xyz"
        bad_md_good_configs_fn = cycle_dir / f"11.2.{pred_prop_prefix}bad_md_traj_good_configs.xyz"
        bad_md_good_configs_sample_train_fn = cycle_dir / f"12.0.{pred_prop_prefix}bad_md_traj_good_configs.sample_for_train.xyz"
        # bad_md_good_configs_sample_train_dft_fn = cycle_dir / f"12.1.{pred_prop_prefix}bad_md_traj_good_configs.sample_for_train.dft.xyz"
        # full_md_fname = cycle_dir / f"08.3.{pred_prop_prefix}optimised.md.{pred_prop_prefix[:-1]}.xyz"
        # full_md_good_geometries_fname = (cycle_dir / f"09.1.{pred_prop_prefix}optimised.md.good_geometries.xyz")
        # full_md_bad_geometries_fname = (cycle_dir / f"09.2.{pred_prop_prefix}optimised.md.bad_geometries.xyz")
        md_with_soap_fname = cycle_dir / f"13.{pred_prop_prefix}optimised.md.good_geometries.soap.xyz"
        test_md_selection_fname = cycle_dir / f"14.1.{pred_prop_prefix}optimised.md.test_sample.xyz"
        train_md_selection_fname = cycle_dir / f"14.2.{pred_prop_prefix}optimised.md.train_sample.xyz"
        test_extra_fname_dft = cycle_dir / f"15.1.{pred_prop_prefix}optimised.md.test_sample.dft.xyz"
        train_extra_fname_dft = cycle_dir / f"15.2.{pred_prop_prefix}optimised.md.extra_train_configs.dft.xyz"

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
        # if not (tests_wdir / f"{pred_prop_prefix}bde_file_with_errors.xyz").exists():
        logger.info("running_tests")
        it.run_tests(
            calculator=calculator,
            pred_prop_prefix=pred_prop_prefix,
            dft_prop_prefix=dft_prop_prefix,
            train_set_fname=train_set_fname,
            test_set_fname=test_set_fname,
            tests_wdir=tests_wdir,
            bde_test_fname=bde_test_fname,
            orca_kwargs=orca_kwargs,
            output_dir = cycle_dir,
        )

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
            output_files=opt_starts_fname,
            force=True,
            all_or_none=True,
            verbose=False,
            set_tags={"iter_no": cycle_idx, "config_type": "rdkit"},
        )

        inputs = it.make_structures(
            extra_smiles_for_this_cycle_csv,
            num_smi_repeat=1,
            outputs=outputs,
            num_rads_per_mol=1,
        )

        # 5. optimise structures with current IP and re-evaluate them
        logger.info(f"optimising structures from {opt_starts_fname} with {ip_type}")
        outputs = ConfigSet_out(
            output_files=opt_traj_fn,
            force=True,
            all_or_none=True,
            set_tags={"dataset_type": f"next_rdkit", "config_type": "opt_traj"},
        )
        # traj_step_interval=None selects only last converged config.
        # traj_step_interval = 1 returns all configs
        inputs = opt.optimise(
            inputs=inputs,
            outputs=outputs,
            calculator=calculator,
            prop_prefix=pred_prop_prefix,
            traj_step_interval=1,
            npool=None
        )
        # need to re-evaluate again, because energy is sometimes not written
        outputs = ConfigSet_out(
            output_files=opt_traj_evaled_fname,
            force=True, 
            all_or_none=True)
        inputs = generic.run(inputs=inputs, 
                             outputs=outputs,
                             calculator=calculator, 
                             properties=["energy", "forces"], 
                             output_prefix=pred_prop_prefix, 
                             npool=None)

        # 6. process optimisation trajectories: check for bad geometry and process accordingly
        good_traj_configs_co = ConfigSet_out(output_files=good_opt_for_md_fname,
                                   force=True, all_or_none=True,
                                   set_tags={"dataset_type": f"next_rdkit_{pred_prop_prefix}opt", "config_type": "optimised"})
        bad_traj_bad_cfg_co = ConfigSet_out (output_files=bad_opt_traj_bad_configs_fname,
                                          force=True, all_or_none=True,
                                          set_tags={"config_type": "bad_opt_bad_config"})
        bad_traj_good_cfg_co = ConfigSet_out(output_files=bad_opt_traj_good_configs_fn,
                                          force=True, all_or_none=True,
                                          set_tags={"config_type": "bad_opt_good_config"})
        it.process_trajs(
            traj_ci=inputs, 
            good_traj_configs_co=good_traj_configs_co,
            bad_traj_bad_cfg_co=bad_traj_bad_cfg_co,
            bad_traj_good_cfg_co=bad_traj_good_cfg_co, 
            traj_sample_rule="last") 

        # 7. slight digression - sub-sample the good geometries from geometry-failed trajectory
        if os.stat(bad_opt_traj_good_configs_fn).st_size != 0:
            bad_traj_good_cfgs_ci = ConfigSet_in(input_files=bad_opt_traj_good_configs_fn)
            sample_co = ConfigSet_out(output_files=bad_opt_traj_good_configs_sample_for_train,
                                            force=True, all_or_none=True)
            it.sample_failed_trajectory(ci=bad_traj_good_cfgs_ci, co=sample_co)


        # 8. evaluate DFT
        logger.info("evaluatig dft on optimised structures")
        outputs = ConfigSet_out(output_files=good_opt_for_md_w_dft_fn, force=True, all_or_none=True)
        inputs = ConfigSet_in(input_files=good_opt_for_md_fname)
        inputs = orca.evaluate(
            inputs=inputs,
            outputs=outputs,
            orca_kwargs=orca_kwargs,
            output_prefix=dft_prop_prefix,
            keep_files=False,
            base_rundir=cycle_dir / "orca_wdir_on_opt")

        # report errors on optimised structures
        logger.info("Errors on optimised structures:")
        error_table.plot(inputs, ref_prefix=dft_prop_prefix, pred_prefix=pred_prop_prefix, info_key="graph_name")

        # 9. filter by energy and force error
        logger.info("Filtering by energy and force errors")
        outputs_large_error = ConfigSet_out(output_files=large_error_configs, force=True, all_or_none=True)
        outputs_small_error = ConfigSet_out(output_files=small_error_configs, force=True, all_or_none=True)
        inputs = it.filter_configs(
            inputs=inputs,
            outputs_large_error=outputs_large_error,
            outputs_small_error=outputs_small_error,
            pred_prop_prefix=pred_prop_prefix,
            e_threshold_per_atom=energy_error_per_atom_threshold,
            e_threshold_total=energy_error_total_threshold,
            max_f_comp_threshold=max_f_comp_error_threshold,
        )
        if inputs is None:
            logger.warning("Found no optimised geometries with large errors, stopping cycles")
            return None

        logger.info(f'{len(read(large_error_configs, ":"))/len(read(good_opt_for_md_fname, ":"))*100:.1f}% structures have large error; running MD on them')
        if len(read(large_error_configs, ':')) == 0:
            logger.info("all new config energy/force evaluations were within threshold, done with iterations?")
            break


        # 10.0 remove old energies and forces
        if not full_md_fname_sanitised.exists():
            logger.info("removing energy&force entries pre-md")
            sanitised_ats = [remove_energy_force_containing_entries(at) for at in inputs]
            write(full_md_fname_sanitised, sanitised_ats)
        inputs = ConfigSet_in(input_files=full_md_fname_sanitised)


        # 10. Run MD
        logger.info(f"Running {ip_type} md")
        outputs = ConfigSet_out(
            output_files=md_traj_fn,
            force=True,
            all_or_none=True,
            set_tags={"config_type": f"{pred_prop_prefix}md"},
        )
        inputs = md.sample(
            inputs=inputs,
            outputs=outputs,
            calculator=calculator,
            verbose=False,
            npool=None,
            **md_params,
        )
        # need to re-evaluate again, because energy is sometimes not written
        outputs = ConfigSet_out(
            output_files=md_traj_evaled_fn,
            force=True, 
            all_or_none=True)
        inputs = generic.run(inputs=inputs, 
                             outputs=outputs,
                             calculator=calculator, 
                             properties=["energy", "forces"], 
                             output_prefix=pred_prop_prefix,
                             npool=None)



        # # 9.1 sample every 100-th config
        # if not full_md_fname.exists():
        #     # put into list of trajectories
        #     all_trajs = []
        #     this_traj = None
        #     for at in inputs:
        #         if at.info["MD_time_fs"]==0: 
        #             if this_traj is not None:
        #                 all_trajs.append(this_traj)
        #             this_traj = []
        #         this_traj.append(at)
            
        #     sample = []
        #     for traj in all_trajs:
        #         sample += traj[::100]
            
        #     write(full_md_fname, sample)

        # inputs = ConfigSet_in(input_files=full_md_fname)

        # # 10. Filter/check for bad geometries
        # outputs_good = ConfigSet_out(output_files=full_md_good_geometries_fname, force=True, all_or_none=True)
        # outputs_bad = ConfigSet_out(output_files=full_md_bad_geometries_fname, force=True, all_or_none=True)
        # inputs = it.filter_configs_by_geometry(inputs=inputs, outputs_good=outputs_good, outputs_bad=outputs_bad)

        # # handle if some configs were found
        # if os.stat(full_md_bad_geometries_fname).st_size == 0:
        #     logger.info("all configs from md are good")
        # else:
        #     num_bad_configs = len([at for at in outputs_bad.to_ConfigSet_in()])
        #     num_good_configs = len([at for at in outputs_good.to_ConfigSet_in()])
        #     if num_bad_configs / (num_bad_configs + num_good_configs) > 0.1:
        #         # raise RuntimeWarning("Too many bad geometries from MD")
        #         logger.warning("Many bad geometries from MD!!!!")
        #     elif num_bad_configs != 0:
        #         logger.warning(f"Some had {num_bad_configs} bad geometries from md")

        # 11. process optimisation trajectories: check for bad geometry and process accordingly
        good_traj_configs_co = ConfigSet_out(output_files=good_md_to_sample_fn,
                                   force=True, all_or_none=True,
                                   set_tags={"dataset_type": f"next_rdkit_{pred_prop_prefix}_md", "config_type": "md"})
        bad_traj_bad_cfg_co = ConfigSet_out(output_files=bad_md_bad_configs_fn,
                                          force=True, all_or_none=True,
                                          set_tags={"config_type": "bad_opt_bad_config"})
        bad_traj_good_cfg_co = ConfigSet_out(output_files=bad_md_good_configs_fn,
                                          force=True, all_or_none=True,
                                          set_tags={"config_type": "bad_md_good_config"})
        it.process_trajs(
            traj_ci=inputs, 
            good_traj_configs_co=good_traj_configs_co,
            bad_traj_bad_cfg_co=bad_traj_bad_cfg_co,
            bad_traj_good_cfg_co=bad_traj_good_cfg_co, 
            traj_sample_rule=100) 

        # 12. slight digression - sub-sample the good geometries from geometry-failed trajectory
        bad_traj_good_cfgs_ci = ConfigSet_in(input_files=bad_md_good_configs_fn)
        sample_co = ConfigSet_out(output_files=bad_md_good_configs_sample_train_fn,
                                          force=True, all_or_none=True)
        it.sample_failed_trajectory(ci=bad_traj_good_cfgs_ci, co=sample_co)


        # 13. Calculate soap descriptor
        logger.info("Calculating SOAP descriptor")
        inputs = ConfigSet_in(input_files=good_md_to_sample_fn)
        outputs = ConfigSet_out(output_files=md_with_soap_fname, force=True, all_or_none=True)
        inputs = wfl.calc_descriptor.calc(
            inputs=inputs,
            outputs=outputs,
            descs=soap_params_for_cur,
            key="small_soap",  # where to store the descriptor
            local=True,
        )  # calculate local descriptor

        # 12. do CUR
        if not train_md_selection_fname.exists():
            logger.info("Selecting configs with CUR")
            outputs = ConfigSet_out(set_tags={"dataset_type": "next_addition_from_md"})
            inputs = cur.per_environment(
                inputs=inputs,
                outputs=outputs,
                num=num_train_environments_per_cycle + num_test_environments_per_cycle,
                at_descs_key="small_soap",
                kernel_exp=3,
                leverage_score_key="cur_leverage_score",
                write_all_configs=False,
            )
            selected_with_cur = list(inputs)
            for at in selected_with_cur:
                del at.arrays["small_soap"]
            random.shuffle(selected_with_cur)
            write(train_md_selection_fname, selected_with_cur[0::2],)
            write(test_md_selection_fname, selected_with_cur[1::2])
        

        # 13. evaluate DFT
        logger.info("evaluatig dft structures selected for next training set")
        
        # first: test set
        inputs = ConfigSet_in(input_files=test_md_selection_fname)
        outputs = ConfigSet_out(output_files=test_extra_fname_dft, force=True, all_or_none=True, 
                                set_tags={"dataset_type": "next_test"})
        orca.evaluate(
            inputs=inputs,
            outputs=outputs,
            orca_kwargs=orca_kwargs,
            output_prefix=dft_prop_prefix,
            keep_files=False,
            base_rundir=cycle_dir / "orca_wdir_on_extra_test")

        # next: all extra for training set
        inputs = ConfigSet_in(
            input_files=[train_md_selection_fname,
                         bad_opt_traj_good_configs_sample_for_train,
                         bad_md_good_configs_sample_train_fn])
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
        it.summary_plots(
            cycle_idx,
            pred_prop_prefix=pred_prop_prefix,
            dft_prop_prefix=dft_prop_prefix,
            train_fname=train_set_fname,
            test_fname=test_set_fname,
            bde_fname=bde_test_fname,
            ip_optimised_fname=good_opt_for_md_w_dft_fn,
            train_extra_fname=train_extra_fname_dft,
            test_extra_fname=test_extra_fname_dft,
            tests_wdir=tests_wdir,
        )

        it.combine_plots(pred_prop_prefix=pred_prop_prefix, dft_prop_prefix=dft_prop_prefix, tests_wdir=tests_wdir, cycle_idx=cycle_idx, figs_dir=figs_dir, wdir=wdir)

        # 15. Combine datasets
        if not next_train_set_fname.exists():
            logger.info("combining old and extra data")
            previous_train = read(train_set_fname, ":")
            previous_test = read(test_set_fname, ":")
            extra_train = read(train_extra_fname_dft, ":")
            extra_test = read(test_extra_fname_dft, ":")

            for at in extra_train:
                at.info["dataset_type"] = "train"
                at.info["iter_no"] = cycle_idx + 1
            for at in extra_test:
                at.info["dataset_type"] = "test"
                at.info["iter_no"] = cycle_idx + 1

            write(next_train_set_fname, previous_train + extra_train)
            write(next_test_set_fname, previous_test + extra_test)

        logger.info(f"cycle {cycle_idx} done. ")


    logger.info("Finished iterations")

    it.summary_plots(
        cycle_idx,
        pred_prop_prefix=pred_prop_prefix,
        dft_prop_prefix=dft_prop_prefix,
        train_fname=train_set_fname,
        test_fname=test_set_fname,
        bde_fname=bde_test_fname,
        ip_optimised_fname=good_opt_for_md_w_dft_fn,
        train_extra_fname=train_extra_fname_dft,
        test_extra_fname=test_extra_fname_dft,
        tests_wdir=tests_wdir)
