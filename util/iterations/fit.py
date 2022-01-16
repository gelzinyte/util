import os
import shutil
import yaml
import logging
import random

from pathlib import Path

from ase.io import read, write

from wfl.calculators import orca
from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.generate_configs import md 
import wfl.calc_descriptor.calc

from util import opt
from util.util_config import Config
from util.iterations import tools as it
from util.configs import cur

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
    num_train_configs_per_cycle=10, 
    num_test_configs_per_cycle=10,
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
        f"GAP parameters from {fit_param_fname}. Fitting to "
        f"{ref_type} energies and forces."
    )

    cfg = Config.load()
    scratch_dir = cfg["scratch_path"]
    logger.info(f"Using:scratch_dir: {scratch_dir}")

    if ip_type == "gap":
        fit_exec_path = cfg["gap_fit_path"]
    elif ip_type == "ace":
        # TODO: do somehow differently?
        fit_exec_path = cfg["ace_fit_path"]
    logger.info(f"Fit exec: {fit_exec_path}")

    wdir = Path(wdir)
    train_set_dir = wdir / "training_sets"
    train_set_dir.make_dirs(parents=True, exists_ok=True)

    shutil.copy(all_extra_smiles_csv, wdir / all_extra_smiles_csv)
    all_extra_smiles_csv = wdir / all_extra_smiles_csv

    # setup orca parameters
    default_kw = Config.from_yaml(os.path.join(cfg["util_root"], "default_kwargs.yml"))

    dft_prop_prefix = "dft_"
    orca_kwargs = default_kw["orca"]
    logger.info(f"orca_kwargs: {orca_kwargs}")
    logger.info(f'orca in environment: {shutil.which("orca")}')

    if ref_type == "dft":
        fit_to_prop_prefix = dft_prop_prefix
        pred_prop_prefix =  ip_type + '_'
        # xtb2_prop_prefix = None
    if ref_type == "dft-xtb2":
        fit_to_prop_prefix = "dft_minus_xtb2_"
        pred_prop_prefix = "gap_plus_xtb2_"
        # xtb2_prop_prefix = "xtb2_"

    with open(fit_param_fname) as yaml_file:
        fit_params_base = yaml.safe_load(yaml_file)
    fit_params_base = it.update_fit_params(fit_params_base, fit_to_prop_prefix)

    # md params
    md_params = {
        "steps":2000,
        'dt':0.5, #fs
        'temperature': md_temp, # K
        'temperature_tau': 200, # fs, somewhat quicker than recommended (???)
        'traj_step_interval':100,
        'results_prefix': pred_prop_prefix
    }

    # soap descriptor for cur params
    with open(soap_params_for_cur_fname, 'r') as f: 
        soap_params_for_cur = yaml.safe_load(f)

    # prepare 0th dataset
    initial_train_fname = train_set_dir / "train_for_fit_0.xyz"
    ci = ConfigSet_in(input_files=base_train_fname)
    co = ConfigSet_out(output_files=initial_train_fname, force=True, all_or_none=True)
    it.prepare_0th_dataset(ci, co)

    initial_test_fname = train_set_dir / "test_for_fit_0.xyz"
    ci = ConfigSet_in(input_files=base_test_fname)
    co = ConfigSet_out(output_files=initial_test_fname, force=True, all_or_none=True)
    it.prepare_0th_dataset(ci, co)

    for cycle_idx in range(0, num_cycles + 1):

        # Check for the final training set from this iteration and skip if found.
        next_train_set_fname = (
            train_set_dir / f"{cycle_idx:02d}_train_for_{ip_type}_{cycle_idx+1:02d}.xyz"
        )
        if os.path.exists(next_train_set_fname):
            logger.info(f"Found {next_train_set_fname}, skipping iteration {cycle_idx}")
            continue

        # define all the filenames
        cycle_dir = wdir / f"iteration_{cycle_idx:02d}"
        cycle_dir.mkdir(exists_ok=True)

        train_set_fname = train_set_dir / f"{cycle_idx - 1:02d}.train_for_fit{cycle_idx:02d}.xyz"
        test_set_fname = train_set_dir / f"{cycle_idx - 1:02d}.test_for_fit{cycle_idx:02d}.xyz"
        
        if cycle_idx == 0:
            train_set_fname = initial_train_fname
            test_set_fname = initial_test_fname

        extra_smiles_for_this_cycle_csv = cycle_dir / "02.extra_smiles.csv"
        opt_starts_fname = cycle_dir / "03.rdkit_mols_rads.xyz"
        opt_fname = cycle_dir / f"04.{pred_prop_prefix}optimised.xyz"
        opt_filtered_fname = cycle_dir / f"05.1.{pred_prop_prefix}optimised.good_geometries.xyz"
        bad_structures_fname = cycle_dir / f"05.2.{pred_prop_prefix}optimised.bad_geometries.xyz"
        opt_fname_w_dft = cycle_dir / f"06.{pred_prop_prefix}optimised.good_geometries.dft.xyz"
        large_error_configs = cycle_dir / f"07.1.{pred_prop_prefix}optimised.good_geometries.dft.large_error.xyz"
        small_error_configs = cycle_dir / f"07.2.{pred_prop_prefix}optimised.good_geometries.dft.small_error.xyz"
        full_md_fname = cycle_dir / f"08.large_error.md.xyz"
        full_md_good_geometries_fname = cycle_dir / f"09.1.large_error.md.good_geometries.xyz"
        full_md_bad_geometries_fname = cycle_dir / f"09.2.large_error.md.bad_geometries.xyz"
        md_with_soap_fname = cycle_dir / f"10.large_error.md.good_geometries.soap.xyz"
        test_md_selection_fname = cycle_dir / f"11.1.large_error.md.test_sample.xyz"
        train_md_selection_fname = cycle_dir / f"11.2.large_error.md.train_sample.xyz"
        test_extra_fname_dft = cycle_dir / f"12.1.large_error.md.test_sample.dft.xyz"
        train_extra_fname_dft = cycle_dir / f"12.2.large_error.md.train_sample.dft.xyz"


        fit_dir = cycle_dir / "fit_dir"
        fit_dir.mkdir(exists_ok=True)

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
            pred_prop_prefix = f"{ip_type}{cycle_idx}_"
        elif ref_type == "dft-xtb2":
            pred_prop_prefix = f"{ip_type}{cycle_idx}_plus_xtb2_"


        # 2. Run tests 
        tests_wdir = cycle_dir / "tests"
        it.run_tests(
            calculator=calculator, 
            pred_prop_prefix=pred_prop_prefix, 
            dft_prop_prefix=dft_prop_prefix, 
            train_set_fname=train_set_fname, 
            test_set_fname=test_set_fname, 
            tests_wdir=tests_wdir, 
            bde_test_fname=bde_test_fname, 
        )

        # 3. Select some smiles from the initial smiles csv
        if not extra_smiles_for_this_cycle_csv.exists():
            it.select_extra_smiles(all_extra_smiles_csv=all_extra_smiles_csv,
                               extra_smiles_for_this_cycle_csv=extra_smiles_for_this_cycle_csv,
                               chunksize=10) 

        # 4. Generate actual structures for optimisation   
        logger.info("generating structures to optimise")
        outputs = ConfigSet_out(
            output_files=opt_starts_fname,
            force=True,
            all_or_none=True,
            verbose=False,
            set_tags={"iter_no":cycle_idx, 
                      "config_type":"rdkit"},
        )
        inputs = it.make_structures(
            extra_smiles_for_this_cycle_csv,
            iter_no=cycle_idx,
            num_smi_repeat=1,
            outputs=outputs,
        )

        # 5. optimise structures with current IP and re-evaluate them
        logger.info(f"optimising structures from {opt_starts_fname} with {ip_type}")
        outputs = ConfigSet_out(output_files=opt_fname, force=True, all_or_none=True, 
                                set_tags={"config_type":f"next_rdkit_{pred_prop_prefix}optimised"})
        # traj_step_interval=None selects only last converged config. 
        inputs = opt.optimise(inputs=inputs, outputs=outputs, calculator=calculator, 
                              prop_prefix=pred_prop_prefix, traj_step_interval=None)

        # 6. filter out insane geometries
        logger.info(f"filtering out bad geometries")
        outputs_good = ConfigSet_out(output_files=opt_filtered_fname, force=True, all_or_none=True)
        outputs_bad = ConfigSet_out(output_files=bad_structures_fname, force=True, all_or_none=True)
        inputs = it.filter_configs_by_geometry(inputs=inputs, outputs_good=outputs_good, outputs_bad=outputs_bad)

        # 7. evaluate DFT
        logger.info("evaluatig dft on optimised structures")
        outputs = ConfigSet_out(output_files=opt_fname_w_dft, force=True, all_or_none=True)
        inputs = orca.evaluate(
            inputs=inputs,
            outputs=outputs,
            orca_kwargs=orca_kwargs,
            output_prefix=dft_prop_prefix,
            keep_files=False,
            base_rundir=cycle_dir / "orca_wdir_1",
        )

        # 8. filter by energy and force error
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
        logger.info(
            f'Number of optimised structures: {len(read(opt_fname, ":"))}; Number '
            f"of structures selected for MD and sampling for training "
            f'{len(read(large_error_configs, ":"))}'
        )

        # 9. Run MD
        logger.info(f"Running {ip_type} md")
        outputs = ConfigSet_out(output_files=full_md_fname, force=True, all_or_none=True, 
                                set_tags={"config_type": f"{pred_prop_prefix}md"})
        inputs = md.sample(inputs=inputs, 
                           outputs=outputs,
                           calculator=calculator, 
                           verbose=True, 
                           **md_params)


        # 10. Filter/check for bad geometries
        outputs_good = ConfigSet_out(output_files=full_md_good_geometries_fname, force=True, all_or_none=True)
        outputs_bad = ConfigSet_out(output_files=full_md_bad_geometries_fname, force=True, all_or_none=True)
        inputs = it.filter_configs_by_geometry(inputs=inputs, outputs_good=outputs_good, outputs_bad=outputs_bad)

        # handle if some configs were found
        num_bad_configs = len([at for at in outputs_bad.to_ConfigSet_in()])
        num_good_configs = len([at for at in outputs_good.to_ConfigSet_in()])
        if num_bad_configs / (num_bad_configs + num_good_configs) > 0.1:
            raise RuntimeWarning("Too many bad geometries from MD")
        elif num_bad_configs != 0:
            logger.error(f"Some had {num_bad_configs} bad geometries from md")

        # 11. Calculate soap descriptor 
        logger.info("Calculating SOAp descriptor")
        outputs = ConfigSet_out(output_files=md_with_soap_fname, force=True, all_or_none=True)
        inputs = wfl.calc_descriptor.calc(inputs=inputs,
                                            outputs=outputs, 
                                            descs=soap_params_for_cur,
                                            key="small_soap",  # where to store the descriptor
                                            local=True) # calculate local descriptor

        # 12. do CUR
        if not train_md_selection.exists():
            logger.info("Selecting configs with CUR")
            outputs = ConfigSet_out(set_tags={"dataset_type":"next_addition", "cycle_idx":cycle_idx})
            inputs = cur.per_environment(inputs=inputs, 
                                        outputs=outputs,
                                        num=num_train_configs_per_cycle+num_test_configs_per_cycle, 
                                        at_descs_key="small_soap", 
                                        kernel_exp=3, 
                                        levarage_score_key="cur_leverage_score", 
                                        write_all_configs=False)
            selected_with_cur = list(inputs)
            del selected_with_cur.info["small_soap"]
            del selected_with_cur.arrays["small_soap"]
            random.shuffle(selected_with_cur)        
            write(train_md_selection_fname, selected_with_cur[:num_train_configs_per_cycle])
            write(test_md_selection_fname, selected_with_cur[num_train_configs_per_cycle:])

        # 13. evaluate DFT
        logger.info("evaluatig dft cur-selected md structures")
        inputs = ConfigSet_in(input_files=[test_md_selection_fname, train_md_selection_fname])
        outputs = ConfigSet_out(output_files={test_md_selection_fname:test_extra_fname_dft,
                                              train_md_selection_fname:train_extra_fname_dft},
                                force=True, 
                                all_or_none=True)
        inputs = orca.evaluate(
            inputs=inputs,
            outputs=outputs,
            orca_kwargs=orca_kwargs,
            output_prefix=dft_prop_prefix,
            keep_files=False,
            base_rundir=cycle_dir / "orca_wdir_2",
        )

        # 14. do summary plots
        it.summary_plots(cycle_idx, 
                         pred_prop_prefix=pred_prop_prefix,
                         dft_prop_prefix=dft_prop_prefix,
                         train_fname=train_set_fname, 
                         test_fname=test_set_fname, 
                         bde_fname=bde_test_fname,
                         ip_optimised_fname=opt_fname_w_dft, 
                         train_extra_fname=train_extra_fname_dft, 
                         test_extra_fname=test_extra_fname_dft, 
                         tests_wdir=tests_wdir)


        # set dataset_types correctly
        # 7. Combine data
        if not os.path.exists(next_train_set_fname):
            logger.info("combining new dataset")
            previous_dataset = read(train_set_fname, ":")
            additional_data = read(nm_sample_fname_for_train_with_dft, ":")
            write(next_train_set_fname, previous_dataset + additional_data)

    logger.info("Finished iterations")

