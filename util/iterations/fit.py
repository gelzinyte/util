import os
import yaml
import logging

from pathlib import Path

from ase.io import read, write

import wfl
from wfl.calculators import orca, generic
from wfl.configset import ConfigSet, OutputSpec

from util.util_config import Config
from util.iterations import tools as it
from util.iterations import plots as ip
import util

logger = logging.getLogger(__name__)


def fit(
    num_cycles,
    initial_train_fname="train.xyz",
    fit_param_fname=None,
    all_extra_smiles_csv=None,
    md_temp=500,
    wdir="runs",
    ref_type="dft",
    ip_type="ace",
    num_extra_smiles_per_cycle=10,
    num_rads_per_mol=0, 
    validation_fname = 'validation.xyz',
    cur_soap_params="cur_soap.yaml",
    md_steps = 2000):
    """ iteratively fits potetnials

    No files are created or modified in the original directory, everything's done in "wdir"

    Parameters
    ---------

    num_cycles: int
        number of fit-optimise cycles to do
    initial_train_fname: str, default='train.xyz'
        fname for fitting the first IP
    fit_param_fname: str, default None
        template ace_fit.jl
    ip_type: str, default "ace"
        "ace" or "gap"
    all_extra_smiles_csv: str, default None
        CSV of id and smiles to generate structures from
    md_temp: float, default 500
        temperature to run MD at
    ref_type: str, default "dft"
        "dft" or "dft-xtb2" for the type of reference energies/forces to
        fit to.
    ip_type: "ace" or "gap", default "ace"
    wdir: wehere to base all the runs in.
    """

    assert ref_type in ["dft"]
    assert ip_type in ["gap", "ace"]

    logger.info('\n', "-"*30, '\n', "Setting up\n", "*"*30)
    logger.info(f"csv file", all_extra_smiles)
    logger.info(f"fitting parameters from {fit_param_fname}")

    cfg = Config.load()
    scratch_dir = cfg["scratch_path"]
    logger.info(f"Using: scratch_dir: {scratch_dir}")


    if ip_type == "gap":
        fit_exec_path = cfg["gap_fit_path"]
    elif ip_type == "ace":
        fit_exec_path = cfg["ace_fit_path"]'
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
    orca_kwargs["keep_files"] = False
    orca_calc = (orca.ORCA, [], orca_kwargs)
    logger.info(f"orca_kwargs: {orca_kwargs}")


    # soap descriptor for cur params
    with open(cur_soap_params, "r") as f:
        cur_soap_params = yaml.safe_load(f)


    fit_to_prop_prefix = dft_prop_prefix
    pred_prop_prefix = ip_type + "_"

    with open(fit_param_fname) as yaml_file:
        fit_params_base = yaml.safe_load(yaml_file)
    fit_params_base = it.update_fit_params(fit_params_base, fit_to_prop_prefix)
    logger.info(f"fit_params_base: {fit_params_base}")

    # md params
    md_params = {
        "steps": md_steps,
        "dt": 0.5,  # fs
        "temperature": md_temp,  # K
        "temperature_tau": 200,  # fs, somewhat quicker than recommended (???)
        "traj_step_interval": 20,
        "results_prefix": pred_prop_prefix,
    }
    logger.info(f"MD params: {md_params}")
 
    initial_train_fname = train_set_dir / f"00.train_for_{ip_type}_01.xyz"
    if not initial_train_fname.exists():
        it.check_dft(initial_train_fname, dft_prop_prefix=dft_prop_prefix, orca_kwargs=orca_kwargs, tests_wdir=wdir/"dft_check_wdir")

    # prepare 1st dataset
    cs = ConfigSet(input_files=initial_train_fname)
    os = OutputSpec(output_files=initial_train_fname, force=True, all_or_none=True,
                       set_tags={"dataset_type": "train"})
    it.prepare_dataset(cs, os, cycle_idx=0)

    #######################################################################

    for cycle_idx in range(1, num_cycles + 0):

        fns = it.get_filenames(no=cycle_idx, wdir=wdir, ip=ip_type, train_set_dir=train_set_dir, val_fn=validation_fname)

        logger.info("-" * 50)
        logger.info(f"ITERATION {cycle_idx}")

        if os.path.exists(fns["next_train"]):
            logger.info(f"Found {fns['next_train']}, skipping iteration {cycle_idx}")
            continue 

        if ip_type == "gap":
            calculator = it.do_gap_fit(
                fit_dir=fit_dir,
                idx=cycle_idx,
                ref_type=ref_type,
                train_set_fname=fns["this_train"],
                fit_params_base=fit_params_base,
                gap_fit_path=fit_exec_path,
            )

        elif ip_type == "ace":
            calculator = it.do_ace_fit(
                fit_dir=fit_dir,
                idx=cycle_idx,
                ref_type=ref_type,
                train_set_fname=fns["this_train"],
                fit_params_base=fit_params_base,
                fit_to_prop_prefix=fit_to_prop_prefix,
                ace_fit_exec=fit_exec_path,
            )


        # 2. Run tests
        tests_wdir = cycle_dir / "tests"
        if not fns["test"]["mlip_on_train"].exists():
            logger.info("running_tests")
            with open(fit_dir / f"ace_{cycle_idx}_params.yaml") as f:
                fit_params = yaml.safe_load(f)

            ip.run_tests(
                pred_calculator=calculator,
                pred_prop_prefix=pred_prop_prefix,
                dft_calculator = orca_calc,
                dft_prop_prefix=dft_prop_prefix,
                fns=fns,
                fit_params = fit_params)


        # 3. Select some smiles from the initial smiles csv
        if not fns["smiles"].exists():
            it.select_extra_smiles(
                all_extra_smiles_csv=all_extra_smiles_csv,
                smiles_selection_csv=fns["smiles"],
                num_extra_smiles=num_extra_smiles)


        # 4. Generate actual structures for md 
        logger.info("generating structures to work with")
        outputs = OutputSpec(
            output_files=fns["md_starts"]["all"],
            set_tags={"iter_no": cycle_idx, "config_type": "rdkit"})

        inputs = it.make_structures(
            smiles_csv=fns["smiles"],
            num_smi_repeat=1,
            outputs=outputs,
            num_rads_per_mol=num_rads_per_mol)


        # 5. Run MD
        outputs = OutputSpec(output_files = fns["md_traj"]["all"])
        inputs = it.run_md(
            inputs=inputs,
            outputs=outputs,
            md_params=md_params,
        )

        # 6. Organise trajectories
        it.organise_md(all_md_trajs=inputs, fns=fns)

        # 7. sample trajectories
        cs_test_cur, cs_train_cur = it.sample_with_cur_soap(fns, soap_params, num_cur_environments, cycle_idx)
        cs_train_from_failed = it.sample_failed(fns)

        # 8. cleanup 
        cs_test_cur = it.cleanup(cs_test_cur, cycle_idx)
        cs_train_cur = it.cleanup(cs_train_cur, cycle_idx)
        cs_train_from_failed = it.cleanup(cs_train_from_failed, cycle_idx)

        # 9. get DFT
        outputs_test = OutputSpec(output_files=fns["extra"]["validation"]["dft"],
            set_tags={"dataset_type": "validation"})
        outputs_train = OutputSpec(output_files=fns["extra"]["train"]["all_dft"],
            set_tags={"dataset_type": "train"})
        inputs_train = ConfigSet(input_configsets = [cs_train_cur, cs_train_from_failed])

        orca_kwargs["directory"] = fns["cycle_dir"] 
        dft_calc = (orca.ORCA, [], orca_kwargs)

        for inp, outp in [(cs_test_cur, outputs_test), (inputs_train, outputs_train)]:

            generic.run(
                inputs=inp, 
                outputs=outp,
                calculator=dft_calc,
                output_prefix=dft_prop_prefix
            )


        # 15. Combine datasets
        if not fns["next_train"].exists():
            logger.info("combining old and extra data")
            previous_train = read(fns["this_train"], ":")
            extra_train = read(fns["extra"]["train"]["all_dft"], ":")
            write(fns["next_train"], previous_train + extra_train)

        logger.info(f"cycle {cycle_idx} done. ")


    logger.info("Finished iterations")

