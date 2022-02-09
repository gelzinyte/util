from pathlib import Path
import os
import yaml
import numpy as np
from util.iterations import tools as it
from quippy.potential import Potential
import ace
import pytest
import shutil
import pandas as pd
from wfl.configset import ConfigSet_in, ConfigSet_out
from ase.build import molecule
from copy import deepcopy
from ase.io import read
from util.util_config import Config
import logging

logger = logging.getLogger(__name__)

def ref_path():
    return Path(__file__).parent.resolve()


def test_do_gap_fit(tmp_path, gap_params):

    fit_dir = tmp_path / "fit_dir"
    train_set_fname = ref_path() / "files/tiny_gap.train_set.xyz"

    calc = it.do_gap_fit(
        fit_dir=fit_dir,
        idx=1,
        ref_type="dft",
        train_set_fname=train_set_fname,
        fit_params_base=gap_params,
        gap_fit_path="gap_fit",
    )

    gap_fname = fit_dir / "gap_1.xml"

    assert gap_fname.exists()
    with open(fit_dir / "gap_1.out", "r") as f:
        t = f.read()
    assert "Bye-Bye!" in t

    assert calc == (Potential, [], {"param_filename": str(gap_fname)})


def test_update_fit_params(gap_params):

    gap_params = it.update_fit_params(gap_params, "fake_prefix_")

    assert gap_params["energy_parameter_name"] == "fake_prefix_energy"

    assert gap_params["force_parameter_name"] == "fake_prefix_forces"


def test_do_ace_fit(tmp_path, ace_params):

    fit_dir = tmp_path / "fit_dir"
    train_set_fname = ref_path() / "files/tiny_gap.train_set.xyz"
    ace_fit_exec = "/home/eg475/dev/workflow/wfl/scripts/ace_fit.jl"
    expected_ace_fname = fit_dir / "ace_1.json"

    del ace_params["dry_run"]

    calc = it.do_ace_fit(
        fit_dir=fit_dir,
        idx=1,
        ref_type="dft",
        train_set_fname=train_set_fname,
        fit_params_base=ace_params,
        fit_to_prop_prefix="dft_",
        ace_fit_exec=ace_fit_exec,
    )

    assert calc == (ace.ACECalculator, [], {"jsonpath": expected_ace_fname})

    assert False

def test_check_for_offset():
    fname = Path(__file__).parent.resolve() / "files/one_test_xyzs/check_for_offset.xyz"

    with pytest.raises(AssertionError):
        it.check_for_offset(fname, 'ace_', 'dft_')

def test_select_extra_smiles(tmp_path):

    fname = Path(__file__).parent.resolve() / "files/extra_smiles.csv"
    in_fname = shutil.copy(fname, tmp_path / "in.csv")
    out_fname = tmp_path / "out.csv"

    df = pd.read_csv(in_fname, delim_whitespace=True)
    assert len(df[df["has_been_used"]]) == 5 
    assert len(df[~df["has_been_used"]]) == 27 

    it.select_extra_smiles(in_fname, out_fname, chunksize=10)

    df = pd.read_csv(in_fname, delim_whitespace=True)
    assert len(df[df["has_been_used"]]) == 15 
    assert len(df[~df["has_been_used"]]) == 17

    df = pd.read_csv(out_fname, delim_whitespace=True)
    assert len(df) == 10

def test_make_structures():

    in_csv = Path(__file__).parent.resolve() / "files" / "extra_smiles.csv"
    co = ConfigSet_out(set_tags={"iter_no":248, 
                      "config_type":"rdkit"})

    ci = it.make_structures(smiles_csv=in_csv, 
                            num_smi_repeat=1, 
                            outputs=co, 
                            num_rads_per_mol=1, 
                            )

    assert len(ci.input_configs) == 64 
    
    at = ci.input_configs[0][0]
    for label in ["config_type", "compound", "mol_or_rad", "rad_num", "graph_name", "iter_no"]:
        assert label in at.info.keys()


def test_filter_configs():

    fake_forces = np.random.rand(5, 3)
    at0 = molecule("CH4")
    at0.info["ref_energy"] = 4.00
    at0.arrays["ref_forces"] = fake_forces

    ats_to_test = []
    at = at0.copy()
    at.info["pred_energy"] = at.info["ref_energy"] + 0.04 
    pred_forces = deepcopy(at.arrays["ref_forces"])
    pred_forces[0][1] += 0.05
    at.arrays["pred_forces"] = pred_forces 

    at.info["test_config_type"] = "small_error"
    ats_to_test.append(at.copy())

    at.arrays["pred_forces"][0][1] += 0.051
    at.info["test_config_type"] = "large_error"
    ats_to_test.append(at.copy())

    at.arrays["pred_forces"][0][1] -= 0.051
    at.info["pred_energy"] += 0.02
    ats_to_test.append(at.copy())
    co_large = ConfigSet_out()
    co_small = ConfigSet_out()
    it.filter_configs(inputs=ats_to_test, 
                     outputs_large_error=co_large,
                     outputs_small_error=co_small, 
                     pred_prop_prefix='pred_', 
                     dft_prefix="ref_",
                     e_threshold_total=0.05, 
                     max_f_comp_threshold=0.1)
    assert len([at for at in co_large.to_ConfigSet_in()]) ==2
    assert len([at for at in co_small.to_ConfigSet_in()]) == 1
    for at in co_large.to_ConfigSet_in():
        assert at.info["test_config_type"] == "large_error"
    for at in co_small.to_ConfigSet_in():
        assert at.info["test_config_type"] == "small_error"

    ats_to_test = []
    at = at0.copy()
    at.info["pred_energy"] = at.info["ref_energy"] + 0.06 
    pred_forces = deepcopy(at.arrays["ref_forces"])
    pred_forces[0][1] += 0.05
    at.arrays["pred_forces"] = pred_forces 

    at.info["test_config_type"] = "small_error"
    ats_to_test.append(at.copy())

    at.arrays["pred_forces"][0][1] += 0.251
    at.info["test_config_type"] = "large_error"
    ats_to_test.append(at.copy())

    co_large = ConfigSet_out()
    co_small = ConfigSet_out()
    it.filter_configs(inputs=ats_to_test, 
                     outputs_large_error=co_large,
                     outputs_small_error=co_small, 
                     pred_prop_prefix='pred_', 
                     dft_prefix="ref_",
                     e_threshold_per_atom=0.05, 
                     max_f_comp_threshold=0.1)
    assert len([at for at in co_large.to_ConfigSet_in()]) == 1
    assert len([at for at in co_small.to_ConfigSet_in()]) == 1
    for at in co_large.to_ConfigSet_in():
        assert at.info["test_config_type"] == "large_error"
    for at in co_small.to_ConfigSet_in():
        assert at.info["test_config_type"] == "small_error"


def test_update_cutoffs():

    cutoffs_mb = {"(:C, :H)": "(1.6, 4.4)",
                  "(:C, :C)": "(0.8, 4.4)",
                  "(:H, :H)": "(1.0, 4.4)"}

    it.update_cutoffs(cutoffs_mb, "CH", 3.14)

    expected_cutoffs_mb = {"(:C, :H)": "(3.14, 4.4)",
                           "(:C, :C)": "(0.8, 4.4)",
                           "(:H, :H)": "(1.0, 4.4)"}

    assert cutoffs_mb == expected_cutoffs_mb


def test_update_ace_params():

    train_set_fname = ref_path() / "files/tiny_train_set.xyz" 
    params_fname = ref_path() / "files/ace_params.yml"
    with open(params_fname) as f:
        params = yaml.safe_load(f)
    params_out = it.update_ace_params(params, read(train_set_fname, ':'))

    expected_cutoffs_mb = {"(:C, :H)": "(1.00, 4.4)",
                           "(:C, :C)": "(0.8, 4.4)",
                           "(:H, :H)": "(1.66, 4.4)"}

    assert params_out['cutoffs_mb'] == expected_cutoffs_mb


def test_check_dft(tmp_path):

    train_set_fname = ref_path() / "files/tiny_train_set.xyz" 

    cfg = Config.load()
    default_kw = Config.from_yaml(os.path.join(cfg["util_root"], "default_kwargs.yml"))
    dft_prop_prefix = "dft_"
    orca_kwargs = default_kw["orca"]
    orca_kwargs["orca_command"] = cfg["orca_path"]
    logger.info(f"orca_kwargs: {orca_kwargs}") 

    it.check_dft(train_set_fname, dft_prop_prefix, orca_kwargs, tmp_path / "dft_check_wdir")


def test_process_trajs():

    at0 = molecule("CH4")

    # check just that everything gets picked up as it should
    traj_1 = [at0.copy() for _ in range(4)]
    for at in traj_1:
        at.info["graph_name"] = "good_traj"
    traj_1[-1].info["config_type"] = "last"

    traj_2 = [at0.copy() for _ in range(4)]
    for at in traj_2:
        at.info["graph_name"] = "bad_traj"

    traj_2[2].set_distance(0, 1, 5)
    traj_2[3].set_distance(0, 1, 5)

    ci = ConfigSet_in(input_configs=traj_1 + traj_2)
    co_good_traj = ConfigSet_out()
    co_bad_traj_good_cfg = ConfigSet_out()
    co_bad_traj_bad_cfg = ConfigSet_out()

    it.process_trajs(ci, co_good_traj, co_bad_traj_bad_cfg, co_bad_traj_good_cfg, "last")

    assert [at for at in co_good_traj.to_ConfigSet_in()][0].info["config_type"] == "last"
    assert [at for at in co_good_traj.to_ConfigSet_in()][0].info["graph_name"] == "good_traj"

    assert len([at for at in co_bad_traj_bad_cfg.to_ConfigSet_in()]) == 2
    assert len([at for at in co_bad_traj_good_cfg.to_ConfigSet_in()]) == 2

    # ~check that can get empty files~
    # actually I'm not writing to files...
    traj_1 = [at0.copy() for _ in range(4)]
    for at in traj_1:
        at.info["graph_name"] = "good_traj"
    traj_1[-1].info["config_type"] = "last"

    traj_2 = [at0.copy() for _ in range(4)]
    for at in traj_2:
        at.info["graph_name"] = "bad_traj"

    ci = ConfigSet_in(input_configs=traj_1 + traj_2)
    co_good_traj = ConfigSet_out()
    co_bad_traj_good_cfg = ConfigSet_out()
    co_bad_traj_bad_cfg = ConfigSet_out()

    it.process_trajs(ci, co_good_traj, co_bad_traj_bad_cfg, co_bad_traj_good_cfg, "last")
    assert len([at for at in co_good_traj.to_ConfigSet_in()]) == 2
    assert len([at for at in co_bad_traj_good_cfg.to_ConfigSet_in()]) == 0











