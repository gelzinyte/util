from pathlib import Path
from util.iterations import tools as it
from quippy.potential import Potential
import ace
import pytest
import shutil
import pandas as pd


def ref_path():
    return Path(__file__).parent.resolve()


# def test_do_ace_fit(tmp_path)


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

