from pathlib import Path
from util.iterations import tools as it


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

    assert False

