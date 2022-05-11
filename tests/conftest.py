import pytest
from ase.build import molecule
from pathlib import Path
from quippy.potential import Potential
import yaml
import util
from util.calculators.orca import ORCA


@pytest.fixture()
def atoms():
    return molecule("CH4")


@pytest.fixture()
def gap_filename(ref_path):
    return Path(ref_path) / "files" / "tiny_gap.xml"


@pytest.fixture()
def ref_path():
    return Path(__file__).parent


@pytest.fixture()
def calculator(gap_filename):
    return (Potential, [], {"param_filename": str(gap_filename)})


@pytest.fixture()
def gap_params():
    params = {
        "default_sigma": [0.001, 0.03, 0.0, 0.0],
        "_gap": [
            {
                "soap": True,
                "l_max": 3,
                "n_max": 6,
                "cutoff": 3,
                "delta": 1,
                "covariance_type": "dot_product",
                "zeta": 2,
                "n_sparse": 40,
                "sparse_method": "cur_points",
                "atom_gaussian_width": 0.4,
                "add_species": True,
            }
        ],
        "config_type_sigma": "isolated_atom:0.001:0.0:0.0:0.0",
        "sparse_separate_file": True,
        "energy_parameter_name": "dft_energy",
        "force_parameter_name": "dft_forces",
        "core_param_file": "/home/eg475/scripts/source_files/glue.xml",
        "core_ip_args": "{IP GLUE}",
    }

    return params


@pytest.fixture()
def ace_params():
    ace_params_filename = Path(__file__).parent.resolve() / "files/ace_params.yml"
    with open(ace_params_filename, 'r') as f:
        params = yaml.safe_load(f)
    return params


@pytest.fixture()
def dft_calculator(tmp_dir):
    orca_kwargs = util.default_orca_params()
    orca_kwargs["workdir_root"] = tmp_dir / "orca_wdir"
    orca_kwargs["keep_files"] = False
    return (ORCA, [], orca_kwargs})


