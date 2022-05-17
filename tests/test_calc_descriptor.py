from pathlib import Path
from ase.build import molecule
from ase.io import read, write

from util.configs import calc_desc


def test_calculate_descriptor():
    param_fname = Path().resolve() / 'test_gap_iter_fit' / 'parameters.yml'
    param_fname = param_fname.as_posix()
    atoms = molecule('CH4')
    calc_desc.from_param_yaml(atoms, param_fname)

