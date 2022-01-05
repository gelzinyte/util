import pytest
from ase.build import molecule
from pathlib import Path
from quippy.potential import Potential

@pytest.fixture()
def atoms():
    return molecule('CH4')

@pytest.fixture()
def gap_filename(ref_path):
    return  Path(ref_path) / 'files' / 'tiny_gap.xml'

@pytest.fixture()
def ref_path():
    return Path(__file__).parent

@pytest.fixture()
def calculator(gap_filename):
    return (Potential, [], {'param_filename': str(gap_filename)})