import os
import numpy as np
from ase.io import read, write
import pytest

from util.bde import plot


def ref_path():
    return os.path.abspath(os.path.dirname(__file__))

@pytest.fixture()
def atoms():

    at_fname = os.path.join(ref_path(), 'files/bde_dft_reopt.xyz')
    return read(at_fname, ':')


def test_bde_table(atoms):

    index = plot.bde_table(atoms, None)
    expected_idx = ['H', 'mol', 'rad4', 'rad5', 'rad6', 'rad7', 'rad8',
                    'rad9', 'rad10', 'rad11', 'rad12', 'rad13']





