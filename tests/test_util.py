import pytest
import numpy as np
from util import remove_energy_force_containing_entries
from ase.build import molecule

def test_remove_energy_force_containing_entries():

    at = molecule("CH4")

    at.info["something"] = 1
    at.info["some_energy_key"] = 3
    at.arrays['some_forces'] = np.random.rand(len(at), 3)
    at.arrays['something_else'] = np.random.rand(len(at), 3)

    remove_energy_force_containing_entries(at)

    assert 'something' in at.info.keys()
    assert 'some_energy_key' not in at.info.keys()
    assert 'some_forces' not in at.arrays.keys()
    assert 'something_else' in at.arrays.keys()
