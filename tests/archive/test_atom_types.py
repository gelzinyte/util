from util import atom_types
import os
import numpy as np


def ref_path():
    return os.path.abspath(os.path.dirname(__file__))

def test_numbers_from_yml():

    filename = os.path.join(ref_path(), 'files', 'limonene_S_mol.yml')
    config_type = 'limonene_S_mol'

    orig_numbers, atom_typed_numbers = atom_types.numbers_from_yaml(
        filename, config_type)

    expected_orig_numbers = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    expected_atom_typed_numbers = np.array([10, 17, 21, 20, 19, 18, 17, 11,
                                            18, 10, 1, 1, 1, 51, 51, 1, 1,
                                            1, 1, 1, 51, 1, 1, 1, 1, 1])

    assert np.all(orig_numbers == expected_orig_numbers)
    assert np.all(atom_typed_numbers == expected_atom_typed_numbers)
