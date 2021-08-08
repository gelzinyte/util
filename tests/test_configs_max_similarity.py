import numpy as np
import os
from ase.io import read, write
from pytest import approx
from util.configs import max_similarity

def ref_path():
    return os.path.abspath(os.path.dirname(__file__))


def test_kernel():

    zeta = 1

    mx1 = np.array([[1, 2, 1], [4, 0.5, 1]])
    mx2 = np.array([[2, 1], [1, 2]])

    non_normalised = max_similarity.kernel_non_normalised(mx1, mx2,
                                                          zeta=zeta)

    expected_mx = np.array([[6, 9], [4.5, 3], [3, 3]])

    assert np.all(non_normalised == expected_mx)


    normalised = max_similarity.kernel(mx1, mx2, zeta=1)
    expected_mx = np.array([[0.65079137, 0.97618706],[0.97618706, 0.65079137],
                            [0.9486833,  0.9486833 ]])
    assert np.all(approx(normalised) == expected_mx)


    non_normalised = max_similarity.kernel_non_normalised(mx2, mx2, zeta=zeta)
    expected_mx = np.array([[5, 4], [4, 5]])

    assert np.all(non_normalised == expected_mx)

    normalised = max_similarity.kernel(mx2, mx2, zeta)
    expected_mx = np.array([[1, 0.8], [0.8, 1]])
    assert np.all(approx(normalised) == expected_mx)

def test_get_descriptor_mx():
    input_files = os.path.join(ref_path(), 'files/mols_soap.xyz')
    at_descs_key = 'SOAP-n4-l3-c2.4-g0.3'
    desc_length = 120
    ats = read(input_files, ':')

    descs = max_similarity.get_descriptor_matrix(ats, at_descs_key)

    assert descs.shape == (desc_length, len(ats))



