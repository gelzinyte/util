import numpy as np
from pytest import approx

from util import electrons


def test_getting_eps_ij(mp2):

    expected = np.array([[-0.00042304, -0.00014149, -0.00023154, -0.00027806, -0.00029513],
                       [-0.00014149, -0.00900552, -0.00914111, -0.007555  , -0.00749152],
                       [-0.00023154, -0.00914111, -0.01986114, -0.01605793, -0.01445048],
                       [-0.00027806, -0.007555  , -0.01605793, -0.01749857, -0.01491374],
                       [-0.00029513, -0.00749152, -0.01445048, -0.01491374, -0.0168985 ]])

    eps_ij = electrons.get_eps_ij(mp2)

    assert np.all(approx(expected, rel=1e-4) == eps_ij)

