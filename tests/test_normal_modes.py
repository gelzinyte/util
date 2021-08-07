import numpy as np
from util import normal_modes as nm
from ase import units
from pytest import approx

def test_sigmoid_energies_for_normal_modes():

    frequencies_eV = np.array([-100,  0, 50, 100, 200, 210]) * units.invcm
    temp = 300

    weighted_energies = nm.downweight_energies(frequencies_eV=frequencies_eV,
                                               temp=temp)

    expected_energies = [0, 0, 0.0003192768610273793, temp * units.kB * 0.5, \
    temp * units.kB, temp * units.kB]

    assert np.all(approx(weighted_energies) == expected_energies)

