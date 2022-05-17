import os
from pytest import approx
import numpy as np
from quippy.potential import Potential
from util.calculators import xtb2_plus_gap
from ase.build import molecule
from xtb.ase.calculator import XTB


def ref_path():
    return os.path.abspath(os.path.dirname(__file__))


def test_calculator():

    ref_at = molecule('CH4')
    gap_fname = os.path.join(ref_path(), 'files/tiny_gap.xml')

    diff_calc = xtb2_plus_gap(gap_filename=gap_fname)
    at = ref_at.copy()
    at.calc = diff_calc
    pred_energy = at.get_potential_energy()
    pred_forces = at.get_forces()

    gap = Potential(param_filename=gap_fname)
    at = ref_at.copy()
    at.calc = gap
    gap_energy = at.get_potential_energy()
    gap_forces = at.get_forces()

    xtb = XTB(method='GFN2-xTB')
    at = ref_at.copy()
    at.calc = xtb
    xtb_energy = at.get_potential_energy()
    xtb_forces = xtb.get_forces()

    assert approx(pred_energy) == gap_energy + xtb_energy
    assert np.all(approx(pred_forces) == gap_forces + xtb_forces)



