import os
try:
    from mace.calculators.mace import MACECalculator
except:
    pass
from ase.build import molecule
from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators import generic

def ref_path():
    return os.path.abspath(os.path.dirname(__file__))


def test_mace_ase_calculator():
    mace_fname = os.path.join(ref_path(), 'files/mace_run-123.model.cpu')
    mace = MACECalculator(mace_fname, r_max=3.0, device='cpu', atomic_numbers=[1,6], default_dtype="float64")
    at = molecule("CH4")
    at.calc = mace
    at.get_potential_energy()
    at.get_forces()


def test_wfl_mace_calculator():

    ats = molecule("CH4")
    ats = [ats] * 10
    inputs = ConfigSet(input_configs=ats)
    outputs = OutputSpec()

    mace_fname = os.path.join(ref_path(), 'files/mace_run-123.model.cpu')
    mace = (MACECalculator, [mace_fname], {"r_max": 3.0, "device": "cpu", "atomic_numbers": [1, 6], "default_dtype":"float64"})

    generic.run(
        inputs=inputs, 
        outputs=outputs, 
        calculator=mace,
        properties=["energy", "forces"],
        output_prefix='wfl_mace_'
    )

    ats = [at for at in outputs.to_ConfigSet()]
    assert "wfl_mace_energy" in ats[0].info
    assert "wfl_mace_forces" in ats[0].arrays




