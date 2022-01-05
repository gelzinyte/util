import os
import numpy as np
import pytest
from pytest import approx
from pathlib import Path

from ase.io import read, write
from ase.build import molecule
from quippy.potential import Potential

from wfl.calculators import generic
from wfl.configset import ConfigSet_out

import util.bde.generate

@pytest.fixture()
def at():
    at = molecule('CH4')
    # add fake data
    at.info['dft_opt_mol_positions_hash'] = 'fake_hash'
    at.info["bde_config_type"] = "dft_optimised"
    at.info['dft_energy'] = -1100.314
    at.arrays['dft_forces'] = np.random.rand(len(at), 3)
    return at

@pytest.fixture()
def gap_filename():
    return  Path(ref_path()) / 'files' / 'tiny_gap.xml'

def ref_path():
    return Path(__file__).parent

@pytest.fixture()
def calculator(gap_filename):
    print(gap_filename)
    return (Potential, [], {'param_filename': str(gap_filename)})

def test_making_isolated_H(tmp_path, calculator):

    wdir = tmp_path / 'bde_wdir'
    output_fname = wdir / "gappy_isolated_h.xyz"

    outputs=ConfigSet_out(output_files=output_fname)
    util.bde.generate.ip_isolated_h(calculator=calculator, 
                                    dft_prop_prefix='dft_',
                                    ip_prop_prefix="gappy_",
                                    outputs=outputs, 
                                    wdir=wdir)

    at = read(output_fname)
    assert at.info["bde_config_type"] == "H"
    assert pytest.approx(at.info["dft_energy"] == -13.547458419057222 )
    assert pytest.approx(at.info["gappy_energy"] == -13.547479108102433)


def test_generate(at, calculator, tmp_path):

    dft_bde_fname = tmp_path / 'input_atoms.xyz'
    at1 = at.copy()
    at1.info["mol_or_rad"] = "mol"
    at1.info["dft_opt_positions_hash"] = "fake_hash_mol"
    at2 = at.copy()
    at2.info["mol_or_rad"] = "rad"
    at2.info["dft_opt_positions_hash"] = "fake_hash_rad"
    write(dft_bde_fname, [at1, at2])

    dft_prop_prefix='dft_'
    gap_prop_prefix='gappy_'
    wdir = tmp_path / 'bde_wdir'

    util.bde.generate.everything(calculator=calculator,
                                 dft_bde_filename=dft_bde_fname,
                                 dft_prop_prefix=dft_prop_prefix,
                                 ip_prop_prefix=gap_prop_prefix,
                                 wdir=wdir)

    # dft-optimised evaluated with gap
    assert (wdir / "input_atoms.gappy.xyz").exists()

    # gap-reoptimised atoms; should have changed config type and removed 
    # outdated energy, etc
    at = read(wdir / "input_atoms.gappy_reoptimised.xyz")
    assert at.info["bde_config_type"] == f'{gap_prop_prefix}optimised'
    assert "dft_energy" not in at.info.keys()

    # evaluate with DFT
    at = read(wdir / "input_atoms.gappy_reoptimised.dft.xyz")
    assert "dft_energy" in at.info.keys()

    # isolated H
    assert (wdir / "gappy_isolated_H.xyz").exists()

    # bdes
    ats = read(wdir / "input_atoms.gappy.bde.xyz", ':')
    assert ats[0].info["bde_config_type"] == "dft_optimised"
    # same as isolated atom energy here, because my test atoms are really 
    # molecule and molecule
    assert approx(ats[1].info["gappy_bde_energy"] == -13.54747910810238)
    assert approx(ats[1].info["dft_bde_energy"] == -13.547458419057193)

    ats = read(wdir / "input_atoms.gappy_reoptimised.dft.bde.xyz", ':')
    assert ats[0].info["bde_config_type"] == "gappy_optimised"
    assert approx(ats[1].info["gappy_bde_energy"] == -13.54747910810238)
    assert approx(ats[1].info["dft_bde_energy"] == -13.547458419057193)

    # check final file is there
    assert (tmp_path / "input_atoms.gappy_bde.xyz").exists()


