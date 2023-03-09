import pytest
from pathlib import Path

from xtb.ase.calculator import XTB

from ase.build import molecule
from ase.constraints import FixBondLength
from ase.io import read

# from wfl.generate import optimize

from util import neb


def ref_path():
    return Path(__file__).parent.resolve()

@pytest.fixture
def xtb_calc():
    return (XTB, [], {'method':'GFN2-xTB'})

@pytest.fixture()
def in_ats():
    pp = ref_path()
    ats = read(pp/"files/neb_rxn_start.xyz", ":")
    return [ats[0], ats[-1]]


# @pytest.fixture()
# def in_ats(xtb_calc):
#     at_start = molecule("CH4")
#     at_end = at_start.copy()
#     at_end.set_distance(0, 1, 8, fix=0)
#     return [at_start, at_end]
    
    # ats = optimize._run_autopara_wrappable(
    #     atoms = [at_start, at_end], 
    #     calculator=xtb_calc,
    #     fmax=0.01,
    #     steps=200,
    #     master=True,
    #     precon=None,
    #     use_armijo=False,
    #     keep_symmetry=False, 
    #     traj_subselect="last_converged")
    # print(ats[1].positions)
    # return ats 



def test_neb(xtb_calc, in_ats):
    print(len(in_ats))
    out = neb.neb_ll(
        inputs = in_ats,
        calculator=xtb_calc,
        num_images = 12,
        fmax=0.01,
        steps=500,
        master=True,
        precon=None,
        use_armijo=False
    )

    print(out)

    import pdb; pdb.set_trace()
