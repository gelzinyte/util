import os
import pytest
from ase.io import read
from util import iter_tools as it


def ref_path():
    return os.path.abspath(os.path.dirname(__file__))

@pytest.mark.skip(reason='outdated iterations tool')
def test_smiles_to_atoms(tmp_path):

    smiles_csv = os.path.join(ref_path(), 'files', 'smiles.csv')
    iter_no = 'test_iter_no'
    num_smi_repeat = 2
    output_fname = os.path.join(tmp_path, 'smi_to_atoms.xyz')

    it.make_structures(smiles_csv=smiles_csv, iter_no=iter_no,
                       num_smi_repeat=num_smi_repeat, output_fname=output_fname)

    atoms = read(output_fname, ':')

    assert len(atoms) == 12






