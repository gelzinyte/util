import util.iterations.fit
import pytest

import logging

from pathlib import Path

from ase.io import read, write

logger = logging.getLogger(__name__)

def ref_path():
    return Path(__file__).parent.resolve()


def test_iterfit(tmp_path):
    """
    TODO
    * check that everything restarts as it should 
    """

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s')

    tmp_path = Path("/home/eg475/scripts/tests/iterations_wdir")
    tmp_path.mkdir(exist_ok=True)


    # sort out train/test sets
    all_ats = read(ref_path() / 'files/tiny_train_set.xyz', ':')
    iso = [at for at in all_ats if len(at) == 1]
    all_ats = [at for at in all_ats if len(at)!=1]
    train_fname = tmp_path / 'train.xyz'
    test_fname = tmp_path / 'test.xyz'

    with pytest.warns(UserWarning):
        write(train_fname, all_ats[0::2] + iso)
        write(test_fname, all_ats[1::2])


    fit_param_fname = ref_path() / 'files/ace_params.yml'
    all_extra_smiles_csv = ref_path() / 'files/extra_smiles.csv'
    bde_test_fname = ref_path() / 'files/bde_test.dft_opt.xyz'
    soap_params_for_cur = ref_path() / 'files/soap_params_for_cur.yml'

    util.iterations.fit.fit(
        num_cycles=1,
        base_train_fname=train_fname,
        base_test_fname=test_fname, 
        fit_param_fname=fit_param_fname,
        all_extra_smiles_csv=all_extra_smiles_csv,
        md_temp=500.0,
        # energy_error_per_atom_threshold=0.0002,
        # energy_error_total_threshold=None,
        # max_f_comp_error_threshold=None,
        wdir= tmp_path / "fits",
        ref_type='dft', 
        ip_type='ace',
        bde_test_fname=bde_test_fname, 
        soap_params_for_cur_fname=soap_params_for_cur,
        num_train_environments_per_cycle=5,
        num_test_environments_per_cycle=5,
        num_extra_smiles_per_cycle=5,
    )
