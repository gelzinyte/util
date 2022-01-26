import util.iterations.fit

from pathlib import Path

from ase.io import read, write

def ref_path():
    return Path(__file__).parent.resolve()


def test_iterfit(tmp_path):

    tmp_path = "/home/eg475/scripts/tests/iterations_wdir"

    all_ats = read(ref_path() / 'files.tiny_gap.train_set.xyz'), ':')
    iso = [at for at in all_ats if len(at) == 1]
    all_ats = [at for at in all_ats if len(at)!=1]
    train_fname = tmp_path / 'train.xyz'
    test_fname = tmp_path / 'test.xyz'
    write(train_fname, all_ats[0::2])
    write(test_fname, all_ats[])
    



