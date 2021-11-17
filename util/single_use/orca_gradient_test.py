import os
import shutil
import tempfile
from wfl.calculators import orca
from ase.io import read
from util import gradient_test
from util.calculators.gap import PopGAP
from util.util_config import Config
import random

def do_orca_grad_test(atoms):

    if not os.path.isdir('some_orca_outputs'):
        os.mkdir('some_orca_outputs')
    nslots=os.environ["NSLOTS"]

    cfg = Config.load()
    scratch_dir = cfg['scratch_path']
    default_kw = Config.from_yaml(os.path.join(cfg['util_root'],
                                               'default_kwargs.yml'))
    # default_kw['orca']
    default_kw['orca']['scratch_path'] = scratch_dir
    default_kw['orca']['orcasimpleinput'] += f" PAL{nslots}"
    orca_command = cfg['orca_path']

    tmp_dir = tempfile.mkdtemp(dir=scratch_dir, prefix='orca_tmp_rundir')

    calc = orca.ExtendedORCA(orca_command=orca_command,
                             orcasimpleinput = default_kw['orca'][
                                 'orcasimpleinput'],
                             orcablocks = default_kw['orca']['orcablocks'],
                              directory=tmp_dir)

    type(calc)

    gradient_test(atoms, calc, start=0, stop=-14)
    shutil.move(tmp_dir, 'some_orca_outputs')
