from ase.io import read
from wfl.fit import ace
import yaml



configs = read("/home/eg475/scripts/tests/files/tiny_train_set.xyz", ':')
with open("ace_params.yml", 'r') as f:
    params = yaml.safe_load(f)

ace.fit(fitting_configs=configs, 
        ACE_name='ACE_name',
        params=params,
        ref_property_prefix='dft_',
        run_dir='ace_rundir',
        formats=['json'])
