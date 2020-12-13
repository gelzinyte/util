import click
import numpy as np
from ase.io import read, write
import subprocess
import os
from ase import Atoms


@click.command()
@click.argument('dimers', nargs=-1)
@click.option('--no_cores', type=int, default=16)
@click.option('--scratch_dir', type=click.Path(), default='/tmp/eg475')
@click.option('--prepare_only', is_flag=True)
def get_dimers(dimers, no_cores, scratch_dir, prepare_only):

    print(f'dimers: {dimers}')

    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    home_dir = os.getcwd()
    distances = np.concatenate([np.arange(0.1, 2.0, 0.02), np.arange(2.0, 6.0, 0.1)])


    for dimer in dimers:

        print(f'preparing dimer {dimer}')


        if not os.path.exists(dimer):
            os.makedirs(dimer)
        os.chdir(dimer)

        #isolated atoms
        isolated_ats = []
        for sym in dimer:
            at = Atoms(sym, positions=[(0, 0, 0)])
            isolated_ats.append(at)
        write(f'isolated_{dimer}.xyz', isolated_ats)

        # dimer
        dimer_ats = []
        for dist in distances:
            at = Atoms(dimer, positions=[(0, 0, 0), (0, 0, dist)])
            dimer_ats.append(at)
        write(f'{dimer}.xyz', dimer_ats)

        #submit a calculation
        sub_script = f'#!/bin/bash\n'\
                     f'#$ -pe smp {no_cores}\n'\
                    f'#$ -l h_rt=12:00:00 \n'\
                    f'#$ -S /bin/bash \n'\
                    f'#$ -N j{dimer}\n'\
                    f'#$ -j yes\n'\
                    f'#$ -cwd \n'\
                    'source /home/eg475/programs/miniconda3/etc/profile.d/conda.sh\n'\
                    'conda activate wo0\n'\
                    'export  OMP_NUM_THREADS=${NSLOTS}\n'

        for in_fname, out_fname in zip([f'{dimer}.xyz', f'isolated_{dimer}.xyz'], [f'{dimer}_out.xyz', f'isolated_{dimer}_out.xyz']):

            sub_script += f'wfl ref-method orca-eval --output-file {out_fname} --base-rundir orca_outputs --keep-files True --kw "smearing=2000"'\
                          f' -tmp {scratch_dir} -nr 1 -nh 30 --orca-simple-input "UKS B3LYP def2-SV(P) def2/J D3BJ" {in_fname}\n'

        with open(f'sub.sh', 'w') as f:
            f.write(sub_script)

        if not prepare_only:
            subprocess.run('qsub sub.sh', shell=True)

        os.chdir(home_dir)

if __name__ == '__main__':
    get_dimers()







