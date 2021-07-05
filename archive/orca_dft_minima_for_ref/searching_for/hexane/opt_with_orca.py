import os
import subprocess
from ase.io import read
import util
import click

@click.command()
@click.option('--at_fname', type=click.Path(exists=True), help='frames to optimise with orca')
def multi_orca_opts(at_fname):

    atl = read(at_fname, ':')

    home_dir = os.getcwd()

    orca_inp = '! UKS B3LYP def2-SV(P) def2/J D3BJ Opt\n' \
               '%scf Convergence VeryTight\n' \
               'SmearTemp 2000\n' \
               'maxiter 500\n' \
               'end\n' \
               '*xyz 0 1\n'



    for idx, atoms in enumerate(atl):

        orca_wdir = f'orca{idx}'
        if not os.path.isdir(orca_wdir):
            os.makedirs(orca_wdir)
        os.chdir(orca_wdir)

        orca_input = f'orca{idx}.inp'
        orca_output = f'orca{idx}.out'
        sub_fname = f'sub{idx}.sh'

        with open(orca_input, 'w') as f:
            f.write(orca_inp)
            for at in atoms:
                f.write(f'{at.symbol} {at.position[0]} {at.position[1]} {at.position[2]}\n')
            f.write('*\n')

        sub_script = '#!/bin/bash\n' \
                     '#$ -pe smp 1 \n' \
                     '#$ -l h_rt=12:00:00\n' \
                     '#$ -S /bin/bash\n' \
                     f'#$ -N opt{idx}\n' \
                     '#$ -j yes\n' \
                     '#$ -cwd\n'

        orca_command = f'/home/eg475/programs/orca/orca_4_2_1_linux_x86-64_openmpi314/orca {orca_input} > {orca_output}\n'
        sub_script += orca_command
        with open(sub_fname, 'w') as f:
            f.write(sub_script)

        subprocess.run(f'qsub {sub_fname}', shell=True)

        os.chdir(home_dir)

if __name__=='__main__':
    multi_orca_opts()
