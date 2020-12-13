import os
import shutil
import subprocess
from ase.io import read, write
import util
import click

# @click.command()
# @click.option('--at_fname', type=click.Path(exists=True), help='frames to optimise with orca')
# @click.option('--combine', default=False, help='only gathers results to one place if True')
# @click.option('--cluster', default='wo0', show_default=True, help='wheter to run on womble or womble0')
def multi_orca_opts(at_fname, combine, cluster):

    atl = read(at_fname, ':')

    home_dir = os.getcwd()

    orca_inp = '! UKS B3LYP def2-SV(P) def2/J D3BJ Opt\n' \
               '%scf Convergence VeryTight\n' \
               'SmearTemp 2000\n' \
               'maxiter 500\n' \
               'end\n' \
               '*xyz 0 2\n'


    if combine:
        all_trj_dir = 'all_orca_opt_trajs'
        if not os.path.isdir(all_trj_dir):
            os.makedirs(all_trj_dir)
        combined_ats = []

    for idx, atoms in enumerate(atl):

        wo0_start = '#!/bin/bash\n' \
                    '#$ -pe smp 1 \n' \
                    '#$ -l h_rt=12:00:00\n' \
                    '#$ -S /bin/bash\n' \
                    f'#$ -N opt{idx}\n' \
                    '#$ -j yes\n' \
                    '#$ -cwd\n'

        wo_start = '#!/bin/sh \n' \
                   f'#$ -N opt{idx} \n' \
                   '#$ -pe smp 1 \n' \
                   '#$ -q "bungo|tomsk|orinoco" \n' \
                   '#$ -l h_rt=12:00:00 \n' \
                   '#$ -S /bin/bash \n' \
                   '#$ -cwd \n' \
                   '#$ -j y \n\n'

        if combine:

            shutil.copy(f'orca/orca{idx}_trj.xyz', all_trj_dir)
            at = read(f'orca/orca{idx}.xyz')
            combined_ats.append(at)

        else:
            orca_wdir = f'orca'
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

            if cluster=='wo0':
                sub_script = wo0_start
            elif cluster == 'wo':
                sub_script = wo_start
            else:
                raise ValueError(f'Unknown cluster={cluster}, choose between "wo0" and "wo"')

            orca_command = f'/home/eg475/programs/orca/orca_4_2_1_linux_x86-64_openmpi314/orca {orca_input} > {orca_output}\n'
            sub_script += orca_command
            with open(sub_fname, 'w') as f:
                f.write(sub_script)

            subprocess.run(f'qsub {sub_fname}', shell=True)

            os.chdir(home_dir)

    if combine:
        write('combined_orca_opt_results.xyz', combined_ats, 'extxyz')

if __name__=='__main__':
    multi_orca_opts()
