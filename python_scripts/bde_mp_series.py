from ase.io import read, write
import click
import subprocess

@click.command()
@click.argument('in_fname')
def generate_bdes(in_fname):

    atoms = read(in_fname, ':')
    for at in atoms:
        name = at.info['config_type']
        h_idx = at.info['h_to_rm']

        at.rattle(stdev=0.1)

        write(f'{name}_bde_in.xyz', at)

        sub = '#!/bin/bash\n' \
              f'#$ -pe smp 2 \n' \
              '#$ -l h_rt=12:00:00\n' \
              '#$ -S /bin/bash\n' \
              f'#$ -N j{name}\n' \
              '#$ -j yes\n' \
              '#$ -cwd\n' \
              'mkdir -p /tmp/eg475 \n' \
              'export OMP_NUM_THREADS=${NSLOTS}\n' \
              'source /home/eg475/programs/miniconda3/etc/profile.d/conda.sh\n'\
              'conda activate wo0\n'


        sub += f'python ~/scripts/python_scripts/generate_bde_files.py --in_fname {name}_bde_in.xyz ' \
               f'--h_list "[{h_idx}]" --prefix {name} --orca_wdir orca_opt_{name}\n'

        with open(f'{name}_sub.sh', 'w') as f:
            f.write(sub)

        print(f'submitting {name}')
        subprocess.run(f'qsub {name}_sub.sh', shell=True)

if __name__=='__main__':
    generate_bdes()
