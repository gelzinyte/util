import click
import os
import subprocess

@click.command('from-pattern')
@click.argument('pattern-fname')
@click.option('--start', '-s',  type=click.INT, default=1, show_default=True,
                help='no to start with')
@click.option('--num', '-n', type=click.INT, help='how many scripts to make')
@click.option('--submit', is_flag=True, help='whether to submit the scripts')
@click.option('--dir-prefix', help='prefix for directory to submit from, if any')
def sub_from_pattern(pattern_fname, start, num, submit, dir_prefix):
    """makes submission scripts from pattern"""

    with open(pattern_fname, 'r') as f:
        pattern = f.read()

    for idx in range(start, num+1):

        if dir_prefix:
            dir_name=f'{dir_prefix}{idx}'
            orig_dir = os.getcwd()
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            os.chdir(dir_name)

        text = pattern.replace('<idx>', str(idx))
        sub_name = f'sub_{idx}.sh'

        with open(sub_name, 'w') as f:
            f.write(text)

        if submit:
            subprocess.run(f'qsub {sub_name}', shell=True)

        if dir_prefix:
            os.chdir(orig_dir)