from quippy.potential import Potential
from ase.io import read, write
import subprocess
import click
import os
from util import tests

@click.command()
@click.option('--gap_fname', type=click.Path(exists=True), help='gap to do optimisation with')
@click.option('--start_fname', type=click.Path(exists=True), help='atoms to optimise')
@click.option('--finish_fname', type=click.Path(), help='atoms output fname')
@click.option('--traj_name', type=str, help='name for the optimisation trajectory')
@click.option('--steps', type=int, help='number of steps to run the optimisation')
@click.option('--fmax', type=float, help='fmax to optimise to.')
def do_opt(gap_fname, start_fname, finish_fname, traj_name, steps, fmax):

    gap = Potential(param_filename=gap_fname)
    at = read(start_fname)
    if not os.path.isfile(f'{traj_name}.xyz'):
        try:
            tests.do_gap_optimisation(at, traj_name, fmax, steps, gap=gap)
            print(f'successful opt {start_fname}')

            traj = read(f'{traj_name}.xyz', ':')
            finish = traj[-1]

            write(finish_fname, finish, 'extxyz', write_results=False)

        except RuntimeError as e:
            print(
                f'Failed to optimise found structure '
                f'{traj_name}.xyz, skipping')



if __name__=='__main__':
    do_opt()