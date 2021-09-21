import re
import numpy as np
import click
import pandas as pd
from ase.io.orca import read_geom_orcainp
from ase.io import read, write
import matplotlib as mpl
from matplotlib import cm



@click.command()
@click.option('--orca_out')
@click.option('--xyz_out')
@click.option('--xyz_in')
@click.option('--orca_in')
@click.option('--pop', default='NA')
@click.option('--cmap', default='Reds')


if __name__ == '__main__':
    print_populations()
