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
def print_populations(orca_out, xyz_in, orca_in, xyz_out, pop, cmap):
    # TODO make reading xyz from orca outptput file
    '''  NA   - Mulliken gross atomic population
  ZA   - Total nuclear charge
  QA   - Mulliken gross atomic charge
  VA   - Mayer's total valence
  BVA  - Mayer's bonded valence
  FA   - Mayer's free valence
'''

    if xyz_in:
        ats = read(xyz_in)
    elif orca_in:
        ats = read_geom_orcainp(orca_in)

    pop_data = get_pop_dict(orca_out)

    if not (np.array(list(ats.symbols)) == np.array(pop_data['elements'])).all():
        raise RuntimeError("atoms and populations' elements don't match")
    del pop_data['elements']

    for key, val in pop_data.items():
        ats.arrays[key] = np.array(val)

    ats = color_by_pop(ats, pop, cmap)


    if not xyz_out:
        write(xyz_in, ats)
    else:
        write(xyz_out, ats)

def color_by_pop(ats, pop, cmap):
    values = ats.arrays[pop]

    # norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values), clip=True)
    norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap))

    colors = np.array([mapper.to_rgba(v)[0:3] for v in values])

    ats.arrays['Color'] = colors
    return ats



def get_pop_dict(in_fname):
    '''  NA   - Mulliken gross atomic population
  ZA   - Total nuclear charge
  QA   - Mulliken gross atomic charge
  VA   - Mayer's total valence
  BVA  - Mayer's bonded valence
  FA   - Mayer's free valence
'''

    d = {'elements':[], 'NA':[], 'QA':[], 'VA':[], 'BVA':[], 'FA':[]}
    # kw_to_idx = {'NA': 2, 'QA':4, 'VA':5, 'BVA':6, 'FA':7}

    with open(in_fname) as f:
        output = f.read().splitlines()

    for idx, line in enumerate(output):
        if 'MAYER POPULATION ANALYSIS' in line:
            start = idx + 11
        if 'Mayer bond orders larger than' in line:
            end = idx - 2
            break


    for line in output[start : end +1]:
        l = line.split()
        d['elements'].append(l[1])
        d['NA'].append(float(l[2]))
        d['QA'].append(float(l[4]))
        d['VA'].append(float(l[5]))
        d['BVA'].append(float(l[6]))
        d['FA'].append(float(l[7]))

    return d


if __name__ == '__main__':
    print_populations()
