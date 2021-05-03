from util import natural_sort
import pandas as pd
from tabulate import tabulate

def atom_sorter(atoms, info_key='mol_or_rad'):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda at: [convert(c) for c in at.info[info_key]]
    return sorted(atoms, key=alphanum_key)


def bde_t(atoms, gap_prefix, dft_prefix, isolated_h, printing=False, precision=3):
    """makes a pd/other t of bde properties for mol/rads:

    -how well gap performs on gap-optimised configs on its own:
    * gap absolute energy error
    * rmse of force error
    * max force error

    -compare with dft-optimised structures
    * max displacement of best rmsd
    * gap-optimised structure's and dft-optimised structure's dft energy difference

    - interesting properties
    * bde error
    * gap BDE
    * dft BDE

    """
    keys = natural_sort([at.info['mol_or_rad'] for at in atoms])
    atoms = [isolated_h] + sorted(atoms, key= lambda x: keys.index(x.info['mol_or_rad']))
    index = [at.info['mol_or_rad'] for at in atoms]


    colums = ['energy_absolute_error',  # meV/atom
              'force_rmse',             # meV/Å
              'max_abs_force_error',    # meV/Å
              'max_distance_best_RMSD', # Å
              'dft_energy_difference',  # between dft equilibrium and gap equilibrium
              'absolute_bde_error',     # meV
              'dft_bde',
              'gap_bde',
              'dft_energy',
              'gap_energy'
              ]


    t = pd.DataFrame(columns=columns, index=index)

    t = add_H_data()

    mol = atoms[0]
    rads = [atoms[1:]]

    assert mol.info['mol_or_rad'] == 'mol'

    # add all molecule stuff to t
    t['energy_absolute_error']['']
    



    print('-'*30)

    pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if int(
        x) == x else f'{{:,.{precision}f}}'.format(x)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(t)




def align_structures(atoms):
    """aligns structures to minimise RMSD and gives back the max displacement"""
    pass


