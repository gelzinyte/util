from util import natural_sort
import pandas as pd
import numpy as np
from tabulate import tabulate

def atom_sorter(atoms, info_key='mol_or_rad'):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda at: [convert(c) for c in at.info[info_key]]
    return sorted(atoms, key=alphanum_key)


def bde_table(atoms, gap_prefix, isolated_h, dft_prefix='dft_',  printing=False, precision=3):
    """makes a pd/other t of bde properties for mol/rads with a given hash:

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

    ---
    expected properties:

    {dft_prefix}opt_positions

    {dft_prefix}opt_{dft_prefix}forces
    {dft_prefix}opt_{dft_prefix}energy

    {dft_prefix}opt_{gap_prefix}forces
    {dft_prefix}opt_{gap_prefix}energy


    {gap_prefix}opt_positions

    {gap_prefix}opt_{gap_prefix}forces
    {gap_prefix}opt_{gap_prefix}energy

    {gap_prefix}opt_{dft_prefix}forces
    {gap_prefix}opt_{dft_prefix}energy

    and for isolated_h simply:
    {gap_prefix}energy
    {dft_prefix}energy

    """
    keys = natural_sort([at.info['mol_or_rad'] for at in atoms])
    atoms = sorted(atoms, key= lambda x: keys.index(x.info['mol_or_rad']))
    index = ['H'] + [at.info['mol_or_rad'] for at in atoms]


    columns = ['energy_absolute_error',  # meV/atom  between gap and dft on gap opt config
              'force_rmse',             # meV/Å between gap and dft on gap opt config
              'max_abs_force_error',    # meV/Å between gap and dft on gap opt config
              'max_distance_best_RMSD', # Å between gap opt and dft opt configs
              'dft_energy_difference',  # between dft equilibrium and gap equilibrium
              'absolute_bde_error',     # meV between gap opt and dft opt configs
              'dft_bde',                # eV on dft opt mol and rad
              'gap_bde',                # eV on gap opt mol and rad
              'dft_opt_dft_energy',             # eV on dft opt mol/rad
              'dft_opt_gap_energy',
              'gap_opt_gap_energy',              # eV on gap opt mol/rad
              'gap_opt_dft_energy'
              ]


    t = pd.DataFrame(columns=columns, index=index)

    add_H_data(t, isolated_h=isolated_h, gap_prefix=gap_prefix,
                   dft_prefix=dft_prefix)

    mol = atoms[0]
    rads = atoms[1:]

    add_mol_data(t, mol=mol, gap_prefix=gap_prefix, dft_prefix=dft_prefix)
    for rad in rads:
        add_rad_data(t, mol=mol, rad=rad, isolated_h=isolated_h,
                     gap_prefix=gap_prefix, dft_prefix=dft_prefix)


    if printing:
        pd.options.display.float_format = lambda x: f'{{:,.{precision}f}}'.format(x) if int(
            x) == x else f'{{:,.{precision}f}}'.format(x)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(t)

    return t

def abs_error(energy1, energy2):
    abs_err = np.abs(energy2 - energy1) * 1e3
    return abs_err

def force_rmse(forces1, forces2):
    return np.sqrt(np.mean(forces1 - forces2)**2) * 1e3

def max_abs_force_error(forces1, forces2):
    return np.max(np.abs(forces1 - forces2)) * 1e3

def add_mol_data(t, mol, gap_prefix, dft_prefix):

    assert mol.info['mol_or_rad'] == 'mol'

    label = 'mol'

    gap_opt_gap_energy = mol.info[f'{gap_prefix}opt_{gap_prefix}energy']
    gap_opt_dft_energy = mol.info[f'{gap_prefix}opt_{dft_prefix}energy']
    dft_opt_dft_energy = mol.info[f'{dft_prefix}opt_{dft_prefix}energy']
    dft_opt_gap_energy = mol.info[f'{dft_prefix}opt_{gap_prefix}energy']

    gap_opt_gap_forces = mol.arrays[f'{gap_prefix}opt_{gap_prefix}forces']
    gap_opt_dft_forces = mol.arrays[f'{gap_prefix}opt_{dft_prefix}forces']

    t.loc[label, 'energy_absolute_error'] = abs_error(gap_opt_gap_energy,
                                                  gap_opt_dft_energy)

    t.loc[label, 'force_rmse'] = force_rmse(gap_opt_gap_forces,
                                        gap_opt_dft_forces)

    t.loc[label, 'max_abs_force_error'] = max_abs_force_error(gap_opt_gap_forces,
                                                          gap_opt_dft_forces)

    t.loc[label, 'max_distance_best_RMSD'] = 'to do'

    t.loc[label, 'dft_energy_difference'] = abs_error(gap_opt_dft_energy,
                                                  dft_opt_dft_energy)

    t.loc[label, 'dft_opt_dft_energy'] = dft_opt_dft_energy
    t.loc[label, 'gap_opt_dft_energy'] = gap_opt_dft_energy
    t.loc[label, 'gap_opt_gap_energy'] = gap_opt_gap_energy
    t.loc[label, 'dft_opt_gap_energy'] = dft_opt_gap_energy

def get_bde(mol_energy, rad_energy, isolated_h_energy):
    return (rad_energy + isolated_h_energy - mol_energy)

def add_rad_data(t, mol, rad, isolated_h, gap_prefix, dft_prefix):
    """ adds radical data"""

    label =  rad.info['mol_or_rad']

    assert 'rad' in label, f'"rad" not in label {label}'

    gap_opt_gap_energy = rad.info[f'{gap_prefix}opt_{gap_prefix}energy']
    gap_opt_dft_energy = rad.info[f'{gap_prefix}opt_{dft_prefix}energy']
    dft_opt_dft_energy = rad.info[f'{dft_prefix}opt_{dft_prefix}energy']
    dft_opt_gap_energy = rad.info[f'{dft_prefix}opt_{gap_prefix}energy']

    gap_opt_gap_forces = rad.arrays[f'{gap_prefix}opt_{gap_prefix}forces']
    gap_opt_dft_forces = rad.arrays[f'{gap_prefix}opt_{dft_prefix}forces']

    # gap performance alone
    t.loc[label, 'energy_absolute_error'] = abs_error(gap_opt_gap_energy,
                                                  gap_opt_dft_energy)

    t.loc[label, 'force_rmse'] = force_rmse(gap_opt_gap_forces,
                                        gap_opt_dft_forces)

    t.loc[label, 'max_abs_force_error'] = max_abs_force_error(gap_opt_gap_forces,
                                                          gap_opt_dft_forces)

    # gap optimisation wrt dft optimisation
    t.loc[label, 'max_distance_best_RMSD'] = 'to do'

    t.loc[label, 'dft_energy_difference'] = abs_error(gap_opt_dft_energy,
                                                  dft_opt_dft_energy)

    # just original energy values
    t.loc[label, 'dft_opt_dft_energy'] = dft_opt_dft_energy
    t.loc[label, 'gap_opt_dft_energy'] = gap_opt_dft_energy
    t.loc[label, 'gap_opt_gap_energy'] = gap_opt_gap_energy
    t.loc[label, 'dft_opt_gap_energy'] = dft_opt_gap_energy

    # bde stuff
    mol_gap_opt_gap_energy = mol.info[f'{gap_prefix}opt_{gap_prefix}energy']
    mol_dft_opt_dft_energy = mol.info[f'{dft_prefix}opt_{dft_prefix}energy']
    isolated_h_gap_energy = isolated_h.info[f'{gap_prefix}energy']
    isolated_h_dft_energy = isolated_h.info[f'{dft_prefix}energy']

    gap_bde = get_bde(mol_energy = mol_gap_opt_gap_energy,
                      rad_energy = gap_opt_gap_energy,
                      isolated_h_energy = isolated_h_gap_energy)

    dft_bde = get_bde(mol_energy = mol_dft_opt_dft_energy,
                      rad_energy = dft_opt_dft_energy,
                      isolated_h_energy = isolated_h_dft_energy)

    t.loc[label, 'absolute_bde_error'] = abs_error(dft_bde, gap_bde)
    t.loc[label, 'dft_bde'] = dft_bde
    t.loc[label, 'gap_bde'] = gap_bde



def add_H_data(t, isolated_h, gap_prefix, dft_prefix):
    """adds H data to the table, where applicable"""

    gap_energy = isolated_h.info[f'{gap_prefix}energy']
    dft_energy = isolated_h.info[f'{dft_prefix}energy']

    t.loc['H', 'energy_absolute_error'] = abs_error(gap_energy, dft_energy)
    t.loc['H', 'dft_opt_dft_energy']= dft_energy
    t.loc['H', 'gap_opt_gap_energy'] = gap_energy




def align_structures(atoms):
    """aligns structures to minimise RMSD and gives back the max displacement"""
    pass


