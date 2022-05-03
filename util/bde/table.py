from util import natural_sort
import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings
import pdb

import logging
logger = logging.getLogger(__name__)

def atom_sorter(atoms, info_key='mol_or_rad'):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda at: [convert(c) for c in at.info[info_key]]
    return sorted(atoms, key=alphanum_key)


def assign_bde_info(all_atoms, prop_prefix, dft_prop_prefix, h_energy=None, isolated_h=None):
    """prop_prefix - which prefix to use for energies for bdes
        dft_prop_prefix - used to get dft_opt_mol_positions_hash"""

    if h_energy is not None:
        assert isolated_h is None
    if isolated_h is not None:
        assert h_energy is None
        h_energy = isolated_h.info[f"{prop_prefix}energy"]

    atoms_by_hash = get_atoms_by_hash_dict(all_atoms, dft_prop_prefix)

    atoms_out = []
    for hash, atoms in atoms_by_hash.items():

        keys = natural_sort([at.info['mol_or_rad'] for at in atoms])
        atoms = sorted(atoms, key=lambda x: keys.index(x.info['mol_or_rad']))

        mol = atoms[0]
        rads = atoms[1:]

        assert mol.info['mol_or_rad'] == 'mol'
        rad_labels = [rad.info['mol_or_rad'] for rad in rads]
        for label in rad_labels:
            assert 'rad' in label

        mol_energy = mol.info[f'{prop_prefix}energy']
        for rad in rads:
            rad_energy = rad.info[f'{prop_prefix}energy']
            bde = get_bde(mol_energy=mol_energy,
                          rad_energy=rad_energy,
                          isolated_h_energy=h_energy)

            rad.info[f'{prop_prefix}bde_energy'] = bde

        atoms_out += [mol] + rads

    return atoms_out



def multiple_tables_from_atoms(all_atoms, isolated_h, pred_prop_prefix,
                              dft_prefix='dft_', printing=False, precision=3):
    """sorts atoms in file by their hash and prints/collects bde tables:
    """

    atoms_by_hash = get_atoms_by_hash_dict(all_atoms, dft_prefix)

    data_out = []

    for hash, atoms in atoms_by_hash.items():

        if printing:
            print(f'\n\n\n{hash}: {len(atoms[0])} atoms')

        data = bde_table(atoms=atoms,
                         pred_prop_prefix=pred_prop_prefix,
                         isolated_h=isolated_h,
                         dft_prefix=dft_prefix,
                         printing=printing,
                         precision=precision)

        if printing:
            print(f'\ndifferences, meV. ')
            print(f'radical')
            dif_table_rad = print_error_diff_table(data, compound="rad")
            print('\nmolecule')
            dif_table_mol = print_error_diff_table(data, compound="mol")
            print(f'\nbde_error:BDE error: {data.loc["rad", "abs_bde_err"]:.3f} eV')
            expected = dif_table_rad.loc["dft_opt_dft_E", "ip_opt_dft_E"] - dif_table_rad.loc["ip_opt_ip_E", "ip_opt_dft_E"] - dif_table_mol.loc["dft_opt_dft_E", "ip_opt_dft_E"] + dif_table_mol.loc["ip_opt_ip_E", "ip_opt_dft_E"]
            print(f'expected bde error: {expected:.3f}')
        
        data_out.append(data)

    return data_out

def print_error_diff_table(data, compound):

    interesting = ["dft_opt_dft_E", "dft_opt_ip_E", "ip_opt_ip_E", "ip_opt_dft_E"]
    names = ["original"] + interesting

    t = pd.DataFrame(index=names, columns=names)

    for col_name in names:
        for row_name in names:
            if col_name == row_name: 
                continue
            if col_name == "original":
                val = data.loc[compound, row_name]
            elif row_name == "original":
                val = data.loc[compound, col_name]
            else:
                val = (data.loc[compound, row_name] - data.loc[compound, col_name]) * 1e3

            t.loc[row_name, col_name] = val

    print_table(t)
    return t
            


def get_atoms_by_hash_dict(atoms, dft_prefix):
    """returns dictionary of hash:[Atoms]"""

    # put everything into a dictionary 
    atoms_by_hash = {}
    for at in atoms:
        hash = at.info[f'{dft_prefix}opt_mol_positions_hash']
        if hash not in atoms_by_hash.keys():
            atoms_by_hash[hash] = []
        atoms_by_hash[hash].append(at)

    # check which hashes don't have a molecule in them
    no_mol_hashes = []
    no_mol_compounds = {}
    for hash, atoms in atoms_by_hash.items():
        mol_or_rads = [at.info['mol_or_rad'] for at in atoms]
        if 'mol' not in mol_or_rads:
            no_mol_hashes.append(hash)
            comp = atoms[0].info["compound"]
            if comp not in no_mol_compounds.keys():
                no_mol_compounds[comp] = len(mol_or_rads)  
            else:
                no_mol_compounds[comp] += len(mol_or_rads)

    if len(no_mol_compounds) > 0:
        warnings.warn(f"not generating BDEs for the following entries, because no molecule"
                f"was found. {no_mol_compounds}")

    # delete entries without corresponding molecule
    for bad_hash in no_mol_hashes:
        del atoms_by_hash[bad_hash]

    return atoms_by_hash


def bde_table(atoms, pred_prop_prefix, isolated_h, dft_prefix='dft_',  printing=False, precision=3):
    """makes a pd/other t of bde properties for mol/rads with a given hash:

    -how well ip performs on ip-optimised configs on its own:
    * ip absolute energy error
    * rmse of force error
    * max force error

    -compare with dft-optimised structures
    * max displacement of best rmsd
    * ip-optimised structure's and dft-optimised structure's dft energy difference

    - interesting properties
    * bde error
    * ip BDE
    * dft BDE

    ---
    expected properties:

    {dft_prefix}opt_positions

    {dft_prefix}opt_{dft_prefix}forces
    {dft_prefix}opt_{dft_prefix}energy

    {dft_prefix}opt_{pred_prop_prefix}forces
    {dft_prefix}opt_{pred_prop_prefix}energy


    {pred_prop_prefix}opt_positions

    {pred_prop_prefix}opt_{pred_prop_prefix}forces
    {pred_prop_prefix}opt_{pred_prop_prefix}energy

    {pred_prop_prefix}opt_{dft_prefix}forces
    {pred_prop_prefix}opt_{dft_prefix}energy

    and for isolated_h simply:
    {pred_prop_prefix}energy
    {dft_prefix}energy

    """
    keys = natural_sort([at.info['mol_or_rad'] for at in atoms])
    atoms = sorted(atoms, key= lambda x: keys.index(x.info['mol_or_rad']))
    index = ['H'] + [at.info['mol_or_rad'] for at in atoms]

    # if index[1] != 'mol':
    # print(index)
    # print(atoms[0].info['dft_opt_mol_positions_hash'])


    columns = ['E_abs_at_err',  # meV/atom  between ip and dft on ip opt config
              'F_rmse',             # meV/Å between ip and dft on ip opt config
              'max_abs_F_err',    # meV/Å between ip and dft on ip opt config
              'max_rmsd', # Å between ip opt and dft opt configs
              'dft_E_diff',  # between dft equilibrium and ip equilibrium
              'abs_bde_err',     # meV between ip opt and dft opt configs
              'dft_bde',                # eV on dft opt mol and rad
              'ip_bde',                # eV on ip opt mol and rad
              'dft_opt_dft_E',     # eV on dft opt mol/rad
              'dft_opt_ip_E',
              'ip_opt_ip_E',     # eV on ip opt mol/rad
              'ip_opt_dft_E'
              ]


    t = pd.DataFrame(columns=columns, index=index)

    add_H_data(t, isolated_h=isolated_h, pred_prop_prefix=pred_prop_prefix,
                   dft_prefix=dft_prefix)

    mol = atoms[0]
    rads = atoms[1:]

    add_mol_data(t, mol=mol, pred_prop_prefix=pred_prop_prefix, dft_prefix=dft_prefix)
    for rad in rads:
        add_rad_data(t, mol=mol, rad=rad, isolated_h=isolated_h,
                     pred_prop_prefix=pred_prop_prefix, dft_prefix=dft_prefix)


    if printing:
        print_table(t, precision=precision)

    return t

def print_table(t, precision=3):
    pd.options.display.float_format = lambda x: f'{{:,.{precision}f}}'.format(x) if int(
        x) == x else f'{{:,.{precision}f}}'.format(x)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(t)


def abs_error(energy1, energy2):
    abs_err = np.abs(energy2 - energy1) * 1e3
    return abs_err


def F_rmse(forces1, forces2):
    return np.sqrt(np.mean(forces1 - forces2)**2) * 1e3


def max_abs_F_err(forces1, forces2):
    return np.max(np.abs(forces1 - forces2)) * 1e3


def add_mol_data(t, mol, pred_prop_prefix, dft_prefix):

    assert mol.info['mol_or_rad'] == 'mol'

    n_atoms = len(mol)

    label = 'mol'

    ip_opt_ip_E = mol.info[f'{pred_prop_prefix}opt_{pred_prop_prefix}energy']
    ip_opt_ip_forces = mol.arrays[f'{pred_prop_prefix}opt_{pred_prop_prefix}forces']

    dft_opt_dft_E = mol.info[f'{dft_prefix}opt_{dft_prefix}energy']
    dft_opt_ip_E = mol.info[f'{dft_prefix}opt_{pred_prop_prefix}energy']

    ip_opt_dft_E = mol.info.get(f'{pred_prop_prefix}opt_{dft_prefix}energy', None)
    ip_opt_dft_forces = mol.arrays.get(f'{pred_prop_prefix}opt_{dft_prefix}forces', None)


    if ip_opt_dft_E is not None:
        t.loc[label, 'E_abs_at_err'] = abs_error(ip_opt_ip_E,
                                                      ip_opt_dft_E) / n_atoms

        t.loc[label, 'F_rmse'] = F_rmse(ip_opt_ip_forces,
                                        ip_opt_dft_forces)

        t.loc[label, 'max_abs_F_err'] = max_abs_F_err(ip_opt_ip_forces,
                                                              ip_opt_dft_forces)

        t.loc[label, 'dft_E_diff'] = abs_error(ip_opt_dft_E,
                                                          dft_opt_dft_E)

    # t.loc[label, 'max_rmsd'] = get_max_rmsd()

    t.loc[label, 'dft_opt_dft_E'] = dft_opt_dft_E
    if ip_opt_dft_E is not None:
        t.loc[label, 'ip_opt_dft_E'] = ip_opt_dft_E
    t.loc[label, 'ip_opt_ip_E'] = ip_opt_ip_E
    t.loc[label, 'dft_opt_ip_E'] = dft_opt_ip_E

def get_bde(mol_energy, rad_energy, isolated_h_energy):
    return (rad_energy + isolated_h_energy - mol_energy)

def add_rad_data(t, mol, rad, isolated_h, pred_prop_prefix, dft_prefix):
    """ adds radical data"""

    label =  rad.info['mol_or_rad']
    assert 'rad' in label, f'"rad" not in label {label}'

    n_atoms = len(rad)

    ip_opt_ip_E = rad.info[f'{pred_prop_prefix}opt_{pred_prop_prefix}energy']
    ip_opt_ip_forces = rad.arrays[f'{pred_prop_prefix}opt_{pred_prop_prefix}forces']

    dft_opt_dft_E = rad.info[f'{dft_prefix}opt_{dft_prefix}energy']
    dft_opt_ip_E = rad.info[f'{dft_prefix}opt_{pred_prop_prefix}energy']

    ip_opt_dft_E = rad.info.get(f'{pred_prop_prefix}opt_{dft_prefix}energy', None)
    ip_opt_dft_forces = rad.arrays.get(f'{pred_prop_prefix}opt_{dft_prefix}forces', None)

    if ip_opt_dft_E is not None:
        # ip performance alone
        t.loc[label, 'E_abs_at_err'] = abs_error(ip_opt_ip_E,
                                                      ip_opt_dft_E) / n_atoms

        t.loc[label, 'F_rmse'] = F_rmse(ip_opt_ip_forces,
                                            ip_opt_dft_forces)

        t.loc[label, 'max_abs_F_err'] = max_abs_F_err(ip_opt_ip_forces,
                                                              ip_opt_dft_forces)

        t.loc[label, 'dft_E_diff'] = abs_error(ip_opt_dft_E,
                                                          dft_opt_dft_E)

    # ip optimisation wrt dft optimisation
    # t.loc[label, 'max_rmsd'] = 'to do'

    # just original energy values
    t.loc[label, 'dft_opt_dft_E'] = dft_opt_dft_E
    if ip_opt_dft_E is not None:
        t.loc[label, 'ip_opt_dft_E'] = ip_opt_dft_E
    t.loc[label, 'ip_opt_ip_E'] = ip_opt_ip_E
    t.loc[label, 'dft_opt_ip_E'] = dft_opt_ip_E

    # bde stuff
    mol_ip_opt_ip_E = mol.info[f'{pred_prop_prefix}opt_{pred_prop_prefix}energy']
    mol_dft_opt_dft_E = mol.info[f'{dft_prefix}opt_{dft_prefix}energy']
    isolated_h_ip_energy = isolated_h.info[f'{pred_prop_prefix}energy']
    isolated_h_dft_energy = isolated_h.info[f'{dft_prefix}energy']

    ip_bde = get_bde(mol_energy = mol_ip_opt_ip_E,
                      rad_energy = ip_opt_ip_E,
                      isolated_h_energy = isolated_h_ip_energy)

    dft_bde = get_bde(mol_energy = mol_dft_opt_dft_E,
                      rad_energy = dft_opt_dft_E,
                      isolated_h_energy = isolated_h_dft_energy)

    t.loc[label, 'abs_bde_err'] = abs_error(dft_bde, ip_bde)
    t.loc[label, 'dft_bde'] = dft_bde
    t.loc[label, 'ip_bde'] = ip_bde



def add_H_data(t, isolated_h, pred_prop_prefix, dft_prefix):
    """adds H data to the table, where applicable"""

    ip_energy = isolated_h.info[f'{pred_prop_prefix}energy']
    dft_energy = isolated_h.info[f'{dft_prefix}energy']

    t.loc['H', 'E_abs_at_err'] = abs_error(ip_energy, dft_energy)
    t.loc['H', 'dft_opt_dft_E']= dft_energy
    t.loc['H', 'ip_opt_ip_E'] = ip_energy




def align_structures(atoms):
    """aligns structures to minimise RMSD and gives back the max displacement"""
    pass


