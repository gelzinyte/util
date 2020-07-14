# General imports
import re
import numpy as np
from itertools import zip_longest

# Packages for atoms and molecules
import ase

# import submodules
# import util.ugap
# import util.itools
# import util.md
# import util.plot
# import util.urdkit
# import util.vib


def hello():
    print('Utils say hi')

def relax(at, calc, fmax=1e-3, steps=0):
    """Relaxes at with given calc with PreconLBFGS"""
    at.set_calculator(calc)
    opt = PreconLBFGS(at)
    opt.run(fmax=fmax, steps=steps)
    return at


def shift0(my_list, by=None):
    """shifts all values in list by first value or by value given"""
    if not by:
        by = my_list[0]
    return [x - by for x in my_list]


def get_counts(at):
    ''' Returns dictionary of chemical formula: dict[at_number] = count'''
    nos = at.get_atomic_numbers()
    unique, counts = np.unique(nos, return_counts=True)
    return dict(zip(unique, counts))


def get_rmse(ar1, ar2):
    sq_error = []
    for val1, val2 in zip(ar1, ar2):
        sq_error.append((val1-val2)**2)
    return np.sqrt(np.mean(sq_error))


def get_std(ar1, ar2):
    sq_error = []
    for val1, val2 in zip(ar1, ar2):
        sq_error.append((val1-val2)**2)
    return np.sqrt(np.var(sq_error))


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def dict_to_vals(my_dict):
    all_values = []
    for type, values in my_dict.items():
        all_values.append(values)
    all_values = np.concatenate(all_values)
    return all_values

def grouper(iterable, n, fillvalue=None):
    """groups a list/etc into chunks of n values, e.g.
    grouper('ABCDEFG', 3, 'x') --> 'ABC' 'DEF' 'Gxx'
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def set_dft_vals(atoms):
    for at in atoms:
        at.info['dft_energy'] = at.info['energy']
        at.arrays['dft_forces'] = at.arrays['forces']
    return atoms


def get_pair_dists(all_dists, at_nos, atno1, atno2):
    dists = []
    for i in range(len(all_dists[at_nos==atno1])):
        dists.append(np.array(all_dists[at_nos==atno1][i][at_nos==atno2]))

    dists = np.array(dists)
    if atno1==atno2:
        dists = np.triu(dists)
        dists = dists[dists!=0]
    else:
        dists = dists.flatten()
    return dists



def distances_dict(at_list):
    dist_hist = {}
    for at in at_list:
        all_dists = at.get_all_distances()

        at_nos = at.get_atomic_numbers()
        at_syms = np.array(at.get_chemical_symbols())

        formula = at.get_chemical_formula()
        unique_symbs = natural_sort(list(ase.formula.Formula(formula).count().keys()))
        for idx1, sym1 in enumerate(unique_symbs):
            for idx2, sym2 in enumerate(unique_symbs[idx1:]):
                label = sym1 + sym2

                atno1 = at_nos[at_syms == sym1][0]
                atno2 = at_nos[at_syms == sym2][0]

                if label not in dist_hist.keys():
                    dist_hist[label] = np.array([])

                distances = get_pair_dists(all_dists, at_nos, atno1, atno2)
                dist_hist[label] = np.concatenate([dist_hist[label], distances])
    return dist_hist

def rattle(at, stdev, natoms):
    '''natoms - how many atoms to rattle'''
    at = at.copy()
    pos = at.positions.copy()
    mask = np.ones(pos.shape).flatten()
    mask[:-natoms] = 0
    np.random.shuffle(mask)
    mask = mask.reshape(pos.shape)

    rng = np.random.RandomState()
    new_pos = pos + rng.normal(scale=stdev, size=pos.shape) * mask
    at.set_positions(new_pos)
    return at


def has_converged(template_path, molpro_out_path='MOLPRO/molpro.out'):
    with open(molpro_out_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'Final alpha occupancy' in line or 'Final occupancy' in line:
                final_iteration_no = int(re.search(r'\d+', lines[i - 2]).group())

    # print(final_iteration_no)
    maxit = 60  # the default
    with open(template_path, 'r') as f:
        for line in f:
            if 'maxit' in line:
                maxit = line.rstrip().split('maxit=')[1]  # take the non-default if present in the input
                break

    # print(maxit)
    if maxit == final_iteration_no:
        print(f'Final iteration no was found to be {maxit}, optimisation has not converged')
        return False
    else:
        return True


def gradient_test(mol, calc, start=-3, stop=-7):
    """gradient test on general molecule with general calculator"""
    unit_d = np.random.random((len(mol), 3))
    unit_d = unit_d / (np.linalg.norm(unit_d))

    num = start - stop + 1
    epsilons = np.logspace(start, stop, num=num)

    mol.set_calculator(calc)
    f_analytical = np.vdot(mol.get_forces(), unit_d)

    print('{:<12s} {:<22s} {:<25s} {:<20s}'.format('', '', 'force', 'initial energy'))
    print('{:<12} {:<22} {:<25.14e} {:<20}'.format('', '', f_analytical, mol.get_potential_energy()))

    print('{:<12s} {:<22s} {:<25s} {:<20s}'.format('epsilon', 'ratio', 'energy gradient', 'displaced energy'))

    for eps in epsilons:
        atoms = mol.copy()
        atoms.set_calculator(calc)

        e0 = atoms.get_potential_energy()

        pos = atoms.get_positions().copy()
        pos += eps * unit_d
        atoms.set_positions(pos)

        e_displ = atoms.get_potential_energy()

        numerator = -(e_displ - e0)
        f_numerical = numerator / eps
        if abs(f_analytical) < 0.1:
            ratio = (1 + f_analytical) / (1 + f_numerical)
        else:
            ratio = f_analytical / f_numerical

        print('{:<12} {:<22} {:<25.14e} {:<20}'.format(eps, ratio, f_numerical, e_displ))


