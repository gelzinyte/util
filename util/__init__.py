# General imports
import re
import numpy as np
import subprocess
from itertools import zip_longest

# Packages for atoms and molecules
import ase
from collections import OrderedDict
from quippy.potential import Potential
from quippy.descriptors import Descriptor
import matplotlib.pyplot as plt
from math import log10, floor

from ase.optimize.precon import PreconLBFGS


from asaplib.data import ASAPXYZ
from asaplib.reducedim import Dimension_Reducers



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


def get_rmse(ref_ar, pred_ar):
    sq_error = []
    for val1, val2 in zip(ref_ar, pred_ar):
        sq_error.append((val1-val2)**2)
    return np.sqrt(np.mean(sq_error))

def get_rmse_over_ref_std(pred_ar, ref_ar):
    rmse = get_rmse(pred_ar, ref_ar)
    std = np.std(ref_ar)
    return rmse/std*100

def get_mae(pred_ar, ref_ar):
    absolute_errors = [np.abs(val1 - val2) for val1, val2 in zip(pred_ar, ref_ar)]
    return np.mean(absolute_errors)


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
    '''sets at.info.'dft_energy' to 'energy' '''
    for at in atoms:
        at.info['dft_energy'] = at.info['energy']
        at.arrays['dft_forces'] = at.arrays['forces']
    return atoms

def round_sig(x, sig=2):
  return round(x, sig-int(floor(log10(abs(x))))-1)

def clear_at_info(at):
    positions = at.arrays['positions']
    numbers = at.arrays['numbers']

    at.info.clear()
    at.arrays.clear()
    at.arrays['positions'] = positions
    at.arrays['numbers'] = numbers
    return at

def str_to_list(str_of_list, type=float):
    list_of_str = str_of_list.strip('][').split(', ')
    if type==str:
        return [elem.strip("'") for elem in list_of_str]
    elif type==float:
        return [float(num) for num in list_of_str]
    else:
        return




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



def get_E_F_dict(atoms, calc_type, param_fname=None):
    '''Returns {'energy': {'config1':[energies],
                           'config2':[energies]},
                'forces':{sym1:{'config1':[forces],
                                'config2':[forces]},
                          sym2:{'config1':[forces],
                                'config2':[forces]}}}'''


    data = dict()
    data['energy'] = OrderedDict()
    data['forces'] = OrderedDict()

    # select which energies and forces to extract
    if calc_type.upper() == 'GAP':
        if param_fname:
            gap = Potential(param_filename=param_fname)
        else:
            raise NameError('GAP filename is not given, but GAP energies requested.')

    else:
        if param_fname:
            print(f"WARNING: calc_type selected as {calc_type}, but gap filename is given, are you sure?")
        energy_name = f'{calc_type}_energy'
        if energy_name not in atoms[0].info.keys():
            print(f"WARNING: '{calc_type}_energy' not found, using 'energy', which might be anything")
            energy_name = 'energy'
        force_name = f'{calc_type}_forces'
        if force_name not in atoms[0].arrays.keys():
            print(f"WARNING: '{calc_type}_forces' not in found, using 'forces', which might be anything")
            force_name = 'forces'


    for atom in atoms:
        at = atom.copy()
        config_type='no_config_type'
        if 'config_type' in at.info.keys():
            config_type = at.info['config_type']


        if len(at) != 1:
            if calc_type.upper() == 'GAP':
                at.set_calculator(gap)
                energy = at.get_potential_energy() / len(at)
                try:
                    data['energy'][config_type] = np.append(data['energy'][config_type], energy)
                except KeyError:
                    data['energy'][config_type] = np.array([])
                    data['energy'][config_type] = np.append(data['energy'][config_type], energy)
                forces = at.get_forces()

            else:
                try:
                    data['energy'][config_type] = np.append(data['energy'][config_type], at.info[energy_name] / len(at))
                except KeyError:
                    data['energy'][config_type] = np.array([])
                    data['energy'][config_type] = np.append(data['energy'][config_type], at.info[energy_name] / len(at))
                forces = at.arrays[force_name]

            sym_all = at.get_chemical_symbols()
            for j, sym in enumerate(sym_all):
                if sym not in data['forces'].keys():
                    data['forces'][sym] = OrderedDict()
                try:
                    data['forces'][sym][config_type] = np.append(data['forces'][sym][config_type], forces[j])
                except KeyError:
                    data['forces'][sym][config_type] = np.array([])
                    data['forces'][sym][config_type] = np.append(data['forces'][sym][config_type], forces[j])

    return data

def dict_to_vals(my_dict):
    ''' concatenates dictionary of multiple dictionary:value  to dictionary:[value1, value2, ...]'''
    all_values = []
    for type, values in my_dict.items():
        all_values.append(values)
    all_values = np.concatenate(all_values)
    return all_values


def desymbolise_force_dict(my_dict):
    '''from dict['forces'][sym]:values makes dict['forces']:[values1, values2...]'''
    force_dict = OrderedDict()
    for sym, sym_dict in my_dict.items():
        for config_type, values in sym_dict.items():
            try:
                force_dict[config_type] = np.append(force_dict[config_type], values)
            except KeyError:
                force_dict[config_type] = np.array([])
                force_dict[config_type] = np.append(force_dict[config_type], values)

    return force_dict

def write_generic_submission_script(script_fname, job_name, command, no_cores=1):

    bash_script_start = '#!/bin/bash \n' + \
                        f'#$ -pe smp {no_cores} \n' + \
                        '#$ -l h_rt=4:00:00 \n' + \
                        '#$ -q  "orinoco|tomsk" \n' + \
                        '#$ -S /bin/bash \n'
    # '#$ -N namename '
    bash_script_middle = '#$ -j yes \n' + \
                         '#$ -cwd \n' + \
                         'echo "-- New Job --"\n ' + \
                         'export  OMP_NUM_THREADS=${NSLOTS} \n' + \
                         'source /home/eg475/programs/miniconda3/etc/profile.d/conda.sh \n' + \
                         'conda activate general \n' + \
                         'echo "running script" \n '
    # molpro command
    # 'echo "-- The End--"


    with open(script_fname, 'w') as f:
        f.write(bash_script_start)
        f.write(f'#$ -N {job_name}\n')
        f.write(bash_script_middle)
        f.write(f'{command} \n')
        f.write('echo "--- The End ---"')

def do_kpca(xyz_fname):

    asapxyz = ASAPXYZ(xyz_fname, periodic=False)
    soap_spec = {'soap1': {'type': 'SOAP',
                           'cutoff': 4.0,
                           'n': 6,
                           'l': 6,
                           'atom_gaussian_width': 0.5,
                           'crossover': False,
                           'rbf': 'gto'}}


    reducer_spec = {'reducer1': {
        'reducer_type': 'average',
        # [average], [sum], [moment_average], [moment_sum]
        'element_wise': False}}


    desc_spec = {'avgsoap': {
        'atomic_descriptor': soap_spec,
        'reducer_function': reducer_spec}}

    asapxyz.compute_global_descriptors(desc_spec_dict=desc_spec,
                                       sbs=[],
                                       keep_atomic=False,
                                       tag='tio2')

    reduce_dict = {}
    reduce_dict['kpca'] = {"type": 'SPARSE_KPCA',
                           'parameter': {"n_components": 10,
                                         "n_sparse": -1,  # no sparsification
                                         "kernel": {"first_kernel": {
                                             "type": 'linear'}}}}

    dreducer = Dimension_Reducers(reduce_dict)
    dm = asapxyz.fetch_computed_descriptors(['avgsoap'])
    proj = dreducer.fit_transform(dm)

    atoms = ase.io.read(xyz_fname, ':')
    atoms_out = []
    for coord, at in zip(proj, atoms):
        at.info[f'pca_d_10'] = coord
        atoms_out.append(at)

    ase.io.write(xyz_fname, atoms_out, 'extxyz', write_results=False)
    # return(atoms_out)


def shell_stdouterr(raw_command, cwd=None):
    """Abstracts the standard call of the commandline, when
    we are only interested in the stdout and stderr
    """
    stdout, stderr = subprocess.Popen(raw_command,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True,
                                      shell=True, cwd=cwd).communicate()
    return stdout.strip(), stderr.strip()


soap_param =  {'name': 'soap',
               'l_max': '4',
               'n_max': '8',
               'cutoff': '3.0',
               'atom_gaussian_width': '0.3',
               'add_species': 'True',
               'average':'True'}
soap = Descriptor(args_str='SOAP', **soap_param)

def get_soap(at, desc):
    return desc.calc_descriptor(at)[0]

def soap_sim(at1, at2, desc=soap):
    return np.dot(get_soap(at1, desc), get_soap(at2, desc))

def soap_dist(at1, at2, desc=soap):
    sp1 = get_soap(at1, desc)
    sp2 = get_soap(at2, desc)
    return np.sqrt(2 - 2 * np.dot(sp1, sp2))

def get_E_F_dict_evaled(atoms, energy_name, force_name):
    '''Returns {'energy': {'config1':[energies],
                           'config2':[energies]},
                'forces':{sym1:{'config1':[forces],
                                'config2':[forces]},
                          sym2:{'config1':[forces],
                                'config2':[forces]}}}'''

    data = dict()
    data['energy'] = OrderedDict()
    data['forces'] = OrderedDict()


    for atom in atoms:
        at = atom.copy()
        config_type='no_config_type'
        if 'config_type' in at.info.keys():
            config_type = at.info['config_type']


        if len(at) != 1:

            try:
                data['energy'][config_type] = np.append(data['energy'][config_type], at.info[energy_name] / len(at))
            except KeyError:
                data['energy'][config_type] = np.array([])
                data['energy'][config_type] = np.append(data['energy'][config_type], at.info[energy_name] / len(at))


            forces = at.arrays[force_name]

            sym_all = at.get_chemical_symbols()
            for j, sym in enumerate(sym_all):
                if sym not in data['forces'].keys():
                    data['forces'][sym] = OrderedDict()
                try:
                    data['forces'][sym][config_type] = np.append(data['forces'][sym][config_type], forces[j])
                except KeyError:
                    data['forces'][sym][config_type] = np.array([])
                    data['forces'][sym][config_type] = np.append(data['forces'][sym][config_type], forces[j])

    return data