from collections import Counter, OrderedDict
import warnings
import numpy as np
from ase.io import write

from wfl.calculators import generic
from wfl.configset import OutputSpec

import pandas as pd


def plot(data, ref_prefix, pred_prefix, calculator=None, output_fname=None, chunksize=10, precision=4, 
         info_key="config_type"):
    """Plots error table by config type

    Parameters
    ----------
    data: list(Atoms) / CoinfigSet_in
        Structures with reference (and predicted) energies and forces
    ref_prefix: str
        Prefix for "energy" in Atoms.info and "forces" in Atoms.arrays for reference
    pred_prefix: str
        Prefix for predicted energy and forces
    calculator: calculator / (initializer, args, kwargs)
        Calculator to evaluate predicted energy & forces
    output_fname: str, default None
        Where to save data with predicted energy & forces, if at all
    chunksize: int, default 10
        How many structures to evaluate sequentially at calculators.generic.run

    Returns
    -------
    """

    assert ref_prefix != pred_prefix

    # Always evaluate configurations if calculator is given
    if calculator is not None:
        data = evaluate_data(data, calculator, pred_prefix, output_fname, chunksize)

    ref_data = read_energies_forces(data, ref_prefix, info_key)
    pred_data = read_energies_forces(data, pred_prefix, info_key)

    config_types = []
    for at in data:
        if len(at) != 1:
            if info_key in at.info.keys():
                config_types.append(at.info[info_key])
            else:
                config_types.append(f'no_{info_key}')

    # prepare table by config type
    config_counts = dict(Counter(config_types))
    total_count = sum([val for key, val in config_counts.items()])
    table = {}
    for config_type in config_counts.keys():
        table[config_type] = {}
        table[config_type]["Count"] = int(config_counts[config_type])
    table['overall'] = {}
    table['overall']["Count"] = total_count

    # read_energies_forces() partitions forces by element and by config_type
    # put all elements together; maybe will introduce force_by_element later
    tmp_no_f_sym_data = desymbolise_force_dict(ref_data['forces'])
    ref_data['forces'].clear()
    ref_data['forces'] = tmp_no_f_sym_data

    tmp_no_f_sym_data = desymbolise_force_dict(pred_data['forces'])
    pred_data['forces'].clear()
    pred_data['forces'] = tmp_no_f_sym_data

    # collect energy RMSEs
    e_key = 'E RMSE, meV/at'
    all_ref_vals = np.array([])
    all_pred_vals = np.array([])
    for ref_config_type, pred_config_type in zip(ref_data['energy'].keys(),
                                                 pred_data['energy'].keys()):
        if ref_config_type != pred_config_type:
            raise ValueError('Reference and predicted config_types do not match')

        ref_vals = ref_data['energy'][ref_config_type]
        pred_vals = pred_data['energy'][pred_config_type]
        all_ref_vals = np.concatenate([all_ref_vals, ref_vals])
        all_pred_vals = np.concatenate([all_pred_vals, pred_vals])

        rmse = get_rmse(ref_vals, pred_vals)
        table[ref_config_type][e_key] = rmse * 1000  # meV
    table['overall'][e_key] = get_rmse(all_ref_vals, all_pred_vals) * 1000

    # collect force RMSEs
    f_key = 'F RMSE, meV/Å'
    f_perf_key = 'F RMSE/STD, %'

    all_ref_vals = np.array([])
    all_pred_vals = np.array([])
    for ref_config_type, pred_config_type in zip(ref_data['forces'].keys(),
                                                 pred_data['forces'].keys()):
        if ref_config_type != pred_config_type:
            raise ValueError(
                'Reference and predicted config_types do not match')

        ref_vals = ref_data['forces'][ref_config_type]
        pred_vals = pred_data['forces'][pred_config_type]
        all_ref_vals = np.concatenate([all_ref_vals, ref_vals])
        all_pred_vals = np.concatenate([all_pred_vals, pred_vals])

        rmse = get_rmse(ref_vals, pred_vals)  # meV/A
        table[ref_config_type][f_key] = rmse * 1000
        table[ref_config_type][f_perf_key] = rmse / np.std(ref_vals) * 100

    overall_rmse = get_rmse(all_ref_vals, all_pred_vals)
    table['overall'][f_key] = overall_rmse * 1000  # meV
    table['overall'][f_perf_key] = overall_rmse / np.std(all_ref_vals) * 100  # %

    # print table
    table = pd.DataFrame(table).transpose()
    pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if int(
        x) == x else f'{{:,.{precision}f}}'.format(x)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(table)

    # useful for processing results later, e.g. for learning curves
    return table


def evaluate_data(data, calculator, pred_prefix, output_fname=None, chunksize=500):
    """Evaluates energies and forces

    Parameters
    ----------
    data: list(Atoms) / CoinfigSet_in
        Structures with reference (and predicted) energies and forces
    pred_prefix: str
        Prefix for predicted energy and forces
    calculator: calculator / (initializer, args, kwargs)
        Calculator to evaluate predicted energy & forces
    output_fname: str, default None
        Where to save evaluated structures, if at all
    chunksize: int, default 500
        How many structures to evaluate sequentially at calculators.generic.run

    Returns
    -------
    list(Atoms): configurations evaluated with the calculator

    """

    output = OutputSpec()
    properties = ['energy', 'forces']
    generic.run(inputs=data, outputs=output, calculator=calculator, properties=properties,
                chunksize=chunksize, output_prefix=pred_prefix)


    if output_fname is not None:
        write(output_fname, output.output_configs)

    return output.output_configs


def read_energies_forces(atoms, prefix, info_key):
    """ Reads out energies and forces into one dictionary

    Parameters
    ---------
    atoms: list(Atoms) / CoinfigSet_in
        Structures with energies and forces
    prefix: str
        Prefix for energy and forces of interest

    Returns
    -------
    OrderedDict: {'energy': {'config1':np.array(energies),
                             'config2':np.array(energies)},
                  'forces':{sym1:{'config1':np.array(forces),
                                  'config2':np.array(forces)},
                            sym2:{'config1':np.array(forces),
                                  'config2':np.array(forces)}}}
    """

    data = dict()
    data['energy'] = OrderedDict()
    data['forces'] = OrderedDict()

    for idx, at in enumerate(atoms):

        if len(at) == 1:
            continue

        config_type = f'no_{info_key}'
        if info_key in at.info.keys():
            config_type = at.info[info_key]

        try:
            data['energy'][config_type] = np.append(data['energy'][config_type],
                                                    at.info[f"{prefix}energy"] / len(at))
        except KeyError:
            data['energy'][config_type] = np.array([])
            data['energy'][config_type] = np.append(data['energy'][config_type],
                                                    at.info[f"{prefix}energy"] / len(at))

        forces = at.arrays[f'{prefix}forces']

        sym_all = at.get_chemical_symbols()
        for j, sym in enumerate(sym_all):
            if sym not in data['forces'].keys():
                data['forces'][sym] = OrderedDict()
            try:
                data['forces'][sym][config_type] = np.append(data['forces'][sym][config_type],
                                                             forces[j])
            except KeyError:
                data['forces'][sym][config_type] = np.array([])
                data['forces'][sym][config_type] = np.append(data['forces'][sym][config_type],
                                                             forces[j])

    return data


def desymbolise_force_dict(my_dict):
    """from dict['forces'][sym]:[values] makes dict['forces']:[values1, values2...]"""
    force_dict = OrderedDict()
    for sym, sym_dict in my_dict.items():
        for config_type, values in sym_dict.items():
            try:
                force_dict[config_type] = np.append(force_dict[config_type], values)
            except KeyError:
                force_dict[config_type] = np.array([])
                force_dict[config_type] = np.append(force_dict[config_type], values)

    return force_dict


def get_rmse(ref_ar, pred_ar):
    return np.sqrt(np.mean((ref_ar - pred_ar) ** 2))
