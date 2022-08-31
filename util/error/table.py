from collections import Counter, OrderedDict
import warnings
import numpy as np
from ase.io import write

from wfl.calculators import generic
from wfl.configset import OutputSpec

from util.plot import rmse_scatter_evaled
from util import error

import pandas as pd


def plot(all_atoms, ref_energy_key, pred_energy_key, pred_forces_key=None,
         ref_forces_key=None, precision=4, info_label="config_type", 
         error_type="rmse", energy_type="atomization_energy", isolated_atoms=None):
    """Plots error table by config type

    Parameters
    ----------
    all_atoms: list(Atoms) / CoinfigSet_in
        Structures with reference (and predicted) energies and forces
    ref_prefix: str
        Prefix for "energy" in Atoms.info and "forces" in Atoms.arrays for reference
    pred_prefix: str
        Prefix for predicted energy and forces
    num_inputs_per_python_subprocess: int, default 10
        How many structures to evaluate sequentially at calculators.generic.run

    Returns
    -------
    """

    assert error_type in ['rmse', 'mae']
    if error_type == 'rmse':
        error_function = util.get_rmse
        error_label = 'RMSE'
    elif error_type == 'mae':
        error_function = util.get_mae
        error_label = "MAE"

    data = error.process(
        all_atoms=all_atoms,
        ref_energy_key=ref_energy_key,
        pred_energy_key = pred_energy_key,
        ref_forces_key = ref_forces_key,
        pred_forces_key = pred_forces_key,
        info_label=info_key,
        energy_type=energy_type,
        isolated_atoms=isolated_atoms) 

    # prepare table by config type
    config_counts = {}
    for label, vals in data.items():
        config_counts[label] = len(vals["energy"]["reference"])

    table = {}
    for config_type in config_counts.keys():
        table[config_type] = {}
        table[config_type]["Count"] = int(config_counts[config_type])

    # collect energy RMSEs
    keys = {
        "energy": f'E {error_label}, meV/at',
        "forces": f'F {error_label}, meV/Å'}

    for label, label_data in data.items():
        for obs, obs_data in label_data.items():
            error = error_function(obs_data["reference"], obs_data["predicted"]) * 1e3
            table[label][keys[obs]] = error

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


def evaluate_data(data, calculator, pred_prefix, output_fname=None, num_inputs_per_python_subprocess=500):
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
    num_inputs_per_python_subprocess: int, default 500
        How many structures to evaluate sequentially at calculators.generic.run

    Returns
    -------
    list(Atoms): configurations evaluated with the calculator

    """

    output = OutputSpec()
    properties = ['energy', 'forces']
    generic.run(inputs=data, outputs=output, calculator=calculator, properties=properties,
                num_inputs_per_python_subprocess=num_inputs_per_python_subprocess, output_prefix=pred_prefix)


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
