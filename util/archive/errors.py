from ase.io import read
from collections import Counter
import numpy as np
import util
import click
from tabulate import tabulate
import pandas as pd

@click.command()
@click.option('--ref_energy_name', '-re')
@click.option('--pred_energy_name', '-pe')
@click.option('--ref_force_name', '-rf')
@click.option('--pred_force_name', '-pf')
@click.option('--evaluated_fname', '-fn')
@click.option('--force_by_element', is_flag=True)
def errors_summary(ref_energy_name, pred_energy_name, ref_force_name, pred_force_name,
                evaluated_fname, force_by_element=False):

    ats = read(evaluated_fname, ':')

    ref_data = util.get_E_F_dict_evaled(ats, energy_name=ref_energy_name, force_name=ref_force_name)
    pred_data = util.get_E_F_dict_evaled(ats, energy_name=pred_energy_name, force_name=pred_force_name)

    config_types = []
    for at in ats:
        if len(at)!=1:
            if 'config_type' in at.info.keys():
                config_types.append(at.info["config_type"])
            else:
                config_types.append('no_config_type')

    config_counts = dict(Counter(config_types))
    total_count = sum([val for key, val in config_counts.items()])
    table = {}
    for config_type in config_counts.keys():
        if config_type == 'isolated_atom':
            continue
        table[config_type] = {}
        table[config_type]["Count"] = int(config_counts[config_type])
    table['overall'] = {}
    table['overall']["Count"] = total_count



    if not force_by_element:
        tmp_no_f_sym_data = util.desymbolise_force_dict(ref_data['forces'])
        ref_data['forces'].clear()
        ref_data['forces'] = tmp_no_f_sym_data

        tmp_no_f_sym_data = util.desymbolise_force_dict(pred_data['forces'])
        pred_data['forces'].clear()
        pred_data['forces'] = tmp_no_f_sym_data


    #energy RMSEs
    e_key = 'E RMSE/meV'
    all_ref_vals = np.array([])
    all_pred_vals = np.array([])
    for ref_config_type, pred_config_type in zip(ref_data['energy'].keys(), pred_data['energy'].keys()):
        if ref_config_type != pred_config_type:
            raise ValueError( 'Reference and predicted config_types do not match')

        ref_vals = ref_data['energy'][ref_config_type]
        pred_vals = pred_data['energy'][pred_config_type]
        all_ref_vals = np.concatenate([all_ref_vals, ref_vals])
        all_pred_vals = np.concatenate([all_pred_vals, pred_vals])

        rmse = util.get_rmse(ref_vals, pred_vals)
        table[ref_config_type][e_key] =  rmse * 1000   #meV
    table['overall'][e_key] = util.get_rmse(all_ref_vals, all_pred_vals) * 1000

    #force RMSEs
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

        rmse = util.get_rmse(ref_vals, pred_vals)  # meV/A
        table[ref_config_type][f_key] = rmse * 1000
        table[ref_config_type][f_perf_key] = rmse / np.std(ref_vals) * 100

    overall_rmse =  util.get_rmse(all_ref_vals,  all_pred_vals)
    table['overall'][f_key] = overall_rmse * 1000  # meV
    table['overall'][f_perf_key] = overall_rmse / np.std(all_ref_vals) * 100  # %



    table = pd.DataFrame(table).transpose()
    pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if int(x) == x else '{:,.4f}'.format(x)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    # pd.options.display.max_rows
    # pd.options.display.float_format = '{:.3f}'.format
    print(table)
    # print(tabulate(table, headers='keys', floatfmt='.3f'))

if __name__ == '__main__':
    errors_summary()


