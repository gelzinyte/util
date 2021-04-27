from ase.io import read, write
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict


def rmsd_table(fname1, fname2, group_key='compound', precision=2):
    print('-'*30)
    print(fname1)
    print('--- vs ---')
    print(fname2)
    print('-'*30)

    atoms1 = read(fname1, ':')
    atoms2 = read(fname2, ':')

    if len(atoms1) != len(atoms2):
        raise RuntimeError('number of structures do not match - no way of accessign which is which')

    groups = []
    for at in atoms1:
        if len(at) == 1:
            continue
        if group_key in at.info.keys():
            groups.append(at.info[group_key])
        else:
            groups.append(f'no_{group_key}')


    data = {}
    for at1, at2 in zip(atoms1, atoms2):

        label1 = at1.info[group_key]
        label2 = at2.info[group_key]

        if label1 != label2:
            raise RuntimeError('labels do not match')

        if label1 not in data.keys():
            data[label1] = []

        rmsd = np.sqrt(np.mean((at1.positions - at2.positions)**2))
        data[label1].append(rmsd)


    label_counts = dict(Counter(groups))
    total_count = sum([val for key, val in label_counts.items()])
    
    table = {}
    all_rmsds = []
    for label in label_counts.keys():
        table[label] = {}
        table[label]["Count"] = int(label_counts[label])
        table[label]["mean RMSD"] = np.mean(data[label])
        table[label]['std of RMSD'] = np.std(data[label])
        all_rmsds += data[label]

    table['overall'] = {}
    table['overall']["mean RMSD"] = np.mean(all_rmsds)
    table['overall']['std of RMSD'] = np.std(all_rmsds)
    table['overall']["Count"] = total_count


    table = pd.DataFrame(table).transpose()
    pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if int(
        x) == x else f'{{:,.{precision}f}}'.format(x)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(table)


