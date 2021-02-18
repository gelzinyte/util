import util
import matplotlib as mpl
from ase.io import read, write
import matplotlib.pyplot as plt
import numpy as np

def kpca(data, tmp_fname='kpca_tmp_atoms.xyz'):
    #     data[label] = atoms

    kpca_atoms_in = []
    for label, atoms in data.items():
        for at in atoms:
            at.info['kpca_label'] = label
        kpca_atoms_in += atoms


    write(tmp_fname, kpca_atoms_in)


    util.do_kpca(tmp_fname)

    ats_out = read(tmp_fname, ':')

    return ats_out



def plot_kpca(atoms, cmap='tab10', title='kpca', skip_labels=None):

    markers = ['+', 'x']
    c_counter = 0
    m_counter = 0

    if skip_labels is None:
        skip_labels = []
    elif isinstance(skip_labels, str):
        skip_labels.split(' ')

    data = {}
    for at in atoms:
        label = at.info['kpca_label']
        if label not in data.keys():
            data[label] = {}
            data[label]['ys'] = []
            data[label]['xs'] = []
        try:
            data[label]['xs'].append(at.info['pca_d_10'][0])
            data[label]['ys'].append(at.info['pca_d_10'][1])
        except:
            print(at.info)
            # return at
            raise

    cmap = mpl.cm.get_cmap(cmap)
    cidx_list = np.linspace(0, 1, 10)

    plt.figure(figsize=(12, 10))

    for idx, (label, vals) in enumerate(data.items()):

        color = cmap(cidx_list[c_counter])
        marker = markers[m_counter]

        plt.scatter(vals['xs'], vals['ys'], color=color, label=label,
                    marker=marker)

        c_counter += 1
        if c_counter == 10:
            m_counter += 1
            c_counter = 0

    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.show()

    return

