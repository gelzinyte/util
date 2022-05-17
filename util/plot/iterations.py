from wfl.generate import vib
from ase.units import invcm
from matplotlib import gridspec
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wfl.configset import ConfigSet
from util import error_table as et
import matplotlib as mpl
from util import bde
from util import suppress_stdout_stderr



def multi_evec_plot(ref_atoms, atoms2, prefix1, prefix2, labels2, suptitle):

    cmap = 'Reds'
    fontsize =14

    N = len(atoms2)

    plt.figure(figsize=( 6 *N, 5))

    grid = gridspec.GridSpec(1, N)

    for idx, (gs, at2, label2) in enumerate(zip(grid, atoms2, labels2)):

        ax = plt.subplot(gs)

        vib1 = vib.Vibrations(at2, prefix2) # x axis
        vib2 = vib.Vibrations(ref_atoms, prefix1) # y axis

        assert np.all(vib1.atoms.symbols == vib2.atoms.symbols)

        dot_products = dict()
        for i, ev1 in enumerate(vib1.evecs):
            dot_products[f'nm1_{i}'] = dict()
            for j, ev2 in enumerate(vib2.evecs):
                dot_products[f'nm1_{i}'][f'nm2_{j}'] = np.abs \
                    (np.dot(ev1, ev2))

        df = pd.DataFrame(dot_products)


        _ = ax.pcolormesh(df, vmin=0, vmax=1, cmap=cmap,
                             edgecolors='lightgrey', linewidth=0.01)

        if idx == 0:
            ax.set_ylabel('DFT modes', fontsize=fontsize)
        ax.set_xlabel(f'{label2} modes', fontsize=fontsize)

    plt.suptitle(suptitle, fontsize=fontsize+2)

def learning_curves(train_fnames, test_fnames, ref_prefix='dft_', pred_prefix=None, pred_prefix_list=None,
                    plot_prefix=None):

    data = {'energy':{'test':[], 'train': []}, 'forces':{'test':[], 'train':[]}, 'dset_sizes':[]}

    if pred_prefix_list is None:
        pred_prefix_list = [pred_prefix] * len(train_fnames)

    e_col = 'E RMSE, meV/at'
    f_col = 'F RMSE, meV/Å'

    for train_fname, test_fname, pred_prefix in zip(train_fnames, test_fnames,  pred_prefix_list):

        cfg_train = ConfigSet(input_files=train_fname)
        cfg_test = ConfigSet(input_files=test_fname)

        with suppress_stdout_stderr():
            err_train = et.plot(cfg_train, ref_prefix, pred_prefix)
            err_test = et.plot(cfg_test, ref_prefix, pred_prefix)

        data['energy']['train'].append(err_train[e_col]['overall'])
        data['energy']['test'].append(err_test[e_col]['overall'])
        data['forces']['train'].append(err_train[f_col]['overall'])
        data['forces']['test'].append(err_test[f_col]['overall'])
        data['dset_sizes'].append(err_train['Count']['overall'])


    plt.figure(figsize=(10, 4))
    gs = mpl.gridspec.GridSpec(1, 2)
    ax_e = plt.subplot(gs[0])
    ax_f = plt.subplot(gs[1])

    xs = data['dset_sizes']
    ax_e.plot(xs, data['energy']['train'], marker='x',  label='energy train')
    ax_e.plot(xs, data['energy']['test'], marker='x', label='energy test')
    ax_f.plot(xs, data['forces']['train'], marker='x', label='forces train')
    ax_f.plot(xs, data['forces']['test'], marker='x', label='forces test')

    for ax in [ax_e, ax_f]:
        ax.legend()
        ax.set_yscale('log')
        ax.set_xlabel('dataset size')
        ax.grid(color='lightgrey', ls=':', which='both')

    ax_f.set_ylabel('Force RMSE, meV/Å')
    ax_e.set_ylabel('Energy RMSE, meV/at')
    plt.suptitle('Learning curves')

    if plot_prefix is None:
        plot_prefix=''

    plt.tight_layout()
    plt.savefig(plot_prefix + 'learning_curves.png', dpi=300)


def bde_related_plots(num_cycles, gap_dir_pattern, dft_dir, metric=None, plot_prefix=None,
                      means=True):
    """ gap_dir_pattern - has {idx} to be substituted"""

    if metric is None:
        metrics =  ['rmsd', 'gap_e_error', 'gap_f_rmse', 'bde']
    else:
        metrics = [metric]

    for metric in metrics:
        all_data = []
        for idx in range(num_cycles):
            gap_dir = gap_dir_pattern.replace('{idx}', str(idx))
            dft_fnames, gap_fnames, _ = bde.dirs_to_fnames(dft_dir, gap_dir)
            with suppress_stdout_stderr():
                data = bde.get_data(dft_fnames, gap_fnames, which_data=metric)
            all_data.append(data)
        bde.iter_plot(all_data, plot_title=plot_prefix.replace('_', ' ') +' ' + metric.replace('_', ' '),
                      which_data=metric, means=means)


