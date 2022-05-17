import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def scatter(measure, all_data, plot_title=None, output_dir='pictures'):
    """ Analyse all of the BDE data

    Parameters
    ----------
    all_data: {label:list(pd.DataFrame)}
        tables with bde data for each compound/geometry
    measure: "bde_error"/"bde_correlation"/"rmsd"/"energy_error"
        what to plot

    """

    if measure == 'bde_error':
        shade = 50
        if plot_title is None:
            plot_title = 'bde_error'
        fill_label = f'<50 meV'
        hline_label = '50 meV'
        ylabel = '|DFT BDE - GAP BDE| / meV'
        xlabel = 'DFT BDE / eV'
        y_column_name = 'absolute_bde_error'
        x_column_name = 'dft_bde'

    elif measure == 'bde_correlation':
        shade = 50
        if plot_title is None:
            plot_title = 'bde_correlation'
        fill_label = f'$\pm$ 50 meV'
        ylabel = 'GAP BDE / eV'
        y_column_name = 'gap_bde'
        xlabel = 'DFT BDE / eV'
        x_column_name = 'dft_bde'

    elif measure == 'rmsd':
        shade = 0.1
        if plot_title is None:
            plot_title = 'gap_vs_dft_opt_rmsd'
        fill_label = f'< 0.1 Å'
        hline_label = '0.1 Å'
        ylabel = 'RMSD / Å'
        y_column_name = 'rmsd'
        xlabel = 'DFT BDE / eV'
        x_column_name = 'dft_bde'

    # elif measure == 'gap_opt_energy_correlation':
    #     shade = 50
    #     if plot_title is None:
    #         plot_title = 'energy_correlation_on_gap_optimised_configs'
    #     fill_label = f'$\pm$ 50 meV'
    #     ylabel = 'GAP energy/ eV'
    #     y_column_name = 'gap_opt_gap_energy'
    #     xlabel = 'DFT energy/ eV'
    #     x_column_name = 'gap_opt_dft_energy'
    #
    # elif measure == 'dft_opt_energy_correlation':
    #     shade = 50
    #     if plot_title is None:
    #         plot_title = 'energy_correlation_on_dft_optimised_configs'
    #     fill_label = f'$\pm$ 50 meV'
    #     ylabel = 'GAP energy/ eV'
    #     y_column_name = 'dft_opt_gap_energy'
    #     xlabel = 'DFT energy/ eV'
    #     x_column_name = 'dft_opt_dft_energy'

    elif measure == 'energy_error':
        shade = 50
        if plot_title is None:
            plot_title = 'energy_error_on_gap_optimised_configs'
        fill_label = f'< {shade} meV'
        hline_label = f'{shade} meV'
        ylabel = 'absolute GAP vs DFT error, meV'
        y_column_name = 'energy_absolute_error'
        xlabel = 'DFT energy / eV'
        x_column_name = 'gap_opt_dft_energy'

    else:
        raise RuntimeError(f'"measure" must be on of [bde_error, bde_scatter, '
                           f'rmsd, energy_error], not {measure}')


    mol_marker='o'
    rad_marker='x'

    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.linspace(0, 1, 10)]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    mol_label = None
    rad_label = None

    for idx, ((data_label, data), color) in enumerate(zip(all_data.items(), colors)):

        if idx == 0:
            print(data[0])

        for table in data:

            mol_y = table.loc['mol', y_column_name]
            mol_x = table.loc['mol', x_column_name]

            rad_y = table.drop(['H', 'mol']).loc[:, y_column_name]
            rad_x = table.drop(['H', 'mol']).loc[:, x_column_name]

            if data_label is not None:
                mol_label = data_label + ' molecules'
                rad_label = data_label + ' radical'

            plt.scatter(mol_x, mol_y, marker=mol_marker,  label=mol_label,
                        color=color)
            plt.scatter(rad_x.values, rad_y.values, marker=rad_marker,
                        label=rad_label, color=color)

            if data_label is not None:
                data_label = None
                mol_label = None
                rad_label = None


    # add shaded area
    xmin, xmax = ax.get_xlim()
    if 'correlation' in measure:
        extend_axis = 0.1
        xmin -=extend_axis
        xmax += extend_axis

        shade /= 1e+3
        plt.plot([xmin, xmax], [xmin, xmax],
                 color='k', lw=0.5)
        plt.fill_between([xmin, xmax], [xmin - shade, xmax - shade],
                        [xmin+ shade, xmax + shade],
                          color='lightgrey', alpha=0.5, zorder=0)

    else:
        plt.fill_between([xmin, xmax], 0, shade, color='lightgrey',
                         label=fill_label, alpha=0.5, zorder=0)
        ax.set_xlim(xmin, xmax)

        plt.yscale('log')

    plt.legend()
    plt.grid(color='grey', ls=':', which='both')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(plot_title.replace('_', ' '))
    plt.tight_layout()


    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = ''

    plt.savefig(os.path.join(output_dir, plot_title + '.png'), dpi=300)







