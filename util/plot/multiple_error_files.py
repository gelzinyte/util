from util.plot.rmse_scatter_evaled import scatter_plot
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker
from pathlib import Path
import util




def main(ref_energy_name, pred_energy_name, ref_force_name, pred_force_name,
         atoms_filenames, output_dir, prefix, color_info_name, xvals=None,
         xlabel=None):
    """only energy_type=bindin energy, energy_shift=False, error_type=rmse for now"""

    if prefix is None:
        prefix="multi"

    # atoms_filenames = util.natural_sort(atoms_filenames)
    all_atoms_list = [read(fname, ':') for fname in atoms_filenames]
    original_prefixes = [Path(fname).stem for fname in atoms_filenames]
    prefixes = [prefix+ '_' + Path(fname).stem for fname in atoms_filenames]


    all_plot_info = [scatter_plot(ref_energy_name=ref_energy_name,
                                  pred_energy_name=pred_energy_name,
                                  ref_force_name=ref_force_name,
                                  pred_force_name=pred_force_name,
                                  all_atoms=all_atoms,
                                  output_dir=output_dir,
                                  prefix=single_prefix,
                                  color_info_name=color_info_name,
                                  isolated_atoms=None,
                                  energy_type="binding_energy")
         for all_atoms, single_prefix in zip(all_atoms_list, prefixes)]


    curves = process_list_of_dicts(all_plot_info)

    if xvals is None:
        xvals = np.arange(1, len(all_plot_info)+1)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.linspace(0, 1, 10)]

    plot_kwargs = {}

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2)
    axes = [plt.subplot(g) for g in gs]

    for ax, (prop_name, prop_curves) in zip(axes, curves.items()):

        for idx, (label, vals) in enumerate(prop_curves.items()):

            ax.plot(xvals, vals, color=colors[idx], label=label, **plot_kwargs)

        if prop_name == "energy":
            ax.set_ylabel("Energy RMSE, meV/at")
        elif prop_name == "forces":
            ax.set_ylabel("Froce component RMSE, meV/Å")

        ax.legend(title=color_info_name)
        ax.grid(color='lightgrey', ls=':')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(ticker.FixedLocator(xvals))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(original_prefixes))
        ax.tick_params(axis='x', labelrotation=90)
        if xlabel is not None:
            ax.set_xlabel(xlabel)


    plt.tight_layout()

    picture_name = f'{prefix}_by_{color_info_name}.png'
    if output_dir:
        picture_name = Path(output_dir) / picture_name
    plt.savefig(picture_name, dpi=300, bbox_inches='tight')




def process_list_of_dicts(all_infos):

    out = {"energy":{}, "forces":{}}

    for info in all_infos:
        for prop in ["energy", "forces"]:
            for label, errors in info[prop].items():
                rmse = rmse_from_errors(errors)
                if label not in out[prop].keys():
                    out[prop][label] = []
                out[prop][label].append(rmse)

    return out

def rmse_from_errors(errors):
    return np.sqrt(np.mean(errors**2))

