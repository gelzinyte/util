from ase.io import read, write
import os
import subprocess
from pathlib import Path
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from util import configs


def main(sub_template, input_fname, aces_dir, ace_fname, temps):

    sub_template = Path(sub_template) 
    assert aces_dir is None or ace_fname is None
    if aces_dir is not None:
        aces_dir = Path(aces_dir)
        aces_fnames = [fn for fn in aces_dir.iterdir()]
    elif ace_fname is not None:
        aces_fnames = [ace_fname]
    
    input_configs = read(input_fname, ":")

    outputs_dir = Path("md_trajs")
    outputs_dir.mkdir(exist_ok=True)

    traj_root = os.getcwd()

    for temp in temps:

        for ace_fname in aces_fnames:
            ace_fname = Path(ace_fname)
            if "ace" not in ace_fname.name:
                continue

            ace_label = ace_fname.stem
            ace_fname = ace_fname.resolve()

            for at in input_configs:
                at_name = at.info["graph_name"]
                at_dir = outputs_dir / at_name
                at_dir.mkdir(exist_ok=True)

                temp_dir = at_dir / str(temp)
                temp_dir.mkdir(exist_ok=True)

                job_dir = temp_dir / ace_label 
                job_dir.mkdir(exist_ok=True)

                label = f"{at_name}.{temp}.{ace_label}"

                at_fname = job_dir / (at_name + '.xyz')
                out_fname = job_dir / (at_name + ".out.xyz")

                write(at_fname, at)

                command = f"util misc md -a {ace_fname} -x {at_fname.resolve()} -t {temp} -o {out_fname.resolve()} -p {ace_label}"

                os.chdir(job_dir)

                with open(sub_template, "r") as f:
                    sub_text = f.read()

                sub_text = sub_text.replace("<command>", command)
                sub_text = sub_text.replace("<label>", label)

                with open("sub.sh", "w") as f:
                    f.write(sub_text)

                subprocess.run("qsub sub.sh", shell=True)

                os.chdir(traj_root)


def plot_struct_graph(mols, extra_info, temps, ace_fname, traj_root, repeats=None):
    """ analyses the long md trajectories"""

    traj_root = Path(traj_root)
    # print(traj_root)

    num_rows = len(mols)
    fig = plt.figure(figsize=(8, num_rows*2))
    gs = gridspec.GridSpec(num_rows, 1)

    axes = [plt.subplot(g) for g in gs]

    for ax, mol_id in zip(axes, mols):
        # print(mol_id)

        plot_ace_graph(ax, ace_fname, extra_info, repeats, temps, mol_id, traj_root)

        # ax.legend()
        ax.grid(color="lightgrey")
        ax.set_xlabel("time, fs")
        ax.set_title(mol_id)
        ax.set_xlim(right=500100)
        # ax.set_xscale('log')

    plt.suptitle(ace_fname)
    plt.tight_layout()
    plt.savefig(mol_id + '.png', dpi=300)

    # return fig


def plot_ace_graph(ax, ace_fname, extra_info, repeats, temps, mol_id, traj_root):

    main_y_vals = np.arange(len(temps))
    for main_y_val, temp in zip(main_y_vals, temps):
        plot_temp_graph(ax, main_y_val, temp, extra_info, repeats, mol_id, ace_fname, traj_root)

    ax.set_yticks(main_y_vals)
    ax.set_yticklabels([str(t) for t in temps])
    ax.set_ylabel("temperature, K")


def plot_temp_graph(ax, main_y_val, temp, extra_info, repeats, mol_id, ace_name, traj_root):

    # dels = (np.linspace(0, 1, len(repeats)) - 0.5) / 3
    if repeats is None:
        dels = [0]
        repeats = [None]

    for del_y, run in zip(dels, repeats):

        y_val = main_y_val + del_y

        ace_fname_bit = str(ace_name).replace('.json', "")
        traj_fname = traj_root / f"{mol_id}/{temp}/{ace_fname_bit}/{mol_id}.traj.xyz" 

        # print(mol_id)

        if run is not None:
            plots_kwargs = extra_info[run]["plot_kwargs"]
        else:
            plots_kwargs = extra_info["plot_kwargs"]

        times, kwargs = process_traj(traj_fname, plots_kwargs)

        yvals = [y_val] * len(times)

        ax.scatter(times, yvals, label=mol_id, **kwargs)
    

def process_traj(traj_fname, plot_kwargs):
    ats = read(traj_fname, ":") 
    results = configs.filter_insane_geometries(ats, mult=1.2)
    good = results["good_geometries"]
    bad = results["bad_geometries"]

    good_colors = [plot_kwargs["color"]] * len(good)
    bad_colors = ['r'] * len(bad)
    colors = good_colors + bad_colors

    good_times = [at.info["MD_time_fs"] for at in good]
    bad_times = [at.info["MD_time_fs"] for at in bad]
    times = good_times + bad_times

    good_sizes = [5] * len(good)
    bad_sizes = [40] * len(bad)
    sizes = good_sizes + bad_sizes

    times, colors, sizes = zip(*sorted(zip(times, colors, sizes)))

    kwargs = {"s": sizes, "c": colors}

    return times, kwargs 


     