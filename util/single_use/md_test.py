from ase.io import read, write
import os
import subprocess
from pathlib import Path
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from util import configs


def main(sub_template, input_fname, aces_dir, ace_fname, output_dir, temps):

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

    home_dir = os.getcwd()

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

                os.chdir(home_dir)


def plot_mol_graph(mol_id, extra_info, runs, temps, aces_nos, home_dir, run_dir):
    """ analyses the long md trajectories"""

    home_dir = Path(home_dir)

    # fig = plt.figure(figsize=(15, 2 * len(aces_nos)))
    fig = plt.figure(figsize=(8, 1 * len(aces_nos)))
    num_columns = int(len(aces_nos)/2)
    if num_columns%2 == 0:
        num_columns += 1
    # gs = gridspec.GridSpec(num_columns, 1)
    gs = gridspec.GridSpec(num_columns, 2)

    axes = [plt.subplot(g) for g in gs]

    for ax, ace_no in zip(axes, aces_nos):
        # print(ace_no)

        plot_ace_graph(ax, ace_no, extra_info, runs, temps, mol_id, home_dir, run_dir)

        # ax.legend()
        ax.grid(color="lightgrey")
        ax.set_xlabel("time, fs")
        ax.set_title(f"ACE {ace_no}")
        ax.set_xlim(right=100000)

    plt.suptitle(mol_id)
    plt.tight_layout()
    plt.savefig(mol_id + '.pdf')

    # return fig


def plot_ace_graph(ax, ace_no, extra_info, runs, temps, mol_id, home_dir, run_dir):

    main_y_vals = np.arange(len(temps))
    for main_y_val, temp in zip(main_y_vals, temps):
        # print(main_y_val, temp)
        plot_temp_graph(ax, main_y_val, temp, extra_info, runs, mol_id, ace_no, home_dir, run_dir)

    ax.set_yticks(main_y_vals)
    ax.set_yticklabels([str(t) for t in temps])
    ax.set_ylabel("temperature, K")



def plot_temp_graph(ax, main_y_val, temp, extra_info, runs, mol_id, ace_no, home_dir, run_dir):

    dels = (np.linspace(0, 1, len(runs)) - 0.5) / 3
    
    for del_y, run in zip(dels, runs):

        # print(run)
        # print(run)

        y_val = main_y_val + del_y

        ace_name = f'ace_{ace_no}.json'
        # print(ace_name)
        all_ace_names = extra_info[run]["aces"]
        # print(all_ace_names)
        if ace_name not in all_ace_names:
            # print("naaaay")
            continue

        # traj_fname = home_dir / f"{run}/md_stuff/md_runs/md_trajs/{mol_id}/{temp}/ace_{ace_no}/{mol_id}.traj.xyz" 
        # traj_fname = home_dir / f"{run}/md_stuff/re_run_failed_trajectories/md_trajs/{mol_id}/{temp}/ace_{ace_no}/{mol_id}.traj.xyz" 
        traj_fname = home_dir / f"{run}/md_stuff/{run_dir}/md_trajs/{mol_id}/{temp}/ace_{ace_no}/{mol_id}.traj.xyz" 

        print(traj_fname)

        times, kwargs = process_traj(traj_fname, extra_info[run]["plot_kwargs"])

        yvals = [y_val] * len(times)

        ax.scatter(times, yvals, label=f"{run}/ace_{ace_no}", **kwargs)
    


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


     