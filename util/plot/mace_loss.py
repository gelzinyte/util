import matplotlib.pyplot as plt
import click
from pathlib import Path
import json
from matplotlib import gridspec
import numpy as np

"""
Plots MACE training results from the `results/model_name-seed_train.txt` file. 

Parameters
----------

* `in_fnames`: list(str) or str 
    (list of) `results/model_name-seed_train.txt` filenames to plot results from
* `fig_name`: str, default "train_summary.png"
    output figure filename 
* `skip_first_n`: int, default None
    don't plot results from the first couple of epochs
* `x_log_scale`: bool, default False
    plot x axis in log scale
"""
def plot_mace_train_summary(in_fnames, fig_name="train_summary.png", skip_first_n=None, x_log_scale=False):

    plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3)
    axes = [plt.subplot(g) for g in gs]
    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.linspace(0, 1, 10)] 


    plot_all_loss = False
    if len(in_fnames) == 1:
        plot_all_loss = True

    for color, fn in zip(colors, in_fnames): 
        d = get_data(fn, skip_first_n)
        label = fn
        if len(in_fnames) > 1:
            label = ('/').join(fn.split('/')[:-2])
        plot_curves(axes, d, color, label, plot_all_loss=plot_all_loss)

    annotate(axes[0], axes[1], axes[2], x_log_scale)

    plt.tight_layout()
    fname = Path(fig_name)
    if fname.suffix != '.png':
        fname.parent / (fname.name + '.png')
    plt.savefig(fname, dpi=300)


def get_data(in_fname, skip_first_n=None):

    with open(in_fname) as f:
        text = f.read()

    d = {}
    d["losses"] = []
    d["epochs"] = []
    d["index"] = []
    d["mean_loss"] = []
    d["energy_rmses"] = []
    d["force_rmses"] = []
    d["epoch_starts"] = []
    d["rmse_xs"] = []

    losses_for_epoch = []
    lines = text.split('\n')
    prev_epoch = 0 
    for idx, entry in enumerate(lines):

        try:
            data = json.loads(entry)
        except json.decoder.JSONDecodeError:
            print(f"error with line: {idx}, entry: {entry}")
        epoch = data["epoch"]
        if skip_first_n is not None and epoch < skip_first_n:
            continue
        loss = data["loss"]
        d["losses"].append(loss)
        d["epochs"].append(epoch)
        d["index"].append(idx)
        losses_for_epoch.append(loss)
        if "rmse_f" in data:
            d["force_rmses"].append(data["rmse_f"]*1e3)
            d["energy_rmses"].append(data["rmse_e"]*1e3)
            d["rmse_xs"].append(epoch)
        if epoch != prev_epoch: 
            prev_epoch = epoch
            d["mean_loss"].append(np.mean(losses_for_epoch))
            losses_for_epoch = []
            d["epoch_starts"].append(idx)

    return d



def plot_curves(axes, d, color, label, plot_all_loss=False):

    plot_kwargs = {"lw": 2, "color": color, "label": label}

    if plot_all_loss:
        cmap = plt.get_cmap('Pastel1')
        num_colors = 5
        ref_colors = [cmap(idx) for idx in np.linspace(0, 1, num_colors)]
        colors = [ref_colors[epoch % num_colors] for epoch in d["epochs"]]
        axes[0].scatter(d["index"], d["losses"], c=colors, s=1, zorder=0)

    axes[0].plot(d["epoch_starts"], d["mean_loss"], **plot_kwargs)
        # energy rmses
    axes[1].plot(d["rmse_xs"], d["energy_rmses"], **plot_kwargs)

    axes[2].plot(d["rmse_xs"], d["force_rmses"], **plot_kwargs)
    
def annotate(ax_loss, ax_e, ax_f, x_log_scale = False):

    ax_loss.set_ylabel("loss")
    ax_loss.set_xlabel("step")
    ax_loss.set_title("mean loss per epoch")

    ax_e.set_ylabel("energy rmse, meV")
    ax_e.set_title("total energy rmse")
    ax_e.set_ylim(bottom=95)

    ax_f.set_ylabel("force rmse, meV/A")
    ax_f.set_title("force rmse")
    ax_f.set_ylim(bottom=9)

    for idx, ax in enumerate([ax_loss, ax_e, ax_f]):
        if x_log_scale:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(color='lightgrey', which="both")
        if idx!=0: 
            ax.set_xlabel("epoch")

    

