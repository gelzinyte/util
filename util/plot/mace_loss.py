import matplotlib.pyplot as plt
import click
from pathlib import Path
import json
from matplotlib import gridspec
import numpy as np


def get_data(in_fname):

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

def plot_loss(fig_name, in_fnames):

    plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3)
    axes = [plt.subplot(g) for g in gs]
    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.linspace(0, 1, 10)] 


    plot_all_loss = False
    if len(in_fnames) == 1:
        plot_all_loss = True

    for color, fn in zip(colors, in_fnames): 
        d = get_data(fn)
        plot_curves(axes, d, color, plot_all_loss=plot_all_loss)

    annotate(axes)

    plt.tight_layout()
    fname = Path(fig_name)
    if fname.suffix != '.png':
        fname.parent / (fname.name + '.png')
    plt.savefig(fname, dpi=300)


def plot_curves(axes, d, color, plot_all_loss=False):

    if plot_all_loss:
        cmap = plt.get_cmap('pastel1')
        ref_colors = [cmap(idx) for idx in np.linspace(0, 1, 20)]
        colors = [ref_colors[epoch % 4] for epoch in d["epochs"]]
        axes[0].scatter(d["index"], d["losses"], c=colors, s=2, zorder=0)

    axes[0].plot(d["epoch_starts"], d["mean_loss"], color='k')
        # energy rmses
    axes[1].plot(d["rmse_xs"], d["energy_rmses"])

    axes[2].plot(d["rmse_xs"], d["force_rmses"])
    
def annotate(ax_loss, ax_e, ax_f):

    ax_loss.set_ylabel("loss")
    ax_loss.set_xlabel("step")
    ax_loss.legend(title="mean loss per epoch")

    ax_e.set_ylabel("energy rmse, meV")

    ax_f.set_ylabel("force rmse, meV/A")

    for idx, ax in enumerate([ax_loss, ax_e, ax_f]):
        ax.set_yscale('log')
        if idx!=0: 
            ax.set_xlabel("epoch")
            ax.grid(color='lightgrey', which="both")

    

