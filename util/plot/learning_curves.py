from pathlib import Path 
import numpy as np
from ase.io import read, write
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from util.plot import rmse_scatter_evaled
from copy import deepcopy


def annotate_ax(ax, obs):
    ax.grid(color='lightgrey', which='both')
    ax.legend()
    ax.set_xlabel("# scalars in training set")
    if obs == "energy":
        ax.set_ylabel("binding energy rmse, meV/at") 
    elif obs == "forces":
        ax.set_ylabel("force component rmse, meV/A")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

def num_scalars_in_dset(ats):
    return np.sum([num_scalars(at) for at in ats])

def num_scalars(at):
    return 1 + len(at) * 3

def add_line(ax_e, ax_f, compound_type, solver, deg, sizes=None):

    if sizes is None:
        sizes = ["0.33", "0.67", "1.00"]

    xs = []

    inner_dict = {
        "mean": [], 
        "std": [],
        "all_errors": []
    }

    data = {
        "energy": {
            "train" : [],
            "validation": [],
            "both_train_rad": [],
            "both_train_mol": []
            },
        "forces": {
            "train" : [],
            "validation": [],
            "both_train_rad": [],
            "both_train_mol": []
            }
    }


    if "N3D" in deg:
        deg = deg.replace("N3D", "")
        wdir = Path("/home/eg475/data/radicals_explore/2_do_fits/2_fits/wdir")
    if "N4D" in deg:
        deg = deg.replace("N4D", "")
        wdir = Path("/home/eg475/data/radicals_explore/2_do_fits/3_larger_basis/wdir")


    for dset_size in sizes:
        identifier = f"{compound_type}.{dset_size}.{solver}.{deg}"
        fn = wdir / f"{solver}/combined.shuffled.{identifier}/both.ace.xyz"
        fn_train = wdir / f"{solver}/combined.shuffled.{identifier}/train.ace.xyz"

        if not fn.exists():
            print(f"Didn't find: {fn}")
            continue
        
        #plot main val/test plot
        ats = read(fn, ":")
        ats_train = read(fn_train, ":")
        xs.append(num_scalars_in_dset(ats_train))

        errors = rmse_scatter_evaled.scatter_plot(
            ref_energy_name="dft_energy", 
            pred_energy_name="ace_energy", 
            ref_force_name="dft_forces", 
            pred_force_name="ace_forces", 
            all_atoms=ats,
            output_dir=None,
            prefix=None,
            color_info_name = "dataset_type",
            isolated_atoms=None,
            energy_type="binding_energy",
            error_type="rmse")

        for key_obs, dd in data.items():
            for key_curve in dd.keys():
                if key_curve in ["both_train_mol", "both_train_rad"]:
                    continue
                data[key_obs][key_curve].append(np.sqrt(np.mean(errors[key_obs][key_curve]**2)))


        if compound_type == "both":
            errors = rmse_scatter_evaled.scatter_plot(
                ref_energy_name="dft_energy", 
                pred_energy_name="ace_energy", 
                ref_force_name="dft_forces", 
                pred_force_name="ace_forces", 
                all_atoms=ats_train,
                output_dir=None,
                prefix=None,
                color_info_name = "mol_or_rad",
                isolated_atoms=None,
                energy_type="binding_energy",
                error_type="rmse")

            for key_obs, dd in data.items():
                for key_curve in dd.keys():
                    if key_curve not in ["both_train_mol", "both_train_rad"]:
                        continue
                    if key_curve == "both_train_mol":
                        key_errors = "mol"
                    if key_curve == "both_train_rad":
                        key_errors = "rad"
                    data[key_obs][key_curve].append(np.sqrt(np.mean(errors[key_obs][key_errors]**2)))
 
    color_both = "black"
    color_rad = "tab:orange"
    color_mol = "tab:green"

    if compound_type == "both":
        color=color_both
    elif compound_type=="mols":
        color=color_mol
    elif compound_type == "rads":
        color=color_rad
    else:
        print('not found')

    ax_e.plot(xs, data["energy"]["train"], c=color, ls="--", label=f"{compound_type} train", marker="o")
    ax_e.plot(xs, data["energy"]["validation"], c=color, ls="-", label=f"{compound_type} validation", marker="o")
    ax_f.plot(xs, data["forces"]["train"], c=color, ls="--", label=f"{compound_type} train", marker="o")
    ax_f.plot(xs, data["forces"]["validation"], c=color, ls="-", label=f"{compound_type} validation", marker="o")

    if compound_type == "both":
        ax_e.plot(xs, data["energy"]["both_train_mol"], c=color_mol, ls=":", label=f"{compound_type} train - mols only", marker="o")
        ax_e.plot(xs, data["energy"]["both_train_rad"], c=color_rad, ls=":", label=f"{compound_type} train - rads only", marker="o")
        ax_f.plot(xs, data["forces"]["both_train_mol"], c=color_mol, ls=":", label=f"{compound_type} train - mols only", marker="o")
        ax_f.plot(xs, data["forces"]["both_train_rad"], c=color_rad, ls=":", label=f"{compound_type} train - rads only", marker="o")
    
