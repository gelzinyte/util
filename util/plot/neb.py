from ase.io import read, write
from util.configs import into_dict_of_labels
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from util import shift0 as sft
from matplotlib import gridspec

def neb(in_fname, num_images, prop_prefix, dft_fname=None, dft_prop_prefixes=['dft_singlet_', 'dft_triplet'],system_label_key="graph_name", out_dir = "neb_overview", shift_by = "dft"):

    assert shift_by in ["dft", "predicted"]

    cmap = plt.get_cmap("viridis")

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    ats = read(in_fname, ":")
    dd = into_dict_of_labels(ats, system_label_key)

    if dft_fname is not None:
        dft_ats = read(dft_fname, ":")
        dft_dd = into_dict_of_labels(dft_ats, system_label_key)

    all_trajs_out = []
    for key, ats in dd.items():
        num_ats = len(ats)
        num_steps = int(num_ats / num_images )
        assert num_ats % num_images == 0
        trajs = [ats[i*num_images:(i+1)*num_images] for i in range(num_steps)]

        colors = [cmap(idx) for idx in np.linspace(0, 1, len(trajs))]

        side = 4.5
        fig = plt.figure(figsize=(side*2, side+2)) 
        gs = gridspec.GridSpec(1, 2)
        axes = [plt.subplot(g) for g in gs]

        refe = trajs[0][0].info[f"{prop_prefix}energy"]

        for idx, traj in enumerate(trajs):

            image_nums = np.arange(28) 
            h_idx = traj[0].info["dissociate_h_idx"]
            c_idx = traj[0].info["dissociate_c_idx"]

            dists = [at.get_distance(c_idx, h_idx) for at in traj]
            energies = [at.info[f"{prop_prefix}energy"] for at in traj]

            fmax = np.max(np.array([at.arrays[f"{prop_prefix}forces"] for at in traj]))

            label = None
            if idx == 0 or idx == len(trajs)-1:
                label = f"neb {idx}"

            if idx == len(trajs) - 1:
                label += f" max F comp {fmax:.1f} eV/Ang"


            axes[0].plot(dists, sft(energies, refe), color=colors[idx], label=label)
            axes[1].plot(image_nums, sft(energies, refe), color=colors[idx], label=label)

            # if idx == len(trajs)-1:
                # axes[0].scatter(dists, sft(energies, refe), color='tab:red', marker='x', zorder=3)
                # axes[1].scatter(image_nums, sft(energies, refe), color='tab:red', marker='x', zorder=3)

        if dft_fname is not None:
            lss = ["--", ":"]
            if key in dft_dd:
                for idx, dft_prop_prefix in enumerate(dft_prop_prefixes):
                    dft_traj = dft_dd[key]
                    dists = [at.get_distance(h_idx, c_idx) for at in dft_traj]
                    dft_energies = [at.info[f"{dft_prop_prefix}energy"] for at in dft_traj]
                    dft_fmax = np.max(np.array([at.arrays[f"{dft_prop_prefix}forces"] for at in dft_traj]))
                    if shift_by == "dft":
                        dft_refe = dft_energies[0]
                    elif shift_by == "predicted":
                        dft_refe = refe

                    label = f"final {dft_prop_prefix} eval; max F comp {dft_fmax:.1f} eV/Ang"
                    axes[0].plot(dists, sft(dft_energies, dft_refe), color='tab:red', ls=lss[idx], label=label)
                    axes[1].plot(image_nums, sft(dft_energies, dft_refe), color='tab:red', ls=lss[idx], label=label)

                    axes[0].scatter(dists, sft(dft_energies, dft_refe), color='tab:red', marker='x', zorder=3)
                    axes[1].scatter(image_nums, sft(dft_energies, dft_refe), color='tab:red', marker='x', zorder=3)


        for ax in axes:
            ax.grid(color='lightgrey', which="both")
            ax.set_ylabel(f"{prop_prefix}energy, shifted wrt {refe:.1f}, eV")
            ax.set_title(key)

            
        axes[0].legend(bbox_to_anchor=(1, -0.2))
        axes[0].set_xlabel(f"Distance between H{h_idx} and C{c_idx}, Ang")
        axes[1].set_xlabel(f"Image number")

        plt.tight_layout()
        plt.savefig(out_dir / f"neb_{key}.png")

        # print fmax
        # plot dft values




