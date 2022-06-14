import matplotlib.pyplot as plt
from util import shift0 as sft
import numpy as np

def curves_from_all_atoms(ats_in, pred_prefix, isolated_at = None, rin=None, rout=None, out_prefix=''):

    if isolated_at is None:
        isolated_ats = [at for at in ats_in if len(at) == 1]
        isolated_at = [at for at in isolated_ats if list(at.symbols)[0] == "H"][0]

    data = {}
    for at in ats_in:
        if len(at) == 1:
            continue
        comp = at.info["compound"]
        if comp not in data.keys():
            data[comp] = {'diss_mol':[], 'iso_h':isolated_at, 'rad':None}
        if at.info["dissociation_test_config_type" ] == "mol":
            data[comp]['diss_mol'].append(at)
            continue
        elif at.info["dissociation_test_config_type"] == "rad":
            data[comp]['rad'] = at
            continue
        else:
            print(f"didn't assign {at.info['dissociation_test_config_type']}")


    for id, vals in data.items():
        print(id)
        c_idx = vals["diss_mol"][0].info["dissociation_test_c_idx"]
        h_idx = vals["diss_mol"][0].info["dissociation_test_h_idx"]
        # c_idx = vals['rad'].info["dissociation_test_c_idx"]
        # h_idx = vals['rad'].info["dissociation_test_h_idx"]
        title = f'{out_prefix}_{pred_prefix}dissociation_curve_{id}_C{c_idx}_H{h_idx}'
        plot_curve(diss_mol = vals["diss_mol"], iso_h = vals['iso_h'], rad=vals['rad'],
                   pred_prefix=pred_prefix, title=title, rin=rin, rout=rout)



def plot_curve(diss_mol, iso_h, rad, pred_prefix, title, rin=None, rout=None,
               bh_prefix='dft_hop_', dft_prefix='dft_'):
    print('rad:', rad)
    # if rad is not None:
    #     c_idx = rad.info["dissociation_test_c_idx"]
    #     h_idx = rad.info["dissociation_test_h_idx"]
    # else:
    c_idx = diss_mol[0].info["dissociation_test_c_idx"]
    h_idx = diss_mol[0].info["dissociation_test_h_idx"]


    distances = [at.get_distance(c_idx, h_idx) for at in diss_mol]
    pred_energies = [at.info[f'{pred_prefix}energy'] for at in diss_mol]

    if rad is not None:
        dft_ref = iso_h.info[f'{dft_prefix}energy'] + np.min([rad.info[f'{bh_prefix}{idx}_energy'] for idx in range(10)])
        pred_ref = iso_h.info[f'{pred_prefix}energy'] + rad.info[f'{pred_prefix}energy']
    else:
        pred_ref = None
        dft_ref = None 

    plt.figure(figsize=(6, 4))

    pred_values = pred_energies
    if dft_ref is not None:
        pred_values = sft(pred_energies, dft_ref)
    plt.plot(distances, pred_values)

    if rad is not None:
        plt.axhline(dft_ref - dft_ref, lw=0.8, color='k', label=f"dft bhop reference: {dft_ref:.3f} eV")
        shifted_pred = pred_ref - dft_ref
        plt.axhline(shifted_pred, lw=0.8, color='tab:orange', ls='--',
                    label=f'{pred_prefix} reference: {shifted_pred*1e3:.3f} meV or {shifted_pred*1e3/(len(rad)+1):.3f} meV/at')

    if rin is not None:
        plt.axvline(rin, color='k', lw=0.8, ls='--', label=f'rin: {rin:2f} Å')
    if rout is not None:
        plt.axvline(rout, color='k', lw=0.8, ls='--', label=f'rout: {rout:2f} Å')

    for at_idx, (dist, at)  in enumerate(zip(distances, diss_mol)):
        d = [dist] * 10
        e = [at.info[f'{bh_prefix}{idx}_energy'] for idx in range(10)]
        vals_to_plot = e
        label = None
        if dft_ref is not None:
            vals_to_plot = sft(e, dft_ref)
            label = f'10 basin hops: {(min(e) - dft_ref)*1e3:.3f} meV' if at_idx == len(distances ) - 1 else None
        plt.scatter(d, vals_to_plot, label=label, color='k', marker='x')

    plt.grid(color='lightgrey')
    plt.xlabel("C-H separation")
    if dft_ref is not None:
        plt.ylabel(f"Shifted energy, eV (w.r.t. {dft_ref:.2f})")
    else:
        plt.ylabel('energy, eV')
    plt.legend(title="(energy - reference) at max separation")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title.replace(' ', '_')+'.png', dpi=300)

