from asyncore import write
from util import shift0 as sft
from ase.io import read, write
from util import configs
import matplotlib.pyplot as plt


def main(dimer_fns, isolated_ats, pred_prop_prefix, output_fn, isolated_at_prop_prefix=None):

    dimer_ats = []
    for dimer_fn in dimer_fns:
        ats = read(dimer_fn, ':')

    for at in isolated_ats:
        assert len(at) == 1
        at.info["at_symbol"] = list(at.symbols)[0]
    isolated_ats = configs.into_dict_of_labels(isolated_ats, info_label="at_symbol")

    plot_dimers(dimer_ats, isolated_ats, pred_prop_prefix, output_fn, isolated_at_prop_prefix)

def plot_dimers(dimer_ats, isolated_ats, pred_prop_prefix, output_fn, isolated_at_prop_prefix=None):
    
    if isolated_at_prop_prefix is None:
        isolated_at_prop_prefix = pred_prop_prefix

    plt.figure()

    for ats in dimer_ats: 
        symbols = list(ats[0].symbols)
        ref_es = 0
        for sym in symbols:
            at = isolated_ats[sym]
            assert len(at) == 1
            ref_es += at[0].info[f"{pred_prop_prefix}energy"]

        dimer_energies = [at.info[f"{pred_prop_prefix}energy"] for at in ats]
        distances = [at.get_distance(0, 1) for at in ats]
        label = ''.join(symbols) + f" wrt {ref_es:.2f} eV"

        plt.plot(distances, sft(dimer_energies, ref_es), label=label)

    plt.grid(color='lightgrey')
    plt.xlabel('distance, A')
    plt.ylabel('energy, eV')
    plt.legend()
    plt.title(f"{pred_prop_prefix} dimer curves (shifted by mace predicted e0's)")
    plt.ylim((-5, 10))
    # plt.ylim(top=1000)
    plt.tight_layout()

    plt.savefig(output_fn, dpi=300)
        

