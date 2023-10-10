import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.configs import into_dict_of_labels


def get_rmse(inputs, ref_key, pred_key):

    ref_vals = []
    pred_vals = [] 

    for at in inputs:
        # import pdb; pdb.set_trace
        if ref_key not in at.arrays or pred_key not in at.arrays:
            continue
        ref = []
        pred = []
        all_ref_vals = at.arrays[ref_key]
        all_pred_vals = at.arrays[pred_key]
        for ref_val, pred_val in zip(all_ref_vals, all_pred_vals):
            if not np.isnan(ref_val) and not np.isnan(pred_val):
                ref.append(ref_val)
                pred.append(pred_val)

        assert len(ref) == len(pred)

        ref_vals += ref
        pred_vals += pred 
    
    return np.sqrt(np.mean((np.array(ref_vals) - np.array(pred_vals))**2)) * 1e3

def get_table(inputs, keys):
    data = {}

    for idx1, key1 in enumerate(keys):
        for key2 in keys:

            # import pdb; pdb.set_trace()
            
            rmse = get_rmse(inputs, ref_key=key1, pred_key=key2)

            if key1 not in data:
                data[key1] = {}

            data[key1][key2] = rmse


    df = pd.DataFrame.from_dict(data)


    precision = 3
    df = df.to_string(
        max_rows = None, 
        max_cols = None, 
        float_format = "{{:.{:d}f}}".format(precision).format,
        sparsify=False
    )

    print("RMSE, meV:")
    print(df)
    return df

def plot_rmsds(rmsds, fname="rmsd_distribution.png", output_dir=None):
    rmsds = np.array(rmsds)

    label = f"mean: {np.mean(rmsds):.2f} max: {np.max(rmsds):.2f}, std: {np.std(rmsds):.3f}"
    
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / fname

    plt.figure()
    bins = np.linspace(0, 1.8, 30)
    plt.hist(rmsds, bins=bins, label=label)
    plt.xlabel("rmsd")
    plt.ylabel("count")
    plt.grid(color="lightgrey", which='both')
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)


def get_rmsds(atl1, atl2):
    rmsds_out = []    

    dd1 = into_dict_of_labels(atl1, "graph_name")
    dd2 = into_dict_of_labels(atl2, "graph_name")


    for key1, val1 in dd1.items():
        assert len(val1) == 1

        if key1 not in dd2:
            continue

        val2 = dd2[key1]
        assert len(val2) == 1

        at1 = val1[0]
        at2 = val2[0]

        assert at1.info["graph_name"] == at2.info["graph_name"]

        rmsd = np.sqrt(np.mean((at1.positions - at2.positions)**2))
        rmsds_out.append(rmsd)
    return rmsds_out

