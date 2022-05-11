import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_ard_scores(fname):
    df = pd.read_csv(fname, header=None)
    vals = np.array(df).flatten()
    xs = np.arange(len(vals))

    # xs = [x for x, v in zip(xs, vals) if v]

    plt.figure()
    plt.plot(xs[-1000:], vals[-1000:])
    plt.grid(color='lightgrey')
    plt.yscale("log")
    plt.xlabel("lml maximisations step")
    plt.ylabel("log marginal likelihood score")
    plt.tight_layout()
    plt.savefig(fname+'.png', dpi=300)
