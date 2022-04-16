import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_ard_scores(fname):
    df = pd.read_csv(fname, header=None)
    vals = np.array(df).flatten()
    xs = np.arange(len(vals))

    # xs = [x for x, v in zip(xs, vals) if v]

    plt.figure()
    plt.plot(xs[-200:], vals[-200:])
    plt.grid(color='lightgrey')
    plt.yscale("log")
    plt.savefig(fname+'.png', dpi=300)
