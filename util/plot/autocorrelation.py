import numpy as np
import matplotlib.pyplot as plt

def autocor(x):
    N = len(x)
    x = x - np.mean(x)
    cor = np.array([1/(N - k)*np.sum([ np.dot(x[i],x[i+k]) for i in range(N-k)]) for k in range(N)])
    return cor / cor[0]


def plot_autocorrelation(ats, idx1, idx2, title, vector_distance=False):
    distances = [at.get_distance(idx1, idx2, vector=vector_distance) for at
                 in ats]
    ac = autocor(distances)
    time = np.arange(1, len(ats) + 1) * 0.5

    plt.figure(figsize=(15, 5))
    plt.plot(time, ac, lw=0.5)
    plt.xlabel("tau, fs")
    plt.ylabel("normalized autocorrelation")
    plt.grid(color='lightgrey')
    plt.axhline(0, color='k')
    plt.title(title)
    plt.savefig(title.replace(' ', '_')+'.pdf')
