import matplotlib.pyplot as plt
import numpy as np
import datetime
import time

def get_time(entry):
    x = time.strptime(entry, '%H:%M:%S')
    entry_sec =  datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
    return entry_sec
 

def plot_fmax(fnames, pic_fname):

    plt.figure

    for label, fname in fnames:

        with open(fname) as f:
            lines = f.read()
        lines = lines.split('\n')

        # if label == "neb_3":
            # import pdb; pdb.set_trace()

        vals = []
        steps = []
        times = []
        prev_time = None 
        for line in lines:
            if "PreconLBFGS" not in line:
                continue
            entries = line.split()
            vals.append(float(entries[-1]))
            steps.append(int(entries[1]))
            if prev_time is None:
                prev_time = get_time(entries[2])
                continue        
            this_time = get_time(entries[2])
            del_time = this_time - prev_time

            # if day switched while this was running
            if del_time > 0:
                times.append(this_time - prev_time)
            prev_time = this_time
        
        step_time = np.mean(times)
        # if label == "neb_3":
            # print(times)
        
        plt.plot(steps, vals, label=f"{label}: {step_time:.1f}", alpha=0.7)

    plt.ylim((0.01, 4))
    plt.grid(color='lightgrey', which='both')
    plt.xlabel("lbfgs step")
    plt.ylabel('fmax')
    plt.legend(title="time per step, s", bbox_to_anchor=(1, 1),
                    loc='upper right')
    plt.tight_layout()
    plt.savefig(pic_fname)