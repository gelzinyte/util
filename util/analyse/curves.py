from ase.io import read
import os
import ace
import matplotlib.pyplot as plt
import numpy as np
import pickle
from util import natural_sort
from pathlib import Path
from wfl.calculators import generic
from wfl.configset import ConfigSet, ConfigSet_out

def save(base_dir, val_configs):
    base_dir = Path(base_dir)

    all_data = collect_run(base_dir, val_configs)

    with open(f"{base_dir.name}.pkl", "wb") as f:
        pickle.dump(all_data, f)


def collect_run(base_dir, val_configs, runs_dir="fits", ):
    runs_dir = base_dir / runs_dir
    dirs = get_iterations_dirs(runs_dir)

    raw_data = {}
    for idx, dir_name in enumerate(dirs):
        dir_name = Path(dir_name)
        expected_name = dir_name.name
        have_name = f"iteration_{idx:02d}"
        assert have_name == expected_name, f" expected {expected_name}, gotten {have_name}"

        eval_potential(dir_name, idx, val_configs)

    for idx, dir_name in enumerate(dirs):
        dir_name = Path(dir_name)
 
        evaled_fname=f"tmp_ace_{idx}.xyz"
        if not os.path.exists(evaled_fname):
            print(f'not found {evaled_fname}')
            continue
        evaled_configs = read(evaled_fname, ":")
        errors = collect_maes_of_single_potential(evaled_configs)

        if idx == 0:
            train_set_fname = runs_dir / f"training_sets/train_for_fit_0.xyz"
        else:
            train_set_fname = runs_dir / f"training_sets/{idx-1:02d}.train_for_ace_{idx:02d}.xyz"

        train_set_size = len(read(train_set_fname, ":"))

        raw_data[train_set_size] = errors

        os.remove(f"tmp_ace_{idx}.xyz")

    return process_data(raw_data)

def process_data(raw_data):

    data_out = {} 
    data_out["energy"] = {}
    data_out["forces"] = {}
    data_out["energy"]["mean"] = []
    data_out["energy"]["std"] = []
    data_out["energy"]["all_errors"] = []
    data_out["forces"]["mean"] = []
    data_out["forces"]["std"] = []
    data_out["forces"]["all_errors"] = []
    data_out["xvals"] = []

    for train_set_size, errors in raw_data.items():

        data_out["xvals"].append(train_set_size)

        energy_errors = errors["energy"]
        data_out["energy"]["mean"].append(np.mean(energy_errors))
        data_out["energy"]["std"].append(np.std(energy_errors))
        data_out["energy"]["all_errors"].append(energy_errors)


        forces_errors = errors["forces"]
        data_out["forces"]["mean"].append(np.mean(forces_errors))
        data_out["forces"]["std"].append(np.std(forces_errors))
        data_out["forces"]["all_errors"].append(forces_errors)

    return data_out
    


def eval_potential(dir_name, idx, val_configs):
    ace_name = dir_name / f"fit_dir/ace_{idx}.json"
    if not os.path.exists(ace_name):
        print(f'not found {ace_name}')
        return
    calc = (ace.ACECalculator, [], {"jsonpath":ace_name, "ACE_version":2})
    co = ConfigSet_out(output_files=f"tmp_ace_{idx}.xyz", force=True, all_or_none=True)
    print(f"evaluating {ace_name}")
    evaled_configs = generic.run(val_configs, co, calc, properties=["energy", "forces"], output_prefix="ace_")
    print(f'evaled configs')

def collect_maes_of_single_potential(evaled_configs):
    errors = get_errors(evaled_configs)
    return errors
    

def get_errors(evaled_configs):
    errors = {}
    energy_absolute_errors = []
    force_comp_absolute_errors = []
    for at in evaled_configs:
        dft_es = at.info["dft_energy"]
        ace_es = at.info["ace_energy"]
        energy_absolute_errors.append(np.abs(dft_es - ace_es)/len(at))

        dft_fs = at.arrays["dft_forces"].flatten()
        ace_fs = at.arrays["ace_forces"].flatten()
        force_comp_absolute_errors += list(np.abs(dft_fs - ace_fs))

    errors["energy"] = np.array(energy_absolute_errors)
    errors["forces"] = np.array(force_comp_absolute_errors)

    return errors


def get_iterations_dirs(runs_dir):
    runs_dir = Path(runs_dir)
    return natural_sort([str(dir_name) for dir_name in runs_dir.iterdir() if "iteration" in str(dir_name)])

