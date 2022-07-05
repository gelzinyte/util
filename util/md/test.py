import os
from pathlib import Path

from ase.io import read, write

from wfl.configset import ConfigSet, OutputSpec 
from wfl.generate import md

from util import configs


def run(workdir_root, in_ats, temp, calc, info_label, steps, sampling_interval, 
        pred_prop_prefix, remote_info):

    workdir_root = Path(workdir_root) 
    workdir_root.mkdir(exist_ok=True)

    tags = {"temp":int(str(temp))}


    ci, co = prepare_inputs(in_ats, info_label, workdir_root, tags)

    os.environ["WFL_DETERMINISTIC_HACK"] = "true"

    md_params = {
        "steps": steps,
        "dt": 0.5,  # fs
        "temperature": temp,  # K
        "temperature_tau": 500,  # fs, somewhat quicker than recommended (???)
        "traj_step_interval": sampling_interval,
        "results_prefix": pred_prop_prefix,
        "update_config_type": False}

    md.sample(
        inputs=ci, 
        outputs=co,
        calculator=calc,
        verbose=False,
        remote_info=remote_info,
        num_inputs_per_python_subprocess=None,
        traj_fn_info_entry="md_traj_label",
        **md_params,
        )



def prepare_inputs(ats, info_label, workdir_root, tags):

    input_files = []
    output_files = {}
    for at in ats:
        hash = configs.hash_atoms(at)
        label = at.info[info_label] + hash
        at.info["md_traj_label"] = label
        fname_in = workdir_root / (label + "_in.xyz")
        fname_out = workdir_root / (label + "_out.xyz")
        input_files.append(fname_in)
        output_files[fname_in] = fname_out
        write(fname_in, at)

    ci = ConfigSet(input_files=input_files)
    co = OutputSpec(
        output_files=output_files,
        all_or_none=True, 
        set_tags=tags)
    return ci, co
