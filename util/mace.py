import numpy as np
from wfl.utils.parallel import construct_calculator_picklesafe
from ase import Atoms
from wfl.autoparallelize import autoparallelize
from ase.io import read, write
import warnings
warnings.filterwarnings("error", module="mace")


def to_cpu(model_path_in, model_path_out):
    import torch
    model = torch.load(model_path_in)
    model = model.to('cpu')
    torch.save(model, model_path_out)


def calc_descriptor_ll(atoms, calculator, prefix='mace_', normalize=True, skip_unnormalizable=True):


    calculator = construct_calculator_picklesafe(calculator)    

    if isinstance(atoms, Atoms):
        atoms = [atoms]

    ats_out = []
    for at in atoms:
        at = at.copy()

        calculator.calculate(at)

        local_desc = calculator.extra_results["descriptor"]
        # global_sum = np.sum(local_desc, axis=0)
        global_mean = np.mean(local_desc, axis=0)

        if normalize:

            local_norm = np.linalg.norm(local_desc, axis=1)

            try:
                local_desc = local_desc / local_norm.reshape(len(local_desc), 1)
            except RuntimeWarning as e:
                write(f"{at.info['dataset_type']}_bad_descriptor.xyz", at)
                if skip_unnormalizable:
                    local_desc = None
                else:
                    raise e 


            # global_sum = global_sum / np.linalg.norm(global_sum)

            try:
                global_mean= global_mean / np.linalg.norm(global_mean)
            except RuntimeWarning as e:
                write(f"{at.info['dataset_type']}_bad_descriptor.xyz", at)
                if skip_unnormalizable:
                    global_mean = None
                else:
                    raise e 



        if local_desc is not None:
            at.arrays[f"{prefix}local_desc"] = local_desc
        else:
            at.info[f"{prefix}local_desc_failed"] = "FAILED_DESCRIPTOR"

        # at.info[f"{prefix}global_sum_desc"] = global_sum
        if global_mean is not None:
            at.info[f"{prefix}global_mean_mace_desc"] = global_mean
        else:
            at.info[f"{prefix}global_desc_failed"] = "FAILED_DESCRIPTOR"


        ats_out.append(at)
    
    return ats_out 


def descriptor(*args, **kwargs):
    return autoparallelize(calc_descriptor_ll, *args, **kwargs)

