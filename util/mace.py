import numpy as np
from wfl.utils.parallel import construct_calculator_picklesafe
from ase import Atoms
from wfl.autoparallelize import autoparallelize


def to_cpu(model_path_in, model_path_out):
    import torch
    model = torch.load(model_path_in)
    model = model.to('cpu')
    torch.save(model, model_path_out)


def calc_descriptor_ll(atoms, calculator, prefix='mace_'):

    calculator = construct_calculator_picklesafe(calculator)    

    if isinstance(atoms, Atoms):
        atoms = [atoms]

    for at in atoms:

        calculator.calculate(at)

        local_desc = calculator.extra_results["descriptor"]
        local_desc = local_desc / np.linalg.norm(local_desc, axis=1).reshape(len(local_desc), 1)

        global_sum = np.sum(local_desc, axis=0)
        global_sum = global_sum / np.linalg.norm(global_sum)

        global_mean = np.mean(local_desc, axis=0)
        global_mean= global_mean / np.linalg.norm(global_mean)

        at.arrays[f"{prefix}local_desc"] = local_desc
        at.info[f"{prefix}global_sum_desc"] = global_sum
        at.info[f"{prefix}global_mean_mace_desc"] = global_mean
    
    return atoms 


def descriptor(*args, **kwargs):
    return autoparallelize(calc_descriptor_ll, *args, **kwargs)

