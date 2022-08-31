

from util import configs
import util

def process(all_atoms, ref_energy_key, pred_energy_key,
             info_label, energy_type, ref_forces_key=None, 
             pred_forces_key=None, isolated_atoms=None):


    assert energy_type in ["atomization_energy", "total_energy"]
    if energy_type == "atomization_energy":
        energy_getter_function = util.get_atomization_energy_per_at
    elif energy_type == "total_energy":
        energy_getter_function = util.read_energy


    if isolated_atoms is None and energy_type=='atomization_energy':
        isolated_atoms = [at for at in all_atoms if len(at) == 1]
    all_atoms = [at for at in all_atoms if len(at) != 1] 

    all_atoms = configs.into_dict_of_labels(all_atoms, info_label=info_label)

    data = get_data_itself(
        all_atoms = all_atoms, 
        energy_getter_function = energy_getter_function,
        ref_energy_key = ref_energy_key,
        pred_energy_key = pred_energy_key,
        isolated_atoms = isolated_atoms,
        ref_forces_key = ref_forces_key, 
        pred_forces_key = pred_forces_key, 
        skip = skip
    )
    return data


def get_data_itself(all_atoms, energy_getter_function, ref_energy_key, pred_energy_key, isolated_atoms, ref_forces_key=None, pred_forces_key=None, skip=False):

    data = {"all" = {
        "energy" = {"predicted": [], "reference":[]},
        "forces" = {"predicted": [], "reference":[]}} 
    }

    assert "all" not in all_atoms

    number_of_skipped_configs = 0
    for label, atoms in all_atoms.items():

        if label not in data:
            data[label] = {
                "energy" = {"predicted": [], "reference":[]},
                "forces" = {"predicted": [], "reference":[]}}

        if skip:
            do_skip = check_skip(at, ref_energy_key, pred_energy_key, ref_forces_key, pred_forces_key)
            number_of_skipped_configs += int(do_skip) 
            if do_skip:
                continue

        ref_e = energy_getter_function(at, isolated_atoms, ref_energy_key)
        pred_e = energy_getter_function(at, isolated_atoms, pred_energy_key) 
        ref_f = list(at.arrays[ref_forces_key].flatten()) 
        pred_f = list(at.arrays[pred_forces_key].flatten())

        data[label]["energy"]["reference"].append(ref_e)
        data[label]["energy"]["predicted"].append(pred_e)
        data[label]["forces"]["reference"] += ref_f
        data[label]["forces"]["predicted"] += pred_f

        data["all"]["energy"]["reference"].append(ref_e)
        data["all"]["energy"]["predicted"].append(pred_e)
        data["all"]["forces"]["reference"] += ref_f
        data["all"]["forces"]["predicted"] += pred_f



    if number_of_skipped_configs > 0: 
        logger.warn(f'skipped {number_of_skipped_configs} configs, because one of {ref_energy_name}, {pred_energy_name}, {ref_forces_key} or {pred_forces_key} was not found.')


    for label in data.keys():
        for obs in ["energy", "forces"]: 
            for val_type in ["predicted", "reference"]:
                vals = data[label][obs][val_type]
                data[label][obs][val_type] = np.array(vals)

    return data

def check_skip(at, ref_energy_key, pred_energy_key, ref_forces_key, pred_forces_key, skip):
    if ref_energy_key not it at.info or pred_energy_key not in at.info:   
        if skip:
            return True 
        else:
            print(at.info)
            raise ValueError(f"did not found property (either {ref_energy_key} or {pred_energy_key}) in atoms")

    if ref_forces_key is not None and pred_forces_key is not None: 
        if ref_forces_key not it at.arrays or pred_forces_key not in at.arrays:   
            if skip:
                return True 
            else:
                print(at.info)
                raise ValueError(f"did not found property (either {ref_forces_key} or {pred_forces_key}) in atoms")
    return False

