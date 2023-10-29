from ase import Atoms
from util import configs
from ase.units import mol, kcal, Ha
from ase.io import write
from collections import Counter
import logging
try:
    import pyanitools as pya
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

def read_ANI(hdf5file, dset_label, elements_to_skip, prop_prefix="comp6_"):

    # from https://github.com/isayev/ANI1_dataset
    isolated_at_energies = {
        "H":  -0.500607632585,
        "C": -37.8302333826,
        "O": -75.0362229210, 
        "N": -54.5680045287
    }

    adl = pya.anidataloader(hdf5file)

    ats_out = []
    for idx, data in enumerate(adl):
        
        printed = False

        coordinates = data["coordinates"]
        elements = data["species"]
        all_energies = data["energies"]
        all_forces = data["forces"]
        test_elements = [el for el in elements if el not in elements_to_skip]
        if len(test_elements) > 0:
            continue


        label = f'{dset_label}_{idx}'

        for energy, forces, coord in zip(all_energies, all_forces, coordinates):
            at = Atoms(elements, positions = coord)
            at.info["original_comp6_hash"] = configs.hash_atoms(at)
            at.info["compound"] = label
            at.info["graph_name"] = f'{label}_mol'
            at.info["mol_or_rad"] = "mol"
            at.info["rad_num"] = "mol"
            at.info["dataset_type"] = label    

            # counted_ats = Counter(elements)
            # isolated_at_energy = 0
            # for symbol, count in counted_ats.items():
                # isolated_at_energy += count * isolated_at_energies[symbol]

            # print(f"isolated_at_energy: {isolated_at_energy} kcal/mol")
            # print(f"energy: {energy} kcal/mol")

            # at.info[f"{prop_prefix}energy"] = (energy + isolated_at_energy) * kcal / mol
            # at.arrays[f"{prop_prefix}forces"] = forces * kcal / mol

            at.info[f"{prop_prefix}energy"] = energy  * Ha
            at.arrays[f"{prop_prefix}forces"] = forces  * Ha 


            ats_out.append(at)
            if not printed:
                print(f"found some for {at.symbols}, {hdf5file}!")
                printed=True

    return ats_out