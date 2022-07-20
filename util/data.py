from ase import Atoms
from util import configs
from ase.io import write
import logging
try:
    import pyanitools as pya
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

def read_ANI(hdf5file, dset_label, elements_to_skip):

    adl = pya.anidataloader(hdf5file)

    ats_out = []
    for idx, data in enumerate(adl):

        printed = False

        coordinates = data["coordinates"]
        elements = data["species"]
        test_elements = [el for el in elements if el not in elements_to_skip]
        if len(test_elements) > 0:
            continue


        label = f'{dset_label}_{idx}'

        for coord in coordinates:
            at = Atoms(elements, positions = coord)
            at.info["original_comp6_hash"] = configs.hash_atoms(at)
            at.info["compound"] = label
            at.info["graph_name"] = f'{label}_mol'
            at.info["mol_or_rad"] = "mol"
            at.info["rad_num"] = "mol"
            at.info["dataset_type"] = label    

            ats_out.append(at)
            if not printed:
                print(f"found some for {at.symbols}, {hdf5file}!")
                printed=True

    return ats_out