from ase.io import read, write
import random
from util import radicals

ats = read("bde_mols.dft_opt.xyz", ":")

ats_out = []
for at in ats:
    del at.info["dft_dipole"]
    del at.info["dft_energy"]
    del at.arrays["dft_forces"]
    # add molecule
    ats_out.append(at.copy())

    rad = at.copy()
    comp = rad.info["compound"]
    sp3_Hs = radicals.get_sp3_h_numbers(rad.copy())
    H_idx_to_del = random.choice(sp3_Hs)
    del rad[H_idx_to_del]

    rad.info["mol_or_rad"] = "rad"
    rad.info["rad_num"] = H_idx_to_del
    rad.info["graph_name"] = str(comp) + '_rad' + str(H_idx_to_del)

    ats_out.append(rad)

write("bde_mols_and_rads.xyz", ats_out)

