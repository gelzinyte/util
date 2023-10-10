import re
from ase.io import read, write
import numpy as np
from ase.units import kJ, mol

def assign_values(mol_at, sub_df, smi):
    # print(smi)
    # print(sub_df)
    only_letters_smi = re.sub('[\W_]+', '', smi)
    only_letters_smi = re.sub('[\d]+', '', only_letters_smi)
    without_h = re.sub('[H]+', '', only_letters_smi)
    no_heavy_atoms = len(without_h)
    # print(without_h)

    # H_positions = [char == "H" for char in list(only_letters_smi)]
    # H_positions = np.where(np.array(list(only_letters_smi)) == "H")[0]
    # print('only_letters_smi:', only_letters_smi)
    # print("h_positions:", H_positions)

    positions = np.array(sub_df["SoM"])
    positions = [int(pos) for pos in positions]
    symbols = [sym for idx, sym in enumerate(mol_at.symbols) if idx in positions]
    if np.any(symbols == "H"):
        print(smi)
        write("whoops.xyz", mol_at)
        raise RuntimeError()


    # print(positions)
    # print("positions:", positions)
    # adjusted_positions = adjust_positions_for_H(H_positions, positions, smi)
    # print("adjusted_positions:", adjusted_positions)

    soms = np.empty(len(mol_at))
    soms.fill(False)
    soms[positions] = True

    # print(soms)
    mol_at.arrays["site_of_metabolism"] = np.array([bool(som) for som in soms])

    orig_eas = [float(ea) for ea in np.array(sub_df["Ea"])]
    eas = np.empty(len(mol_at))
    eas.fill(np.nan)
    for pos, ea in zip(positions, orig_eas):
        eas[pos] = ea

    eas = eas * kJ / mol
    
    mol_at.arrays["Optibrium_Ea"] = eas

    # add sites of metabolism 

    labels = ["primary_exp_site", "secondary_exp_site", "tertiary_exp_site"]
    for label, series in zip(labels, [sub_df["PM"], sub_df["SM"], sub_df["TM"]]):
        selected_pos = []
        for pos, entry in zip(sub_df["SoM"], series):
            if isinstance(entry, str):
                selected_pos.append(int(pos))
        array_to_add = np.empty(len(mol_at))
        array_to_add.fill(False)
        array_to_add[selected_pos] = True
        mol_at.arrays[label] = np.array([bool(pos) for pos in array_to_add])

    mol_at.info["smiles"] = smi

    return mol_at
     


def adjust_positions_for_H(H_positions, orig_positions, smi):
    new_positions = orig_positions.copy()

    for idx, pos in enumerate(orig_positions):
        if pos in H_positions:
            print(pos, smi)
            raise ValueError(f"matched H at position {pos}")
        
        # print("h_positions:", H_positions)
        # print(type(pos))
        count = np.sum(H_positions < pos)
        new_positions[idx] += count 
    return new_positions


