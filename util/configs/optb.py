import re
import numpy as np

def assign_values(mol, sub_df, smi):
    print(smi)
    only_letters_smi = re.sub('[\W_]+', '', smi)
    only_letters_smi = re.sub('[\d]+', '', only_letters_smi)
    without_h = re.sub('[H]+', '', only_letters_smi)
    no_heavy_atoms = len(without_h)

    H_positions = [char == "H" for char in list(only_letters_smi)]
    H_positions = np.where(np.array(list(only_letters_smi)) == "H")[0]
    print(only_letters_smi)
    print(H_positions)

    positions = np.array(sub_df["Site_of_metabolism"])
    positions = [int(pos) for pos in positions]
    print(positions)
    adjusted_positions = adjust_positions_for_H(H_positions, positions)
    print(adjusted_positions)

def adjust_positions_for_H(H_positions, orig_positions):
    new_positions = orig_positions.copy()

    for idx, pos in enumerate(orig_positions):
        if pos in H_positions:
            raise RuntimeError("matched H")
        
        # print((H_positions))
        # print(type(pos))
        count = np.sum(H_positions < pos)
        new_positions[idx] += count 
    return new_positions


