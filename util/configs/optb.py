import re
import numpy as np

def assign_values(mol, sub_df, smi):
    # print(smi)
    only_letters_smi = re.sub('[\W_]+', '', smi)
    only_letters_smi = re.sub('[\d]+', '', only_letters_smi)
    without_h = re.sub('[H]+', '', only_letters_smi)
    no_heavy_atoms = len(without_h)

    H_positions = [char == "H" for char in list(only_letters_smi)]
    H_positions = np.where(np.array(list(only_letters_smi)) == "H")[0]
    # print('only_letters_smi:', only_letters_smi)
    # print("h_positions:", H_positions)

    positions = np.array(sub_df["SoM"])
    positions = [int(pos) for pos in positions]
    # print("positions:", positions)
    adjusted_positions = adjust_positions_for_H(H_positions, positions, smi)
    # print("adjusted_positions:", adjusted_positions)

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


