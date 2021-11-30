import os
import pandas as pd
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except ModuleNotFoundError:
    pass
import numpy as np

def draw_mols(mols, legend=None, molsPerRow=4):
    if not legend:
        legend = [f'{i}.' for i in range(len(mols))]
    else:
        legend = [f'{i}. {name}' for i, name in enumerate(legend)]
    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(250,250),
                                legends=legend)#, maxMols=400)

def main(input_csv, name_col, smiles_col):
    # df = pd.read_csv(input_csv, index_col='Unnamed: 0')
    df = pd.read_csv(input_csv, delim_whitespace=True)
    fname = os.path.basename(os.path.splitext(input_csv)[0]) + '.png'
    from_df(df, name_col, smiles_col, fname)

def from_df(df, name_col, smiles_col, fname):
    mols = [Chem.MolFromSmiles(s) for s in df[smiles_col]]
    legend = [f'{name}' for i, name in enumerate(df[name_col])]
    img = draw_mols(mols, legend)


    img.save(fname)
    return img

