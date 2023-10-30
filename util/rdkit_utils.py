from rdkit.Chem import rdDetermineBonds
from io import StringIO
import rdkit.Chem
from util import smarts_cho_patterns
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs


def xyz_to_mol(fname):

    with open(fname) as fileobj:
        fileobj = StringIO(fileobj.read())

    frames = []
    while True:
        line = fileobj.readline()
        if line.strip() == '':
                break
        natoms = int(line)
        fileobj.readline()
        atom_lines = []
        for _ in range(natoms):
            line = fileobj.readline()
            atom_lines.append(line)
        assert len(atom_lines) == natoms
        frames.append((natoms, atom_lines))

    rdkit_mols = []
    for frame in frames:
        xyz_block = str(frame[0]) + "\n\n" + "".join(frame[1])
        mol = rdkit.Chem.rdmolfiles.MolFromXYZBlock(xyz_block)
        if mol is None:
            import pdb; pdb.set_trace()
            continue
        rdDetermineBonds.DetermineBonds(mol)
        rdkit_mols.append(mol)
    return rdkit_mols


maccsKeys = None

def GenMyMACCSKeys(mol):

    global maccsKeys

    if maccsKeys is None:
        maccsKeys = [(None, 0)] * len(smarts_cho_patterns.smartsPatt.keys())
        MACCSkeys._InitKeys(maccsKeys, smarts_cho_patterns.smartsPatt)

    ctor =  DataStructs.SparseBitVect
    res = ctor(len(maccsKeys) + 1)
    for i, (patt, count) in enumerate(maccsKeys):
        if patt is not None:
            if count == 0:
                res[i + 1] = mol.HasSubstructMatch(patt)
            else:
                matches = mol.GetSubstructMatches(patt)
                if len(matches) > count:
                    res[i + 1] = 1
        else:
            raise ValueError("Expected not None patt")
    return res
