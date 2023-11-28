from rdkit.Chem import rdDetermineBonds
from io import StringIO
import rdkit.Chem
from util import smarts_cho_patterns
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from ase.io import read, write
import util

def xyz_to_mol(ats=None, fname=None, additional_arrays=None):

    def_additional_arrays = ["cur_leverage_score", "magmoms"]
    if additional_arrays is None:
        additional_arrays = def_additional_arrays
    else:
        additional_arrays += def_additional_arrays

    if not fname is None:
        assert ats is None
        out_fname = fname.replace(".xyz", ".cleaned_for_rdkit.xyz")
        ats = read(fname, ":")
    elif ats is not None:
        out_fname = "cleaned_for_rdkit.xyz"
        assert fname is None


    # ats = [util.remove_energy_force_containing_entries(at, additional_arrays=additional_arrays) for at in ats]

    for at in ats:
        arrays_keys = list(at.arrays.keys())
        for key in arrays_keys:
            if key in ["positions", "numbers"]: 
                continue
            del at.arrays[key]

    # import pdb; pdb.set_trace()

    write(out_fname, ats)

    with open(out_fname) as fileobj:
        fileobj = StringIO(fileobj.read())

    frames = []
    while True:
        line = fileobj.readline()
        if line.strip() == '':
                break
        natoms = int(line)
        info_line = fileobj.readline()
        atom_lines = []
        for _ in range(natoms):
            line = fileobj.readline()
            atom_lines.append(line)
        assert len(atom_lines) == natoms
        frames.append((natoms, atom_lines, info_line))

    rdkit_mols = []
    for frame in frames:
        xyz_block = str(frame[0]) + "\n\n" + "".join(frame[1])
        mol = rdkit.Chem.rdmolfiles.MolFromXYZBlock(xyz_block)
        if mol is None:
            import pdb; pdb.set_trace()
            continue
        try:
            if "mol_or_rad=rad" in frame[2]:
                import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            rdDetermineBonds.DetermineBonds(mol, allowChargedFragments=False, useHueckel=False, useAtomMap=False)
        except:
            import pdb; pdb.set_trace()
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
