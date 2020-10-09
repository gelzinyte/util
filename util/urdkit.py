
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io import read, write

def get_xyz_str(mol):
    """For RDKit molecule, gets ase-compatable string with species and positions defined"""
    insert = 'Properties=species:S:1:pos:R:3'
    xyz = Chem.rdmolfiles.MolToXYZBlock(mol)
    xyz = xyz.split(sep='\n\n')
    return f'{xyz[0]}\n{insert}\n{xyz[1]}'


def write_rdkit_xyz(mol, filename):
    """Writes RDKit molecule to xyz file with filenam"""
    xyz_str = get_xyz_str(mol)
    with open(filename, 'w') as f:
        f.write(xyz_str)


def smi_to_xyz(smi, fname, useBasicKnowledge=True, useExpTorsionAnglePrefs=True):
    """Converts smiles to 3D and writes to xyz file"""
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    out = AllChem.EmbedMolecule(mol, useBasicKnowledge=useBasicKnowledge, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs)
    write_rdkit_xyz(mol, fname)
    mol = read(fname)
    return mol