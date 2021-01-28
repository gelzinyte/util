
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io import read
import io

def smi_to_xyz(smi, useBasicKnowledge=True, useExpTorsionAnglePrefs=True):
    """Converts smiles to 3D and writes to xyz file"""
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    out = AllChem.EmbedMolecule(mol, useBasicKnowledge=useBasicKnowledge, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs)

    insert = 'Properties=species:S:1:pos:R:3'
    xyz = Chem.rdmolfiles.MolToXYZBlock(mol)
    xyz = xyz.split(sep='\n\n')
    xyz = f'{xyz[0]}\n{insert}\n{xyz[1]}'
    xyz_file = io.StringIO(xyz)

    atoms = read(xyz_file, format='xyz')
    return atoms

