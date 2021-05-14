import os
import tempfile
import shutil

import pytest
from pytest import approx

import numpy as np

from pyscf import gto
from pyscf import ao2mo

from ase.build import molecule



@pytest.fixture(scope='session')
def mol():

    ase_atoms = molecule('H2O')

    mol = gto.Mole()  # molecule object
    mol.atom = [[at.symbol, at.position] for at in
                ase_atoms]  # in ['O', [0.0, 0.0, 0.0]] form
    mol.unit = 'Ang'  # reads in positions with their units in Angstroms
    mol.symmetry = True  # makes it work with symmetry
    mol.verbose = 4
    mol.basis = 'ccpvdz'
    mol.build()

    return mol

@pytest.fixture(scope='session')
def mol_h2():
    ase_atoms = molecule('H2')
    ase_atoms.set_distance(0, 1, 1.4)

    mol = gto.Mole()  # molecule object
    mol.atom = [[at.symbol, at.position] for at in
                ase_atoms]  # in ['O', [0.0, 0.0, 0.0]] form
    mol.unit = 'Bohr'  # reads in positions with their units in Angstroms
    mol.symmetry = True  # makes it work with symmetry
    mol.verbose = 4
    # mol.basis = 'ccpvdz'
    mol.build()
    return mol

@pytest.fixture(scope='session')
def mf_h2(mol_h2):

    # tmp_path = tempfile.mkdtemp()

    mf = mol_h2.RHF()  # initialise the mean field method
    # mf.chkfile = os.path.join(tmp_path, 'pyscf_rhf_h2_checkfile.ch')
    mf.kernel()  # run the calculation
    return mf

@pytest.fixture(scope='session')
def mf(mol):
    """ does the HF"""
    mf = mol.RHF()
    # mf.chkfile = os.path.join(tmp_path, 'pyscf_rhf_checkfile.ch')
    mf.kernel()
    return mf

@pytest.fixture(scope='session')
def mp2(mf):
    mp2 = mf.MP2().run()
    return mp2

@pytest.fixture(scope='session')
def e2o4_mo( mol, mp2):

    tmp_path = tempfile.mkdtemp()

    lmo_fname = os.path.join(tmp_path, 'integrals_in_MO')
    ao2mo.kernel(mol, mp2.mo_coeff, lmo_fname)
    with ao2mo.load(lmo_fname) as eri:
        e2o4_mo = ao2mo.restore(1, np.asarray(eri), mp2.mo_coeff.shape[1])

    shutil.rmtree(tmp_path)

    return e2o4_mo
