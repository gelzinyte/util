import os

import pytest
from pytest import approx

import numpy as np

from pyscf import gto
from pyscf import ao2mo

from ase.build import molecule


def test_from_mo_coefficients_to_density_matrix(mf):

    my_dm = mo_to_dm(mf)
    mf_dm = mf.from_chk()

    assert np.all(approx(my_dm) == mf_dm)

def mo_to_dm(mf):
    """from mo coefficients to density matrix"""
    N_electron = mf.mo_occ.sum()
    C = mf.mo_coeff
    N_mo = len(C)

    density_mx = np.zeros(C.shape)

    for i in range(N_mo):
        for j in range(N_mo):
            density_mx[i][j] = 2 * np.sum(
                [C[i][k] * C[j][k] for k in range(int(N_electron / 2))])

    return density_mx

def test_veff_core_to_fock(mf):

    fock = mf.get_fock()
    veff = mf.get_veff()
    core = mf.get_hcore()

    assert np.all(approx(fock) == core + veff)

def test_mf_mol_overlap(mf, mol):
    mf_overlap = mf.get_ovlp()
    mol_overlap = mol.intor('int1e_ovlp')

    assert np.all(approx(mf_overlap) == mol_overlap)


@pytest.mark.skip(reason='Is only true for minimal basis H2, need to '
                         'figure out why')
def test_FC_to_SCE(mf):
    # FC == SCE ?

    fock = mf.get_fock()
    coeff = mf.mo_coeff
    overlap = mf.get_ovlp()
    energies = np.eye(fock.shape[0]) * mf.mo_energy


    FC = fock @ coeff
    SCE = overlap @ coeff @ energies

    comparison = approx(FC) == SCE

    print('FC shape:', FC.shape)
    print('SCE shape:', SCE.shape)
    print('total no entries:', FC.shape[0]**2)
    print('number of truths in comparison:', np.sum(comparison))

    assert np.all(comparison)


def test_hcore(mf, mol):
    # h_core = kin + vnuc

    # !!!!!!!!!!!!!!!!!!!!!!!!1
    #   shouldn't these integrals change before and after HF orbitals were
    #   found?
    # I guess not - these are just expressions of the (T and V) operators
    # in our "AO" basis,
    # It's the coefficients in F(C)C=SCE that do all the work
    # !!!!!!!!!!!!!!!!!!!!!!!!!!

    h_core = mf.get_hcore()
    kin = mol.intor('int1e_kin')
    vnuc = mol.intor('int1e_nuc')

    assert np.all(approx(h_core) == kin + vnuc)


def test_Hcore_G_to_F(mol, mf):
    # F = Hcore + G ?

    # G_mu_nu from 2-electron integrals and density matrix
    density_mx = mf.from_chk()
    four_center_integrals = mol.intor(
        'int2e')  # they call this 'eri' everywhere for some reason



    fock = mf.get_fock()

    h_core = mf.get_hcore()
    G = G_from_dmx_and_integrals(density_mx, four_center_integrals)

    np.all(approx(fock) == h_core + G)


def G_from_dmx_and_integrals(density_mx, four_center_integrals):
    # eqn 3.154

    N = density_mx.shape[0]

    G = np.zeros((N, N))

    for idx1 in range(N):
        for idx2 in range(N):

            entries_to_sum_over = []

            for idx3 in range(N):
                for idx4 in range(N):
                    coulomb = four_center_integrals[idx1][idx2][idx3][
                        idx4]
                    exchange = four_center_integrals[idx1][idx4][idx3][
                        idx2]

                    coeff = density_mx[idx3][idx4]

                    item = coeff * (coulomb - 0.5 * exchange)

                    entries_to_sum_over.append(item)

            G[idx1][idx2] = np.sum(entries_to_sum_over)
    return G


def test_veff_from_two_el_integrals(mf, mol):
    # veff from 2 electron integrals
    # ------
    # from mf.get_veff docummentation
    # |      Returns:
    #  |          matrix Vhf = 2*J - K.  Vhf can be a list matrices,
    #  corresponding to the
    #  |          input density matrices.


    density_mx = mf.from_chk()
    four_center_integrals = mol.intor('int2e')

    G = G_from_dmx_and_integrals(density_mx, four_center_integrals)


    j, k = mf.get_jk()
    veff = mf.get_veff()


    # -> J and K here should be weighted by the density matrix
    # TODO

    assert not np.all(approx(veff) == 2 * j - k)

    # instead to get veff need to sum over appropriate 2-electron integrals
    assert np.all(approx(veff) == G)


def test_JK_from_dm_4c2e_ints(mf, mol):
    # J and K vs 2e/4c integrals, similarly to G above

    density_mx = mf.from_chk()
    four_center_integrals = mol.intor('int2e')


    my_K = K_from_ints(density_mx, four_center_integrals)
    my_J = J_from_ints(density_mx, four_center_integrals)

    j, k = mf.get_jk()

    assert np.all(approx(my_J) == j)
    assert np.all(approx(my_K) == k)

def J_from_ints(density_mx, four_center_integrals):

    N = four_center_integrals.shape[0]

    J = np.zeros((N, N))

    for a in range(N):
        for b in range(N):

            to_sum_over = []
            for i in range(N):
                for j in range(N):
                    entry = density_mx[i][j] * \
                            four_center_integrals[i][j][a][b]
                    to_sum_over.append(entry)

            J[b][a] = np.sum(to_sum_over)

    return J

def K_from_ints(density_mx, four_center_integrals):

    N = four_center_integrals.shape[0]

    K = np.zeros((N, N))

    for a in range(N):
        for b in range(N):

            to_sum_over = []
            for i in range(N):
                for j in range(N):
                    entry = density_mx[i][j] * \
                            four_center_integrals[a][i][j][b]
                    to_sum_over.append(entry)

            K[b][a] = np.sum(to_sum_over)

    return K



def test_total_e_from_dm_hcore_F(mf, mol):
    # total elelctron energy in terms of density, core and fock
    # eqn 3.184 when F is non-diagonal and expressed in AO basis  (I think
    # that's right?)

    h = mf.get_hcore()
    fock = mf.get_fock()
    density_mx = mf.from_chk()
    nuclear_repulsion = mol.energy_nuc()

    N = h.shape[0]

    total_energy = mf.e_tot

    HF_energy = 0.5 * np.sum(
        [[density_mx[mu][nu] * (h[nu][mu] + fock[nu][mu]) for mu in range(N)]
         for nu in range(N)])

    assert approx(total_energy) == HF_energy + nuclear_repulsion


@pytest.mark.skip(reason='Doesn\'t test whether I can get a Fock in MO basis')
def test_diagonalise_fock(mf):
    ## diagonalise fock

    fock = mf.get_fock()
    # print('non-canonical fock')
    # print(fock)

    orb_es, mo_coeff = mf.canonicalize(mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
    # print('canonicalisation:', orb_es, '\n', mo_coeff)


    assert np.all(approx(mf.mo_energy) == orb_es)



def test_ao_to_mo(tmp_path, mol_h2, mf_h2):
    ## in general, getting from 4 orbital integrals in AO basis to
    # integrals in MO basis
    mol = mol_h2
    mf = mf_h2

    lmo_fname = os.path.join(tmp_path, 'integrals_in_MO')
    mo_coeff = mf.mo_coeff
    ao2mo.kernel(mol, mf.mo_coeff, lmo_fname)

    e2o4_ao = mol.intor('int2e')
    C = mf.mo_coeff
    N = C.shape[0]

    e2o4_mo_my = np.zeros((N, N, N, N))



    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):

                    to_sum = []

                    for mu in range(N):
                        for nu in range(N):
                            for gamma in range(N):
                                for sigma in range(N):
                                    entry = C[mu][i] * \
                                            C[nu][j] * \
                                            C[gamma][k] * \
                                            C[sigma][l] * \
                                            e2o4_ao[mu][nu][gamma][sigma]

                                    to_sum.append(entry)

                    e2o4_mo_my[i][j][k][l] = np.sum(to_sum)

    with ao2mo.load(lmo_fname) as eri:
        e2o4_mo = ao2mo.restore(1, np.asarray(eri), mo_coeff.shape[1])

    assert  np.all(approx(e2o4_mo) == e2o4_mo_my)



def test_correlation_e_from_integrals(mf, mp2, e2o4_mo):

    verbose = False

    N_occ = sum(mf.mo_occ != 0)
    N = len(mf.mo_occ)
    N_virt = N - N_occ

    orb_energies = mf.mo_energy


    e2_mx = np.zeros((N_occ, N_occ, N_virt, N_virt))

    for i in range(N_occ):
        for j in range(N_occ):
            for a in range(N_occ, N):
                for b in range(N_occ, N):

                    if verbose:
                        print(f'i {i} j {j} a {a} b {b}')

                    denom = orb_energies[a] + orb_energies[b] - orb_energies[
                        i] - orb_energies[j]

                    int_iajb = e2o4_mo[i][a][j][b]
                    int_ibja = e2o4_mo[i][b][j][a]

                    entry = -1 * int_iajb * (2 * int_iajb - int_ibja)

                    if verbose:
                        print('numerator:', entry)
                    entry /= denom

                    if verbose:
                        print('int_iajb', int_iajb)
                        print('int_ibja', int_ibja)

                        print('denom', denom)
                        print('entry', entry)

                    e2_mx[i][j][a - N_occ][b - N_occ] = entry

    e_corr_my = np.sum(e2_mx)

    assert np.all(approx(mp2.e_corr) == e_corr_my)


def test_eris_to_ovov(mf, mp2, e2o4_mo):

    N_occ = sum(mf.mo_occ != 0)
    N = len(mf.mo_occ)
    N_virt = N - N_occ

    eris = mp2.ao2mo(mp2.mo_coeff)

    reshaped_ovov = eris.ovov.reshape((N_occ, N_virt, N_occ, N_virt))
    ovov_from_e2o4_mo = e2o4_mo[:N_occ, N_occ:, :N_occ, N_occ:]

    assert  np.all(approx(reshaped_ovov) == ovov_from_e2o4_mo)

def test_K_to_ovov(mp2, e2o4_mo):
    # K_iajb == ovov?
    # same as above cell

    N_occ = sum(mp2.mo_occ != 0)
    N = len(mp2.mo_occ)
    N_virt = N - N_occ

    K_iajb = e2o4_mo[:N_occ, N_occ:, :N_occ, N_occ:]
    ovov = mp2.ao2mo(mp2.mo_coeff).ovov.reshape(N_occ, N_virt, N_occ, N_virt)

    assert np.all(approx(K_iajb) == ovov)


def test_K_ijab_oovv(mp2, e2o4_mo):
    # K_ijab == oovv?

    N_occ = sum(mp2.mo_occ != 0)
    N = len(mp2.mo_occ)
    N_virt = N - N_occ

    K_ijab = e2o4_mo[:N_occ, :N_occ, N_occ:, N_occ:]

    ovov = mp2.ao2mo(mp2.mo_coeff).ovov.reshape(N_occ, N_virt, N_occ, N_virt)
    oovv = ovov.transpose(0, 2, 1, 3)

    # ofc not equal, because I'm taking different elements of K, but just
    # permuting ovov
    assert not np.all(approx(K_ijab) == oovv)


def correlation_e_from_mp2_amplitudes(mp2):
    # correlation energy from amplitudes?

    N_occ = sum(mp2.mo_occ != 0)
    N = len(mp2.mo_occ)
    N_virt = N - N_occ

    # K_ijab = e2o4_mo[:N_occ, :N_occ, N_occ:, N_occ:]
    ovov = mp2.ao2mo(mp2.mo_coeff).ovov.reshape(N_occ, N_virt, N_occ, N_virt)
    oovv = ovov.transpose(0, 2, 1, 3)

    t2 = mp2.t2

    # doing ijab, ijba over ovov transposed into oovv
    # is the same as
    # doing iajb, ibja over ovov
    # which is my equation!
    # E^{(2)} = - \sum_{ijab}\frac{(ia|jb)(2(ia|jb) - (ib|ja))}{\varepsilon_a +
    # \varepsilon_b - \varepsilon_i - \varepsilon_j}
    positive_entry = np.einsum('ijab, ijab', oovv, t2) * 2
    negative_entry = np.einsum('ijab, ijba', oovv, t2) * -1

    my_e_corr = positive_entry + negative_entry

    assert approx(my_e_corr) == mp2.e_corr









