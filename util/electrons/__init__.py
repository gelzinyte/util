import numpy as np
from pytest import approx

def get_eps_ij(mp2):
    """returns MP2 correlation energy for each pair of occupied orbitals"""

    N_occ = mp2.get_nocc()
    N =  mp2.get_nmo()  
    N_virt = N - N_occ

    t2 = mp2.t2
    ovov = mp2.ao2mo(mp2.mo_coeff).ovov.reshape(N_occ, N_virt, N_occ, N_virt)
    oovv = ovov.transpose(0, 2, 1, 3)

    positive = np.einsum('ijab, ijab -> ij', oovv, t2) * 2
    negative = np.einsum('ijab, ijba -> ij', oovv, t2) * -1

    eps_ij = positive + negative

    # eps_ij should sum to correlation energy
    assert approx(mp2.e_corr) == eps_ij.sum()

    return eps_ij

