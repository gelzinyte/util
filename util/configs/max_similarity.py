import click
import os
import numpy as np
from ase.io import read
from ase import Atoms
from tqdm import tqdm
import logging
from wfl.configset import ConfigSet_out, ConfigSet_in
from wfl.pipeline import iterable_loop

logger = logging.getLogger(__name__)


def get_descriptor_matrix(ats, descriptor_arrays_key):

    if isinstance(ats, Atoms):
        ats = [ats]

    at_descs = None
    for at in ats:
        if at_descs is None:
            at_descs = at.arrays[descriptor_arrays_key]
        else:
            at_descs = np.concatenate([at_descs, at.arrays[
                descriptor_arrays_key]])

    # shape (desc_len x n_descs)
    at_descs = np.asarray(at_descs).T
    return at_descs


def kernel_non_normalised(mx1, mx2, zeta):
    return np.matmul(mx1.T, mx2) ** zeta

def norm_for_kernel(mx, zeta):
    return np.sqrt(np.array([np.dot(i, i) ** zeta for i in mx.T]))
    # return np.sqrt(np.diagonal(kernel_non_normalised(mx, mx, zeta)))

def kernel(mx1, mx2, zeta):
    norm_1 = norm_for_kernel(mx1, zeta)
    norm_2 = norm_for_kernel(mx2, zeta)
    norm = np.outer(norm_1, norm_2)
    non_normalised = kernel_non_normalised(mx1, mx2, zeta)
    assert norm.shape == non_normalised.shape
    return np.divide(non_normalised, norm)


def main(train_set, set_to_eval, output_fname, zeta=1,
         max_similarity_key=None, remove_descriptor=False):

    if max_similarity_key is None:
        logger.info(f'train_set filename: {train_set}')
        train_set_base = os.path.splitext(os.path.basename(train_set))[0]
        logger.info(f'train_set_base: {train_set_base}')
        max_similarity_key = f'max_SOAP_z{zeta}_to_{train_set_base}'

    descriptor_arrays_key = 'SOAP-n4-l3-c2.4-g0.3'

    logger.info(f'train set {train_set}, set to evaluate {set_to_eval}, '
                f'output fname {output_fname}, descriptors key '
                f'{descriptor_arrays_key}')

    outputs=ConfigSet_out(output_files=output_fname)
    inputs = ConfigSet_in(input_files=set_to_eval)
    ats_train = read(train_set, ':')
    # ats_compare = read(set_to_eval, ':')

    train_descs = get_descriptor_matrix(ats_train, descriptor_arrays_key)

    logger.info('assigning similarities')

    assign_max_similarity(inputs=inputs,
                          outputs=outputs,
                          train_descs=train_descs,
                          zeta=zeta,
                          descriptor_arrays_key=descriptor_arrays_key,
                          max_similarity_key=max_similarity_key,
                          remove_descriptor=remove_descriptor)


def assign_max_similarity(inputs, outputs, train_descs, zeta,
                          descriptor_arrays_key, max_similarity_key,
                          remove_descriptor):
    return iterable_loop(iterable=inputs, configset_out=outputs,
                         op=assign_max_similarity_op,
                         train_descs=train_descs,
                         zeta=zeta,
                         descriptor_arrays_key=descriptor_arrays_key,
                         max_similarity_key=max_similarity_key,
                         remove_descriptor=remove_descriptor)


def assign_max_similarity_op(atoms, train_descs, zeta,
                            descriptor_arrays_key, max_similarity_key,
                            remove_descriptor):

    if isinstance(atoms, Atoms):
        atoms = [atoms]

    atoms_out = []
    for at in atoms:
        at_descs = get_descriptor_matrix(at, descriptor_arrays_key)

        kernel_mx = kernel(at_descs, train_descs, zeta)

        max_similarity_per_atom = kernel_mx.max(axis=1)
        at.arrays[max_similarity_key] = max_similarity_per_atom
        at.info[max_similarity_key] = np.max(max_similarity_per_atom)

        if remove_descriptor:
            del at.arrays[descriptor_arrays_key]
            del at.info[descriptor_arrays_key]

    return atoms_out







