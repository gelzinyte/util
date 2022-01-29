import logging

import numpy as np

from wfl.select_configs import by_descriptor

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s')

def per_environment(inputs, outputs, num,
                        at_descs_key=None, kernel_exp=None,
                        stochastic=True,
                        keep_descriptor_arrays=True, center=True,
                        leverage_score_key='leverage_score',
                        write_all_configs=False
                        ):
    """Select atoms from a list or iterable using CUR on per-atom descriptors

    Parameters
    ----------
    inputs: ConfigSet_in
        atomic configs to select from
    outputs: ConfigSet_out
        where to write output to
    num: int
        number to select
    stochastic_seed: bool, default None
        fix seed of random number generator
    at_descs_key: str
        key to Atoms.arrays dict containing per-environment descriptor vector
    kernel_exp: float, default None
        exponent to compute kernel
    stochastic: bool, default True
        use stochastic selection
    keep_descriptor_arrays: bool, default True
        do not delete descriptor from arrays
    center: bool, default True
        center data before doing SVD, as generally required for PCA
    leverage_score_key: str, default 'leverage_score'
        if not None, info key to store leverage score in
    write_all_configs: bool, default False
        whether to return all configs or those selected by cur only

    Returns
    -------
        ConfigSet_in corresponding to configs that contain selected
        environments
    """

    if outputs.is_done():
        logger.info('output is done, returning')
        return outputs.to_ConfigSet_in()

    logger.info('preparing descriptors')

    at_descs, parent_at_idx = prepare_descriptors(inputs, at_descs_key)

    # do SVD on kernel if desired
    if kernel_exp is not None:
        logger.info('computing kernel matrix')
        descs_mat = np.matmul(at_descs.T, at_descs) ** kernel_exp
        if center:
            # centering like that used for kernel-PCA
            row_of_col_means = np.mean(descs_mat, axis=0)
            col_of_row_means = np.mean(descs_mat, axis=1)
            descs_mat -= row_of_col_means
            descs_mat = (descs_mat.T - col_of_row_means).T
            descs_mat += np.mean(col_of_row_means)
    else:
        if center:
            descs_mat = (at_descs.T - np.mean(at_descs, axis=1)).T
        else:
            descs_mat = at_descs

    logger.info('calling workflow CUR')

    selected, leverage_scores = by_descriptor.CUR(mat=descs_mat, num=num,
                                 stochastic=stochastic)


    logger.info('processing cur results')

    clean_and_write_selected(inputs=inputs, 
                             outputs=outputs, 
                             selected=selected,
                             parent_at_idx=parent_at_idx, 
                             at_descs_key=at_descs_key,
                            keep_descriptor_arrays=keep_descriptor_arrays, 
                            leverage_score_key=leverage_score_key,
                             leverage_scores=leverage_scores,
                             write_all_configs=write_all_configs
                             )

    return outputs.to_ConfigSet_in()

def clean_and_write_selected(inputs, outputs, selected,
                             parent_at_idx, at_descs_key,
                             keep_descriptor_arrays,
                             leverage_score_key, 
                             leverage_scores,
                             write_all_configs):
    """Writes configs with selected environments to output configset

    Parameters
    ----------
    inputs: ConfigSet_in
        input configuration set
    outputs: ConfigSet_out
        target for output of selected configurations
    selected: list(int)
        list of indices to be selected, cannot have duplicates
    parent_at_idx: list(int)
        list of parent atoms id's to trace back which environment's
        descriptor came from which structure.
    at_descs_key: str, default None
        key in info dict to delete if keep_descriptor_arrays is False
    keep_descriptor_arrays: bool, default True
        keep descriptor in info dict
    """

    if not keep_descriptor_arrays and at_descs_key is None:
        raise RuntimeError('Got False \'keep_descriptor_arrays\' but not the info key \'at_descs_key\' to wipe')

    selected_s = set(selected)
    assert len(selected) == len(selected_s)

    selected_parents = parent_at_idx[selected]
    selected_parents = np.asarray(list(set(selected_parents)))

    logger.info(f'Selected {len(selected)} environments from '
                f'{len(selected_parents)} structures ({len(selected_parents) / len(list(inputs))*100:.0f}% of all structures from md).')

    # make a list of leverage scores for each of individual environments
    # and subdivide into list of lists to be assigned to atoms.
    leverage_scores_arrays = leverage_scores_into_arrays(inputs,
                                                         leverage_scores,
                                                         selected,
                                                         parent_at_idx)

    counter = 0
    # inputs is an iterable, can't directly reference specific # configs,
    # so loop through and search in set (should be fast)
    for at_i, (at, scores) in enumerate(zip(inputs, leverage_scores_arrays)):
        if not write_all_configs:
            if at_i not in selected_parents:
                continue

        if not keep_descriptor_arrays:
            del at.arrays[at_descs_key]
            del at.info[at_descs_key]
        if leverage_score_key:
            at.arrays[leverage_score_key] = scores

        outputs.write(at)
        counter += 1

        if not write_all_configs:
            if counter >= len(selected):
                # skip remaining iterator if we've used entire selected list
                break
    outputs.end_write()

def leverage_scores_into_arrays(inputs, leverage_scores, selected,
                             parent_at_idx):
    """Goes from leverage scores for the selected configs into list of
    lists to be written into Atoms.arrays for each structure

    Parameters
    ----------
    inputs: ConfigSet_in
        all of the configs that went into cur
    leverage_scores: list(float)
        scores for selected environments
    selected: list(int)
        indices for all of the environments
    parent_at_idx: list(int)
        list of len(total_environments) with indices of inputs from to
        which the environments correspond

    Returns
    -------
        list(list(float)) - leverage scores to be assigned to output
        at.arrays.
     """

    # list of all environments with leverage scores for those that have
    # been selected and zeros otherwise
    scores_to_assign = np.zeros(len(parent_at_idx))
    scores_to_assign[selected] = leverage_scores


    # chop up list of all environments into sections that have length of
    # the parent Atoms
    stride_start = 0
    output_scores = []
    for at in inputs:
        n_at = len(at)
        scores = scores_to_assign[stride_start:stride_start + n_at]
        output_scores.append(np.array(scores))
        stride_start += n_at

    return output_scores


def prepare_descriptors(inputs, at_descs_key):
    """ processes configs or input descriptor array to produce descriptors
    colun array

    Parameters
    ----------
    inputs: ConfigSet_in
        input configurations
    at_descs_key: str, default None
        key into Atoms.info dict for descriptor vector of each config

    Returns
    -------
        at_descs: np.ndarray (desc_len x n_atoms)
            array of descriptors (as columns) for each config
        parent_idx: np.array
            array of indices of parent structures from which the
            environments were selected.

    """

    # select environment descriptors and make note of the structure index they
    # correspond to
    at_descs = None
    parent_idx = []
    for at_idx, at in enumerate(inputs):
        # at_descs.append(at.arrays[at_descs_key])
        if at_descs is None:
            at_descs = at.arrays[at_descs_key]
        else:
            at_descs  = np.concatenate([at_descs, at.arrays[at_descs_key]])

        parent_idx += [at_idx] * len(at)

    parent_idx = np.asarray(parent_idx)
    at_descs = np.asarray(at_descs).T

    # normalise
    at_descs = np.apply_along_axis(normalisation, axis=0, arr=at_descs)

    assert len(parent_idx) == at_descs.shape[1]

    return at_descs, parent_idx

def normalisation(a, *args, **kwargs):
    return a / np.sqrt(np.dot(a, a))














