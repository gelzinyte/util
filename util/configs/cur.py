import logging
from wfl.select_configs import by_descriptor

logger = logging.getLogger(__name__)

def cur_per_environment(inputs, outputs, num,
                        at_descs_arrays_key=None, kernel_exp=None,
                        stochastic=True, stochastic_seed=None,
                        keep_descriptor_arrays=True, center=True,
                        leverage_score_key=None,
                        selected_environment_info_label=None):
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
    at_descs_arrays_key: str, mutually exclusive with at_descs
        key to Atoms.arrays dict containing per-environment descriptor vector
    kernel_exp: float, default None
        exponent to compute kernel (if other than 1)
    stochastic: bool, default True
        use stochastic selection
    keep_descriptor_info: bool, default True
        do not delete descriptor from arrays
    center: bool, default True
        center data before doing SVD, as generally required for PCA
    leverage_score_key: str, default None
        if not None, info key to store leverage score in
    selected_environment_info_label: str, None
        if not None, arrays key to mark CUR-selected environment

    Returns
    -------
        ConfigSet_in corresponding to configs that contain selected
        environments
    """

    if outputs.is_done():
        logger.info('output is done, returning')
        return outputs.to_ConfigSet_in()

    at_descs, parent_at_idx = prepare_descriptors(inputs, at_descs_arrays_key)

    # do SVD on kernel if desired
    if kernel_exp is not None:
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

    selected, _ = by_descriptor.CUR(mat=descs_mat, num=num,
                                 stochastic=stochastic,
                      stochastic_seed=stochastic_seed, exclude_list=exclude_ind_list)

    clean_and_write_selected(inputs, outputs, selected,
                             parent_at_idx, at_descs_arrays_key,
            keep_descriptor_info, selected_environment_info_label)

    return outputs.to_ConfigSet_in()

def clean_and_write_selected(inputs, outputs, selected,
                             parent_at_idx, at_descs_arrays_key,
                             keep_descriptor_arrays,
                             selected_environment_info_label):
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
    at_descs_arrays_key: str, default None
        key in info dict to delete if keep_descriptor_info is False
    keep_descriptor_arrays: bool, default True
        keep descriptor in info dict
    """

    if not keep_descriptor_arrays and at_descs_arrays_key is None:
        raise RuntimeError('Got False \'keep_descriptor_info\' but not the info key \'at_descs_arrays_key\' to wipe')

    selected_s = set(selected)
    assert len(selected) == len(selected_s)

    selected_parents = parent_at_idx[selected_s]
    selected_parents = np.asarray(list(set(selected_parents)))

    logger.info(f'Selected {len(selected_s)} environments from '
                f'{len(selected_parents)} structures.')

    counter = 0
    # inputs is an iterable, can't directly reference specific # configs,
    # so loop through and search in set (should be fast)
    for at_i, at in enumerate(inputs):
        if at_i in selected_parents:
            if not keep_descriptor_arrays:
                del at.arrays[at_descs_arrays_key]
            outputs.write(at)
            counter += 1
            if counter >= len(selected_s):
                # skip remaining iterator if we've used entire selected list
                break
    outputs.end_write()





def prepare_descriptors(inputs, at_descs_arrays_key):
    """ processes configs or input descriptor array to produce descriptors
    colun array

    Parameters
    ----------
    inputs: ConfigSet_in
        input configurations
    at_descs_arrays_key: str, default None
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
    at_descs = []
    parent_idx = []
    for at_idx, at in enumerate(inputs):
        at_descs.append(at.arrays[at_descs_arrays_key])
        parent_idx.append([at_idx] * len(at))

    parent_idx = np.asarray(parent_idx)
    at_descs = np.asarray(at_descs).T

    assert len(parent_idx) == at_descs.shape[1]

    return at_descs, parent_idx














