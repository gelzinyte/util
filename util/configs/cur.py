from wfl.select_configs import by_descriptor

def cur_per_environment(inputs, outputs, num, at_descs=None,
                        at_descs_arrays_key=None, kernel_exp=None,
                        stochastic=True, stochastic_seed=None,
                        keep_descriptor_info=True, center=True,
                        leverage_score_key=None,
                        mark_slected_environment=True):
    """Select atoms from a list or iterable using CUR on  descriptors
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
    at_descs: np.array(n_descs, desc_len), mutually exclusive with at_descs_info_key
        list of descriptor vectors
    at_descs_info_key: str, mutually exclusive with at_descs
        key to Atoms.info dict containing per-config descriptor vector
    kernel_exp: float, default None
        exponent to compute kernel (if other than 1)
    stochastic: bool, default True
        use stochastic selection
    keep_descriptor_info: bool, default True
        do not delete descriptor from info
    exclude_list: iterable(Atoms)
        list of Atoms to exclude from CUR selection.  Needs to be _exactly_ the same as
        actual Atoms objects in inputs, to full machine precision
    center: bool, default True
        center data before doing SVD, as generally required for PCA
    leverage_score_key: str, default None
        if not None, info key to store leverage score in
    Returns
    -------
        ConfigSet_in corresponding to selected configs output
    """