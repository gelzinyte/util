import os

import numpy as np

from ase.build import molecule

from wfl.configset import ConfigSet_in, ConfigSet_out

from util.configs import cur


def ref_path():
    return os.path.abspath(os.path.dirname(__file__))

def test_atomic_cur():
    # just check it works

    input_files = os.path.join(ref_path(), 'files/mols_soap.xyz')
    inputs = ConfigSet_in(input_files=input_files)
    outputs = ConfigSet_out()

    at_descs_key = 'SOAP-n4-l3-c2.4-g0.3'
    kernel_exp = 1
    keep_descriptor_arrays=False
    leverage_score_key='leverage_scores'
    n_sparse = 10

    cur.per_environment(inputs=inputs,
                        outputs=outputs,
                        num=n_sparse,
                        at_descs_key=at_descs_key,
                        kernel_exp=kernel_exp,
                        keep_descriptor_arrays=keep_descriptor_arrays,
                        leverage_score_key=leverage_score_key
                        )


def test_leverage_scores_arrays():

    mols = [molecule("CH4"), molecule("H2O"), molecule("C6H6")]

    #random dummy leverage_scores
    leverage_scores = (np.arange(6)+1)/10

    #make note of which environment corresponds to which molecule
    parent_at_idx = []
    for idx, at in enumerate(mols):
        parent_at_idx += [idx] * len(at)

    # random environments to be selected
    selected = [2, 3, 6, 13, 14, 15]

    output_scores =  cur.leverage_scores_into_arrays(inputs=mols,
                  leverage_scores=leverage_scores,
                  selected=selected,
                      parent_at_idx=parent_at_idx)


    expected_scores = [
        np.array([0, 0, 0.1, 0.2, 0]),
        np.array([0, 0.3, 0]),
        np.array([0, 0, 0, 0, 0, 0.4, 0.5, 0.6, 0, 0, 0, 0]),

    ]

    for out, exp in zip(output_scores, expected_scores):
        assert np.all(out == exp)


