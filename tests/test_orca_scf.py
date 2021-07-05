import os
from util import qm

def ref_path():
    return os.path.abspath(os.path.dirname(__file__))


def test_plot_scf_convergence(tmp_path):

    output_fname = os.path.join(tmp_path, 'output_fig.png')

    input = os.path.join(ref_path(), 'files/orca.out')

    qm.orca_scf_plot(input_fname=input,
                     method='dft',
                     fname=output_fname)


def test_plot_cc_convergence(tmp_path):

    output_fname = os.path.join(tmp_path, 'output_fig.png')

    input = os.path.join(ref_path(), 'files/orca_cc.out')

    output_fname='test.png'

    qm.orca_scf_plot(input_fname=input,
                     method='cc',
                     fname=output_fname)


