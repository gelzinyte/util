import subprocess
import pytest
from ase.io import read, write
import numpy as np
import os

def ref_path():
    return os.path.abspath(os.path.dirname(__file__))

@pytest.fixture
def at_in():
    return read('files/cu_quip_test.xyz')


def calc_desc(at_in, q_list, quip_version='modified', cleanup=True):
    temp_fname = 'tmp.xyz'
    prep_at(at_in, q_list, temp_fname)
    desc = do_desc(quip_version)

    if cleanup:
        os.remove(temp_fname)
        os.remove('desc.out')
    return desc


def do_desc(quip_version='modified'):
    if quip_version == 'modified':
        quip = '/home/eg475/dev/dev_QUIP/build/linux_x86_64_gfortran_openmp' \
               '/quip'
    elif quip_version == 'original':
        quip = '/opt/womble/QUIP/2021_07_06//bin/quip'

    command = f" {quip} atoms_filename=tmp.xyz " \
              f'descriptor_str={{soap l_max=6 n_max=12 cutoff=3 delta=1 ' \
              f'covariance_type=dot_product zeta=4 ' \
              f'n_sparse=100 sparse_method=cur_points ' \
              f'atom_gaussian_width=0.3 cutoff_transition_width=0.5  ' \
              f'add_species=True}} > desc.out'

    print(command)

    subprocess.run(command, shell=True)

    return parse_output('desc.out')


def parse_output(output_fname):
    with open(output_fname, 'r') as f:
        lines = f.readlines()

    descs_out = []
    for line in lines:
        if 'DESC' in line:
            line = line.split()
            descs_out.append([np.float64(l) for l in line[1:]])
    return np.array(descs_out)


def prep_at(at, q_list, temp_fname):
    at.arrays["at_gaussian_weight"] = q_list
    write(temp_fname, at)

def test_modified_quip_same_as_not(at_in):
    original_quip_desc = calc_desc(at_in, np.array([1.0, 1.0, 1.0, 1.0]), quip_version='original')
    modified_quip_desc = calc_desc(at_in, np.array([1.0, 1.0, 1.0, 1.0]), quip_version='modified')
    assert np.all(original_quip_desc == modified_quip_desc)


def test_downweighting_all_gaussians_doesnt_do_anything(at_in):
    modified_quip_desc_ones = calc_desc(at_in, np.array([1.0, 1.0, 1.0, 1.0]),
                                        quip_version='original')
    modified_quip_desc_halves = calc_desc(at_in,
                                          np.array([1.0, 1.0, 1.0, 1.0])*0.5,
                               quip_version='modified', cleanup=False)
    assert np.all(modified_quip_desc_ones == modified_quip_desc_halves)

def test_downweigting_single_gaussian_changes_descriptor(at_in):
    modified_quip_desc_ones = calc_desc(at_in, np.array([1.0, 1.0, 1.0, 1.0]),
                                        quip_version='original')
    modified_quip_desc_halves = calc_desc(at_in, np.array([1.0, 1.0, 1.0, 0.5]),
                                          quip_version='modified')
    assert not np.all(modified_quip_desc_ones == modified_quip_desc_halves)








