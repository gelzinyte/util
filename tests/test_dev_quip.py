import subprocess
import pytest
from ase.io import read, write
import numpy as np
import os
from quippy.descriptors import Descriptor
from quippy.potential import Potential

def ref_path():
    return os.path.abspath(os.path.dirname(__file__))

@pytest.fixture
def at_in():
    return read('files/cu_quip_test.xyz')

# @pytest.mark.xfail()
def calc_desc(at_in, q_list, quip_version='modified', cleanup=True):
    temp_fname = 'tmp.xyz'
    prep_at(at_in, q_list, temp_fname)
    desc = do_desc(quip_version)

    if cleanup:
        os.remove(temp_fname)
        os.remove('desc.out')
    return desc


# @pytest.mark.xfail()
def do_desc(quip_version='modified'):
    if quip_version == 'modified':
        quip = '/home/eg475/dev/dev_QUIP/build/linux_x86_64_gfortran_openmp' \
               '/quip'
        # quip = '/home/eg475/dev/dev_QUIP/build/linux_x86_64_gfortran' \
        #            '/quip'

        command = f" {quip} atoms_filename=tmp.xyz " \
                  f'descriptor_str={{soap l_max=6 n_max=12 cutoff=3 delta=1 ' \
                  f'' \
                  f'covariance_type=dot_product zeta=4 ' \
                  f'n_sparse=100 sparse_method=cur_points ' \
                  f'atom_gaussian_width=0.3 cutoff_transition_width=0.5  ' \
                  f'add_species=True }} ' \
                  f'calc_args={{atom_gaussian_weight_name' \
                  f'=atom_gaussian_weight}} > ' \
                  f'desc.out'

    elif quip_version == 'original':
        quip = '/opt/womble/QUIP/2021_07_06//bin/quip'

        command = f" {quip} atoms_filename=tmp.xyz " \
                  f'descriptor_str={{soap l_max=6 n_max=12 cutoff=3 delta=1 ' \
                  f'covariance_type=dot_product zeta=4 ' \
                  f'n_sparse=100 sparse_method=cur_points ' \
                  f'atom_gaussian_width=0.3 cutoff_transition_width=0.5  ' \
                  f'add_species=True }} > desc.out'

    assert os.path.isfile(quip)


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
    at.arrays["atom_gaussian_weight"] = q_list
    write(temp_fname, at)


# @pytest.mark.xfail()
def test_modified_quip_same_as_not(at_in):
    original_quip_desc = calc_desc(at_in, np.array([1.0, 1.0, 1.0, 1.0]), quip_version='original')
    modified_quip_desc = calc_desc(at_in, np.array([1.0, 1.0, 1.0, 1.0]), quip_version='modified')
    assert np.all(original_quip_desc == modified_quip_desc)


# @pytest.mark.xfail()
def test_downweighting_all_gaussians_doesnt_do_anything(at_in):
    modified_quip_desc_ones = calc_desc(at_in, np.array([1.0, 1.0, 1.0, 1.0]),
                                        quip_version='original')
    modified_quip_desc_halves = calc_desc(at_in,
                                          np.array([1.0, 1.0, 1.0, 1.0])*0.5,
                               quip_version='modified', cleanup=False)
    assert np.all(modified_quip_desc_ones == modified_quip_desc_halves)


# @pytest.mark.xfail()
def test_downweigting_single_gaussian_changes_descriptor(at_in):
    modified_quip_desc_ones = calc_desc(at_in, np.array([1.0, 1.0, 1.0, 1.0]),
                                        quip_version='original')
    modified_quip_desc_halves = calc_desc(at_in, np.array([1.0, 1.0, 1.0, 0.5]),
                                          quip_version='modified')
    assert not np.all(modified_quip_desc_ones == modified_quip_desc_halves)


# @pytest.mark.xfail()
def test_calculate_descriptor(at_in):

    modified_quip_desc_halves = calc_desc(at_in,
                                          np.array([1.0, 1.0, 1.0, 0.5]),
                                          quip_version='modified')

    at_in.arrays["atom_gaussian_weight"] = np.array([1.0, 1.0, 1.0, 0.5])
    d = Descriptor(args_str="soap l_max=6 n_max=12 cutoff=3 delta=1 covariance_type=dot_product zeta=4 "\
                   "n_sparse=100 sparse_method=cur_points atom_gaussian_width=0.3 cutoff_transition_width=0.5  "\
                   "add_species=True", ase_to_quip_kwargs={'add_arrays':"atom_gaussian_weight"})
    my_desc = d.calc(at_in,
                     args_str="atom_gaussian_weight_name=atom_gaussian_weight")

    assert pytest.approx(my_desc['data']) == modified_quip_desc_halves


# @pytest.mark.xfail()
def test_quip_energies_forces():

    ats_in_fname = os.path.join(ref_path(),
                                'files/atoms_for_modified_gap.xyz')
    gap_fname = os.path.join(ref_path(), 'files/modified_gap.xml')
    quip_command = f'/home/eg475/dev/dev_QUIP/build' \
                   f'/linux_x86_64_gfortran_openmp/quip ' \
                   f'atoms_filename={ats_in_fname} param_filename={gap_fname} ' \
                   f'calc_args={{atom_gaussian_weight_name=at_gaussian_weight}} E F'


    result = subprocess.run(quip_command, shell=True, capture_output=True,
                            text=True)
    assert "got atom gaussian weight" in result.stdout


# @pytest.mark.xfail()
def test_quippy_energies_forces():

    gap_fname = os.path.join(ref_path(), 'files/modified_gap.xml')
    gap = Potential(param_filename=gap_fname, add_arrays="at_gaussian_weight",
                    calc_args="atom_gaussian_weight_name=at_gaussian_weight")

    at = read(os.path.join(ref_path(),
                                'files/atoms_for_modified_gap.xyz'))

    quip_energy = at.info['gap_energy']
    at.calc = gap
    quippy_energy = at.get_potential_energy()
    assert pytest.approx(quippy_energy) == quip_energy
