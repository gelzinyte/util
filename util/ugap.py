
from copy import deepcopy
from ase.io.extxyz import key_val_str_to_dict
from ase.io.extxyz import key_val_dict_to_str
from quippy.potential import Potential
from ase.build import molecule
import numpy as np
import xml.etree.ElementTree as et
import re
from ase.io import read, write
import subprocess
import util
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from util import plot



def make_descr_str(descr_dict):
    """ given dictionary of {"name":"desc_name", "param":"value", ...} converts to string "desc_name param=value ...", suitable for GAP"""
    descr_dict = deepcopy(descr_dict)
    string = f"{descr_dict['name']} "
    del descr_dict['name']
    string += key_val_dict_to_str(descr_dict)
    return string


def make_config_type_sigma_str(dct):
    """Converts config_type sigma values to string to be included in GAP command"""
    '''dct should be {config_type,str:values,float list}'''
    string = ' config_type_kernel_regularisation={'

    for i, key in enumerate(dct.keys()):
        string += key
        for value in dct[key]:
            string += f':{value}'
        if i != len(dct) - 1:
            string += ':'

    string += '}'
    return string


def make_gap_command(gap_filename, training_filename, descriptors_dict, default_sigma, config_type_sigma=None, \
                     gap_fit_path=None, output_filename=False, glue_fname=None):
    """Makes GAP command to be called in shell"""
    descriptor_strs = [make_descr_str(descriptors_dict[key]) for key in descriptors_dict.keys()]

    default_sigma = f'{{{default_sigma[0]} {default_sigma[1]} {default_sigma[2]} {default_sigma[3]}}}'

    if not gap_fit_path:
        gap_fit_path = 'gap_fit'

    if glue_fname:
        glue_command = f' core_param_file={glue_fname} core_ip_args={{IP Glue}}'
    else:
        glue_command = ''

    if not config_type_sigma:
        config_type_sigma = ''
    else:
        if type(config_type_sigma) != dict:
            raise TypeError('config_type_sigma should be a dictionary of config_type(str):values(list of floats)')
        config_type_sigma = make_config_type_sigma_str(config_type_sigma)

    gap_command = f'{gap_fit_path} gp_file={gap_filename} atoms_filename={training_filename} energy_parameter_name=dft_energy force_parameter_name=dft_forces sparse_separate_file=F default_sigma='
    gap_command += default_sigma + config_type_sigma + glue_command + ' gap={'

    for i, desc in enumerate(descriptor_strs):
        gap_command += desc
        if i != len(descriptor_strs) - 1:
            gap_command += ': '
        else:
            gap_command += '}'

    if output_filename:
        gap_command += f' > {output_filename}'

    return gap_command


def gap_gradient_test(gap_fname, start=-3, stop=-7):
    """Gradient test on methane molecule for given GAP"""
    methane = molecule('CH4')
    rattle_std = 0.01
    seed = np.mod(int(time.time() * 1000), 2 ** 32)
    methane.rattle(stdev=rattle_std, seed=seed)

    calc = Potential(param_filename=gap_fname)

    gradient_test(methane, calc, start, stop)



def get_gap_2b_dict(param_fname):
    '''dictionary to get descriptor no corresponding for element pairs'''
    desc_dict = {}
    elem_symb_dict = {'H':1, 'C':6, 'N':7, 'O':8, 1:'H', 6:'C', 7:'N', 8:'O'}
    root = et.parse(param_fname).getroot()
    for i, descriptor in enumerate(root.iter('descriptor')):
        descriptor = descriptor.text
        if 'distance_2b' in descriptor:
            Z1 = re.search(r'Z1=\d', descriptor).group()
            Z1 = int(Z1.split('=')[1])
            Z2 = re.search(r'Z2=\d', descriptor).group()
            Z2 = int(Z2.split('=')[1])
            entry = ''.join(util.natural_sort([elem_symb_dict[Z1], elem_symb_dict[Z2]]))
            desc_dict[entry] = i+1
    return desc_dict


def get_soap_params(param_fname):
    root = et.parse(param_fname).getroot()
    soaps = []
    for desc in root.iter('descriptor'):
        soap = key_val_str_to_dict(desc.text)
#         print(soap)
        if 'soap' not in soap.keys():
            continue
        del soap['Z']
        soap = key_val_dict_to_str(soap)
        soaps.append(soap)

    soaps = list(set(soaps))
    soaps = [key_val_str_to_dict(soap) for soap in soaps]

    if len(soaps)>1:
        print("WARNING: different soaps encoutnered, taking the first one")
    return soaps[0]


def atoms_from_gap(param_fname, tmp_atoms_fname='tmp_atoms.xyz'):
    root = et.parse(param_fname).getroot()
    xyz_string = root.find('GAP_params').find('XYZ_data').text[5:]
    with open(tmp_atoms_fname, 'w') as f:
        f.write(xyz_string)
    return read(tmp_atoms_fname, ':')






