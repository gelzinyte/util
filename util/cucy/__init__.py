import numpy as np
import os
from ase.io import read, write
import click
from wfl.calculators import orca
from ase.constraints import FixedPlane, FixInternals

# @click.command()
# @click.option('--opt_xyzs_dir', default='/data/eg475/carbyne/optimised_structures')
# @click.option('--outputs_dir',
#               default='/data/eg475/carbyne/calculate/aligned_structures')
# @click.option('--output_fname', default='opt_aligned_with_energies.xyz')
def assign_energies(opt_xyzs_dir, outputs_dir, output_fname):

    opt_methods = ['triplet_opt', 'singlet_opt']
    structures = ['HC30H', 'HC31H', 'HC32H', 'HC33H', 'H2C30H2_flat',
                'H2C30H2_angle', 'H2C31H2_flat', 'H2C31H2_angle']

    eval_methods = ['dlpno-ccsd_cc-pvdz', 'uks_cc-pvdz']
    eval_multiplicities = ['1-let', '3-let']
    eval_mult_names = ['singlet', 'triplet']

    atoms_out = []

    for structure in structures:
        for opt_method in opt_methods:

            opt_fname = f'{structure}.{opt_method}.uks_def2-svp_opt.xyz'
            at = read(os.path.join(opt_xyzs_dir, opt_fname))

            at.info['opt_method'] = opt_method
            at.info['structure'] = structure
            at.info['ful_opt_fname'] = opt_fname

            for eval_method in eval_methods:
                for eval_mult, eval_mult_name in zip(eval_multiplicities,
                                                     eval_mult_names):

                    eval_dir_name = f'{structure}.' \
                                    f'{opt_method}.uks_def2-svp_opt'

                    orca_output_label = eval_dir_name \
                                       + f'.{eval_method}.{eval_mult_name}'

                    orca_output_label = os.path.join(outputs_dir,
                                                    eval_dir_name,
                                                    orca_output_label,
                                                     orca_output_label)

                    energy = read_energy(orca_output_label)

                    at.info[f'{eval_method}.{eval_mult_name}_energy'] = \
                        energy

            atoms_out.append(at)
    write(output_fname, atoms_out)


def read_energy(out_label):
    calc = orca.ORCA()
    calc.label = out_label
    calc.read_energy()
    return calc.results['energy']


def rotate_comulene(inputs, outputs, angles=None):
    if outputs.all_written():
        return outputs.to_ConfigSet()
    if angles is None:
        angles = [0, 30, 45, 60, 90]
    for at in inputs:
        ats_out = []
        ea = get_at_nos_for_dihedral(at)
        # print(ea)
        # end atoms 
        for angle in angles:
            aa = at.copy()
            aa.set_dihedral(ea["c2"], ea["h22"], ea["c1"], ea["h12"], angle, indices=[ea["c1"], ea["h12"], ea["h11"]])
            aa.info["dihedral_angle"] = angle
            ats_out.append(aa)
        outputs.store(ats_out)
    outputs.close()
    return outputs.to_ConfigSet()

def set_cu_constraint_plane(inputs, outputs):
    for group in inputs.groups():
        out = []
        for at in group:
            ea = get_at_nos_for_dihedral(at)
            # set constraints
            ch1 = at.get_distance(ea["c1"], ea["h11"], vector=True)
            ch2 = at.get_distance(ea["c1"], ea["h12"], vector=True)
            direction = np.cross(ch1, ch2)
            direction = direction / np.linalg.norm(direction)
            end1 = FixedPlane([ea["h11"], ea["h12"]], direction = direction)

            ch1 = at.get_distance(ea["c2"], ea["h21"], vector=True)
            ch2 = at.get_distance(ea["c2"], ea["h22"], vector=True)
            direction = np.cross(ch1, ch2)
            direction = direction / np.linalg.norm(direction)
            end2 = FixedPlane([ea["h21"], ea["h22"]], direction = direction)

            at.set_constraint([end1, end2])
            out.append(at)
        outputs.store(out)
    outputs.close() 
    return outputs.to_ConfigSet()

def set_cu_constraint_dihedrals(inputs, outputs):
    for group in inputs.groups():
        out = []
        for at in group:
            ea = get_at_nos_for_dihedral(at)
            dih_constraints = []
            for first_h in ["h11", "h12"]:
                for second_h in ["h21", "h22"]:
                    dihedral_indices = ea["c1"], ea[first_h], ea["c2"], ea[second_h]
                    constraint = [at.get_dihedral(*dihedral_indices), dihedral_indices]
                    dih_constraints.append(constraint)
            overal_constraint = FixInternals(dihedrals_deg = dih_constraints, epsilon=1e-1)
            at.set_constraint(overal_constraint)
            out.append(at)
        outputs.store(out)
    outputs.close()
    return outputs.to_ConfigSet()

 

def get_at_nos_for_dihedral(at):
    if len(at) == 6:
        return {"c1":1, "h11": 4, "h12": 5, "c2":0, "h21":2, "h22":3} 
    c1 = 0
    h11 = 1
    h12 = 2
    max_pos = 0
    c2 = None 
    h21 = None
    h22 = None
    for  aa in at:
        if aa.symbol == "H":
            if aa.index == h11 or aa.index == h12:
                continue
            if h21 is None:
                h21 = aa.index
            elif h22 is None:
                h22 = aa.index
            continue
        if aa.position[0] > max_pos:
            max_pos = aa.position[0]
            c2 = aa.index
    return {"c1":c1, "h11": h11, "h12": h12, "c2":int(c2), "h21":int(h21), "h22":int(h22)}
    
        


