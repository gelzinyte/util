import subprocess
import util
import shutil
import os
from ase.io import read, write
from ase.io.orca import read_geom_orcainp


def orca_par_opt(at_list, no_cores, at_or_traj='at'):

    orca_header  =  '! UKS B3LYP def2-SV(P) def2/J D3BJ Opt\n' \
               '%scf Convergence VeryTight\n' \
               'SmearTemp 2000\n' \
               'maxiter 500\n' \
               'end\n' \

    orca_exec = '/home/eg475/programs/orca/orca_4_2_1_linux_x86-64_openmpi314/orca'

    orca_wdir = 'orca_optimisations'
    if not os.path.isdir(orca_wdir):
        os.makedirs(orca_wdir)
    home_dir = os.getcwd()
    os.chdir(orca_wdir)

    unoptimised_at_list = []
    for at_idx, at in enumerate(at_list):
        orca_output = f'orca{at_idx}.out'
        if os.path.exists(orca_output):
            print(f'found {orca_output}, not re-optimising')
            #check if input file matches current atoms
            input_atoms = read_geom_orcainp(f'orca{at_idx}.inp')
            if (at.positions != input_atoms.positions).all():
                raise RuntimeError(f'found output for structure {at_idx}, but the input and atoms positions do not match')
            #otherwise positions match and we can read out without optimisation later on.
        else:
            unoptimised_at_list.append((at_idx, at))

    for unopt_at_group in util.grouper(unoptimised_at_list, no_cores):
        unopt_at_group = [group for group in unopt_at_group if group is not None]

        glob_at_idx_group = [group[0] for group in unopt_at_group]
        at_group = [group[1] for group in unopt_at_group]

        sub = ''
        for idx, atoms in zip(glob_at_idx_group, at_group):
            if len(atoms) != 1:

                mult = get_multiplicity(atoms)
                orca_input = f'orca{idx}.inp'
                orca_output = f'orca{idx}.out'

                with open(orca_input, 'w') as f:
                    f.write(orca_header)
                    f.write(f'*xyz 0 {mult}\n')
                    for at in atoms:
                        f.write(
                            f'{at.symbol} {at.position[0]} {at.position[1]} '
                            f'{at.position[2]}\n')
                    f.write('*\n')

                sub += f'{orca_exec} {orca_input} > {orca_output} &\n'

            sub += 'wait\n'

        print('submitting to optimise')
        subprocess.run(sub, shell=True)


    opt_ats = []
    if at_or_traj == 'at':
        for idx, at in enumerate(at_list):
            if len(at)!=1:
                opt_at = read(f'orca{idx}.xyz')
                opt_at.info.clear()
                pos = opt_at.arrays['positions']
                num = opt_at.arrays['numbers']
                opt_at.arrays.clear()
                opt_at.arrays['positions'] = pos
                opt_at.arrays['numbers'] = num
            else:
                opt_at = at
            opt_ats.append(opt_at)

    elif at_or_traj == 'traj':
        for idx, orig_at in enumerate(at_list):
            if len(orig_at)!=1:
                opt_traj = read(f'orca{idx}_trj.xyz', ':')
                traj = []
                for at in opt_traj:
                    at.info.clear()
                    pos = at.arrays['positions']
                    num = at.arrays['numbers']
                    at.arrays.clear()
                    at.arrays['positions'] = pos
                    at.arrays['numbers'] = num
                    traj.append(at)
                opt_ats.append(traj)
            else:
                opt_at = [orig_at]
                opt_ats.append(opt_at)

    else:
        raise ValueError(f"Should have either 'at' or 'traj', not {at_or_traj}")

    os.chdir(home_dir)

    return opt_ats



def get_multiplicity(atoms):
    # TODO very primitive, REDO
    symbols = list(atoms.symbols)
    no_H = symbols.count('H')
    if no_H % 2 == 1:
        return 2
    else:
        return 1

def add_my_decorations(nm_data, at_info=None, del_ens_fs=True):
    '''prepares data to go into gap_fit'''

    for at in nm_data:
        at.cell = [40, 40, 40]
        at.info['dft_energy'] = at.info['energy']
        at.arrays['dft_forces'] = at.arrays['forces']
        if at_info:
            for key, value in at_info.items():
                at.info[key] = value

        if del_ens_fs:
            try:
                del at.info['energy']
            except:
                print('Could not delete "energy"')
            try:
                del at.arrays['forces']
            except:
                print('Could not delete "forces"')

    return nm_data



def get_dft_energies(in_atoms, keep_files=False):
    '''get dft energies via wfl'''

    smearing = 2000
    n_wfn_hop = 1
    task = 'gradient'
    orcasimpleinput = 'UKS B3LYP def2-SV(P) def2/J D3BJ'

    # scratch_dir = '/scratch/eg475'
    scratch_dir = '/tmp/eg475'

    n_run_glob_op = 1
    kw_orca = f'smearing={smearing}'

    tmp_prefix='tmp_orca_'
    tmp_suffix='.xyz'
    tmp_dir = '.'

    # os_handle, tmp_fname = tempfile.mkstemp(suffix=tmp_suffix, prefix=tmp_prefix, dir=tmp_dir)

    tmp_fname = os.path.join(tmp_dir, tmp_prefix + util.rnd_string(8) + tmp_suffix)

    wfl_command = f'wfl -v ref-method orca-eval --output-file {tmp_fname} ' \
                  f'-tmp ' \
                  f'{scratch_dir} --keep-files {keep_files} --base-rundir ' \
                  f'orca_outputs_{util.rnd_string(8)} ' \
                  f'-nr {n_run_glob_op} -nh {n_wfn_hop} --kw "{kw_orca}" ' \
                  f'--orca-simple-input "{orcasimpleinput}"'


    atoms_out = orca_par_data(in_atoms, tmp_fname, wfl_command)

    if keep_files:
        print(f'keeping {os.path.basename(tmp_fname)}')
        if not os.path.isfile(os.path.basename(tmp_fname)):
            shutil.move(tmp_fname, '.')
            os.remove(tmp_fname)
    else:
        os.remove(tmp_fname)

    return atoms_out


def orca_par_data(atoms_in, out_fname, wfl_command):
    '''calls workflow command to get orca energies '''

    in_fname = os.path.splitext(out_fname)[0] + '_to_eval.xyz'
    write(in_fname, atoms_in, 'extxyz', write_results=False)

    wfl_command += f' {in_fname}'
    print(f'Workflow command:\n{wfl_command}')

    stdout, stderr = util.shell_stdouterr(wfl_command)
    print(f'---wfl stdout\n{stdout}')
    print(f'---wfl stderr\n{stderr}')

    atoms_out = read(out_fname, ':')
    os.remove(in_fname)

    return atoms_out
