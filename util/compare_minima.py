from util import iter_tools as it
from wfl.generate_configs import vib
from quippy.potential import Potential
import os
from wfl.configset import ConfigSet_in, ConfigSet_out
from ase.io import read, write


def opt_and_normal_modes(dft_dir, gap_fnames, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if isinstance(gap_fnames, str):
        gap_fnames = gap_fnames.split()

    input_fnames = [os.path.join(dft_dir, fname) for fname in
                    os.listdir(dft_dir)]

    for gap_fname in gap_fnames:

        basename = os.path.splitext(os.path.basename(gap_fname))[0]
        opt_output_fname = basename + 'dft_eq_reopt.xyz'
        opt_output_fname = os.path.join(output_dir, opt_output_fname)

        traj_fname = basename + 'dft_eq_reopt_traj.xyz'
        traj_fname = os.path.join(output_dir, traj_fname)

        nm_output_fname = basename + 'normal_modes_from_dft_eq.xyz'
        nm_output_fname = os.path.join(output_dir, nm_output_fname)

        if not os.path.isfile(opt_output_fname):
            print(f'optimising with {gap_fname}')
            optimise(gap_fname, input_fnames, traj_fname,
                                 opt_output_fname)

        if not os.path.isfile(nm_output_fname):
            print(f'normal-moding with {gap_fname}')
            derive_normal_modes_parallel_atoms(opt_output_fname, gap_fname, nm_output_fname)


def optimise(gap_fname, input_fnames, traj_fname, output_fname):
    calculator = (Potential, [], {'param_filename': gap_fname})

    inputs = ConfigSet_in(input_files=input_fnames)
    outputs = ConfigSet_out(output_files=traj_fname)

    it.run_opt(inputs=inputs,
               outputs=outputs,
               calculator=calculator, return_traj=True, logfile=None)

    opt_traj = read(traj_fname, ':')

    opt_ats = [at for at in opt_traj if 'minim_config_type' in
               at.info.keys() and 'converged' in at.info['minim_config_type']]

    write(output_fname, opt_ats)


def derive_normal_modes_parallel_atoms(opt_fname, gap_fname, output_fname):
    calculator = (Potential, [], {'param_filename': gap_fname})

    print(output_fname)

    inputs = ConfigSet_in(input_files=opt_fname)
    outputs = ConfigSet_out(output_files=output_fname)
    vib.generate_normal_modes_parallel_atoms(inputs=inputs,
                         outputs=outputs,
                         calculator=calculator,
                         prop_prefix='gap_')
