import pandas as pd
import os
import sys
from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.generate_configs import smiles, radicals
from ase.optimize.precon import PreconLBFGS
from wfl.pipeline import iterable_loop
from wfl.utils.parallel import construct_calculator_picklesafe
from wfl.utils.at_copy_spc import at_copy_SPC
from wfl.utils.misc import atoms_to_list
from ase.io import write, read


def make_structures(smiles_csv, iter_no, calculator, no_opt, non_opt_fname,
                    opt_fname, opt_traj_fname, logfile=None):

    if logfile is None:
        logfile = f'opt_log{iter_no}'

    # generate molecules
    df = pd.read_csv(smiles_csv)
    smiles_to_convert = []
    smi_names = []
    for smi, name in zip(df['SMILES'], df['Name']):
        smiles_to_convert += [smi] * no_opt
        smi_names += [name] * no_opt

    if not os.path.exists(non_opt_fname):

        molecules = ConfigSet_out()
        smiles.run(outputs=molecules, smiles=smiles_to_convert, extra_info={'iter_no':iter_no})
        for at, name in zip(molecules.output_configs, smi_names):
            at.info['config_type'] = name
            at.cell = [40, 40, 40]

        # generate radicals
        mols_rads = ConfigSet_out(output_files=non_opt_fname)
        radicals.multi_mol_remove_h(molecules.output_configs, output=mols_rads)

    else:
        print(f' found {non_opt_fname}, not generating structures for optimisation')


    # optimise
    optimised_mols_rads = ConfigSet_out(output_files=opt_traj_fname)
    run_opt(inputs=ConfigSet_in(input_files=non_opt_fname),
                    outputs=optimised_mols_rads,
              calculator=calculator, return_traj=True, logfile=logfile)

    optimised_mols_rads = read(opt_traj_fname, ':')

    opt_ats = [at for at in optimised_mols_rads if 'minim_config_type' in
               at.info.keys() and 'converged' in at.info['minim_config_type']]
    write(opt_fname, opt_ats)

    return ConfigSet_in(input_configs=opt_ats)

def make_bde_files(bde_start_dir, output_dir, calculator, gap_name='gap'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_fnames = [os.path.join(bde_start_dir, name) for name in os.listdir(bde_start_dir)]
    basenames = [os.path.basename(start_fname) for start_fname in start_fnames]
    gap_fnames = [basename.replace('_optimised.xyz', f'_{gap_name}_optimised.xyz')
                  for basename in basenames]

    in_to_out_fnames = {}
    for start_fname, gap_fname in zip(start_fnames, gap_fnames):
        in_to_out_fnames[start_fname] = gap_fname


    gap_optimised = ConfigSet_out(output_files=in_to_out_fnames, file_root=output_dir)
    inputs = ConfigSet_in(input_files=start_fnames)

    gap_optimised = run_opt(inputs=inputs, outputs=gap_optimised, calculator=calculator,
               return_traj=False, logfile=os.path.join(output_dir, f'opt_log_{gap_name}'))

    with_gap_results = ConfigSet_out(output_files=in_to_out_fnames, file_root=output_dir,
                                     force=True)
    prefix_results('gap', inputs=gap_optimised, outputs=with_gap_results)


def prefix_results(prefix, inputs, outputs):
    
    for at in inputs:
        at.info[f'{prefix}energy'] = at.info['energy']
        at.arrays[f'{prefix}forces'] = at.info['forces']
        outputs.write(at)
    
    outputs.end_write()
    return outputs.to_ConfigSet_in()


def run_opt(inputs, outputs, calculator, fmax=1.0e-3, steps=1000, precon='auto',
            use_armijo=False, chunksize=10, return_traj=False, logfile='-'):

    return iterable_loop(iterable=inputs, configset_out=outputs, op=optimise,
                         chunksize=chunksize, calculator=calculator, fmax=fmax,
                         steps=steps, precon=precon,
                         use_armijo=use_armijo, return_traj=return_traj, logfile=logfile)

def optimise(atoms, calculator, fmax=1.0e-3, steps=1000, precon='auto',
             use_armijo=False, return_traj=False, logfile='-'):

    calculator = construct_calculator_picklesafe(calculator)

    all_trajs = []

    for at in atoms_to_list(atoms):

        at.calc = calculator

        opt = PreconLBFGS(at, precon=precon, master=True, use_armijo=use_armijo, logfile=logfile)

        traj = []

        def process_step():

            if len(traj) > 0 and traj[-1] == at:
                # Some minimization algorithms sometimes seem to repeat, perhaps
                # only in weird circumstances, e.g. bad gradients near breakdown.
                # Do not store those duplicate configs.
                return

            traj.append(at_copy_SPC(at))

        opt.attach(process_step)

        # preliminary value
        final_status = 'unconverged'

        try:
            opt.run(fmax=fmax, steps=steps)
        except Exception as exc:
            # label actual failed minims
            # when this happens, the atomic config somehow ends up with a 6-vector stress,
            # which can't be
            # read by xyz reader.
            # that should probably never happen
            final_status = 'exception'
            # raise

        if len(traj) == 0 or traj[-1] != at:
            traj.append(at_copy_SPC(at))

        # set for first config, to be overwritten if it's also last config
        traj[0].info['minim_config_type'] = 'minim_initial'

        if opt.converged():
            final_status = 'converged'

        traj[-1].info['minim_config_type'] = f'minim_last_{final_status}'
        traj[-1].info['minim_n_steps'] = opt.get_number_of_steps()

        all_trajs.append(traj)

    if not return_traj:
        all_trajs = [traj[-1] for traj in all_trajs]

    return all_trajs



