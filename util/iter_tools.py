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
from util.config import Config
from wfl.calculators import orca
import os

from util import error_table as et
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def make_structures(smiles_csv, iter_no, num_smi_repeat, opt_starts_fname):

    # generate molecules
    df = pd.read_csv(smiles_csv)
    smiles_to_convert = []
    smi_names = []
    for smi, name in zip(df['SMILES'], df['Name']):
        smiles_to_convert += [smi] * num_smi_repeat
        smi_names += [name] * num_smi_repeat


    molecules = ConfigSet_out()
    smiles.run(outputs=molecules, smiles=smiles_to_convert, extra_info={'iter_no':iter_no})
    for at, name in zip(molecules.output_configs, smi_names):
        at.info['config_type'] = name
        at.cell = [40, 40, 40]

    # generate radicals
    mols_rads = ConfigSet_out(output_files=opt_starts_fname)
    radicals.multi_mol_remove_h(molecules.output_configs, output=mols_rads)



    # # determine optimal chunksize
    # no_structures = len(read(opt_starts_fname, ':'))
    # no_cores = int(os.environ['AUTOPARA_NPOOL'])
    # chunksize = int(no_structures/no_cores) + 1
    #
    #
    # # optimise
    # opt_trajectory = ConfigSet_out(output_files=opt_traj_fname)
    # run_opt(inputs=ConfigSet_in(input_files=opt_starts_fname),
    #                 outputs=opt_trajectory, chunksize=chunksize,
    #           calculator=calculator, return_traj=True, logfile=logfile)
    #
    # opt_trajectory = read(opt_traj_fname, ':')
    #
    # opt_ats = [at for at in opt_trajectory if 'minim_config_type' in
    #            at.info.keys() and 'converged' in at.info['minim_config_type']]
    # write(opt_fname, opt_ats)
    #
    # return ConfigSet_in(input_configs=opt_ats)



def  optimise(calculator, opt_starts_fname,
                    opt_fname, opt_traj_fname, logfile=None):

    # determine optimal chunksize
    no_structures = len(read(opt_starts_fname, ':'))
    no_cores = int(os.environ['AUTOPARA_NPOOL'])
    chunksize = int(no_structures/no_cores) + 1

    # optimise
    opt_trajectory = ConfigSet_out(output_files=opt_traj_fname)
    inputs = ConfigSet_in(input_files=opt_starts_fname)

    run_opt(inputs=inputs,
               outputs=opt_trajectory, chunksize=chunksize,
              calculator=calculator, return_traj=True, logfile=logfile)

    opt_trajectory = read(opt_traj_fname, ':')

    opt_ats = [at for at in opt_trajectory if 'minim_config_type' in
               at.info.keys() and 'converged' in at.info['minim_config_type']]
    
    write(opt_fname, opt_ats)


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

    return iterable_loop(iterable=inputs, configset_out=outputs, op=opt,
                         chunksize=chunksize, calculator=calculator, fmax=fmax,
                         steps=steps, precon=precon,
                         use_armijo=use_armijo, return_traj=return_traj, logfile=logfile)

def opt(atoms, calculator, fmax=1.0e-3, steps=1000, precon='auto',
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


def reeval_dft(gap_dirs):

    if isinstance(gap_dirs, str):
        gap_dirs = gap_dirs.split(' ')

    cfg = Config.load()
    default_kw = Config.from_yaml(
        os.path.join(cfg['util_root'], 'default_kwargs.yml'))

    output_prefix = 'dft_'
    orca_kwargs = default_kw['orca']
    orca_kwargs['smearing'] = 2000

    print(f'orca_kwargs: {orca_kwargs}')

    for gap_dir in gap_dirs:

        print(f'---reevaluating {gap_dir}')

        dft_fnames = os.listdir(gap_dir)
        dft_fnames = [os.path.join(gap_dir, fname) for fname in dft_fnames if 'opt_trajectories' not in fname]

        print(dft_fnames)

        file_mapping = {}
        for fname in dft_fnames:
            if 'opt_trajectories' in fname:
                continue
            file_mapping[fname] = fname

        print(file_mapping)

        inputs = ConfigSet_in(input_files=dft_fnames)
        outputs = ConfigSet_out(output_files=file_mapping, force=True)

        orca.evaluate(inputs, outputs, base_rundir='orca_bde_reevaluations',
                      orca_kwargs=orca_kwargs, output_prefix=output_prefix)


def overall_errors(all_errs, title):
    e_errs = []
    f_errs = []
    counts = []

    e_col = 'E RMSE, meV/at'
    f_col = 'F RMSE, meV/Å'

    for df in all_errs:
        e_errs.append(df[e_col]['overall'])
        f_errs.append(df[f_col]['overall'])
        counts.append(int(df['Count']['overall']))

    if np.std(counts) == 0:
        xs = np.arange(len(all_errs))
        int_label = True
        xlabel = 'iteration'
    else:
        xs = counts
        int_label = False
        xlabel = 'Training set size'

    fig = plt.figure()
    ax1 = fig.gca()

    ax1.plot(xs, e_errs, color='tab:red', marker='o', label=e_col)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()

    ax2.plot(xs, f_errs, color='tab:blue', marker='x', label=f_col)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    # ax1.legend(bbox_to_anchor=(0, 0, 1, 0.8))
    # ax2.legend(bbox_to_anchor=(0, 0, 1, 0.8))
    ax1.set_ylabel(e_col, color='tab:red')
    ax1.set_xlabel(xlabel)
    if int_label:
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax2.set_ylabel(f_col, color='tab:blue')
    plt.title(title)
    plt.show()

def rmses_by_mol(all_errs, label_addition=''):

    roots = ['butane', 'ethanol', 'trans_phenylpropene', 'limonene',
             'alpha_thujone', 'safrole', 'ethoxy_coumarin',
             'myristicin', 'butoxy_coumarin']
    plot_dict = {}
    for rt in roots:
        plot_dict[rt] = {}

    e_col = 'E RMSE, meV/at'
    f_col = 'F RMSE, meV/Å'

    for df in all_errs:
        for label, row in df.iterrows():
            if label == 'overall':
                continue

            for rt in roots:
                if rt in label:
                    compound = rt
                    break
            else:
                raise RuntimeError(f'missing {label} among roots')

            if label not in plot_dict[compound].keys():
                plot_dict[compound][label] = {}
                plot_dict[compound][label]['energy'] = []
                plot_dict[compound][label]['forces'] = []

            plot_dict[compound][label]['energy'].append(row[e_col])
            plot_dict[compound][label]['forces'].append(row[f_col])

    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.arange(10)]

    for idx, (compound, plot_vals) in enumerate(plot_dict.items()):

        check =  sum([len(line['energy']) for line in plot_vals.values()])
        if check == 0:
            continue

        color = colors[idx % 10]

        plt.figure(figsize=(10, 4))

        gs = mpl.gridspec.GridSpec(1, 2)
        ax_e = plt.subplot(gs[0])
        ax_f = plt.subplot(gs[1])

        for line_label, line in plot_vals.items():

            ys_e = line['energy']
            ys_f = line['forces']

            if len(ys_e) == len(all_errs) - 1:
                xs = np.arange(1, len(all_errs))
            elif len(ys_e) == len(all_errs):
                xs = np.arange(len(all_errs))

            else:
                print(f'skipping {line_label}')
                continue

            ax_e.plot(xs, ys_e, label=line_label)
            ax_f.plot(xs, ys_f, label=line_label)

        for ax in [ax_e, ax_f]:
            ax.set_xlabel('iteration')
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.set_ylim(bottom=0)

        plt.suptitle(compound + ' ' + label_addition, color=color,
                     size=16)
        #     ax_f.set_title('Force RMSE', color=color)

        ax_e.set_ylabel(e_col)
        ax_f.set_ylabel(f_col)
        ax_f.legend(bbox_to_anchor=[1, 1], labelcolor=color)
        plt.tight_layout()



