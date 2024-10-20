import os
import random
import logging
from pathlib import Path
from copy import deepcopy

from os.path import join as pj


from ase.io import read, write
from ase import Atoms

from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators import generic
from wfl.autoparallelize.autoparainfo import AutoparaInfo

from util.util_config import Config
from util import opt
from util import remove_energy_force_containing_entries
from util.bde import table

logger = logging.getLogger(__name__)

cfg = Config.load()
scratch_dir = cfg['scratch_path']


def ip_isolated_h(pred_calculator, dft_calculator, dft_prop_prefix, ip_prop_prefix, outputs,
                   wdir='ip_bde_wdir', no_dft=False, remote_info=None):

    dft_h = Atoms('H', positions=[(0, 0, 0)])
    dft_h.info['bde_config_type'] = 'H'
    # for ACE
    dft_h.cell = [50, 50, 50]

    if ip_prop_prefix=='mace_':
        if outputs.is_done():
            logger.info("isolated h is done")
            return outputs.to_ConfigSet()
        dft_h.info["dft_energy"] = -13.547478676206193
        dft_h.info["mace_energy"] = -13.547478676206193
        outputs.store(dft_h)
        outputs.close()
        return outputs.to_ConfigSet()

    inputs = ConfigSet(dft_h)
    interim_outputs = OutputSpec()

    remote_info = None

    if not no_dft:
        inputs = generic.run(inputs=inputs,
                    outputs=interim_outputs, 
                    calculator=dft_calculator, 
                    properties=["energy"],
                    output_prefix=dft_prop_prefix,
                    autopara_info = AutoparaInfo(
                        num_inputs_per_python_subprocess=1,
                        remote_info=remote_info,
                        remote_label = None)
                    )

    generic.run(inputs=inputs,
                outputs=outputs,
                calculator=pred_calculator, properties=['energy', 'forces'],
                output_prefix=ip_prop_prefix,
                autopara_info = AutoparaInfo(
                    num_python_subprocesses=None,
                    remote_info=remote_info,
                    remote_label = None))

    return outputs.to_ConfigSet() 



def everything(pred_calculator, dft_calculator, dft_bde_filename,
               dft_prop_prefix, ip_prop_prefix, wdir='ip_bde_wdir',
               num_inputs_per_python_subprocess=1, output_dir='.', no_dft=False, remote_info=None):
    """

     1. evaluate dft structures with ip
         - should have dft-optimised mol positions' hashses
           to trace back which radicals belong to which molecule
         - should have bde_config_type = f'{prop_prefix}optimised
         - should have dft-optimised structure hashes
           to trace back which compound was optimised
     2. Duplicate and label structures
         - relabel config_type
         - keep dft-optimised mol positions' hashes
     3. ip-optimise and evaluate with ip
     4. evaluate with dft
     5. add isolated hydrogen atom 
     6. calculate BDEs
     7. Put everything to the same file, so we can plot ip_bdes vs dft_bdes

    So by the end will have {dft_prop_prefix/ip_prop_prefix}optimised geometries and
    {dft_prop_prefix/ip_prop_prefix}energy/forces

    by the end should have
    dft_opt_position_hash


    Parameters
    ----------
    pred_calculator - (calculator, calc_args, calc_kwargs) construct for wfl

    """
    dft_bde_filename = Path(dft_bde_filename)
    # check that expected info labels are there
    random_at = random.choice(read(dft_bde_filename, ':50'))
    assert f'{dft_prop_prefix}opt_mol_positions_hash' in random_at.info.keys()
    assert 'bde_config_type' in random_at.info.keys() and random_at.info['bde_config_type'] == f'{dft_prop_prefix}optimised'
    assert f'{dft_prop_prefix}opt_positions_hash' in random_at.info.keys()
    assert isinstance(dft_calculator, tuple)
    assert len(dft_calculator) == 3

    # if remote_info is not None:
        # orig_job_name = remote_info.job_name
    

    # deal with needed paths
    wdir = Path(wdir)
    wdir.mkdir(parents=True, exist_ok=True)

    #set calculator path 
    calc_kwargs = deepcopy(dft_calculator[2])
    calc_kwargs["workdir_root"] = pj(wdir, 'orca_outputs')
    dft_calculator = (dft_calculator[0], dft_calculator[1], calc_kwargs)


    stem = Path(dft_bde_filename).stem
    dft_bde_with_ip_fname = wdir / (stem + '.' + ip_prop_prefix[:-1] + '.xyz')
    ip_reopt_fname = wdir / (stem + '.' + ip_prop_prefix + 'reoptimised.xyz')
    ip_reopt_fname_with_ip = wdir / (stem + '.' + ip_prop_prefix + 'reoptimised.' + ip_prop_prefix[-1] + '.xyz')
    ip_reopt_with_dft_fname = wdir / (ip_reopt_fname.stem + '.' + dft_prop_prefix[:-1] + '.xyz')
    isolated_h_fname = wdir / (ip_prop_prefix + "isolated_H.xyz")
    dft_opt_bde_fname =  wdir / (dft_bde_with_ip_fname.stem + '.bde.xyz') 
    ip_reopt_bde_fname = wdir / (ip_reopt_with_dft_fname.stem + '.bde.xyz')
    summary_file = Path(output_dir) / Path(dft_bde_filename).parent / (stem + '.' + ip_prop_prefix + "bde.xyz")

    # for p in [dft_bde_with_ip_fname, ip_reopt_fname, ip_reopt_with_dft_fname, isolated_h_fname,
    #           dft_opt_bde_fname, ip_reopt_bde_fname, summary_file]:
    #     print(p)

    # 1. evaluate structures with pred_calculator
    # if remote_info is not None:
    #     remote_info.job_name = orig_job_name + f'_{ip_prop_prefix}ef'
    logger.info("evaluating IP on dft-optimised structures")
    inputs = ConfigSet(dft_bde_filename)
    outputs = OutputSpec(dft_bde_with_ip_fname)
    inputs = generic.run(inputs=inputs,
                        outputs=outputs,
                        calculator=pred_calculator,
                        properties=['energy', 'forces'],
                        output_prefix=ip_prop_prefix,
                        autopara_info = AutoparaInfo(
                            remote_info=remote_info["pred_single_point"],
                            num_inputs_per_python_subprocess = 50))
    inputs = _set_tags(inputs, {"dataset_type":f"bde_{dft_prop_prefix}optimised"})


    # 2. Duplicate and relabel structures in-memory
    outputs = OutputSpec()
    inputs = _prepare_structures(inputs, outputs)

    # 3. Optimise with interatomic potential
    # if remote_info is not None:
    #     remote_info.job_name = orig_job_name + f'_{ip_prop_prefix}opt'
    logger.info('IP-optimising DFT structures')
    outputs = OutputSpec(ip_reopt_fname)
    inputs = opt.optimise(inputs=inputs,
                        outputs=outputs,
                        calculator=pred_calculator,
                        output_prefix=ip_prop_prefix,
                        autopara_info = AutoparaInfo(
                            num_inputs_per_python_subprocess=num_inputs_per_python_subprocess,
                            remote_info=remote_info["pred_opt"]))

    # inputs = _set_tags(inputs, {'bde_config_type': f"{ip_prop_prefix}optimised",
    #                                 'dataset_type': f"bde_{ip_prop_prefix}reoptimised"})

    # 3.1 evaluate with interatomic potential
    # if remote_info is not None:
    #     remote_info.job_name = orig_job_name + f'_{ip_prop_prefix}ef'
    outputs = OutputSpec(ip_reopt_fname_with_ip)
    inputs = generic.run(inputs=inputs,
                        outputs=outputs,
                        calculator=pred_calculator,
                        properties=['energy', 'forces'],
                        output_prefix=ip_prop_prefix,
                        autopara_info=AutoparaInfo(
                            remote_info=remote_info["pred_single_point"],
                            num_inputs_per_python_subprocess = 50))

    if not no_dft:
        # 4. evaluate with DFT
        logger.info('Re-evaluating ip-optimised structures with DFT')
        outputs = OutputSpec(ip_reopt_with_dft_fname)
        generic.run(inputs=inputs, 
                    outputs=outputs,
                    calculator=dft_calculator,
                    output_prefix=dft_prop_prefix, 
                    properties=["energy", "forces"],
                    autopara_info = AutoparaInfo(
                        num_inputs_per_python_subprocess=1,
                        remote_info=remote_info["dft_single_point"],
                        remote_label="dft_single_point"))

    # 5. construct isolated atom 
    logger.info("Constructing isolated_h")
    # if remote_info is not None:
    #     remote_info.job_name = orig_job_name + f'_H'
    outputs = OutputSpec(isolated_h_fname)
    ip_isolated_h(pred_calculator=pred_calculator,
                  dft_calculator=dft_calculator,
                  dft_prop_prefix=dft_prop_prefix, 
                  ip_prop_prefix=ip_prop_prefix, 
                  outputs=outputs,
                  wdir=wdir,
                  no_dft=no_dft,
                  remote_info=remote_info)


    # 6. assign BDEs
    logger.info("Assigning BDEs")
    isolated_h = read(isolated_h_fname)
    if not dft_opt_bde_fname.exists(): 
        ## on dft-optimised structures
        all_atoms_dft_opt = read(dft_bde_with_ip_fname, ':')
        ### ip bdes
        table.assign_bde_info(all_atoms=all_atoms_dft_opt,
                            prop_prefix=ip_prop_prefix, 
                            hash_label=f'{dft_prop_prefix}opt_mol_positions_hash', 
                            isolated_h=isolated_h)
        ### dft bdes
        table.assign_bde_info(all_atoms=all_atoms_dft_opt,
                            prop_prefix=dft_prop_prefix, 
                            hash_label=f'{dft_prop_prefix}opt_mol_positions_hash', 
                            isolated_h=isolated_h)
        write(dft_opt_bde_fname, all_atoms_dft_opt)

    if not ip_reopt_bde_fname.exists(): 
        ## on ip-reoptimised structures
        all_atoms_ip_reopt = read(ip_reopt_with_dft_fname, ':')
        table.assign_bde_info(all_atoms=all_atoms_ip_reopt,
                            prop_prefix=ip_prop_prefix, 
                            hash_label=f'{dft_prop_prefix}opt_mol_positions_hash', 
                            isolated_h=isolated_h)
        if not no_dft:
            ### dft bdes
            table.assign_bde_info(all_atoms=all_atoms_ip_reopt,
                                prop_prefix=dft_prop_prefix, 
                                hash_label=f'{dft_prop_prefix}opt_mol_positions_hash', 
                                isolated_h=isolated_h)
        write(ip_reopt_bde_fname, all_atoms_ip_reopt)


    # 7. gather results in one file for comparison
    logger.info("gathering results to one file")
    dft_opt_ats = ConfigSet(dft_opt_bde_fname)
    ip_reopt_ats = ConfigSet(ip_reopt_bde_fname)
    outputs = OutputSpec(summary_file)
    inputs = collect_bde_results(dft_opt_ats=dft_opt_ats, 
                        ip_reopt_ats=ip_reopt_ats, 
                        dft_prop_prefix=dft_prop_prefix,
                        ip_prop_prefix=ip_prop_prefix, 
                        outputs=outputs)

    return inputs




def collect_bde_results(dft_opt_ats, ip_reopt_ats, dft_prop_prefix, ip_prop_prefix,
                        outputs):

    for dft_opt_at, ip_reopt_at in zip(dft_opt_ats, ip_reopt_ats):

        # we have preserved order in two files, so in the first instance
        # can just check that's the case
        # eventually will want to match the corresponding files

        assert dft_opt_at.info[f'{dft_prop_prefix}opt_positions_hash'] == \
              ip_reopt_at.info[f'{dft_prop_prefix}opt_positions_hash']

        at = dft_opt_at.copy()
        remove_energy_force_containing_entries(at)
        at.info["bde_config_type"] = "collected"

        # info from dft_opt atoms
        at.info[f'{dft_prop_prefix}opt_{dft_prop_prefix}energy'] = dft_opt_at.info[f'{dft_prop_prefix}energy']
        at.arrays[f'{dft_prop_prefix}opt_{dft_prop_prefix}forces'] = dft_opt_at.arrays[f'{dft_prop_prefix}forces']
        at.info[f'{dft_prop_prefix}opt_{ip_prop_prefix}energy'] = dft_opt_at.info[f'{ip_prop_prefix}energy']
        at.arrays[f'{dft_prop_prefix}opt_{ip_prop_prefix}forces'] = dft_opt_at.arrays[f'{ip_prop_prefix}forces']

        #info from ip-opt atoms
        at.info[f'{ip_prop_prefix}opt_{dft_prop_prefix}energy'] = ip_reopt_at.info[f'{dft_prop_prefix}energy']
        at.arrays[f'{ip_prop_prefix}opt_{dft_prop_prefix}forces'] = ip_reopt_at.arrays[f'{dft_prop_prefix}forces']
        at.info[f'{ip_prop_prefix}opt_{ip_prop_prefix}energy'] = ip_reopt_at.info[f'{ip_prop_prefix}energy']
        at.arrays[f'{ip_prop_prefix}opt_{ip_prop_prefix}forces'] = ip_reopt_at.arrays[f'{ip_prop_prefix}forces']

        if at.info["mol_or_rad"] == "rad":
            at.info[f'{dft_prop_prefix}opt_{dft_prop_prefix}bde_energy'] = dft_opt_at.info[f'{dft_prop_prefix}bde_energy']
            at.info[f'{dft_prop_prefix}opt_{ip_prop_prefix}bde_energy'] = dft_opt_at.info[f'{ip_prop_prefix}bde_energy']
            at.info[f'{ip_prop_prefix}opt_{dft_prop_prefix}bde_energy'] = ip_reopt_at.info[f'{dft_prop_prefix}bde_energy']
            at.info[f'{ip_prop_prefix}opt_{ip_prop_prefix}bde_energy'] = ip_reopt_at.info[f'{ip_prop_prefix}bde_energy']


        at.arrays[f'{dft_prop_prefix}opt_positions'] = dft_opt_at.positions.copy()
        at.arrays[f'{ip_prop_prefix}opt_positions'] = ip_reopt_at.positions.copy()

        outputs.store(at)

    outputs.close()

    return outputs.to_ConfigSet() 

def _prepare_structures(inputs, outputs):

    for at in inputs:
        fresh_at = at.copy()
        # remove keys that will change upon geometry optimisation
        remove_energy_force_containing_entries(fresh_at)
        outputs.store(fresh_at)
    outputs.close()
    return outputs.to_ConfigSet()


def setup_orca_kwargs():

    cfg=Config.load()
    default_kw = Config.from_yaml(
        os.path.join(cfg['util_root'], 'default_kwargs.yml'))

    orca_kwargs = default_kw['orca']
    orca_kwargs['scratch_path'] = scratch_dir
    orca_kwargs["orca_command"] = cfg["orca_path"]
    logger.info(f"using orca_kwargs: {orca_kwargs}")

    return  orca_kwargs


def _set_tags(inputs, tags):
    outputs = OutputSpec()
    for at in inputs:
        for key, tag in tags.items():
            at.info[key] = tag
        outputs.store(at)
    outputs.close()
    return outputs.to_ConfigSet()

