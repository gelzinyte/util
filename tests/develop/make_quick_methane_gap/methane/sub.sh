#!/bin/bash
#$ -pe smp 32        # number of cores requested
#$ -l h_rt=168:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N ch4 
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted

echo $(date)
smiles="C"
config_type=methane
mol_non_opt_fn='mol_non_opt.xyz'
mol_rad_non_opt_fn='mol_rad_non_opt.xyz'
opt_fn="opt_mol_rad.xyz"
nm_fn='mol_rad_nms.xyz'

scratch_path=/scratch-ssd/eg475
opt_workdir_root=orca_opt
orca_simple_input="UKS B3LYP def2-SV(P) def2/J D3BJ"
smearing=5000
opt_calc_kwargs="task=opt smearing=${smearing}"

n_run=1
n_hop=1
nm_calc_kwargs="orcasimpleinput='${orca_simple_input}'  scratch_path='$scratch_path' n_run='${n_run}' n_hop='${n_hop}' smearing='$smearing'"


mkdir -p $scratch_path
export  OMP_NUM_THREADS=${NSLOTS}
export AUTOPARA_NPOOL=${NSLOTS}

#
#conda activate wo
#wfl -v generate-configs smiles -o ${mol_non_opt_fn} -i "config_type=${config_type}" $smiles
#wfl -v generate-configs remove-sp3-Hs -o ${mol_rad_non_opt_fn} ${mol_non_opt_fn}
#
#conda activate py3.8
#echo 'optimising'
#wfl -v ref-method orca-eval --output-prefix 'dft_' --orca-simple-input "${orca_simple_input}" --calc-kwargs "${opt_calc_kwargs}" --scratch-path $scratch_path --base-rundir $opt_workdir_root  --output-file ${opt_fn}  ${mol_rad_non_opt_fn}
#
#
#conda activate wo
#echo 'generating normal modes'
#wfl -v generate-configs derive-normal-modes --calc-kwargs "${nm_calc_kwargs}" --calc-name orca -p dft_ --parallel-hessian -o ${nm_fn} ${opt_fn}


wfl -v generate-configs sample-normal-modes -n 50 -t 300 -p dft_ -i "smiles config_type" -o methane_sample.xyz $nm_fn

conda activate py3.8
wfl -v ref-method orca-eval --output-prefix 'dft_' --orca-simple-input "${orca_simple_input}" --calc-kwargs "smearing=${smearing}" --scratch-path $scratch_path --base-rundir $opt_workdir_root  --output-file methane_sample_dft.xyz methane_sample.xyz 




