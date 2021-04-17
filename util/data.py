import pandas as pd
import subprocess
import os

# skip_first = 30
# how_many = 68
#
# submit = True
# overwrite = False
# df = pd.read_csv('../CHO_train_compounds.csv')


everything = """mol_non_opt_fn='mol_non_opt.xyz'
mol_rad_non_opt_fn='mol_rad_non_opt.xyz'
opt_fn="opt_mol_rad.xyz"
nm_fn='mol_rad_nms.xyz'

scratch_path=/scratch-ssd/eg475
opt_base_rundir=orca_opt
orca_simple_input="UKS B3LYP def2-SV(P) def2/J D3BJ"
smearing=5000
opt_calc_kwargs="task=opt smearing=${smearing}"

n_run=1
n_hop=1
nm_calc_kwargs="orcasimpleinput='${orca_simple_input}'  scratch_path='$scratch_path' n_run='${n_run}' n_hop='${n_hop}' smearing='$smearing'"


mkdir -p $scratch_path
export  OMP_NUM_THREADS=${NSLOTS}
export AUTOPARA_NPOOL=${NSLOTS}


#conda activate wo
#wfl -v generate-configs smiles -o ${mol_non_opt_fn} -i "config_type=${config_type}" $smiles
#wfl -v generate-configs remove-sp3-Hs -o ${mol_rad_non_opt_fn} ${mol_non_opt_fn}

#conda activate py3.8
#echo 'optimising'
#wfl -v ref-method orca-eval --output-prefix 'dft_' --calc-kwargs "${opt_calc_kwargs}" --scratch-path $scratch_path --base-rundir $opt_base_rundir  --output-file ${opt_fn}  ${mol_rad_non_opt_fn}


conda activate wo
echo 'generating normal modes'
wfl -v generate-configs derive-normal-modes --calc-kwargs "${nm_calc_kwargs}" --calc-name orca -p dft_ --parallel-hessian -o ${nm_fn} ${opt_fn}
"""

def sub_data(df_name, how_many, skip_first, submit, overwrite_sub, hours=48, no_cores=16, script_name='sub.sh'):

    df = pd.read_csv(df_name)

    homedir = os.getcwd()
    for idx, row in df.iterrows():

        if skip_first is not None and idx < skip_first:
            continue
        if idx == how_many:
            break

        name = row['Name']
        smi = row['SMILES']


        if not os.path.isdir(name):
            os.makedirs(name)

        if not overwrite_sub:
            if os.path.isdir(os.path.join(name, script_name)):
                continue

        os.chdir(name)

        print(f'making {name} with {smi}')

        with open(script_name, 'w') as f:
            f.write(f"""#!/bin/bash
#$ -pe smp {no_cores}        # number of cores requested
#$ -l h_rt={hours}:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N {name} 
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted

echo $(date)
""")
            f.write(f'smiles="{smi}"\n')
            f.write(f'config_type={name}\n')

            f.write(everything)

        if submit:
            subprocess.run(f"qsub {script_name}", shell=True)

        os.chdir(homedir)

    print('Finished script')
