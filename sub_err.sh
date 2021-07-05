#!/bin/bash
#$ -pe smp 28       # number of cores requested
#$ -l h_rt=168:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N err_diff 
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted


mkdir -p /tmp/eg475

echo $(date)
conda activate wo
export  OMP_NUM_THREADS=1
export WFL_AUTOPARA_NPOOL=${NSLOTS}

mkdir -p xyzs 

gap_name=gap_dftb_diff
gap_filename=${gap_name}.xml

train_name=train_${idx}
train_filename=${train_name}.xyz
evaled_train=${gap_name}_on_${train_name}.xyz
train_plot=scatter_train.pdf

test_name=test_012
test_filename=${test_name}.xyz
evaled_test=${gap_name}_on_${test_name}.xyz
test_plot=scatter_${gap_name}_${test_name}.pdf



util plot error-table -r dft_ -p ${gap_name}_ --kw "param_filename=${gap_filename}" -o ${evaled_train} --chunksize 50 ${train_filename}  > err_${gap_name}_${train_name}.txt
util plot error-table -r dft_ -p ${gap_name}_ --kw "param_filename=${gap_filename}" -o ${evaled_test}  --chunksize 50 ${test_filename} > err_${gap_name}_${test_name}.txt

wfl plot ef-thinpoints --plot-fn ${test_plot}  --gap-fn $gap_filename -re dft_energy -ge ${gap_name}_energy -rf dft_forces -gf ${gap_name}_forces ${evaled_test} 
wfl plot ef-thinpoints --plot-fn ${train_plot}  --gap-fn $gap_filename -re dft_energy -ge ${gap_name}_energy -rf dft_forces -gf ${gap_name}_forces ${evaled_train}

