#!/bin/bash
#$ -pe smp 32        # number of cores requested
#$ -l h_rt=4:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N gf
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted


mkdir -p /tmp/eg475

echo $(date)
conda activate wfldev

export WFL_AUTOPARA_NPOOL=${NSLOTS}
export OMP_NUM_THREADS=1

start=`date +%s`

util -v gap fit --num-cycles 1 --train-fname train.xyz --gap-param-filename  parameters.yml  --smiles-csv smiles.csv --num-smiles-opt 2 --num-nm-displacements-per-temp 2 --num-nm-temps 2 --energy-filter-threshold 0.05 --max-force-filter-threshold 0.1 

end=`date +%s`
runtime=$((end-start))

echo "script took ${runtime} s" 
