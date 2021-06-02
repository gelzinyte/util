#!/bin/bash                                          
#$ -pe smp 8        # number of cores requested
#$ -l h_rt=12:00:00  # time requested in HH:MM:SS format                                                          
#$ -S /bin/bash      # shell to run the job in
#$ -N mol17            # name of job (will appear in output of qstat)
#$ -j yes            # combine error and output logs       
#$ -cwd              # execute job in directory from which it was submitted

mkdir -p /tmp/eg475
source /home/eg475/programs/miniconda3/etc/profile.d/conda.sh
conda activate wo0
export  OMP_NUM_THREADS=${NSLOTS}


wfl -v ref-method orca-eval --output-file out.xyz -tmp /tmp/eg475 --keep-files True --base-rundir ORCA_out -nr 1 -nh 1 --kw "smearing=2000" --orca-simple-input "UKS B3LYP def2-SV(P) def2/J D3BJ" "in.xyz"


echo "finished script and removed scratch directory"
