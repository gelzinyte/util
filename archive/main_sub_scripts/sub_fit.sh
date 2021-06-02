#!/bin/bash                                          
#$ -pe smp 2        # number of cores requested
#$ -l h_rt=12:00:00  # time requested in HH:MM:SS format                                                          
#$ -S /bin/bash      # shell to run the job in
#$ -N r            # name of job (will appear in output of qstat)
#$ -j yes            # combine error and output logs       
#$ -cwd              # execute job in directory from which it was submitted

mkdir -p /scratch/eg475
source /home/eg475/programs/miniconda3/etc/profile.d/conda.sh
conda activate wo0
export  OMP_NUM_THREADS=${NSLOTS}

python fit.py --n_iterations 2 --n_dpoints 2 --n_rattle_atoms 2  C

rm -r /scratch/eg475
echo "finished script and removed scratch directory"
