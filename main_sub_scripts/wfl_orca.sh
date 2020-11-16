#!/bin/bash                                          
#$ -pe smp 8        # number of cores requested
#$ -l h_rt=12:00:00  # time requested in HH:MM:SS format                                                          
#$ -S /bin/bash      # shell to run the job in
#$ -N nm100K            # name of job (will appear in output of qstat)
#$ -j yes            # combine error and output logs       
#$ -cwd              # execute job in directory from which it was submitted

mkdir -p /scratch/eg475
source /home/eg475/programs/miniconda3/etc/profile.d/conda.sh
conda activate wo0
export  OMP_NUM_THREADS=${NSLOTS}

wfl -v ref-method orca-eval  -o nm_displaced_100K_out.xyz -tmp "/scratch/eg475" -n 1 -p 32 --kw="n_hop=1 smearing=2000 maxiter=200 " --orca-simple-input "UKS B3LYP def2-SV(P) def2/J D3BJ" nm_displaced_100K.xyz 

rm -r /scratch/eg475
echo "finished script and removed scratch directory"
