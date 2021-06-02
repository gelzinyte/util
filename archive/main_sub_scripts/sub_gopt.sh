#!/bin/bash
#$ -pe smp 8        # number of cores requested
#$ -l h_rt=12:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N gopt7.2            # name of job (will appear in output of qstat)
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted

source /home/eg475/programs/miniconda3/etc/profile.d/conda.sh
conda activate wo0
export  OMP_NUM_THREADS=${NSLOTS}

python /home/eg475/programs/my_scripts/gopt_test/gap_geo_opt_test.py --gap_fname ../gaps/gap_10.xml --dft_eq_xyz dft_min_nbu.xyz --stds '[0.01, 0.03, 0.1, 0.3, 1]' 'CCCC' 

echo "finished script"

