#!/bin/bash
#$ -pe smp 32        # number of cores requested
#$ -l h_rt=12:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N r300.1           # name of job (will appear in output of qstat)
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted

echo $(date)

mkdir -p /scratch/eg475
source /home/eg475/programs/miniconda3/etc/profile.d/conda.sh
conda activate wo0
export  OMP_NUM_THREADS=${NSLOTS}

echo "running gap fitting"
python /home/eg475/programs/my_scripts/gap_fitting_scripts/gap_from_dft_min.py --dft_min_fname dft_min_butane.xyz --no_dpoints 300 --stdev 0.03 --n_rattle_atoms 14 

echo "running gap testing"
OMP_NUM_THREADS=1
python /home/eg475/programs/my_scripts/gopt_test/gap_geo_opt_test.py --no_cores ${NSLOTS}  --gap_fname gap.xml --dft_eq_xyz dft_min_butane.xyz --stds '[0.01, 0.03, 0.1, 0.3, 1]' 'CCCC' 'C(C)(C)C'


#rm -r /scratch/eg475
echo "finished script and removed scratch directory"
