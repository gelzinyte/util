#!/bin/bash
#$ -pe smp 8     # number of cores requested
#$ -l h_rt=4:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N fit_test 
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted


echo $(date)
hostmane

export OMP_NUM_THREADS=${NSLOTS}
export JULIA_NUM_THREADS=${NSLOTS}

start=`date +%s`
 
python call_fit.py
julia ../evaluate_fit.jl --param-fname ace_rundir/ACE_name.json  > evaluate_fit.out

end=`date +%s`
runtime=$((end-start))
echo "script took ${runtime} s"

