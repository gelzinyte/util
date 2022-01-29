#!/bin/bash
#$ -pe smp 16     # number of cores requested
#$ -l h_rt=4:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N fit_old
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted


echo $(date)
hostmane

export OMP_NUM_THREADS=${NSLOTS}
export JULIA_NUM_THREADS=${NSLOTS}

start=`date +%s`

julia fit.jl 
julia ../evaluate_fit.jl --param-fname ace_old_script.json  > evaluate_fit.out

end=`date +%s`
runtime=$((end-start))
echo "script took ${runtime} s"

