#!/bin/bash
#$ -pe smp 1 
#$ -l h_rt=12:00:00
#$ -S /bin/bash
#$ -N opt6
#$ -j yes
#$ -cwd
/home/eg475/programs/orca/orca_4_2_1_linux_x86-64_openmpi314/orca orca6.inp > orca6.out
