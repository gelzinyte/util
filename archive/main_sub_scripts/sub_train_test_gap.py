#!/bin/bash
#$ -pe smp 32        # number of cores requested
#$ -l h_rt=12:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N j1.1           # name of job (will appear in output of qstat)
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted

echo $(date)
gap_name=gap1.1_delta0.5
train_dset=../xyzs/dset1.xyz

source /home/eg475/programs/miniconda3/etc/profile.d/conda.sh
conda activate wo0
export  OMP_NUM_THREADS=${NSLOTS}


if ! test -f ${gap_name}.xml; then

	echo "Fitting GAP"

	/home/eg475/programs/QUIPwo0/build/linux_x86_64_gfortran_openmp/gap_fit energy_parameter_name=dft_energy force_parameter_name=dft_forces sparse_separate_file=F default_sigma={0.001 0.01 0.0 0.0} config_type_kernel_regularisation={isolated_atom:0.0001:0.0:0.0:0.0} core_param_file=/home/eg475/programs/my_scripts/source_files/glue_orca.xml core_ip_args={IP Glue} gap={soap l_max=4 n_max=8 cutoff=4.0 delta=0.5 covariance_type=dot_product zeta=4.0 n_sparse=300 sparse_method=cur_points atom_gaussian_width=0.3 add_species=True} atoms_filename=$train_dset gp_file=${gap_name}.xml  2>&1 | tee out_${gap_name}.txt

else
	echo "Found ${gap_name} not fitting"

fi

echo "plotting plots"
gap_plots --param_fname ${gap_name}.xml --test_fname ../xyzs/test_dset4.xyz --dimer_plot False

python -c "from util import plot; plot.evec_plot('${gap_name}.xml', 'dft_min_tbu.xyz', at_fname_to_opt='dft')"

echo "optimising geometries"
OMP_NUM_THREADS=1
python /home/eg475/programs/my_scripts/gopt_test/gap_geo_opt_test.py --cleanup False --no_cores ${NSLOTS} --gap_fname "${gap_name}.xml" --dft_eq_xyz dft_min_tbu.xyz --stds '[0.01, 0.03, 0.1, 0.3]' --temps '[30, 100, 300, 1000]'  'C(C)(C)C'


echo "Tadaa"
