#!/bin/bash
#$ -pe smp 16        # number of cores requested
#$ -l h_rt=12:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N r32_3 
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted

echo $(date)

test_dset=xyzs/test.xyz
gap_name=gap
train_dset=xyzs/train.xyz

config_type_sigma={isolated_atom:0.0001:0.0:0.0:0.0}

e_sigma=0.001
f_sigma=0.01

l_max=6
n_max=12
delta=1
zeta=4
n_sparse=400

cutoff_soap_1=3
atom_gaussian_width_1=0.3
cutoff_transition_width_1=0.5

cutoff_soap_2=6
atom_gaussian_width_2=0.6
cutoff_transition_width_2=1

glue_fname=/home/eg475/scripts/source_files/glue_repulsive_fitted.xml

source /home/eg475/programs/miniconda3/etc/profile.d/conda.sh
conda activate wo0 
export  OMP_NUM_THREADS=${NSLOTS}

mkdir -p xyzs 


eval_train_name=xyzs/${gap_name}_on_train.xyz
eval_test_name=xyzs/${gap_name}_on_test.xyz

echo "gap_name: ${gap_name};"
echo "train_fname: ${train_dset}"


/home/eg475/programs/QUIPwo0/build/linux_x86_64_gfortran_openmp/gap_fit energy_parameter_name=dft_energy force_parameter_name=dft_forces  sparse_separate_file=F default_sigma={${e_sigma} ${f_sigma} 0.0 0.0} config_type_kernel_regularisation=$config_type_sigma core_param_file=${glue_fname} core_ip_args={IP Glue} gap={soap l_max=$l_max n_max=$n_max cutoff=$cutoff_soap_1 delta=$delta covariance_type=dot_product zeta=$zeta n_sparse=$n_sparse sparse_method=cur_points atom_gaussian_width=$atom_gaussian_width_1 cutoff_transition_width=$cutoff_transition_width_1 add_species=True : soap l_max=$l_max n_max=$n_max cutoff=$cutoff_soap_2 delta=$delta covariance_type=dot_product zeta=$zeta n_sparse=$n_sparse sparse_method=cur_points atom_gaussian_width=$atom_gaussian_width_2 cutoff_transition_width=$cutoff_transition_width_2 add_species=True} atoms_filename=$train_dset gp_file=${gap_name}.xml > out_${gap_name}.txt 



/home/eg475/programs/QUIPwo0/build/linux_x86_64_gfortran_openmp/quip E=T F=T atoms_filename=${train_dset} param_filename=${gap_name}.xml  | grep AT | sed 's/AT//' > $eval_train_name
/home/eg475/programs/QUIPwo0/build/linux_x86_64_gfortran_openmp/quip E=T F=T atoms_filename=${test_dset} param_filename=${gap_name}.xml  | grep AT | sed 's/AT//' > $eval_test_name

# scatter plots
#python ~/scripts/gap_plots_evaled.py --ref_energy_name dft_energy --ref_force_name dft_forces --pred_energy_name energy --pred_force_name force --evaluated_train_fname $eval_train_name --evaluated_test_fname $eval_test_name  --prefix ${gap_name}_F_by_element

#python ~/scripts/gap_plots_evaled.py --ref_energy_name dft_energy --ref_force_name dft_forces --pred_energy_name energy --pred_force_name force --evaluated_train_fname $eval_train_name --evaluated_test_fname $eval_test_name  --prefix ${gap_name}_F_overview --force_by_element False

#python ~/scripts/gap_plots_evaled.py --ref_energy_name dft_energy --ref_force_name dft_forces --pred_energy_name energy --pred_force_name force --evaluated_train_fname $eval_train_name --evaluated_test_fname $eval_test_name  --prefix ${gap_name}_by_config_type --force_by_element False --by_config_type True

export  OMP_NUM_THREADS=${NSLOTS}

# gap-optimise structures for getting bond dissociation energies
python ~/scripts/python_scripts/bde_summary.py --gap_fname ${gap_name}.xml --bde_start_fname ~/data/bde_files/starts/32_myristicin_non-optimised.xyz --dft_bde_fname ~/data/bde_files/dft/32_myristicin_optimised.xyz --bde_out_dir gap_bdes --gap_bde_out_fname 32_myristicin_${gap_name}_optimised.xyz


# plot summary again 
python ~/scripts/python_scripts/bde_summary.py  --bde_start_fname ~/data/bde_files/starts/32_myristicin_non-optimised.xyz --dft_bde_fname ~/data/bde_files/dft/32_myristicin_optimised.xyz --bde_out_dir gap_bdes --gap_bde_out_fname 32_myristicin_${gap_name}_optimised.xyz

# bde/gopt summary plots
#python ~/scripts/util/plot/bde.py -g GAP_BDE -m rmsd 
#python ~/scripts/util/plot/bde.py -g GAP_BDE -m soap 
#python ~/scripts/util/plot/bde.py -g GAP_BDE -m energy

# error summary
echo '------------------------ TRAIN -----------------------------'
python ~/scripts/util/errors.py -re dft_energy -pe energy -rf dft_forces -pf force -fn $eval_train_name

echo '------------------------ TEST -----------------------------'
python ~/scripts/util/errors.py -re dft_energy -pe energy -rf dft_forces -pf force -fn $eval_test_name


echo 'done with the script'

