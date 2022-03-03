dir_name="all_200x200x200_densities/*"

# singlet_triplet_opt 
rename 's/triplet_opt.uks_def2-svp_opt/uDFT_triplet_optimised/g' $dir_name 
rename 's/singlet_opt.uks_def2-svp_opt/uDFT_singlet_optimised/g' $dir_name

# uks or cc to more clearly 
rename 's/uks_cc-pvdz/uDFT_single_point/g' $dir_name
rename 's/dlpno-ccsd_cc-pvdz/DLPNO-CCSD_single_point/' $dir_name

# densities kind 
rename 's/mdci/CC_density/' $dir_name
rename 's/uhf_scf/uHF_density/' $dir_name
rename 's/uks_scf/uDFT_density/' $dir_name

# spin vs electron densities
rename 's/spindens/spin_density/' $dir_name
rename 's/eldens/electron_density/' $dir_name






