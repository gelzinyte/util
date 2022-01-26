
export WFL_AUTOPARA_REMOTEINFO=$PWD/remoteinfo.json
cat <<EOF > remoteinfo.json
{
"orca.py::evaluate" : {
  "sys_name": "local",
  "job_name": "train_wfl_orca_eval",
  "resources": {"n" : [1, "tasks"], "partitions":"any$", "max_time": "4h", "ncores_per_task":4},
  "partial_node":true,
  "job_chunksize": 1 
  }
}

EOF



echo $(date)
#conda activate wfl_dev
echo $HOSTNAME

tmp_dir=/scratch-ssd/eg475
mkdir -p $tmp_dir

opt_base_rundir=orca_base_rundir
orca_simple_input="UKS B3LYP def2-SV(P) def2/J D3BJ PAL4"
smearing=5000
orca_additional_blocks="%scf Convergence Tight SmearTemp ${smearing} end "
output_prefix='dft_'
orca_command="/opt/womble/orca/orca_5.0.0/orca"
calc_kwargs="task=opt"


input="bde_mols.xyz"
output="bde_mols.dft_opt.xyz"

start=`date +%s`

wfl -v ref-method orca-eval -tmp ${tmp_dir} --base-rundir $opt_base_rundir --output-prefix "${output_prefix}" --calc-kwargs "${calc_kwargs}" --orca-simple-input "${orca_simple_input}" --orca-additional-blocks "${orca_additional_blocks}" --orca-command "${orca_command}" --output-file $output --keep-files default $input 


end=`date +%s`
runtime=$((end-start))

echo "script took ${runtime} s" 
