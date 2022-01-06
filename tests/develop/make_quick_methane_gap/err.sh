source ~/.bashrc
gap_name=tiny_gap
gap_fname=${gap_name}.xml
train_name=train
train_fname=${train_name}.xyz
test_name=test
test_fname=${test_name}.xyz

util plot error-table -r dft_ -p tiny_gap_ --kw "param_filename=${gap_fname}" -o ${gap_name}_on_${test_name}.xyz --chunksize 50  $test_fname > err_${test_name}.txt
util plot error-table -r dft_ -p tiny_gap_ --kw "param_filename=${gap_fname}" -o ${gap_name}_on_${train_name}.xyz --chunksize 50  $train_fname > err_${train_name}.txt



