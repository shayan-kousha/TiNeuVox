# bash prepare_data.sh /data_dnerf/water_bottle_2/ 20220908_184708.mp4

data_dir=${N3DR_DATA_PATH}$1
mp4_file=${data_dir}$2

echo ${data_dir}
echo ${mp4_file}

ns-process-data video --data ${mp4_file} --output-dir ${data_dir} --no-gpu 

cp -r ${data_dir}transforms.json ${data_dir}transforms_val.json
cp -r ${data_dir}transforms.json ${data_dir}transforms_train.json
cp -r ${data_dir}transforms.json ${data_dir}transforms_test.json

rm -r ${data_dir}images_2
rm -r ${data_dir}images_4
rm -r ${data_dir}images_8
rm -r ${data_dir}colmap
rm ${data_dir}transforms.json
