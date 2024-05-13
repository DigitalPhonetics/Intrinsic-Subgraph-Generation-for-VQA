#!/usr/bin/env bash

gpu_id=1

file_name=mgat_bs_128
batch_size=128
output_dir=/mount/arbeitsdaten53/projekte/simtech/tillipl/results/isubgvqa/$file_name

eval "$(conda shell.bash hook)"
conda activate isubgvqa

CUDA_VISIBLE_DEVICES=$gpu_id nohup python -u main_imle.py --batch-size=128 --output_dir=$output_dir > $file_name.out 2>&1 &
