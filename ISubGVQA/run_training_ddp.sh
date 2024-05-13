#!/usr/bin/env bash

cuda () {
    local devs=$1
    shift
    CUDA_VISIBLE_DEVICES="$devs" "$@"
}

file_name=mgat_ddp_bs_128
batch_size=128
num_workers=8
output_dir=/mount/arbeitsdaten53/projekte/simtech/tillipl/results/isubgvqa/$file_name

eval "$(conda shell.bash hook)"
conda activate isubgvqa

cuda 0,2,3,4 torchrun --standalone --nproc_per_node=4 --nnodes=1 main_imle.py --distributed --batch-size=$batch_size --num_workers=$num_workers > $file_name.out 2>&1 &
