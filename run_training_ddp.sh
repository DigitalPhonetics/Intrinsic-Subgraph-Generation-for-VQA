#!/usr/bin/env bash

cuda () {
    local devs=$1
    shift
    CUDA_VISIBLE_DEVICES="$devs" "$@"
}


batch_size=256
num_workers=6
sampler=simple
sample_k=4
n_epochs=100
file_name="mgat_ddp_bs_${batch_size}_${sampler}"
output_dir=/mount/arbeitsdaten53/projekte/simtech/tillipl/results/isubgvqa/$file_name


eval "$(conda shell.bash hook)"
conda activate isubgvqa

# --text_sampling \
cuda 0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py \
    --distributed \
    --epochs=$n_epochs \
    --batch-size=$batch_size \
    --num_workers=$num_workers \
    --output_dir=$output_dir \
    --sampler_type=$sampler \
    --sample_k=$sample_k \
    > $file_name.out 2>&1 &
