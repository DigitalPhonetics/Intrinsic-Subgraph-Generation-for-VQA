#!/usr/bin/env bash

gpu_id=6

batch_size=128

sampler=imle
sample_k=2
n_epochs=50
n_workers=8
nb_samples=1
alpha=1.0
beta=10.0
tau=1.0
MGAT_MASKS=('1.0' '1.0' '1.0' '0.1')
# resume="/mount/arbeitsdaten53/projekte/simtech/tillipl/results/isubgvqa/mgat_bs_256_simple_k4/checkpoint.pth"

if [ "$sampler" = "imle" ]; then
    file_name="mgat_bs_${batch_size}_${sampler}_k${sample_k}_nb_samples_${nb_samples}_alpha_${alpha}_beta_${beta}_tau_${tau}_v1"
elif [ "$sampler" = "aimle" ]; then
    file_name="mgat_bs_${batch_size}_${sampler}_k${sample_k}_nb_samples_${nb_samples}_alpha_${alpha}_tau_${tau}_v1"
elif [ "$sampler" = "simple" ]; then
    file_name="mgat_${sampler}_bs${batch_size}_k${sample_k}_v1"
elif [ "$sampler" = "gumbel" ]; then
    file_name="mgat_${sampler}_bs${batch_size}_k${sample_k}_v1"
fi

output_dir=/mount/arbeitsdaten53/projekte/simtech/tillipl/results/isubgvqa/$file_name

export TOKENIZERS_PARALLELISM=true

eval "$(conda shell.bash hook)"
conda activate isubgvqa
# --text_sampling \ --resume=$resume \
CUDA_VISIBLE_DEVICES=$gpu_id nohup python -u main.py \
    --num_workers=$n_workers \
    --epochs=$n_epochs \
    --batch-size=$batch_size \
    --output_dir=$output_dir \
    --sampler_type=$sampler \
    --sample_k=$sample_k \
    --nb_samples=$nb_samples \
    --alpha=$alpha \
    --beta=$beta \
    --tau=$tau \
    --mgat_masks ${MGAT_MASKS[@]} \
    > $file_name.out 2>&1 &
