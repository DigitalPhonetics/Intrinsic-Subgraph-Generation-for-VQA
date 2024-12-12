#!/usr/bin/env bash

gpu_id=5

file_name="mgat_bs_512_imle_k5_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1_token_coo_top_ckpt"

export TOKENIZERS_PARALLELISM=true

eval "$(conda shell.bash hook)"
conda activate isubgvqa
CUDA_VISIBLE_DEVICES=$gpu_id nohup python -u run_token_coo.py \
    > $file_name.out 2>&1 &
