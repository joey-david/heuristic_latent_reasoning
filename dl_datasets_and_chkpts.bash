#!/bin/bash

# run data/preprocessing/gsm_icot.bash to prepare the GSM8K dataset for Internalize CoT
# if the data/gsm_* files don't exist
if [ ! -f data/gsm_train.json ]; then
    echo "data/gsm_training set not found, running data/preprocessing/gsm_icot.bash to prepare the GSM8K dataset for Internalize CoT"
    bash preprocessing/gsm_icot.bash
else
    echo "data/gsm_training set found, skipping data/preprocessing/gsm_icot.bash"
fi

# download and prepare the checkpoints from hf in data/checkpoints
# https://huggingface.co/Esther22/coconut_Reproduction/discussions

mkdir -p data/checkpoints/gsm/

# stage 0 (pre-CoT) checkpoint, courtesy of Esther22 on HF
if [ ! -f data/checkpoints/gsm/gsm-cot ]; then
    wget https://huggingface.co/Esther22/coconut_Reproduction/resolve/main/stage_0_training_ck/checkpoint_6 \
        -O data/checkpoints/gsm/gsm-cot
else 
    echo "data/checkpoints/gsm/gsm-cot found, skipping download"
fi

# stage 1 (post-CoT) checkpoint, courtesy of Esther22 on HF
if [ ! -f data/checkpoints/gsm/gsm-coconut ]; then
    wget https://huggingface.co/Esther22/coconut_Reproduction/resolve/main/stage_1_training_ck/checkpoint_5 \
        -O data/checkpoints/gsm/gsm-coconut
else 
    echo "data/checkpoints/gsm/gsm-coconut found, skipping download"
fi
