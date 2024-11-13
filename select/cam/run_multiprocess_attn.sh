#!/bin/bash
set -ux
export CHUNK_SIZE=128
export WINDOW_SIZE=64000
# hyper-parameters
MODEL_PATH=THUDM/LongAlign-7B-64k-base

SINGLE_PPL_BATCH_SIZE=32
SEED=11

# output settings
# make sure to make ori_LongALign.json into jsonl format
DATA_FILE_PATH=../data/ori_LongAlign.jsonl
ROOT_PATH=../data/attention_final

python ./run_batch_multi_process.py \
    --data_file $DATA_FILE_PATH \
    --root_path $ROOT_PATH \
    --model_name $MODEL_PATH \
    --chunk_size $CHUNK_SIZE \
    --window_size $WINDOW_SIZE \
    --single_ppl_batch_size $SINGLE_PPL_BATCH_SIZE \
    --seed $SEED \
    --do_attn



    
