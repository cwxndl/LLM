#!/bin/bash

# 清除屏幕
clear

# 设置训练参数
MODEL_SAVE_DIR="./model_save/sft_qwen1"
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
NUM_TRAIN_EPOCHS=3
WEIGHT_DECAY=0.1
LEARNING_RATE=6e-5
SAVE_STEPS=50
LOGGING_STEPS=20
WARMUP_STEPS=1000
SFT_DATA_PATH='/root/autodl-tmp/zh1.json'
MODEL_PATH_NAME='qwen/Qwen-1_8B'
# USE_LORA=False
MAX_LEN=512
cache_data_path='/root/autodl-tmp/zh1.pkl'
# 执行训练脚本
accelerate launch --multi_gpu --config_file accelerate_multi_gpu.yaml finetune_qwen.py \
    --model_save_dir $MODEL_SAVE_DIR \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --learning_rate $LEARNING_RATE \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS\
    --warmup_steps $WARMUP_STEPS\
    --sft_data_path $SFT_DATA_PATH\
    --max_len $MAX_LEN\
    --model_path_name $MODEL_PATH_NAME\
    --cache_data_path $cache_data_path\
    
# torchrun --standalone --nproc_per_node=8 pretrain.py \
#     --model_save_dir $MODEL_SAVE_DIR \
#     --train_batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --num_train_epochs $NUM_TRAIN_EPOCHS \
#     --weight_decay $WEIGHT_DECAY \
#     --learning_rate $LEARNING_RATE \
#     --save_steps $SAVE_STEPS \
#     --logging_steps $LOGGING_STEPS\
#     --warmup_steps $WARMUP_STEPS\
    