#!/bin/bash

# 清除屏幕
clear

# 设置训练参数
MODEL_SAVE_DIR="./model_save/pretrain_moe"
BATCH_SIZE=10
GRADIENT_ACCUMULATION_STEPS=5
NUM_TRAIN_EPOCHS=1
WEIGHT_DECAY=0.1
LEARNING_RATE=1e-4
SAVE_STEPS=50
LOGGING_STEPS=20
WARMUP_STEPS=1000

# 执行训练脚本
accelerate launch --multi_gpu --config_file accelerate_multi_gpu.yaml moe_pretrain.py \
    --model_save_dir $MODEL_SAVE_DIR \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --learning_rate $LEARNING_RATE \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS\
    --warmup_steps $WARMUP_STEPS\
    
    