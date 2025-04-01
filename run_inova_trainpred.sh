#!/bin/bash
# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_BLOCK=0
export CUDA_VISIBLE_DEVICES=0

# Set the output directory to a variable for easy change
DATE=$(date +"%Y-%m-%d-%H-%M-%S")
COMMENT=dci-connector
MODEL=llava-next-interleave-qwen-7b
MODEL_NAME_OR_PATH="/home/nnguyen/spinning-storage/nnguyen/inova_icme25/foundation_models/$MODEL"
OUTPUT_DIR="output/${MODEL}_${COMMENT}_${DATE}"
ANSWER_FILE="output/result_dir_${MODEL}_${COMMENT}_${DATE}/answer_file.jsonl"
RESULT_DIR="output/result_dir_${MODEL}_${COMMENT}_${DATE}"
DATA_SPLIT="train"

# Run the Python training script with the specified output directory
python inova_train.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_split $DATA_SPLIT \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --tf32 True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to wandb \
    --model_max_length 4096 \
    --run_name ${MODEL}_${COMMENT}_${DATE} \
    --logging_steps 10 \
    --mm_dense_connector_type "dci"


# Run the prediction script
python inova_pred.py \
    --data-set "valid" \
    --answers-file ${ANSWER_FILE} \
    --model-path ${OUTPUT_DIR} \
    --temperature 0.0

# Run the metrics computation script
python inova_compute_metrics.py --answers-file ${ANSWER_FILE} --result-dir ${RESULT_DIR}
