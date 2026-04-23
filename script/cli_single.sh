#!/bin/bash

print_usage() {
    echo "Usage: $0 <request_rate> <num_prompts> <dataset> <port> <cook_flag> [model]"
    echo "Example: bash cli_single.sh 20 2000 sharegpt 8006 0"
    echo "Example: bash cli_single.sh 20 2000 loogles 8006 0"
}

if [ "$#" -ne 5 ] && [ "$#" -ne 6 ]; then
    print_usage
    exit 1
fi

REQUEST_RATE=$1
NUM_PROMPTS=$2
DATASET=$3
PORT=$4
COOK_FLAG=$5
MODEL_NAME=${6:-"qwen3-8b"}

declare -A MODEL_PATHS=(
    ["llama"]="./models/Meta-Llama-3.1-8B-Instruct"
    ["llama70b"]="./models/Meta-Llama-3.1-70B-Instruct"
    ["qwen"]="./models/Qwen2.5-14B-Instruct"
    ["mistral"]="./models/Mistral-7B-Instruct-v0.1"
    ["qwen3-8b"]="./models/Qwen3-8B/"
)

MODEL=${MODEL_PATHS[$MODEL_NAME]}
if [ -z "$MODEL" ]; then
    echo "Invalid model name"
    exit 1
fi

declare -A DATASET_PATHS=(
    ["sharegpt"]="./datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    ["loogles"]="./datasets/loogle/output_short.jsonl"
    ["looglel"]="./datasets/loogle/output_long.jsonl"
    ["burstgpt"]="./datasets/BurstGPT/"
)

DATASET_PATH=${DATASET_PATHS[$DATASET]}
if [ -z "$DATASET_PATH" ]; then
    echo "Invalid dataset name"
    exit 1
fi

TIME=0
RESULT_DIR="./outputs/vllm/single$(TZ=Asia/Shanghai date +%m%d)"
RESULT_FILENAME="$PORT-$DATASET-qps$REQUEST_RATE-req$NUM_PROMPTS-$MODEL_NAME-$TIME.json"
LOG_PATH="./outputs/log/single_P$COOK_FLAG-$DATASET-$PORT-$TIME.txt"

if [ "$DATASET" == "looglel" ] || [ "$DATASET" == "loogles" ]; then
    DATASET="loogle"
fi

vllm bench serve \
    --model $MODEL \
    --dataset-name $DATASET \
    --dataset-path $DATASET_PATH \
    --request-rate $REQUEST_RATE \
    --num-prompts $NUM_PROMPTS \
    --port $PORT \
    --ignore-eos --save-result --save-detailed \
    --result-dir $RESULT_DIR \
    --result-filename $RESULT_FILENAME \
    --disable-shuffle \
    --cf $COOK_FLAG \
    >> $LOG_PATH 2>&1 &

echo $LOG_PATH
