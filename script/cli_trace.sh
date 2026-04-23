#!/bin/bash

print_usage() {
    echo "Usage: $0 <trace> <rate> <num_prompts> <dataset> <port> [model]"
    echo "Example: bash cli_trace.sh sheet94 20 2000 sharegpt 8000"
    echo "Example: bash cli_trace.sh burstgpt 20 2000 loogles 8001"
    exit 1
}

if [ "$#" -ne 5 ] && [ "$#" -ne 6 ]; then
    print_usage
    exit 1
fi

TRACE=$1
SCALE_PEAK_RATE=$2
NUM_PROMPTS=$3
DATASET=$4
PORT=$5
MODEL_NAME=${6:-"llama"}

declare -A MODEL_PATHS=(
    ["llama"]="./models/Meta-Llama-3.1-8B-Instruct"
    ["llama70b"]="./models/Meta-Llama-3.1-70B-Instruct"
    ["qwen"]="./models/Qwen2.5-14B-Instruct"
    ["mistral"]="./models/Mistral-7B-Instruct-v0.1"
)

MODEL=${MODEL_PATHS[$MODEL_NAME]}
if [ -z "$MODEL" ]; then
    echo "Invalid model name"
    exit 1
fi

declare -A TRACE_PATHS=(
    ["burstgpt"]="./traces/BurstGPT_without_fails_2.csv"
    ["sheet94"]="./traces/format/formatted_sheet94.xlsx"
)

# sheet94 dataset is the industry dataset as fig.2, which is not allowed to be opened.

TRACE_PATH=${TRACE_PATHS[$TRACE]}
if [ -z "$TRACE_PATH" ]; then
    echo "Invalid trace name"
    exit 1
fi

declare -A DATASET_PATHS=(
    ["sharegpt"]="./datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
    ["looglel"]="./datasets/loogle/output.jsonl"
    ["loogles"]="./datasets/loogle/output_short.jsonl"
    ["burstgpt"]="./traces/BurstGPT_without_fails_2.csv"
)

DATASET_PATH=${DATASET_PATHS[$DATASET]}
if [ -z "$DATASET_PATH" ]; then
    echo "Invalid dataset name"
    exit 1
fi

PRIORITY=${PRIORITY:-0}

TIME=0
RESULT_DIR="./outputs/vllm/trace_$TRACE$(TZ=Asia/Shanghai date +%m%d)"
RESULT_FILENAME="trace$PORT-$DATASET-peak$SCALE_PEAK_RATE-req$NUM_PROMPTS-$MODEL_NAME-$TIME.json"
LOG_PATH="./outputs/log/trace_$TRACE$(TZ=Asia/Shanghai date +%m%d)-$DATASET-$PORT-$TIME.txt"

python ../benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $MODEL \
    --dataset-name $DATASET  \
    --dataset-path $DATASET_PATH \
    --trace $TRACE_PATH \
    --request-rate $SCALE_PEAK_RATE \
    --num-prompts $NUM_PROMPTS \
    --port $PORT \
    --priority $PRIORITY \
    --save-result \
    --result-dir $RESULT_DIR \
    --result-filename $RESULT_FILENAME \
    >> $LOG_PATH 2>&1 &

echo $LOG_PATH
