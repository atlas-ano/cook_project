#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <MODEL_ID> <MAX_TOKEN>"
    echo "  MODEL_ID: 1-5 (1:Llama-3.1-70B-Instruct, 2:Llama-3.1-8B-Instruct, 3:Qwen2___5-72B-Instruct, 4:Qwen3-32B, 5:Qwen2___5-72B-Instruct-FP8)"
    echo "  MAX_TOKEN: positive integer"
    exit 1
fi

MODEL_ID=$1
MAX_TOKEN=$2

if [[ ! $MODEL_ID =~ ^[1-5]$ ]]; then
    echo "Error: MODEL_ID must be between 1 and 5"
    exit 1
fi

if [[ ! $MAX_TOKEN =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_TOKEN must be a positive integer"
    exit 1
fi

declare -A MODEL_MAP
MODEL_MAP[1]="Llama-3.1-70B-Instruct"
MODEL_MAP[2]="Llama-3.1-8B-Instruct"
MODEL_MAP[3]="Qwen2___5-72B-Instruct"
MODEL_MAP[4]="Qwen3-32B"
MODEL_MAP[5]="Qwen2___5-72B-Instruct-FP8"

MODEL_NAME=${MODEL_MAP[$MODEL_ID]}
DATE_TIME=$(date +"%m%d%H%M")
BS_VALUES=(1 2 4 8 16 32 48 64 96 128)

echo "Starting benchmark for model: $MODEL_NAME"
echo "MAX_TOKEN: $MAX_TOKEN"
echo "Date/Time: $DATE_TIME"
echo "----------------------------------------"

for BS in "${BS_VALUES[@]}"; do
    if [ $BS -gt $MAX_TOKEN ]; then
        echo "Skipping BS=$BS (BS > MAX_TOKEN)"
        continue
    fi
    
    LEN=$((MAX_TOKEN / BS))
    RESULT_DIR="./outputs/${MODEL_NAME}_maxtoken${MAX_TOKEN}_${DATE_TIME}"
    mkdir -p "$RESULT_DIR"
    
    CMD="vllm bench serve --model ./models/${MODEL_NAME}/ --num-prompts ${BS} --dataset-name random --random-input-len 2 --random-output-len ${LEN} --save-detailed --ignore-eos --request-rate inf --save-result --result-dir ${RESULT_DIR} --result-filename ${MODEL_NAME}-bs${BS}-len${LEN}.json --port 8801"
    
    echo "Executing: BS=${BS}, LEN=${LEN}"
    echo "Command: $CMD"
    echo "----------------------------------------"
    
    eval "$CMD"
    
    if [ $? -ne 0 ]; then
        echo "Error: Command failed for BS=${BS}, LEN=${LEN}"
        echo "Stopping execution."
        exit 1
    fi
    
    echo "Completed: BS=${BS}, LEN=${LEN}"
    echo "----------------------------------------"
done

echo "All benchmarks completed successfully!"
