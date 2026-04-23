#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <MODEL_ID> <MAX_INPUT_LEN>"
    echo "  MODEL_ID: 1-5 (1:Llama-3.1-70B-Instruct, 2:Llama-3.1-8B-Instruct, 3:Qwen2___5-72B-Instruct, 4:Qwen3-32B, 5:Qwen2___5-72B-Instruct-FP8)"
    echo "  MAX_INPUT_LEN: positive integer"
    exit 1
fi

MODEL_ID=$1
MAX_INPUT_LEN=$2

if [[ ! $MODEL_ID =~ ^[1-5]$ ]]; then
    echo "Error: MODEL_ID must be between 1 and 5"
    exit 1
fi

if [[ ! $MAX_INPUT_LEN =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_INPUT_LEN must be a positive integer"
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
INPUT_LENS=()

if [ $MAX_INPUT_LEN -ge 32 ]; then
    val=512
    while [ $val -le $MAX_INPUT_LEN ]; do
        INPUT_LENS+=($val)
        val=$((val + 512))
    done
fi

echo "Starting benchmark for model: $MODEL_NAME"
echo "Max Input Length: $MAX_INPUT_LEN"
echo "Output Length: 1 (fixed)"
echo "Date/Time: $DATE_TIME"
echo "----------------------------------------"

NUM_PROMPTS=1
OUTPUT_LEN=1

for LEN in "${INPUT_LENS[@]}"; do
    RESULT_DIR="./outputs/${MODEL_NAME}_prefill_maxtoken${MAX_INPUT_LEN}_${DATE_TIME}"
    mkdir -p "$RESULT_DIR"

    CMD="vllm bench serve \
        --model ./models/${MODEL_NAME}/ \
        --num-prompts ${NUM_PROMPTS} \
        --dataset-name random \
        --random-input-len ${LEN} \
        --random-output-len ${OUTPUT_LEN} \
        --save-detailed \
        --ignore-eos \
        --request-rate inf \
        --save-result \
        --result-dir ${RESULT_DIR} \
        --result-filename ${MODEL_NAME}-len${LEN}.json \
        --port 8801"

    echo "Executing: input_len=${LEN}, output_len=${OUTPUT_LEN}"
    echo "Command: $CMD"
    echo "----------------------------------------"

    eval "$CMD"

    if [ $? -ne 0 ]; then
        echo "Error: Command failed for input_len=${LEN}"
        echo "Stopping execution."
        exit 1
    fi

    echo "Completed: input_len=${LEN}"
    echo "----------------------------------------"
done

echo "All benchmarks completed successfully!"
