#!/bin/bash

echo_usage() {
    echo "Usage: $0 <device> <cook_level> [model]"
    echo "Example: bash server.sh 5 10"
    echo "Example: bash server.sh '0,1,2' 11 qwen3-8b"
    echo -e "Available cook_level:\t 00 (no cook), 10 (enable cook), 11 (enable cook + hybrid free queue)"
    echo -e "Available models:\t qwen3-8b (default), llama, llama70b, qwen, mistral"
}

if [ "$#" -lt 2 ]; then
    echo_usage
    exit 1
fi

DEVICE="$1"
COOK_LEVEL="$2"
MODEL_NAME="${3:-qwen3-8b}"

FIRST_GPU=$(echo "$DEVICE" | cut -d',' -f1)
if ! [[ "$FIRST_GPU" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid device specification: $DEVICE"
    exit 1
fi
PORT=$((8000 + FIRST_GPU))

declare -A MODEL_PATHS=(
    ["llama"]="./models/Meta-Llama-3.1-8B-Instruct"
    ["llama70b"]="./models/Meta-Llama-3.1-70B-Instruct"
    ["qwen"]="./models/Qwen2.5-14B-Instruct"
    ["mistral"]="./models/Mistral-7B-Instruct-v0.1"
    ["qwen3-8b"]="./models/Qwen3-8B/"
)

MODEL="${MODEL_PATHS[$MODEL_NAME]}"
if [ -z "$MODEL" ]; then
    echo "Invalid model name: $MODEL_NAME"
    echo "Available models: ${!MODEL_PATHS[@]}"
    exit 1
fi

BASE_COMMAND="vllm serve \"$MODEL\" --config ./config/server_config.yaml --port $PORT"

case "$COOK_LEVEL" in
    00)
        CMD="CUDA_VISIBLE_DEVICES=$DEVICE $BASE_COMMAND --no-enable-cook --no-enable-hybrid-free-queue"
        ;;
    10)
        CMD="CUDA_VISIBLE_DEVICES=$DEVICE $BASE_COMMAND --enable-cook --no-enable-hybrid-free-queue"
        ;;
    11)
        CMD="CUDA_VISIBLE_DEVICES=$DEVICE $BASE_COMMAND --enable-cook --enable-hybrid-free-queue"
        ;;
    *)
        echo "Invalid cook_level: $COOK_LEVEL (must be 00, 10, or 11)"
        exit 1
        ;;
esac

echo "Running: $CMD"
eval $CMD
