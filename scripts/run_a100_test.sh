#!/usr/bin/env bash 

set -euo pipefail

export WANDB_PROJECT="text2sql-spider-a100-tests"
export HF_HOME="/data/huggingface-cache"

run_experiment(){
    local MODEL="$1"
    local TRAINING_MODE="$2"
    local LEARNING_RATE="$3"

    TRANSFORMERS_VERBOSITY=error uv run python train_grpo.py \
    --train \
    --model "$MODEL" \
    --training-mode "$TRAINING_MODE" \
    --learning-rate "$LEARNING_RATE" \
    --report-to wandb \
    --split train \
    --limit 200 \
    --max-steps 200 \
    --batch-size 4 \
    --num-generations 4 \
    --max-completion-length 128 \
    --temperature 1.0 \
    --seed 13 \
    --logging-steps 1 \
    --debug \
    --save-steps 50 \
    --save-total-limit 2 \
    --output-dir "outputs/training-runs"
}

evaluate_lora() {
    local MODEL="$1"
    local ADAPTER="$2"

    TRANSFORMERS_VERBOSITY=error uv run python grpo_eval.py \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --split dev \
    --limit 1034 \
    --batch-size 4 \
    --num-generations 4 \
    --max-completion-length 128 \
    --temperature 1.0 \
    --seed 13 \
    --output outputs/grpo_evals
}

evaluate_full() {
    local CHECKPOINT="$1"

    TRANSFORMERS_VERBOSITY=error uv run python grpo_eval.py \
    --model "$CHECKPOINT" \
    --split dev \
    --limit 1034 \
    --batch-size 4 \
    --num-generations 4 \
    --max-completion-length 128 \
    --temperature 1.0 \
    --seed 13 \
    --output outputs/grpo_evals
}

# Completed 0.5B sweep:
# run_experiment \
#     "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
#     "full" \
#     "1e-6"

# run_experiment \
#     "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
#     "lora" \
#     "1e-6"

# run_experiment \
#     "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
#     "lora" \
#     "1e-5"

BEST_LORA_LR="1e-5"

run_experiment \
    "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
    "full" \
    "1e-6"

run_experiment \
    "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
    "lora" \
    "$BEST_LORA_LR"
