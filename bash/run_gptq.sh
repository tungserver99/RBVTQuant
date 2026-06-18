#!/usr/bin/env bash
set -euo pipefail

# Quantized-only GPTQ runs.
# Float baselines are launched separately via bash/run_float.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rbvtquant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
GPTQ_DEVICE="${GPTQ_DEVICE:-cuda:0}"

MODELS=(
  "meta-llama/Llama-3.1-8B"
  "mistralai/Mistral-7B-v0.3"
  "Qwen/Qwen2.5-7B"
)

QUANTIZERS=(
  "nf4"
  "nf3"
)

LM_EVAL_TASKS=(
  "arc_easy"
  "arc_challenge"
  "hellaswag"
  "piqa"
  "winogrande"
  "boolq"
  "rte"
  "openbookqa"
  "lambada_openai"
  "mmlu"
  "gsm8k"
)

slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s//./_}"
  s="${s//-/_}"
  echo "$s"
}

for model in "${MODELS[@]}"; do
  model_slug="$(slugify "$model")"
  echo "=== GPTQ quantized runs for ${model} ==="
  WANDB_ARGS=()
  if [ "$USE_WANDB" = "1" ]; then
    WANDB_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
    if [ -n "$WANDB_ENTITY" ]; then
      WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi

  for quantizer in "${QUANTIZERS[@]}"; do
    python main.py \
      --model-path "$model" \
      --device "$GPTQ_DEVICE" \
      --method gptq \
      --quantizer "$quantizer" \
      --output-dir "./outputs/${model_slug}_gptq_${quantizer}" \
      --calib-dataset c4 \
      --max-length 2048 \
      --eval-max-length 2048 \
      --include-lm-eval \
      --lm-eval-tasks "${LM_EVAL_TASKS[@]}" \
      "${WANDB_ARGS[@]}"
  done
done
