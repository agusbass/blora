#!/bin/bash
# run_suite.sh — Production-grade PEFT eval orchestrator (1x L4, disk-safe, resumable)
# =============================================================================
# For each (method, seed): SFT-train -> bake to standard checkpoint -> evaluate
# with lm-eval-harness on the Open LLM Leaderboard v1 suite -> record JSON ->
# delete checkpoint (only one ~16GB checkpoint alive at a time, fits 100GB disk).
# Re-running skips finished (method,seed) cells -> safe to resume after preempt.
#
# Usage:  bash run_suite.sh
set -u
cd ~/blora
export HF_TOKEN=$(cat ~/.hf_token)
export TOKENIZERS_PARALLELISM=false

RESULTS=~/results
CKPT=/tmp/ckpt_active
mkdir -p "$RESULTS"
MAX_STEPS=${MAX_STEPS:-1000}
N_EX=${N_EX:-16000}

# (method seed) matrix: 3-way ablation x 3 seeds + SOTA context x 1 seed
MATRIX=(
  "bcms 0"   "bcms 1"   "bcms 2"
  "lora16 0" "lora16 1" "lora16 2"
  "blora 0"  "blora 1"  "blora 2"
  "lora 0"   "dora 0"   "pissa 0"
)

# OLLv1: task:fewshot
TASKS=( "arc_challenge:25" "hellaswag:10" "mmlu:5" "winogrande:5" "gsm8k:5" "truthfulqa_mc2:0" )

run_eval () {  # $1=ckpt_dir  $2=out_prefix
  for t in "${TASKS[@]}"; do
    task="${t%%:*}"; fs="${t##*:}"
    done_marker="${2}__${task}.json"
    [ -f "$done_marker" ] && { echo "  [skip] $task"; continue; }
    echo "  [eval] $task (${fs}-shot)"
    python3 -m lm_eval --model hf \
      --model_args "pretrained=${1},dtype=bfloat16" \
      --tasks "$task" --num_fewshot "$fs" --batch_size auto \
      --device cuda:0 --output_path "${2}__${task}.raw" 2>&1 \
      | grep -iE "^\|${task}|Value|ERROR" | tail -4
    # lm-eval writes to a nested dir; flatten newest json to our marker
    latest=$(ls -t ${2}__${task}.raw/**/*.json 2>/dev/null | head -1)
    [ -n "$latest" ] && cp "$latest" "$done_marker"
  done
}

for cell in "${MATRIX[@]}"; do
  set -- $cell; method=$1; seed=$2
  prefix="${RESULTS}/${method}_s${seed}"
  if [ -f "${prefix}.DONE" ]; then echo "=== [done] ${method} seed ${seed} ==="; continue; fi
  echo ""
  echo "############################################################"
  echo "### ${method} seed ${seed}   ($(date -u +%H:%M:%S) UTC)"
  echo "############################################################"
  rm -rf "$CKPT"
  python3 scripts/sft_train.py --method "$method" --seed "$seed" \
    --out-dir "$CKPT" --max-steps "$MAX_STEPS" --n-examples "$N_EX" 2>&1 \
    | grep -vE "examples/s|it/s|%\||Map:|Generating|Downloading" | tail -12
  if [ ! -f "$CKPT/config.json" ]; then echo "  TRAIN FAILED -> skip"; continue; fi
  cp "$CKPT/train_meta.json" "${prefix}_train_meta.json" 2>/dev/null
  run_eval "$CKPT" "$prefix"
  touch "${prefix}.DONE"
  rm -rf "$CKPT"
  echo "### ${method} seed ${seed} COMPLETE"
done

echo ""
echo "ALL CELLS DONE -> $RESULTS"
