#!/usr/bin/env bash
set -euo pipefail

# Minimal AutoDL experiment pack.
# Usage:
#   bash scripts/run_thesis_minimum_pack.sh HAI data/HAI 59 59
#   bash scripts/run_thesis_minimum_pack.sh MSL data/MSL 55 55

DATASET=${1:-HAI}
DATA_PATH=${2:-data/HAI}
INPUT_C=${3:-59}
OUTPUT_C=${4:-59}

EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-64}
ANORM_RATIO=${ANORM_RATIO:-1.0}
RESULT_DIR=${RESULT_DIR:-results/paper_main}
THRESHOLD_MODE=${THRESHOLD_MODE:-val_percentile}
THRESHOLD_PERCENTILE=${THRESHOLD_PERCENTILE:-95}
RUN_SWEEP=${RUN_SWEEP:-1}
RUN_EXTENDED=${RUN_EXTENDED:-0}

DMODEL=${DMODEL:-}
DROPOUT=${DROPOUT:-}
TEMPERATURE=${TEMPERATURE:-}
SCORE_MODE=${SCORE_MODE:-combined}

TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT=${LOG_ROOT:-logs/thesis_pack_${DATASET}_${TS}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints/thesis_pack_${DATASET}_${TS}}
mkdir -p "$LOG_ROOT" "$CHECKPOINT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

die_oom_help() {
  local tag=$1
  cat <<MSG
[OOM] ${tag} failed, likely due to insufficient GPU memory.
Try one of:
  TEST_BATCH_SIZE=32
  BATCH_SIZE=64
  RUN_SWEEP=0
Example:
  BATCH_SIZE=64 TEST_BATCH_SIZE=32 RUN_SWEEP=0 bash scripts/run_thesis_minimum_pack.sh ${DATASET} ${DATA_PATH} ${INPUT_C} ${OUTPUT_C}
MSG
}

run_train_test() {
  local tag=$1
  local dgr_mode=$2
  local win_size=$3
  local k_value=$4

  local ckpt="${CHECKPOINT_ROOT}/${tag}"
  local log_train="${LOG_ROOT}/${tag}_train.log"
  local log_test="${LOG_ROOT}/${tag}_test.log"

  local extra_args=()
  [[ -n "$DMODEL" ]] && extra_args+=(--d_model "$DMODEL")
  [[ -n "$DROPOUT" ]] && extra_args+=(--dropout "$DROPOUT")
  [[ -n "$TEMPERATURE" ]] && extra_args+=(--temperature "$TEMPERATURE")
  [[ -n "$SCORE_MODE" ]] && extra_args+=(--score_mode "$SCORE_MODE")

  echo "[RUN] ${tag} mode=${dgr_mode} win=${win_size} k=${k_value}"

  if ! python main.py \
    --anormly_ratio "$ANORM_RATIO" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --mode train \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --input_c "$INPUT_C" \
    --output_c "$OUTPUT_C" \
    --model_save_path "$ckpt" \
    --result_dir "$RESULT_DIR" \
    --experiment_name "$tag" \
    --threshold_mode "$THRESHOLD_MODE" \
    --threshold_percentile "$THRESHOLD_PERCENTILE" \
    --dgr_mode "$dgr_mode" \
    --win_size "$win_size" \
    --k "$k_value" \
    "${extra_args[@]}" 2>&1 | tee "$log_train"; then
    die_oom_help "$tag/train"
    exit 1
  fi

  if ! python main.py \
    --anormly_ratio "$ANORM_RATIO" \
    --num_epochs "$EPOCHS" \
    --batch_size "$TEST_BATCH_SIZE" \
    --mode test \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --input_c "$INPUT_C" \
    --output_c "$OUTPUT_C" \
    --model_save_path "$ckpt" \
    --result_dir "$RESULT_DIR" \
    --experiment_name "$tag" \
    --threshold_mode "$THRESHOLD_MODE" \
    --threshold_percentile "$THRESHOLD_PERCENTILE" \
    --dgr_mode "$dgr_mode" \
    --win_size "$win_size" \
    --k "$k_value" \
    "${extra_args[@]}" 2>&1 | tee "$log_test"; then
    die_oom_help "$tag/test"
    exit 1
  fi
}

run_train_test "${DATASET}_E1_PA" "none" 100 3
run_train_test "${DATASET}_E2_PA" "dynamic" 100 3
run_train_test "${DATASET}_E3_PA" "multiscale" 100 3
run_train_test "${DATASET}_E4_PA" "static" 100 3

if [[ "$RUN_SWEEP" == "1" ]]; then
  for win in 50 100 150 200; do
    run_train_test "${DATASET}_E1_win${win}_k3" "none" "$win" 3
  done

  for k_value in 1 3 5 10; do
    run_train_test "${DATASET}_E1_win100_k${k_value}" "none" 100 "$k_value"
  done
fi

if [[ "$RUN_EXTENDED" == "1" ]]; then
  echo "[INFO] RUN_EXTENDED=1 is reserved for extra no-PA and score-ablation runs."
fi

echo "[DONE] logs: ${LOG_ROOT}"
echo "[DONE] checkpoints: ${CHECKPOINT_ROOT}"
echo "[DONE] structured results: ${RESULT_DIR}"

AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}
SHUTDOWN_CMD=${SHUTDOWN_CMD:-poweroff}
if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
  echo "[INFO] AUTO_SHUTDOWN=1, running: ${SHUTDOWN_CMD}"
  bash -lc "${SHUTDOWN_CMD}" || echo "[WARN] shutdown command failed"
fi
