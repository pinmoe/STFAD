#!/usr/bin/env bash
set -euo pipefail

# 最小可行实验包（按你老师给的优先级）：
# 1) E1/E2：PA（当前仓库原生）
# 2) 参数敏感性：window size、k
# 3) 预留：No-PA 与异常分数组成消融（需你已有对应开关/分支）

DATASET=${1:-HAI}
DATA_PATH=${2:-data/HAI}
INPUT_C=${3:-59}
OUTPUT_C=${4:-59}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-64}
ANORM_RATIO=${ANORM_RATIO:-1.0}

# === 方案1+3 新增参数（可通过环境变量覆盖 main.py 的 dataset-aware 默认值） ===
DMODEL=${DMODEL:-}      # 留空则使用 main.py 中 _DATASET_OVERRIDES 的默认值
DROPOUT=${DROPOUT:-}    # 留空则使用 main.py 的 dataset-aware 默认值
TEMPERATURE=${TEMPERATURE:-}  # 留空则使用 main.py 默认值 50
SCORE_MODE=${SCORE_MODE:-assoc+recon}  # 异常分数组合模式

TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT=${LOG_ROOT:-logs/thesis_pack_${DATASET}_${TS}}
mkdir -p "$LOG_ROOT" "$LOG_ROOT/checkpoints"

# 减少显存碎片导致的 OOM 风险
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

die_oom_help () {
  local tag=$1
  cat <<MSG
[OOM] 运行 ${tag} 失败（显存不足）。
建议重试（按顺序）：
  1) TEST_BATCH_SIZE=32
  2) BATCH_SIZE=64
  3) 先仅跑 E1/E2：RUN_SWEEP=0
示例：
  BATCH_SIZE=64 TEST_BATCH_SIZE=32 RUN_SWEEP=0 bash scripts/run_thesis_minimum_pack.sh ${DATASET} ${DATA_PATH} ${INPUT_C} ${OUTPUT_C}
MSG
}

run_train_test () {
  local tag=$1
  local dgr_mode=$2
  local win_size=$3
  local k=$4

  local ckpt="${LOG_ROOT}/checkpoints/${tag}"
  local log_train="${LOG_ROOT}/${tag}_train.log"
  local log_test="${LOG_ROOT}/${tag}_test.log"

  # 构建可选额外参数
  local extra_args=()
  [[ -n "$DMODEL" ]]       && extra_args+=(--d_model "$DMODEL")
  [[ -n "$DROPOUT" ]]      && extra_args+=(--dropout "$DROPOUT")
  [[ -n "$TEMPERATURE" ]]  && extra_args+=(--temperature "$TEMPERATURE")
  [[ -n "$SCORE_MODE" ]]   && extra_args+=(--score_mode "$SCORE_MODE")

  echo "[RUN] ${tag} (mode=${dgr_mode}, win=${win_size}, k=${k})" | tee -a "$log_train" "$log_test"
  echo "       extra: d_model=${DMODEL:-auto} dropout=${DROPOUT:-auto} temp=${TEMPERATURE:-auto} score=${SCORE_MODE}"

  if ! python main.py \
    ${ANORM_RATIO:+--anormly_ratio "$ANORM_RATIO"} \
    ${EPOCHS:+--num_epochs "$EPOCHS"} \
    --batch_size "$BATCH_SIZE" \
    --mode train \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --input_c "$INPUT_C" \
    --output_c "$OUTPUT_C" \
    --model_save_path "$ckpt" \
    --dgr_mode "$dgr_mode" \
    --win_size "$win_size" \
    --k "$k" \
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
    --dgr_mode "$dgr_mode" \
    --win_size "$win_size" \
    --k "$k" \
    "${extra_args[@]}" 2>&1 | tee "$log_test"; then
    die_oom_help "$tag/test"
    exit 1
  fi
}

RUN_SWEEP=${RUN_SWEEP:-1}

# E1 / E2 / E3 / E4（PA）
run_train_test "${DATASET}_E1_PA" "none" 100 3
run_train_test "${DATASET}_E2_PA" "dynamic" 100 3
run_train_test "${DATASET}_E3_PA" "multiscale" 100 3
run_train_test "${DATASET}_E4_PA" "static" 100 3

if [[ "$RUN_SWEEP" == "1" ]]; then
  for win in 50 100 150 200; do
    run_train_test "${DATASET}_E1_win${win}_k3" "none" "$win" 3
  done

  for kval in 1 3 5 10; do
    run_train_test "${DATASET}_E1_win100_k${kval}" "none" 100 "$kval"
  done
fi

RUN_EXTENDED=${RUN_EXTENDED:-0}
if [[ "$RUN_EXTENDED" == "1" ]]; then
  echo "[INFO] 进入扩展实验（No-PA/异常分数消融/seed稳定性）"
  echo "[TODO] 在你支持 No-PA 的分支运行：E1/E2 的 No-PA F1 或 AUC"
  echo "[TODO] 在你支持 score_mode 的分支运行：assoc_only / recon_only / assoc+recon"
  echo "[TODO] 稳定性：seed=42,43,44 复现 E1/E2，统计 mean±std"
fi

echo "[DONE] 实验完成。日志目录：${LOG_ROOT}"

AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}
SHUTDOWN_CMD=${SHUTDOWN_CMD:-"poweroff"}
if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
  echo "[INFO] AUTO_SHUTDOWN=1，准备执行关机命令: ${SHUTDOWN_CMD}"
  bash -lc "${SHUTDOWN_CMD}" || echo "[WARN] 关机命令执行失败，请在平台面板手动关机。"
fi