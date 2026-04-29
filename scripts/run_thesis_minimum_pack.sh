#!/usr/bin/env bash
set -euo pipefail

# 最小可行实验包（按你老师给的优先级）：
# 1) E1/E2：PA（当前仓库原生）
# 2) 参数敏感性：window size、k
# 3) 预留：No-PA 与异常分数组成消融（需你已有对应开关/分支）
#
# 用法：
#   bash scripts/run_thesis_minimum_pack.sh HAI data/HAI 59 59
#   bash scripts/run_thesis_minimum_pack.sh MSL data/MSL 55 55

DATASET=${1:-HAI}
DATA_PATH=${2:-data/HAI}
INPUT_C=${3:-59}
OUTPUT_C=${4:-59}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
ANORM_RATIO=${ANORM_RATIO:-1.0}

TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT=${LOG_ROOT:-logs/thesis_pack_${DATASET}_${TS}}
mkdir -p "$LOG_ROOT"

run_train_test () {
  local tag=$1
  local dgr_mode=$2
  local win_size=$3
  local k=$4

  local ckpt="checkpoints/${tag}"
  local log_train="${LOG_ROOT}/${tag}_train.log"
  local log_test="${LOG_ROOT}/${tag}_test.log"

  echo "[RUN] ${tag} (mode=${dgr_mode}, win=${win_size}, k=${k})"
  python main.py \
    --anormly_ratio "$ANORM_RATIO" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --mode train \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --input_c "$INPUT_C" \
    --output_c "$OUTPUT_C" \
    --model_save_path "$ckpt" \
    --dgr_mode "$dgr_mode" \
    --win_size "$win_size" \
    --k "$k" 2>&1 | tee "$log_train"

  python main.py \
    --anormly_ratio "$ANORM_RATIO" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --mode test \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --input_c "$INPUT_C" \
    --output_c "$OUTPUT_C" \
    --model_save_path "$ckpt" \
    --dgr_mode "$dgr_mode" \
    --win_size "$win_size" \
    --k "$k" 2>&1 | tee "$log_test"
}

# -----------------------------
# 第一部分：E1 / E2（PA）
# E1 = dgr_mode=none；E2 = dgr_mode=dynamic
# -----------------------------
run_train_test "${DATASET}_E1_PA" "none" 100 3
run_train_test "${DATASET}_E2_PA" "dynamic" 100 3

# -----------------------------
# 第二部分：参数敏感性（先在 HAI 跑）
# window size: 50/100/150/200
# k: 1/3/5/10
# -----------------------------
for win in 50 100 150 200; do
  run_train_test "${DATASET}_E1_win${win}_k3" "none" "$win" 3
done

for kval in 1 3 5 10; do
  run_train_test "${DATASET}_E1_win100_k${kval}" "none" 100 "$kval"
done

# -----------------------------
# 第三部分（预留）：No-PA、异常分数消融、多随机种子
# 说明：当前主干代码默认在 test 中做 PA 校正。
# 若你有 no-pa/score-ablation 分支，把下面开关改为1并填对应命令。
# -----------------------------
RUN_EXTENDED=${RUN_EXTENDED:-0}
if [[ "$RUN_EXTENDED" == "1" ]]; then
  echo "[INFO] 进入扩展实验（No-PA/异常分数消融/seed稳定性）"
  echo "[TODO] 在你支持 No-PA 的分支运行：E1/E2 的 No-PA F1 或 AUC"
  echo "[TODO] 在你支持 score_mode 的分支运行：assoc_only / recon_only / assoc+recon"
  echo "[TODO] 稳定性：seed=42,43,44 复现 E1/E2，统计 mean±std"
fi

echo "[DONE] 实验完成。日志目录：${LOG_ROOT}"
echo "[TIP] 你可以先提交最小包：E1/E2 (PA) + 参数敏感性；再补 No-PA 与消融分支结果。"
