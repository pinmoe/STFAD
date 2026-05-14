#!/usr/bin/env bash
# =============================================================================
# tune_msl.sh  —  MSL 专项调参脚本（两阶段）
#
# 阶段一：对已有 checkpoint 仅改 score_mode / score_alpha 重测（不重训练）
#         → 约 10-20 分钟跑完
# 阶段二：针对性重训 4 个新配置，专攻 Recall 恢复问题
#         → 约 60-90 分钟
#
# 使用方式（在远程服务器上执行）：
#   bash scripts/tune_msl.sh 2>&1 | tee logs/tune_msl_$(date '+%Y%m%d_%H%M').log
#
# 结果汇总：
#   grep "F-score" logs/tune_msl_*.log
# =============================================================================

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate adp 2>/dev/null || true

DATASET="MSL"
DATA_PATH="data/MSL"
INPUT_C=55
OUTPUT_C=55
RATIO=1.0       # anormly_ratio=1.0，与现有 E1-E4 完全一致
EPOCHS=10
BATCH=256
WIN=100
K=3
LR=0.0001

LOG_DIR="logs/tune_msl_$(date '+%Y%m%d_%H%M')"
mkdir -p "${LOG_DIR}"
echo "==== MSL 调参日志目录: ${LOG_DIR} ===="

# --------------------------------------------------------------------------- #
# 工具函数：test-only，复用已有 checkpoint
# --------------------------------------------------------------------------- #
retest() {
    local TAG="$1"
    local CKPT="$2"          # checkpoints/xxx
    local DGR_MODE="$3"
    local FUSION="$4"
    local ALPHA="$5"
    local ALPHA_LEARN="$6"
    local SCORE_MODE="$7"
    local SCORE_ALPHA="$8"
    local SMOOTH="${9:-1}"
    local LOG="${LOG_DIR}/${TAG}.log"

    echo "------------------------------------------------------------"
    echo " RETEST  ${TAG}  ckpt=${CKPT}  score=${SCORE_MODE}"
    echo "------------------------------------------------------------"
    python main.py \
        --mode test \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" \
        --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --dgr_mode "${DGR_MODE}" \
        --prior_fusion "${FUSION}" \
        --prior_alpha "${ALPHA}" \
        --prior_alpha_learnable "${ALPHA_LEARN}" \
        --score_mode "${SCORE_MODE}" \
        --score_alpha "${SCORE_ALPHA}" \
        --score_smooth_k "${SMOOTH}" \
        --model_save_path "${CKPT}" \
        --num_epochs "${EPOCHS}" \
        --batch_size "${BATCH}" \
        --win_size "${WIN}" \
        --k "${K}" \
        --lr "${LR}" \
        2>&1 | tee "${LOG}"
    echo " → ${LOG}"
    echo ""
}

# --------------------------------------------------------------------------- #
# 工具函数：完整 train + test
# --------------------------------------------------------------------------- #
run_full() {
    local TAG="$1"
    local DGR_MODE="$2"
    local FUSION="$3"
    local ALPHA="$4"
    local ALPHA_LEARN="$5"
    local K_VAL="$6"
    local EPOCHS_VAL="$7"
    local RATIO_VAL="$8"
    local SCORE_MODE="${9:-combined}"
    local CKPT="checkpoints/${TAG}_${DATASET}"
    local LOG="${LOG_DIR}/${TAG}.log"

    echo "============================================================"
    echo " TRAIN+TEST  ${TAG}  dgr=${DGR_MODE}  fusion=${FUSION}"
    echo "  alpha=${ALPHA}  k=${K_VAL}  epochs=${EPOCHS_VAL}  ratio=${RATIO_VAL}"
    echo "============================================================"

    python main.py \
        --mode train \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" \
        --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO_VAL}" \
        --dgr_mode "${DGR_MODE}" \
        --prior_fusion "${FUSION}" \
        --prior_alpha "${ALPHA}" \
        --prior_alpha_learnable "${ALPHA_LEARN}" \
        --score_mode "${SCORE_MODE}" \
        --model_save_path "${CKPT}" \
        --num_epochs "${EPOCHS_VAL}" \
        --batch_size "${BATCH}" \
        --win_size "${WIN}" \
        --k "${K_VAL}" \
        --lr "${LR}" \
        2>&1 | tee -a "${LOG}"

    python main.py \
        --mode test \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" \
        --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO_VAL}" \
        --dgr_mode "${DGR_MODE}" \
        --prior_fusion "${FUSION}" \
        --prior_alpha "${ALPHA}" \
        --prior_alpha_learnable "${ALPHA_LEARN}" \
        --score_mode "${SCORE_MODE}" \
        --model_save_path "${CKPT}" \
        --num_epochs "${EPOCHS_VAL}" \
        --batch_size "${BATCH}" \
        --win_size "${WIN}" \
        --k "${K_VAL}" \
        --lr "${LR}" \
        2>&1 | tee -a "${LOG}"

    echo " → ${LOG}"
    echo ""
}

# ============================================================================
# ■ 阶段一：已跳过
#   原因：旧 checkpoint 用 head_dim=8 训练，当前代码改为
#         head_dim = in_channels // n_heads（MSL: 55//8=6），结构不兼容。
#         直接进入阶段二完整重训。
# ============================================================================
echo ""
echo "████  阶段一跳过（checkpoint 与当前代码结构不兼容）  ████"
echo ""

# ============================================================================
# ■ 阶段二：针对性重训 4 个新配置
#
# 分析：DGR 变体在 MSL 上主要问题是 Recall 下降（Precision 反而略升）。
# 以下配置均从"减少 DGR 先验对序列关联的干扰"角度设计：
#
# BM1 — blend alpha=0.9（先验 90% 高斯 + 10% 动态DGR），k=3
#        逻辑：几乎保留高斯先验，仅小剂量注入结构约束
#
# BM2 — blend alpha=0.7，k=1（KL 权重从 3 降到 1）
#        逻辑：重建损失主导，KL 分支权重减半，降低先验对 Recall 的压制
#
# BM3 — learnable alpha，k=1，epochs=15
#        逻辑：让模型自己找最优融合比，同时降低 KL 权重
#
# BM4 — static DGR + blend alpha=0.85，k=3
#        逻辑：静态先验在推理时不受异常输入污染，+大权重高斯保证 Recall
#
# 预计耗时：~60-80 分钟
# ============================================================================
echo ""
echo "████  阶段二：针对性重训  ████"
echo ""

# BM1: 90% Gaussian + 10% DGR dynamic, k=3
run_full "BM1" "dynamic"    "blend"  0.9  "false" 3  10  1.0  "combined"

# BM2: 70% Gaussian + 30% DGR dynamic, k=1 (降低KL权重)
run_full "BM2" "dynamic"    "blend"  0.7  "false" 1  10  1.0  "combined"

# BM3: learnable alpha, k=1, 15 epochs
run_full "BM3" "dynamic"    "blend"  0.5  "true"  1  15  1.0  "combined"

# BM4: static DGR + 85% Gaussian, k=3
run_full "BM4" "static"     "blend"  0.85 "false" 3  10  1.0  "combined"

# ============================================================================
# 结果汇总
# ============================================================================
echo ""
echo "████████  MSL 调参全部完成  ████████"
echo ""
echo "——— 阶段二（重训）结果 ———"
grep "F-score" "${LOG_DIR}"/BM*.log | \
    sed 's|.*BM||; s|\.log:| |' | \
    sort -t= -k2 -rn

echo ""
echo "——— E1 基线对照 ———"
echo "E1_MSL : F-score : 0.9143  (Prec=0.9023, Recall=0.9266)"
