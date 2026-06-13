#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh
# E1–E5 + B1–B3 全量实验，完整 train + test，覆盖 SKAB / MSL / SMAP 三个数据集。
#
# 使用方式（AutoDL + tmux）：
#   tmux new -s exp          # 新建 tmux 会话
#   bash scripts/run_all_experiments.sh 2>&1 | tee run_main.log
#   # Ctrl+B D 分离会话，关闭 SSH 不影响运行
#   # tmux attach -t exp     重新连接
#
# 日志自动保存到 logs/ 目录，checkpoint 保存到 checkpoints/{EXP}_{数据集}/
# 若某单个实验失败，脚本会记录错误并继续运行后续实验（不会中途退出）。
#
# 新增指标（自动输出，无需额外参数）：
#   [逐点(无PA)]  Precision / Recall / F1  —— 严格点级指标
#   [AUPRC]       不依赖阈值，越高越好
#   [事件级(PA)]  以连续异常段为粒度的 Precision / Recall / F1
#   注：SKAB/MSL/SMAP 为滑窗数据集，[窗口级AUPRC] 打印时请忽略（无语义意义）。
# =============================================================================

FAILED_EXPERIMENTS=()

# AutoDL / conda 环境激活（若已在 conda env 中运行可注释掉此行）
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate adp 2>/dev/null || true

# 减少显存碎片导致的 OOM 风险
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG_DIR="logs/logs_$(date '+%Y%m%d_%H%M')"
mkdir -p "${LOG_DIR}"
echo "日志目录：${LOG_DIR}"

# --------------------------------------------------------------------------- #
# 工具函数：train + test 一对（基础版）
# 参数: $1=实验名  $2=数据集  $3=data_path  $4=input_c  $5=output_c
#       $6=anormly_ratio  $7=dgr_mode
#       $8=dgr_feature_mode（可选，默认 diff）
#       $9=lambda_diff（可选，默认 0.0）
#       $10=diff_beta（可选，默认 0.0）
#       $11=use_memory_bank（可选，默认 false）
# --------------------------------------------------------------------------- #
run_one() {
    local EXP="$1"
    local DATASET="$2"
    local DATA_PATH="$3"
    local INPUT_C="$4"
    local OUTPUT_C="$5"
    local RATIO="$6"
    local DGR_MODE="$7"
    local DGR_FEAT_MODE="${8:-diff}"
    local LAMBDA_DIFF="${9:-0.0}"
    local DIFF_BETA="${10:-0.0}"
    local USE_MB="${11:-false}"
    local SAVE_PATH="checkpoints/${EXP}_${DATASET}"
    local LOG_FILE="${LOG_DIR}/${EXP}_${DATASET}.log"

    echo "========================================================"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] START  ${EXP} / ${DATASET}"
    echo "========================================================"

    # ----- TRAIN -----
    python main.py \
        --mode train \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" \
        --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --dgr_mode "${DGR_MODE}" \
        --dgr_feature_mode "${DGR_FEAT_MODE}" \
        --lambda_diff "${LAMBDA_DIFF}" \
        --model_save_path "${SAVE_PATH}" \
        --num_epochs 10 \
        --batch_size 256 \
        --win_size 100 \
        --k 3 \
        --lr 0.0001 \
        2>&1 | tee -a "${LOG_FILE}"

    # ----- TEST -----
    python main.py \
        --mode test \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" \
        --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --dgr_mode "${DGR_MODE}" \
        --dgr_feature_mode "${DGR_FEAT_MODE}" \
        --diff_beta "${DIFF_BETA}" \
        --use_memory_bank "${USE_MB}" \
        --model_save_path "${SAVE_PATH}" \
        --num_epochs 10 \
        --batch_size 256 \
        --win_size 100 \
        --k 3 \
        --lr 0.0001 \
        2>&1 | tee -a "${LOG_FILE}"

    local EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "[错误] ${EXP}/${DATASET} 失败（退出码 ${EXIT_CODE}），继续下一个实验。" | tee -a "${LOG_FILE}"
        FAILED_EXPERIMENTS+=("${EXP}_${DATASET}")
    fi

    echo ""
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] DONE   ${EXP} / ${DATASET}  →  ${LOG_FILE}"
    echo ""
}

# --------------------------------------------------------------------------- #
# 工具函数：带融合策略的 train + test（用于 B1-B3）
# 参数: $1=实验名  $2=数据集  $3=data_path  $4=input_c  $5=output_c
#       $6=anormly_ratio  $7=dgr_mode  $8=prior_fusion  $9=prior_alpha
#       $10=prior_alpha_learnable（true/false）
#       $11=dgr_feature_mode（可选，默认 diff）
#       $12=lambda_diff（可选，默认 0.0）
#       $13=diff_beta（可选，默认 0.0）
#       $14=use_memory_bank（可选，默认 false）
# --------------------------------------------------------------------------- #
run_blend() {
    local EXP="$1"
    local DATASET="$2"
    local DATA_PATH="$3"
    local INPUT_C="$4"
    local OUTPUT_C="$5"
    local RATIO="$6"
    local DGR_MODE="$7"
    local FUSION="$8"
    local ALPHA="$9"
    local ALPHA_LEARNABLE="${10}"
    local DGR_FEAT_MODE="${11:-diff}"
    local LAMBDA_DIFF="${12:-0.0}"
    local DIFF_BETA="${13:-0.0}"
    local USE_MB="${14:-false}"
    local SAVE_PATH="checkpoints/${EXP}_${DATASET}"
    local LOG_FILE="${LOG_DIR}/${EXP}_${DATASET}.log"

    echo "========================================================"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] START  ${EXP} / ${DATASET}"
    echo "========================================================"

    # ----- TRAIN -----
    python main.py \
        --mode train \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" \
        --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --dgr_mode "${DGR_MODE}" \
        --prior_fusion "${FUSION}" \
        --prior_alpha "${ALPHA}" \
        --prior_alpha_learnable "${ALPHA_LEARNABLE}" \
        --dgr_feature_mode "${DGR_FEAT_MODE}" \
        --lambda_diff "${LAMBDA_DIFF}" \
        --model_save_path "${SAVE_PATH}" \
        --num_epochs 10 \
        --batch_size 256 \
        --win_size 100 \
        --k 3 \
        --lr 0.0001 \
        2>&1 | tee -a "${LOG_FILE}" || true

    # ----- TEST -----
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
        --prior_alpha_learnable "${ALPHA_LEARNABLE}" \
        --dgr_feature_mode "${DGR_FEAT_MODE}" \
        --diff_beta "${DIFF_BETA}" \
        --use_memory_bank "${USE_MB}" \
        --model_save_path "${SAVE_PATH}" \
        --num_epochs 10 \
        --batch_size 256 \
        --win_size 100 \
        --k 3 \
        --lr 0.0001 \
        2>&1 | tee -a "${LOG_FILE}"

    local EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "[错误] ${EXP}/${DATASET} 失败（退出码 ${EXIT_CODE}），继续下一个实验。" | tee -a "${LOG_FILE}"
        FAILED_EXPERIMENTS+=("${EXP}_${DATASET}")
    fi

    echo ""
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] DONE   ${EXP} / ${DATASET}  →  ${LOG_FILE}"
    echo ""
}

# =============================================================================
# E1：原始高斯先验（baseline），dgr_mode=none
# =============================================================================
echo "############################################################"
echo "#  E1  baseline（高斯先验，无 DGR）"
echo "############################################################"

run_one "E1" "SKAB" "data/SKAB" 8 8 1.0 "none"
run_one "E1" "MSL"  "data/MSL"  55 55 1.0 "none"
run_one "E1" "SMAP" "data/SMAP" 25 25 1.0 "none"

# =============================================================================
# E2：动态 DGR 先验，dgr_mode=dynamic
# =============================================================================
echo "############################################################"
echo "#  E2  动态 DGR 先验（DGRPrior）"
echo "############################################################"

run_one "E2" "SKAB" "data/SKAB" 8 8 1.0 "dynamic"
run_one "E2" "MSL"  "data/MSL"  55 55 1.0 "dynamic"
run_one "E2" "SMAP" "data/SMAP" 25 25 1.0 "dynamic"

# =============================================================================
# E3：多尺度动态 DGR 先验，dgr_mode=multiscale
# =============================================================================
echo "############################################################"
echo "#  E3  多尺度动态 DGR 先验（MultiScaleDGRPrior）"
echo "############################################################"

run_one "E3" "SKAB" "data/SKAB" 8 8 1.0 "multiscale"
run_one "E3" "MSL"  "data/MSL"  55 55 1.0 "multiscale"
run_one "E3" "SMAP" "data/SMAP" 25 25 1.0 "multiscale"

# =============================================================================
# E4：静态可学习 DGR 先验，dgr_mode=static
# =============================================================================
echo "############################################################"
echo "#  E4  静态可学习 DGR 先验（StaticDGRPrior）"
echo "############################################################"

run_one "E4" "SKAB" "data/SKAB" 8 8 1.0 "static"
run_one "E4" "MSL"  "data/MSL"  55 55 1.0 "static"
run_one "E4" "SMAP" "data/SMAP" 25 25 1.0 "static"

# =============================================================================
# E5：sigma_offset 调制高斯核宽度
# =============================================================================
echo "############################################################"
echo "#  E5  sigma_offset 先验（DGRSigmaOffset）"
echo "############################################################"

run_one "E5" "SKAB" "data/SKAB" 8 8 1.0 "sigma_offset"
run_one "E5" "MSL"  "data/MSL"  55 55 1.0 "sigma_offset"
run_one "E5" "SMAP" "data/SMAP" 25 25 1.0 "sigma_offset"

# =============================================================================
# B1：Dynamic + Blend(alpha=0.7, fixed)
# =============================================================================
echo "############################################################"
echo "#  B1  Dynamic + Blend (alpha=0.7 固定)"
echo "############################################################"

run_blend "B1" "SKAB" "data/SKAB" 8 8 1.0 "dynamic" "blend" 0.7 "false"
run_blend "B1" "MSL"  "data/MSL"  55 55 1.0 "dynamic" "blend" 0.7 "false"
run_blend "B1" "SMAP" "data/SMAP" 25 25 1.0 "dynamic" "blend" 0.7 "false"

# =============================================================================
# B2：Dynamic + Blend(alpha learnable)
# =============================================================================
echo "############################################################"
echo "#  B2  Dynamic + Blend (alpha 可学习)"
echo "############################################################"

run_blend "B2" "SKAB" "data/SKAB" 8 8 1.0 "dynamic" "blend" 0.5 "true"
run_blend "B2" "MSL"  "data/MSL"  55 55 1.0 "dynamic" "blend" 0.5 "true"
run_blend "B2" "SMAP" "data/SMAP" 25 25 1.0 "dynamic" "blend" 0.5 "true"

# =============================================================================
# B3：Dynamic + Entropy-Gate
# =============================================================================
echo "############################################################"
echo "#  B3  Dynamic + Entropy-Gate 自适应融合"
echo "############################################################"

run_blend "B3" "SKAB" "data/SKAB" 8 8 1.0 "dynamic" "entropy_gate" 0.5 "false"
run_blend "B3" "MSL"  "data/MSL"  55 55 1.0 "dynamic" "entropy_gate" 0.5 "false"
run_blend "B3" "SMAP" "data/SMAP" 25 25 1.0 "dynamic" "entropy_gate" 0.5 "false"

# =============================================================================
echo "============================================================"
echo "  ALL EXPERIMENTS DONE"
echo "  日志目录: ${LOG_DIR}/"
echo "  模型目录: checkpoints/"
echo "============================================================"

if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    echo "所有实验均成功完成！"
else
    echo "以下实验失败，请检查对应日志："
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - ${exp}"
    done
fi
