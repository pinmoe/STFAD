#!/usr/bin/env bash
# =============================================================================
# run_st330ir001_cp001.sh
# 针对 ST330IR001_CP001 焊枪故障数据集的全量实验脚本
# 实验覆盖 E1–E5（先验消融）+ B1–B3（融合策略），格式与 run_all_experiments.sh 一致。
#
# 使用方式（AutoDL + tmux）：
#   tmux new -s st330         # 新建 tmux 会话
#   bash scripts/run_st330ir001_cp001.sh 2>&1 | tee run_st330.log
#   # Ctrl+B D 分离会话，关闭 SSH 不影响运行
#   # tmux attach -t st330    重新连接
#
# 前置条件：
#   data/ST330IR001_CP001/ 目录下需存在以下文件：
#     ST330IR001_CP001_train.npy       shape (N_train, 56, 29)
#     ST330IR001_CP001_test.npy        shape (N_test,  56, 29)
#     ST330IR001_CP001_test_label.npy  shape (N_test,  56, 1)
#   如尚未生成，请先运行：
#     python scripts/prepare_st330ir001_cp001.py
#
# 日志自动保存到 logs/ 目录，checkpoint 保存到 checkpoints/{EXP}_ST330IR001_CP001/
# 某单个实验失败时记录错误并继续，不中断全部实验。
# =============================================================================

FAILED_EXPERIMENTS=()

# AutoDL / conda 环境激活（若已在 conda env 中运行可注释掉此行）
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate adp 2>/dev/null || true

# 减少显存碎片导致的 OOM 风险
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG_DIR="logs/logs_st330_$(date '+%Y%m%d_%H%M')"
mkdir -p "${LOG_DIR}"
echo "日志目录：${LOG_DIR}"

# --------------------------------------------------------------------------- #
# 数据集固定参数
# --------------------------------------------------------------------------- #
DATASET="ST330IR001_CP001"
DATA_PATH="data/ST330IR001_CP001"
INPUT_C=29
OUTPUT_C=29
WIN_SIZE=56
RATIO=45.93
# --------------------------------------------------------------------------- #
# 工具函数：train + test 一对（基础版）
# 参数: $1=实验名 $2=dgr_mode $3=dgr_feature_mode（可选，默认 diff）
#       $4=lambda_diff（可选，默认 0.0） $5=diff_beta（可选，默认 0.0）
#       $6=use_memory_bank（可选，默认 false）
# --------------------------------------------------------------------------- #
run_one() {
    local EXP="$1"
    local DGR_MODE="$2"
    local DGR_FEAT_MODE="${3:-diff}"
    local LAMBDA_DIFF="${4:-0.0}"
    local DIFF_BETA="${5:-0.0}"
    local USE_MB="${6:-false}"
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
        --win_size "${WIN_SIZE}" \
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
        --win_size "${WIN_SIZE}" \
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
# 参数: $1=实验名 $2=dgr_mode $3=prior_fusion $4=prior_alpha
#       $5=prior_alpha_learnable（true/false）$6=dgr_feature_mode（可选，默认 diff）
#       $7=lambda_diff（可选，默认 0.0）$8=diff_beta（可选，默认 0.0）
#       $9=use_memory_bank（可选，默认 false）
# --------------------------------------------------------------------------- #
run_blend() {
    local EXP="$1"
    local DGR_MODE="$2"
    local FUSION="$3"
    local ALPHA="$4"
    local ALPHA_LEARNABLE="$5"
    local DGR_FEAT_MODE="${6:-diff}"
    local LAMBDA_DIFF="${7:-0.0}"
    local DIFF_BETA="${8:-0.0}"
    local USE_MB="${9:-false}"
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
        --win_size "${WIN_SIZE}" \
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
        --win_size "${WIN_SIZE}" \
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
run_one "E1" "none"

# =============================================================================
# E2：动态 DGR 先验，dgr_mode=dynamic
# =============================================================================
echo "############################################################"
echo "#  E2  动态 DGR 先验（DGRPrior）"
echo "############################################################"
run_one "E2" "dynamic"

# =============================================================================
# E3：多尺度动态 DGR 先验，dgr_mode=multiscale
# =============================================================================
echo "############################################################"
echo "#  E3  多尺度动态 DGR 先验（MultiScaleDGRPrior）"
echo "############################################################"
run_one "E3" "multiscale"

# =============================================================================
# E4：静态可学习 DGR 先验，dgr_mode=static
# =============================================================================
echo "############################################################"
echo "#  E4  静态可学习 DGR 先验（StaticDGRPrior）"
echo "############################################################"
run_one "E4" "static"

# =============================================================================
# E5：sigma_offset 调制高斯核宽度
# =============================================================================
echo "############################################################"
echo "#  E5  sigma_offset 先验（DGRSigmaOffset）"
echo "############################################################"
run_one "E5" "sigma_offset"

# =============================================================================
# B1：Dynamic + Blend(alpha=0.7, fixed)
# =============================================================================
echo "############################################################"
echo "#  B1  Dynamic + Blend (alpha=0.7 固定)"
echo "############################################################"
run_blend "B1" "dynamic" "blend" 0.7 "false"

# =============================================================================
# B2：Dynamic + Blend(alpha learnable)
# =============================================================================
echo "############################################################"
echo "#  B2  Dynamic + Blend (alpha 可学习)"
echo "############################################################"
run_blend "B2" "dynamic" "blend" 0.5 "true"

# =============================================================================
# B3：Dynamic + Entropy-Gate
# =============================================================================
echo "############################################################"
echo "#  B3  Dynamic + Entropy-Gate 自适应融合"
echo "############################################################"
run_blend "B3" "dynamic" "entropy_gate" 0.5 "false"

# =============================================================================
# 改进实验（针对 ST330IR001_CP001 高异常率 + 固定窗口的特殊性）
#
# 三个假设分别验证：
#   F1: 差分评分（diff_beta）能否提升 E1 baseline 的 Precision？
#       → 不需重训练，复用 E1 checkpoint，仅改 test 评分方式
#       → 但本脚本 train+test 一起跑，所以还是会重训练一次
#   F2: 记忆库净化（use_memory_bank）能否修复 E2 的先验污染？
#       → E2 在此数据集上 F1 低于 E1，主因是 45.93% 异常率污染先验
#   F3: 差分正则（lambda_diff）+ 差分评分 + 记忆库全部开启，E2 的上界？
# =============================================================================
echo "############################################################"
echo "#  F1  E1 + diff_beta=0.3（差分评分放大突变信号）"
echo "############################################################"
run_one "F1" "none" "diff" 0.0 0.3 "false"

echo "############################################################"
echo "#  F2  E2 + memory_bank（记忆库净化先验污染）"
echo "############################################################"
run_one "F2" "dynamic" "diff" 0.0 0.0 "true"

echo "############################################################"
echo "#  F3  E2 + lambda_diff=0.05 + diff_beta=0.3 + memory_bank"
echo "############################################################"
run_one "F3" "dynamic" "diff" 0.05 0.3 "true"

# =============================================================================
# G 系列：针对固定窗口 + 分布偏移型异常的结构性改进
#
# 背景：
#   - 训练集 100% 正常焊枪周期，测试集含 45.93% 故障周期
#   - 故障是整个焊接周期的工艺偏差（分布偏移），非点突变
#   - 当前 diff 模式捕捉时间差分（点突变），不适合此类故障
#   - 当前 max 通道聚合对噪声通道敏感，均值 (rec_mean) 更稳健
#
# 实验设计：
#   G1: E1 + score_mode=rec_mean（无需重训练，仅改评分）
#   G2: E2 + dgr_feature_mode=raw + score_mode=rec_mean
#       raw 模式捕捉传感器间相关性（压力/电流/电压的协同模式），
#       故障周期的传感器互相关结构与正常周期不同 → 先验判别力更强
#   G3: B1(blend) + dgr_feature_mode=raw + score_mode=rec_mean
#       混合融合保留高斯先验稳定性，raw DGR 提供传感器关系信息
# =============================================================================

# G1 复用 E1 checkpoint，只跑 test（直接调 python，不用 run_one）
echo "############################################################"
echo "#  G1  E1 + score_mode=rec_mean（复用 E1 checkpoint，无需重训练）"
echo "############################################################"
LOG_FILE="${LOG_DIR}/G1_${DATASET}.log"
echo " [$(date '+%Y-%m-%d %H:%M:%S')] START  G1 (test-only) / ${DATASET}"
python main.py \
    --mode test \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --input_c "${INPUT_C}" \
    --output_c "${OUTPUT_C}" \
    --anormly_ratio "${RATIO}" \
    --dgr_mode "none" \
    --score_mode "rec_mean" \
    --model_save_path "checkpoints/E1_${DATASET}" \
    --num_epochs 10 \
    --batch_size 256 \
    --win_size "${WIN_SIZE}" \
    --k 3 \
    --lr 0.0001 \
    2>&1 | tee -a "${LOG_FILE}"
echo " [$(date '+%Y-%m-%d %H:%M:%S')] DONE   G1 / ${DATASET}  →  ${LOG_FILE}"

echo "############################################################"
echo "#  G2  E2 + dgr_feature_mode=raw + score_mode=rec_mean"
echo "############################################################"
run_one "G2" "dynamic" "raw" 0.0 0.0 "false"
# 注意：G2 test 阶段需要指定 score_mode=rec_mean，但 run_one 目前不支持。
# 单独跑 test：
LOG_FILE="${LOG_DIR}/G2_${DATASET}_test_recmean.log"
python main.py \
    --mode test \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --input_c "${INPUT_C}" \
    --output_c "${OUTPUT_C}" \
    --anormly_ratio "${RATIO}" \
    --dgr_mode "dynamic" \
    --dgr_feature_mode "raw" \
    --score_mode "rec_mean" \
    --model_save_path "checkpoints/G2_${DATASET}" \
    --num_epochs 10 \
    --batch_size 256 \
    --win_size "${WIN_SIZE}" \
    --k 3 \
    --lr 0.0001 \
    2>&1 | tee -a "${LOG_FILE}"

echo "############################################################"
echo "#  G3  B1(blend 0.7) + dgr_feature_mode=raw + score_mode=rec_mean"
echo "############################################################"
run_blend "G3" "dynamic" "blend" 0.7 "false" "raw"
LOG_FILE="${LOG_DIR}/G3_${DATASET}_test_recmean.log"
python main.py \
    --mode test \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --input_c "${INPUT_C}" \
    --output_c "${OUTPUT_C}" \
    --anormly_ratio "${RATIO}" \
    --dgr_mode "dynamic" \
    --prior_fusion "blend" \
    --prior_alpha 0.7 \
    --dgr_feature_mode "raw" \
    --score_mode "rec_mean" \
    --model_save_path "checkpoints/G3_${DATASET}" \
    --num_epochs 10 \
    --batch_size 256 \
    --win_size "${WIN_SIZE}" \
    --k 3 \
    --lr 0.0001 \
    2>&1 | tee -a "${LOG_FILE}"

# =============================================================================
echo "============================================================"
echo "  ALL EXPERIMENTS DONE  [ST330IR001_CP001]"
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
