#!/usr/bin/env bash
# =============================================================================
# run_diag_msl_smap.sh
# 诊断 MSL / SMAP 上 DGR 先验退化原因的快速实验脚本。
#
# 核心假设：退化可能来自以下任意一项
#   H1  先验权重过强 → 降低 prior_alpha（blend 模式，α=0.1/0.3）
#   H2  DGR 特征构建不匹配 → 换 dgr_feature_mode=raw
#   H3  高维数据 DGR 信号噪声大 → 换 prior_fusion=entropy_gate（自适应门控）
#   H4  评分公式掩盖了真实提升 → 测试 diff_beta / score_mode 后处理
#
# 使用方式（AutoDL + tmux）：
#   bash scripts/run_diag_msl_smap.sh 2>&1 | tee logs/diag_msl_smap.log
#
# 每个实验约 5-10 分钟，全部约 2-3 小时。
# 重点看：[逐点(无PA)] F1 和 [AUPRC] 两列（不受 PA 虚高影响）。
# =============================================================================

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate adp 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG_DIR="logs/diag_$(date '+%Y%m%d_%H%M')"
mkdir -p "${LOG_DIR}"
echo "诊断日志目录：${LOG_DIR}"
echo ""

FAILED=()

# --------------------------------------------------------------------------- #
# 基础工具函数：仅需要指定诊断变量的参数，其余走 Auto 配置
#   $1  实验标签（自定义，写进 log 文件名）
#   $2  数据集 (MSL / SMAP)
#   $3  input_c / output_c
#   $4  data_path
#   $5  dgr_mode
#   $6  prior_fusion
#   $7  prior_alpha
#   $8  dgr_feature_mode
#   $9  diff_beta（仅 test 阶段，0=关）
#   $10 score_mode
# --------------------------------------------------------------------------- #
run_diag() {
    local TAG="$1"
    local DATASET="$2"
    local CH="$3"
    local DATA_PATH="$4"
    local DGR_MODE="$5"
    local FUSION="$6"
    local ALPHA="$7"
    local FEAT_MODE="$8"
    local DIFF_BETA="$9"
    local SCORE_MODE="${10:-combined}"
    local SAVE="checkpoints/diag_${TAG}_${DATASET}"
    local LOG="${LOG_DIR}/${TAG}_${DATASET}.log"

    echo "========================================================"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] START  ${TAG} / ${DATASET}"
    echo "  dgr_mode=${DGR_MODE}  fusion=${FUSION}  alpha=${ALPHA}"
    echo "  feat=${FEAT_MODE}  diff_beta=${DIFF_BETA}  score=${SCORE_MODE}"
    echo "========================================================"

    python main.py \
        --mode train \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${CH}" --output_c "${CH}" \
        --anormly_ratio 1.0 \
        --dgr_mode "${DGR_MODE}" \
        --prior_fusion "${FUSION}" \
        --prior_alpha "${ALPHA}" \
        --dgr_feature_mode "${FEAT_MODE}" \
        --model_save_path "${SAVE}" \
        --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
        2>&1 | tee -a "${LOG}"

    python main.py \
        --mode test \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${CH}" --output_c "${CH}" \
        --anormly_ratio 1.0 \
        --dgr_mode "${DGR_MODE}" \
        --prior_fusion "${FUSION}" \
        --prior_alpha "${ALPHA}" \
        --dgr_feature_mode "${FEAT_MODE}" \
        --diff_beta "${DIFF_BETA}" \
        --score_mode "${SCORE_MODE}" \
        --model_save_path "${SAVE}" \
        --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
        2>&1 | tee -a "${LOG}"

    local EC=$?
    [[ ${EC} -ne 0 ]] && FAILED+=("${TAG}_${DATASET}")

    echo " [$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${TAG} / ${DATASET}  →  ${LOG}"
    echo ""
}

# --------------------------------------------------------------------------- #
# 快速只跑 test（复用已有 checkpoint，节省时间；用于后处理参数扫描）
# --------------------------------------------------------------------------- #
run_test_only() {
    local TAG="$1"
    local DATASET="$2"
    local CH="$3"
    local DATA_PATH="$4"
    local DGR_MODE="$5"
    local FUSION="$6"
    local ALPHA="$7"
    local FEAT_MODE="$8"
    local DIFF_BETA="$9"
    local SCORE_MODE="${10:-combined}"
    local SAVE="checkpoints/diag_${TAG}_${DATASET}"   # 复用同名 checkpoint
    local LOG="${LOG_DIR}/${TAG}_${DATASET}_testonly.log"

    echo " [$(date '+%Y-%m-%d %H:%M:%S')] TEST-ONLY  ${TAG} / ${DATASET}"
    python main.py \
        --mode test \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${CH}" --output_c "${CH}" \
        --anormly_ratio 1.0 \
        --dgr_mode "${DGR_MODE}" \
        --prior_fusion "${FUSION}" \
        --prior_alpha "${ALPHA}" \
        --dgr_feature_mode "${FEAT_MODE}" \
        --diff_beta "${DIFF_BETA}" \
        --score_mode "${SCORE_MODE}" \
        --model_save_path "${SAVE}" \
        --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
        2>&1 | tee -a "${LOG}"
    echo " DONE  ${TAG} / ${DATASET}  →  ${LOG}"
    echo ""
}

# =============================================================================
# REF：从 run_main.log 复现的 baseline（E1）和最优 DGR（E2），用作对照基准
# =============================================================================
echo "############################################################"
echo "#  REF  对照组（baseline E1 + 最优 DGR E2）"
echo "############################################################"

# E1: 高斯先验，无 DGR
run_diag "REF_E1"   "MSL"  55 "data/MSL"  "none"    "replace" 0.5 "diff" 0.0
run_diag "REF_E1"   "SMAP" 25 "data/SMAP" "none"    "replace" 0.5 "diff" 0.0

# E2: 动态 DGR，replace 融合（与 run_main.log 一致，确认可复现）
run_diag "REF_E2"   "MSL"  55 "data/MSL"  "dynamic" "replace" 0.5 "diff" 0.0
run_diag "REF_E2"   "SMAP" 25 "data/SMAP" "dynamic" "replace" 0.5 "diff" 0.0

# =============================================================================
# H1：先验权重过强 —— blend 模式，大幅降低 DGR 权重
#   思路：alpha=1.0 时退化为纯高斯（≈E1），alpha 越小 DGR 权重越大。
#   先测 alpha=0.9/0.7/0.5/0.3，找到 F1 开始下降的拐点。
# =============================================================================
echo "############################################################"
echo "#  H1  Blend 融合：扫描 prior_alpha（DGR 权重）"
echo "############################################################"

for ALPHA in 0.9 0.7 0.5 0.3; do
    TAG="H1_a${ALPHA/./}"   # e.g. H1_a09
    run_diag "${TAG}" "MSL"  55 "data/MSL"  "dynamic" "blend" "${ALPHA}" "diff" 0.0
    run_diag "${TAG}" "SMAP" 25 "data/SMAP" "dynamic" "blend" "${ALPHA}" "diff" 0.0
done

# =============================================================================
# H2：DGR 特征构建不匹配 —— 改用 raw（原始值）代替 diff（差分）
#   思路：MSL/SMAP 异常多为持续段（不是点突变），raw 可能比 diff 更合适。
# =============================================================================
echo "############################################################"
echo "#  H2  特征模式：raw vs diff（replace 融合）"
echo "############################################################"

run_diag "H2_raw"  "MSL"  55 "data/MSL"  "dynamic" "replace" 0.5 "raw" 0.0
run_diag "H2_raw"  "SMAP" 25 "data/SMAP" "dynamic" "replace" 0.5 "raw" 0.0

# raw + 低权重 blend 组合
run_diag "H2_raw_blend07" "MSL"  55 "data/MSL"  "dynamic" "blend" 0.7 "raw" 0.0
run_diag "H2_raw_blend07" "SMAP" 25 "data/SMAP" "dynamic" "blend" 0.7 "raw" 0.0

# =============================================================================
# H3：高维数据 DGR 信号噪声大 —— entropy_gate 自适应门控
#   思路：高熵区域（不确定时）退回高斯先验，只在低熵（DGR 置信高）时使用 DGR。
# =============================================================================
echo "############################################################"
echo "#  H3  Entropy-Gate 自适应门控"
echo "############################################################"

# 默认参数（tau=0.6, gamma=12）
run_diag "H3_egate" "MSL"  55 "data/MSL"  "dynamic" "entropy_gate" 0.5 "diff" 0.0
run_diag "H3_egate" "SMAP" 25 "data/SMAP" "dynamic" "entropy_gate" 0.5 "diff" 0.0

# 更保守：tau 调高（更多区域回退到高斯）
run_diag "H3_egate_tau08" "MSL"  55 "data/MSL"  "dynamic" "entropy_gate" 0.5 "diff" 0.0
# 注：tau 通过 --prior_entropy_tau 控制，此处手动追加参数
# （run_diag 不含该参数，下面直接写完整命令）
python main.py --mode train  --dataset MSL  --data_path data/MSL  --input_c 55 --output_c 55 \
    --anormly_ratio 1.0 --dgr_mode dynamic --prior_fusion entropy_gate \
    --prior_alpha 0.5 --dgr_feature_mode diff --prior_entropy_tau 0.8 --prior_entropy_gamma 6.0 \
    --model_save_path "checkpoints/diag_H3_egate_tau08_MSL" --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
    2>&1 | tee -a "${LOG_DIR}/H3_egate_tau08_MSL.log"
python main.py --mode test   --dataset MSL  --data_path data/MSL  --input_c 55 --output_c 55 \
    --anormly_ratio 1.0 --dgr_mode dynamic --prior_fusion entropy_gate \
    --prior_alpha 0.5 --dgr_feature_mode diff --prior_entropy_tau 0.8 --prior_entropy_gamma 6.0 \
    --model_save_path "checkpoints/diag_H3_egate_tau08_MSL" --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
    2>&1 | tee -a "${LOG_DIR}/H3_egate_tau08_MSL.log"

python main.py --mode train  --dataset SMAP --data_path data/SMAP --input_c 25 --output_c 25 \
    --anormly_ratio 1.0 --dgr_mode dynamic --prior_fusion entropy_gate \
    --prior_alpha 0.5 --dgr_feature_mode diff --prior_entropy_tau 0.8 --prior_entropy_gamma 6.0 \
    --model_save_path "checkpoints/diag_H3_egate_tau08_SMAP" --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
    2>&1 | tee -a "${LOG_DIR}/H3_egate_tau08_SMAP.log"
python main.py --mode test   --dataset SMAP --data_path data/SMAP --input_c 25 --output_c 25 \
    --anormly_ratio 1.0 --dgr_mode dynamic --prior_fusion entropy_gate \
    --prior_alpha 0.5 --dgr_feature_mode diff --prior_entropy_tau 0.8 --prior_entropy_gamma 6.0 \
    --model_save_path "checkpoints/diag_H3_egate_tau08_SMAP" --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
    2>&1 | tee -a "${LOG_DIR}/H3_egate_tau08_SMAP.log"

# =============================================================================
# H4：评分后处理 —— 不重训练，直接用 REF_E2 的 checkpoint 换参数测试
#   思路：diff_beta 在差分域放大突变异常；score_local_z_win 提升对比度
#   注意：复用 REF_E2 的 checkpoint（需先确认 REF_E2 已完成）
# =============================================================================
echo "############################################################"
echo "#  H4  后处理扫描（复用 REF_E2 checkpoint，无需重训练）"
echo "############################################################"

# diff_beta 扫描（差分辅助评分）
for BETA in 0.5 1.0 2.0; do
    BTAG="H4_beta${BETA/./}"
    LOG="${LOG_DIR}/${BTAG}_MSL.log"
    python main.py --mode test --dataset MSL --data_path data/MSL --input_c 55 --output_c 55 \
        --anormly_ratio 1.0 --dgr_mode dynamic --prior_fusion replace \
        --dgr_feature_mode diff --diff_beta "${BETA}" --score_mode combined \
        --model_save_path "checkpoints/diag_REF_E2_MSL" --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
        2>&1 | tee -a "${LOG}"
    echo "  DONE H4 diff_beta=${BETA} / MSL"

    LOG="${LOG_DIR}/${BTAG}_SMAP.log"
    python main.py --mode test --dataset SMAP --data_path data/SMAP --input_c 25 --output_c 25 \
        --anormly_ratio 1.0 --dgr_mode dynamic --prior_fusion replace \
        --dgr_feature_mode diff --diff_beta "${BETA}" --score_mode combined \
        --model_save_path "checkpoints/diag_REF_E2_SMAP" --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
        2>&1 | tee -a "${LOG}"
    echo "  DONE H4 diff_beta=${BETA} / SMAP"
done

# local_z 后处理（提升局部对比度，抑制背景能量带来的假阳性）
for ZWIN in 200 500; do
    ZTAG="H4_lz${ZWIN}"
    LOG="${LOG_DIR}/${ZTAG}_MSL.log"
    python main.py --mode test --dataset MSL --data_path data/MSL --input_c 55 --output_c 55 \
        --anormly_ratio 1.0 --dgr_mode dynamic --prior_fusion replace \
        --dgr_feature_mode diff --diff_beta 0.0 --score_local_z_win "${ZWIN}" --score_mode combined \
        --model_save_path "checkpoints/diag_REF_E2_MSL" --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
        2>&1 | tee -a "${LOG}"
    echo "  DONE H4 local_z_win=${ZWIN} / MSL"
done

# =============================================================================
# 汇总：从各 log 提取关键指标，方便快速对比
# =============================================================================
echo ""
echo "############################################################"
echo "#  实验完成，提取关键指标"
echo "############################################################"
echo ""
echo "=== 关键指标速览（[逐点F1] 和 [AUPRC]）==="
echo ""

# 遍历日志，提取每个实验的结果行
for LOG in "${LOG_DIR}"/*.log; do
    EXPNAME=$(basename "${LOG}" .log)
    # 找 DONE 行确认完成
    DONE_LINE=$(grep "DONE\|TEST MODE\|F-score" "${LOG}" | tail -3 || true)
    if grep -q "F-score" "${LOG}"; then
        echo "--- ${EXPNAME} ---"
        grep -E "F-score|逐点|AUPRC|窗口级AUPRC" "${LOG}" | tail -6
        echo ""
    fi
done

# --------------------------------------------------------------------------- #
# 失败列表
# --------------------------------------------------------------------------- #
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "[警告] 以下实验失败："
    printf '  %s\n' "${FAILED[@]}"
fi

echo ""
echo "全部诊断实验完成。日志目录：${LOG_DIR}"
echo "建议重点关注："
echo "  1. H1 系列：哪个 alpha 值让 MSL/SMAP 的 AUPRC 不再退化于 E1？"
echo "  2. H2 系列：raw 特征模式是否改善了 SMAP 的逐点F1？"
echo "  3. H3 系列：entropy_gate 能否在高维数据上自适应保持性能？"
echo "  4. H4 系列：仅靠后处理能否在不重训练的情况下提升 MSL 分数？"
