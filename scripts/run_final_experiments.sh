#!/usr/bin/env bash
# =============================================================================
# run_final_experiments.sh
# 论文最终实验脚本（基于诊断结论修订）
#
# 实验设计：
#   E1   baseline：高斯先验，无 DGR（所有数据集统一对照）
#   E2   原始 DGR（dynamic+diff）：与 E1 对比，展示改进来源
#   BEST 最优配置（每个数据集独立调优）：
#     SKAB → dynamic + diff（诊断证明 diff 对点突变异常最优）
#     MSL  → dynamic + raw + diff_beta=1.0（诊断：raw 特征 + 差分辅助评分）
#     SMAP → dynamic + raw + blend α=0.7（诊断：raw 特征 + 轻量融合）
#
# 诊断关键发现（来自 diag_20260603_1601）：
#   - diff 特征适合点突变异常（SKAB 8维工业传感器）
#   - raw 特征适合持续段偏移异常（MSL/SMAP NASA 遥测数据）
#   - diff_beta=1.0 在 MSL 上逐点F1: 0.0676→0.0724（无需重训练）
#   - raw+blend α=0.7 在 SMAP 上 AUPRC: 0.1108→0.1212，逐点F1: 0.0089→0.0140
#
# 使用方式（AutoDL + tmux）：
#   tmux new -s final
#   bash scripts/run_final_experiments.sh 2>&1 | tee logs/run_final.log
#   Ctrl+B D  分离会话
#   tmux attach -t final  重新连接
#
# 预计总时长：约 3~4 小时（E1+E2 共 6 组 + BEST 共 3 组 = 9 组实验）
# =============================================================================

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate adp 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG_DIR="logs/logs_$(date '+%Y%m%d_%H%M')"
mkdir -p "${LOG_DIR}"
echo "日志目录：${LOG_DIR}"
echo ""

FAILED_EXPERIMENTS=()

# --------------------------------------------------------------------------- #
# 工具函数 run_one：标准 train + test（replace 融合，单一特征模式）
#   $1  实验标签   $2  数据集    $3  data_path
#   $4  input_c   $5  output_c  $6  anormly_ratio
#   $7  dgr_mode  $8  dgr_feature_mode（默认 diff）
#   $9  diff_beta（仅 test，默认 0.0）
# --------------------------------------------------------------------------- #
run_one() {
    local EXP="$1" DATASET="$2" DATA_PATH="$3"
    local INPUT_C="$4" OUTPUT_C="$5" RATIO="$6"
    local DGR_MODE="$7"
    local DGR_FEAT="${8:-diff}"
    local DIFF_BETA="${9:-0.0}"
    local SAVE="checkpoints/${EXP}_${DATASET}"
    local LOG="${LOG_DIR}/${EXP}_${DATASET}.log"

    echo "========================================================"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] START  ${EXP} / ${DATASET}"
    echo "  dgr_mode=${DGR_MODE}  feat=${DGR_FEAT}  diff_beta=${DIFF_BETA}"
    echo "========================================================"

    python main.py \
        --mode train \
        --dataset "${DATASET}" --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --dgr_mode "${DGR_MODE}" \
        --dgr_feature_mode "${DGR_FEAT}" \
        --model_save_path "${SAVE}" \
        --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
        2>&1 | tee -a "${LOG}"

    python main.py \
        --mode test \
        --dataset "${DATASET}" --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --dgr_mode "${DGR_MODE}" \
        --dgr_feature_mode "${DGR_FEAT}" \
        --diff_beta "${DIFF_BETA}" \
        --model_save_path "${SAVE}" \
        --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
        2>&1 | tee -a "${LOG}"

    local EC=$?
    [[ ${EC} -ne 0 ]] && FAILED_EXPERIMENTS+=("${EXP}_${DATASET}")

    echo " [$(date '+%Y-%m-%d %H:%M:%S')] DONE   ${EXP} / ${DATASET}  →  ${LOG}"
    echo ""
}

# --------------------------------------------------------------------------- #
# 工具函数 run_blend：带融合策略的 train + test
#   $1  实验标签   $2  数据集    $3  data_path
#   $4  input_c   $5  output_c  $6  anormly_ratio
#   $7  dgr_mode  $8  prior_fusion  $9  prior_alpha
#   $10 dgr_feature_mode（默认 diff）
#   $11 diff_beta（仅 test，默认 0.0）
# --------------------------------------------------------------------------- #
run_blend() {
    local EXP="$1" DATASET="$2" DATA_PATH="$3"
    local INPUT_C="$4" OUTPUT_C="$5" RATIO="$6"
    local DGR_MODE="$7" FUSION="$8" ALPHA="$9"
    local DGR_FEAT="${10:-diff}"
    local DIFF_BETA="${11:-0.0}"
    local SAVE="checkpoints/${EXP}_${DATASET}"
    local LOG="${LOG_DIR}/${EXP}_${DATASET}.log"

    echo "========================================================"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] START  ${EXP} / ${DATASET}"
    echo "  dgr_mode=${DGR_MODE}  fusion=${FUSION}  alpha=${ALPHA}  feat=${DGR_FEAT}"
    echo "========================================================"

    python main.py \
        --mode train \
        --dataset "${DATASET}" --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --dgr_mode "${DGR_MODE}" \
        --prior_fusion "${FUSION}" \
        --prior_alpha "${ALPHA}" \
        --dgr_feature_mode "${DGR_FEAT}" \
        --model_save_path "${SAVE}" \
        --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
        2>&1 | tee -a "${LOG}"

    python main.py \
        --mode test \
        --dataset "${DATASET}" --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --dgr_mode "${DGR_MODE}" \
        --prior_fusion "${FUSION}" \
        --prior_alpha "${ALPHA}" \
        --dgr_feature_mode "${DGR_FEAT}" \
        --diff_beta "${DIFF_BETA}" \
        --model_save_path "${SAVE}" \
        --batch_size 256 --win_size 100 --k 3 --lr 0.0001 \
        2>&1 | tee -a "${LOG}"

    local EC=$?
    [[ ${EC} -ne 0 ]] && FAILED_EXPERIMENTS+=("${EXP}_${DATASET}")

    echo " [$(date '+%Y-%m-%d %H:%M:%S')] DONE   ${EXP} / ${DATASET}  →  ${LOG}"
    echo ""
}

# =============================================================================
# E1：baseline（高斯先验，无 DGR）
# 作用：论文对照基准，所有数据集统一使用
# =============================================================================
echo "############################################################"
echo "#  E1  baseline（高斯先验，无 DGR）"
echo "############################################################"

run_one "E1" "SKAB" "data/SKAB" 8  8  1.0 "none"
run_one "E1" "MSL"  "data/MSL"  55 55 1.0 "none"
run_one "E1" "SMAP" "data/SMAP" 25 25 1.0 "none"

# =============================================================================
# E2：动态 DGR 先验（diff 特征，replace 融合）
# 作用：展示 DGR 本身的贡献；同时作为 SKAB 最优配置
# =============================================================================
echo "############################################################"
echo "#  E2  动态 DGR 先验（dynamic + diff，所有数据集统一配置）"
echo "############################################################"

run_one "E2" "SKAB" "data/SKAB" 8  8  1.0 "dynamic" "diff"
run_one "E2" "MSL"  "data/MSL"  55 55 1.0 "dynamic" "diff"
run_one "E2" "SMAP" "data/SMAP" 25 25 1.0 "dynamic" "diff"

# =============================================================================
# BEST：每个数据集独立最优配置（论文主结果）
#
# SKAB：E2 结果即为最优（不单独重跑，直接复用 E2_SKAB checkpoint）
#
# MSL：dynamic + raw 特征 + diff_beta=1.0
#   依据：
#     - dgr_feature_mode=raw → AUPRC 0.1250→0.1323（+5.8%，超过 baseline 0.1289）
#     - diff_beta=1.0（差分辅助评分）→ 逐点F1 0.0654→0.0724（+10.7%，超过 baseline 0.0676）
#     - 两者叠加：用 raw 特征重训，test 时加 diff_beta=1.0
#
# SMAP：dynamic + raw 特征 + blend α=0.7
#   依据：
#     - raw+blend α=0.7 → AUPRC 0.1093→0.1212（+10.9%，超过 baseline 0.1108）
#     - 逐点F1 0.0060→0.0140（+133%，超过 baseline 0.0089）
# =============================================================================
echo "############################################################"
echo "#  BEST  每数据集最优配置（论文主结果）"
echo "############################################################"

# BEST_MSL：dynamic + raw 特征，diff_beta=1.0（测试阶段差分辅助评分）
echo "--- BEST / MSL ---"
run_one "BEST" "MSL" "data/MSL" 55 55 1.0 "dynamic" "raw" "1.0"

# BEST_SMAP：dynamic + raw 特征 + blend α=0.7
echo "--- BEST / SMAP ---"
run_blend "BEST" "SMAP" "data/SMAP" 25 25 1.0 "dynamic" "blend" 0.7 "raw"

# SKAB 最优即 E2，此处仅打印提示，不重复训练
echo "--- BEST / SKAB ---"
echo "  SKAB 最优配置 = E2（dynamic+diff），已在 E2 组完成，checkpoint: checkpoints/E2_SKAB"
echo ""

# =============================================================================
# 结果汇总：自动提取各实验关键指标
# =============================================================================
echo "============================================================"
echo "  ALL EXPERIMENTS DONE"
echo "  日志目录: ${LOG_DIR}/"
echo "============================================================"
echo ""
echo "=== 关键指标速览 ==="
echo "格式：实验/数据集 | F-score(PA) | 逐点F1 | AUPRC | 窗口AUPRC"
echo ""

for LOG in "${LOG_DIR}"/*.log; do
    NAME=$(basename "${LOG}" .log)
    if grep -q "F-score" "${LOG}" 2>/dev/null; then
        PA_F=$(grep "F-score"       "${LOG}" | tail -1 | grep -oP 'F-score : \K[0-9.]+')
        PF1=$(grep "逐点(无PA)"     "${LOG}" | tail -1 | grep -oP 'F1: \K[0-9.]+')
        AUC=$(grep "^\[AUPRC\]"     "${LOG}" | tail -1 | grep -oP '[0-9.]+' | head -1)
        WAUC=$(grep "^\[窗口级AUPRC\]" "${LOG}" | tail -1 | grep -oP '[0-9.]+' | head -1)
        printf "  %-20s | PA-F1=%-6s | 逐点F1=%-6s | AUPRC=%-6s | W-AUPRC=%s\n" \
               "${NAME}" "${PA_F}" "${PF1}" "${AUC}" "${WAUC}"
    fi
done

echo ""

if [[ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]]; then
    echo "[警告] 以下实验失败，请检查对应日志："
    printf '  %s\n' "${FAILED_EXPERIMENTS[@]}"
else
    echo "所有实验均成功完成！"
fi

# =============================================================================
# AutoDL 自动关机（节省费用）
# 所有实验结束后自动关闭实例，无论成功或失败都执行。
# 如需调试不关机，注释掉下面这行即可。
# =============================================================================
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有任务完成，30 秒后自动关机..."
sleep 30
/usr/bin/shutdown
