#!/usr/bin/env bash
# =============================================================================
# run_abc_experiments.sh
# 三个新方向 A / B / C 的消融实验脚本
#
# 方向 A（diff_beta）    : 差分重建辅助评分，纯测试阶段，无需重训练
# 方向 B（local_z_win）  : 局部 z-score 后处理，纯测试阶段，无需重训练
# 方向 C（lambda_diff）  : 训练时差分重建正则项，仅 MSL 重训练
#
# 依赖：run_all_experiments.sh 已跑完，E1_{SKAB,MSL,SMAP} checkpoint 已存在。
#       SKAB/SMAP 用 A+B 验证不影响其他数据集（无需重训练）。
#
# 使用方式（AutoDL + tmux）：
#   tmux new -s abc
#   bash run_abc_experiments.sh 2>&1 | tee run_abc.log
#   Ctrl+B D 离开会话
# =============================================================================

FAILED_EXPERIMENTS=()

LOG_DIR="logs/logs_abc_$(date '+%Y%m%d_%H%M')"
mkdir -p "${LOG_DIR}"
echo "日志目录：${LOG_DIR}"

# --------------------------------------------------------------------------- #
# 工具函数：仅跑 test（复用现有 checkpoint，无需重训练）
# 参数: $1=实验名 $2=数据集 $3=data_path $4=input_c $5=output_c
#       $6=anormly_ratio $7=checkpoint_path $8+=额外 test 参数
# --------------------------------------------------------------------------- #
run_test_only() {
    local EXP="$1"; local DATASET="$2"; local DATA_PATH="$3"
    local INPUT_C="$4"; local OUTPUT_C="$5"; local RATIO="$6"
    local CKPT_PATH="$7"
    shift 7
    local EXTRA_ARGS="$*"
    local LOG_FILE="${LOG_DIR}/${EXP}_${DATASET}.log"

    echo "========================================================"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] TEST  ${EXP} / ${DATASET}"
    echo " checkpoint: ${CKPT_PATH}  extra: ${EXTRA_ARGS}"
    echo "========================================================"

    python main.py \
        --mode test \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" \
        --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --model_save_path "${CKPT_PATH}" \
        --num_epochs 10 \
        --batch_size 256 \
        --win_size 100 \
        --k 3 \
        --lr 0.0001 \
        ${EXTRA_ARGS} \
        2>&1 | tee -a "${LOG_FILE}"

    local EXIT_CODE=$?
    [ ${EXIT_CODE} -ne 0 ] && FAILED_EXPERIMENTS+=("${EXP}_${DATASET}")
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${EXP} / ${DATASET}  → ${LOG_FILE}"
    echo ""
}

# --------------------------------------------------------------------------- #
# 工具函数：train + test（方向 C，需要重训练）
# 参数: $1=实验名 $2=数据集 $3=data_path $4=input_c $5=output_c
#       $6=anormly_ratio $7=lambda_diff $8+=额外 test 参数
# --------------------------------------------------------------------------- #
run_train_test_C() {
    local EXP="$1"; local DATASET="$2"; local DATA_PATH="$3"
    local INPUT_C="$4"; local OUTPUT_C="$5"; local RATIO="$6"
    local LAMBDA_DIFF="$7"
    shift 7
    local EXTRA_TEST_ARGS="$*"
    local SAVE_PATH="checkpoints/${EXP}_${DATASET}"
    local LOG_FILE="${LOG_DIR}/${EXP}_${DATASET}.log"

    echo "========================================================"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] TRAIN ${EXP} / ${DATASET}  lambda_diff=${LAMBDA_DIFF}"
    echo "========================================================"

    python main.py \
        --mode train \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" \
        --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --lambda_diff "${LAMBDA_DIFF}" \
        --model_save_path "${SAVE_PATH}" \
        --num_epochs 10 \
        --batch_size 256 \
        --win_size 100 \
        --k 3 \
        --lr 0.0001 \
        2>&1 | tee -a "${LOG_FILE}"

    echo " [$(date '+%Y-%m-%d %H:%M:%S')] TEST  ${EXP} / ${DATASET}"

    python main.py \
        --mode test \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --input_c "${INPUT_C}" \
        --output_c "${OUTPUT_C}" \
        --anormly_ratio "${RATIO}" \
        --lambda_diff "${LAMBDA_DIFF}" \
        --model_save_path "${SAVE_PATH}" \
        --num_epochs 10 \
        --batch_size 256 \
        --win_size 100 \
        --k 3 \
        --lr 0.0001 \
        ${EXTRA_TEST_ARGS} \
        2>&1 | tee -a "${LOG_FILE}"

    local EXIT_CODE=$?
    [ ${EXIT_CODE} -ne 0 ] && FAILED_EXPERIMENTS+=("${EXP}_${DATASET}")
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${EXP} / ${DATASET}  → ${LOG_FILE}"
    echo ""
}

# =============================================================================
# PART 1  方向 A 单独消融——MSL，复用 E1 checkpoint，扫 diff_beta
# =============================================================================
echo "############################################################"
echo "#  PART 1  方向 A (diff_beta) 单独消融 — MSL"
echo "############################################################"

E1_MSL_CKPT="checkpoints/E1_MSL"

run_test_only "A_beta0.5"  "MSL" "data/MSL" 55 55 1.0 "${E1_MSL_CKPT}" --diff_beta 0.5
run_test_only "A_beta1.0"  "MSL" "data/MSL" 55 55 1.0 "${E1_MSL_CKPT}" --diff_beta 1.0
run_test_only "A_beta2.0"  "MSL" "data/MSL" 55 55 1.0 "${E1_MSL_CKPT}" --diff_beta 2.0

# =============================================================================
# PART 2  方向 B 单独消融——MSL，复用 E1 checkpoint，扫 local_z_win
# =============================================================================
echo "############################################################"
echo "#  PART 2  方向 B (local_z_win) 单独消融 — MSL"
echo "############################################################"

run_test_only "B_win200"   "MSL" "data/MSL" 55 55 1.0 "${E1_MSL_CKPT}" --score_local_z_win 200
run_test_only "B_win300"   "MSL" "data/MSL" 55 55 1.0 "${E1_MSL_CKPT}" --score_local_z_win 300
run_test_only "B_win500"   "MSL" "data/MSL" 55 55 1.0 "${E1_MSL_CKPT}" --score_local_z_win 500

# =============================================================================
# PART 3  方向 A+B 联合——MSL，复用 E1 checkpoint，几组典型组合
# =============================================================================
echo "############################################################"
echo "#  PART 3  方向 A+B 联合 — MSL"
echo "############################################################"

run_test_only "AB_b0.5_w200" "MSL" "data/MSL" 55 55 1.0 "${E1_MSL_CKPT}" --diff_beta 0.5 --score_local_z_win 200
run_test_only "AB_b1.0_w300" "MSL" "data/MSL" 55 55 1.0 "${E1_MSL_CKPT}" --diff_beta 1.0 --score_local_z_win 300
run_test_only "AB_b2.0_w500" "MSL" "data/MSL" 55 55 1.0 "${E1_MSL_CKPT}" --diff_beta 2.0 --score_local_z_win 500
run_test_only "AB_b1.0_w500" "MSL" "data/MSL" 55 55 1.0 "${E1_MSL_CKPT}" --diff_beta 1.0 --score_local_z_win 500

# =============================================================================
# PART 4  跨数据集验证——A+B 对 SKAB / SMAP 的影响（不重训练）
#         使用上面 MSL 最典型的参数：diff_beta=1.0, local_z_win=300
#         证明：该策略对其他数据集无负面影响
# =============================================================================
echo "############################################################"
echo "#  PART 4  跨数据集验证 A+B — SKAB / SMAP（复用 E1 checkpoint）"
echo "############################################################"

run_test_only "AB_b1.0_w300" "SKAB" "data/SKAB" 8  8  1.0 "checkpoints/E1_SKAB" --diff_beta 1.0 --score_local_z_win 300
run_test_only "AB_b1.0_w300" "SMAP" "data/SMAP" 25 25 1.0 "checkpoints/E1_SMAP" --diff_beta 1.0 --score_local_z_win 300

# 也单独验证 A、B 各自对其他数据集的影响
run_test_only "A_beta1.0"   "SKAB" "data/SKAB" 8  8  1.0 "checkpoints/E1_SKAB" --diff_beta 1.0
run_test_only "B_win300"    "SKAB" "data/SKAB" 8  8  1.0 "checkpoints/E1_SKAB" --score_local_z_win 300
run_test_only "A_beta1.0"   "SMAP" "data/SMAP" 25 25 1.0 "checkpoints/E1_SMAP" --diff_beta 1.0
run_test_only "B_win300"    "SMAP" "data/SMAP" 25 25 1.0 "checkpoints/E1_SMAP" --score_local_z_win 300

# =============================================================================
# PART 5  方向 C 单独消融——仅 MSL 重训练（扫 lambda_diff），不带 A/B
#         目的：确认 lambda_diff 本身的训练增益
# =============================================================================
echo "############################################################"
echo "#  PART 5  方向 C (lambda_diff) 单独消融 — 仅 MSL 重训练"
echo "############################################################"

run_train_test_C "C_ld0.1"  "MSL" "data/MSL" 55 55 1.0 0.1
run_train_test_C "C_ld0.3"  "MSL" "data/MSL" 55 55 1.0 0.3
run_train_test_C "C_ld0.5"  "MSL" "data/MSL" 55 55 1.0 0.5
run_train_test_C "C_ld1.0"  "MSL" "data/MSL" 55 55 1.0 1.0

# =============================================================================
# PART 6  A+B+C 全组合——使用 PART 5 最佳 C checkpoint + 最佳 A/B 参数
#         说明：lambda_diff 选 0.3（与 0.5 结果比较后选优，调整下方 BEST_LD）
#              diff_beta 和 local_z_win 根据 PART 3 结果调整
# =============================================================================
echo "############################################################"
echo "#  PART 6  A+B+C 全组合 — MSL"
echo "############################################################"

# ← 根据 PART 3/5 的日志结果，手动调整这两个变量再运行 PART 6
BEST_LD=0.3
BEST_BETA=1.0
BEST_WIN=300

# PART 6 仅做 test（复用 PART 5 训好的 checkpoint，无需再训）
run_test_only "ABC_ld${BEST_LD}_b${BEST_BETA}_w${BEST_WIN}" \
    "MSL" "data/MSL" 55 55 1.0 \
    "checkpoints/C_ld${BEST_LD}_MSL" \
    --lambda_diff "${BEST_LD}" \
    --diff_beta "${BEST_BETA}" \
    --score_local_z_win "${BEST_WIN}"

# =============================================================================
echo "============================================================"
echo "  ALL EXPERIMENTS DONE"
echo "  日志目录: ${LOG_DIR}/"
echo "============================================================"

if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    echo "所有实验均成功完成！"
else
    echo "以下实验失败，请检查对应日志："
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - ${exp}"
    done
fi

# =============================================================================
# 自动关机（AutoDL）：60 秒内可 Ctrl+C 取消
# =============================================================================
echo ""
echo "[关机] 全部完成。60 秒后自动关机，如需取消请在 tmux 中按 Ctrl+C"
sleep 60
sudo shutdown -h now 2>/dev/null || poweroff 2>/dev/null || \
    echo "[警告] 自动关机失败，请手动在 AutoDL 控制台停止实例。"
