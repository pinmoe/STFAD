#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
#   bash scripts/run_ablation_3ds.sh
#   bash scripts/run_ablation_3ds.sh --run_blend
#   bash scripts/run_ablation_3ds.sh --run_e5 --run_entropy
#   bash scripts/run_ablation_3ds.sh --run_e5 --run_entropy --shutdown   # 完成后自动关机（AutoDL）
#   bash scripts/run_ablation_3ds.sh --python python --epochs 20 --batch_size 256 --anormly_ratio 1.0
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_ablation_3ds.sh --run_blend

PYTHON_BIN="python"
EPOCHS=10
BATCH_SIZE=256
ANORMLY_RATIO=1.0
RUN_BLEND=0
RUN_E5=0
RUN_ENTROPY=0
ENTROPY_TAU=0.6
ENTROPY_GAMMA=12.0
LOG_DIR="logs"
SHUTDOWN_AFTER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"; shift 2;;
    --epochs)
      EPOCHS="$2"; shift 2;;
    --batch_size)
      BATCH_SIZE="$2"; shift 2;;
    --anormly_ratio)
      ANORMLY_RATIO="$2"; shift 2;;
    --run_blend)
      RUN_BLEND=1; shift 1;;
    --run_e5)
      RUN_E5=1; shift 1;;
    --run_entropy)
      RUN_ENTROPY=1; shift 1;;
    --entropy_tau)
      ENTROPY_TAU="$2"; shift 2;;
    --entropy_gamma)
      ENTROPY_GAMMA="$2"; shift 2;;
    --log_dir)
      LOG_DIR="$2"; shift 2;;
    --shutdown)
      SHUTDOWN_AFTER=1; shift 1;;
    *)
      echo "未知参数: $1"; exit 1;;
  esac
done

mkdir -p "$LOG_DIR"

run_one() {
  local ds_name="$1"
  local data_path="$2"
  local input_c="$3"
  local output_c="$4"
  local exp_name="$5"
  local dgr_mode="$6"
  local prior_fusion="$7"
  local prior_alpha="$8"
  local prior_alpha_learnable="$9"
  local dgr_input_mode="${10}"
  local mode="${11}"

  local save_path="checkpoints/${ds_name}_${exp_name}"
  local log_path="${LOG_DIR}/${exp_name}_${ds_name}.log"
  local now
  now="$(date '+%Y-%m-%d %H:%M:%S')"

  # 每次 train 先清空同实验日志，确保 parse_results.py 读取的是本轮结果。
  if [[ "$mode" == "train" ]]; then
    : > "$log_path"
  fi

  echo "========================================================================"
  echo "[${ds_name}] [${exp_name}] [${mode}]"
  echo "model_save_path: ${save_path}"
  echo "log_path: ${log_path}"

  {
    echo "========================================================================"
    echo "[${now}] [${ds_name}] [${exp_name}] [${mode}]"
    echo "model_save_path: ${save_path}"
    echo "cmd: $PYTHON_BIN main.py --mode $mode --dataset $ds_name --data_path $data_path ..."

    "$PYTHON_BIN" main.py \
      --mode "$mode" \
      --dataset "$ds_name" \
      --data_path "$data_path" \
      --input_c "$input_c" \
      --output_c "$output_c" \
      --num_epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --anormly_ratio "$ANORMLY_RATIO" \
      --model_save_path "$save_path" \
      --dgr_mode "$dgr_mode" \
      --prior_fusion "$prior_fusion" \
      --prior_alpha "$prior_alpha" \
      --prior_alpha_learnable "$prior_alpha_learnable" \
      --dgr_input_mode "$dgr_input_mode" \
      --prior_entropy_tau "$ENTROPY_TAU" \
      --prior_entropy_gamma "$ENTROPY_GAMMA"
  } 2>&1 | tee -a "$log_path"
}

# 数据集配置（HAI=59, MSL=55, SKAB=8）
declare -a DS_NAMES=("HAI" "MSL" "SKAB")
declare -a DS_PATHS=("data/HAI" "data/MSL" "data/SKAB")
declare -a DS_IN=(59 55 8)
declare -a DS_OUT=(59 55 8)

# 基础实验 E1-E4（与你论文定义一致）
declare -a EXP_NAMES=("E1" "E2" "E3" "E4")
declare -a EXP_DGR=("none" "dynamic" "multiscale" "static")
declare -a EXP_FUSION=("replace" "replace" "replace" "replace")
declare -a EXP_ALPHA=(0.5 0.5 0.5 0.5)
declare -a EXP_ALPHA_L=("false" "false" "false" "false")
declare -a EXP_INPUT=("raw" "raw" "raw" "raw")

for i in "${!DS_NAMES[@]}"; do
  ds_name="${DS_NAMES[$i]}"
  ds_path="${DS_PATHS[$i]}"
  ds_in="${DS_IN[$i]}"
  ds_out="${DS_OUT[$i]}"

  for j in "${!EXP_NAMES[@]}"; do
    run_one "$ds_name" "$ds_path" "$ds_in" "$ds_out" \
      "${EXP_NAMES[$j]}" "${EXP_DGR[$j]}" "${EXP_FUSION[$j]}" \
      "${EXP_ALPHA[$j]}" "${EXP_ALPHA_L[$j]}" "${EXP_INPUT[$j]}" "train"

    run_one "$ds_name" "$ds_path" "$ds_in" "$ds_out" \
      "${EXP_NAMES[$j]}" "${EXP_DGR[$j]}" "${EXP_FUSION[$j]}" \
      "${EXP_ALPHA[$j]}" "${EXP_ALPHA_L[$j]}" "${EXP_INPUT[$j]}" "test"
  done

  if [[ "$RUN_BLEND" -eq 1 ]]; then
    # B1: 动态DGR + 融合(alpha=0.7 固定)
    run_one "$ds_name" "$ds_path" "$ds_in" "$ds_out" \
      "B1" "dynamic" "blend" 0.7 "false" "raw" "train"
    run_one "$ds_name" "$ds_path" "$ds_in" "$ds_out" \
      "B1" "dynamic" "blend" 0.7 "false" "raw" "test"

    # B2: 动态DGR + 融合(alpha可学习)
    run_one "$ds_name" "$ds_path" "$ds_in" "$ds_out" \
      "B2" "dynamic" "blend" 0.5 "true" "raw" "train"
    run_one "$ds_name" "$ds_path" "$ds_in" "$ds_out" \
      "B2" "dynamic" "blend" 0.5 "true" "raw" "test"
  fi

  if [[ "$RUN_ENTROPY" -eq 1 ]]; then
    # B3: 动态DGR + entropy_gate（自适应选择高斯/DGR，高熵→高斯，低熵→DGR）
    run_one "$ds_name" "$ds_path" "$ds_in" "$ds_out" \
      "B3" "dynamic" "entropy_gate" 0.5 "false" "raw" "train"
    run_one "$ds_name" "$ds_path" "$ds_in" "$ds_out" \
      "B3" "dynamic" "entropy_gate" 0.5 "false" "raw" "test"
  fi

  if [[ "$RUN_E5" -eq 1 ]]; then
    # E5: DGR Sigma Offset（零初始退化，调制高斯核宽度，不替换先验分布）
    run_one "$ds_name" "$ds_path" "$ds_in" "$ds_out" \
      "E5" "sigma_offset" "replace" 0.5 "false" "raw" "train"
    run_one "$ds_name" "$ds_path" "$ds_in" "$ds_out" \
      "E5" "sigma_offset" "replace" 0.5 "false" "raw" "test"
  fi
done

echo "========================================================================"
echo "全部任务完成"

if [[ "$SHUTDOWN_AFTER" -eq 1 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] --shutdown 已指定，60 秒后关机..."
  shutdown -h +1 "AutoDL 实验完成，即将关机"
fi
