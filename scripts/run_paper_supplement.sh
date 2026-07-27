#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/paper_supplement}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/paper_supplement}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/paper_supplement}"
SEEDS="${SEEDS:-2024 2025 2026}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-256}"
RESUME="${RESUME:-1}"
RUN_STFAD="${RUN_STFAD:-1}"
RUN_SCORE_ABLATION="${RUN_SCORE_ABLATION:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

mkdir -p "${RESULT_ROOT}" "${CKPT_ROOT}" "${LOG_ROOT}"
FAILED="${RESULT_ROOT}/failed_runs.tsv"
MASTER_LOG="${LOG_ROOT}/master.log"
: > "${FAILED}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"
}

record_failed() {
  local dataset="$1" config="$2" seed="$3" stage="$4" code="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${config}" "${seed}" "${stage}" "${code}" >> "${FAILED}"
}

run_cmd() {
  local dataset="$1" config="$2" seed="$3" stage="$4" logfile="$5"
  shift 5
  log "RUN ${dataset}/${config}/seed_${seed}/${stage}"
  "$@" > >(tee -a "${logfile}") 2> >(tee -a "${logfile}" >&2)
  local code=$?
  if [[ ${code} -ne 0 ]]; then
    log "FAIL ${dataset}/${config}/seed_${seed}/${stage} exit=${code}"
    record_failed "${dataset}" "${config}" "${seed}" "${stage}" "${code}"
  fi
  return ${code}
}

metrics_complete() {
  local metrics="$1" dataset="$2" config="$3" seed="$4"
  [[ -f "${metrics}" ]] || return 1
  python - "$metrics" "$dataset" "$config" "$seed" <<'PY'
import json, sys
path, dataset, config, seed = sys.argv[1:5]
m = json.load(open(path, encoding="utf-8"))
ok = str(m.get("dataset")) == dataset and str(m.get("experiment_name")) == config and str(m.get("seed")) == seed
raise SystemExit(0 if ok else 1)
PY
}

save_environment() {
  {
    echo "commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "status_start"
    git status --short 2>/dev/null || true
    echo "gpu_start"
    nvidia-smi 2>/dev/null || true
    echo "python_env_start"
    python - <<'PY'
import sys, torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_cuda", torch.version.cuda)
PY
  } > "${RESULT_ROOT}/environment.txt"
}

save_environment

DATASETS=("MSL:55:55:100" "SMAP:25:25:100" "SKAB:8:8:100" "ST330IR001_CP001:29:29:56")
CONFIGS=("E1_gaussian:none:replace:diff:combined" "E3_multiscale:multiscale:replace:diff:combined")

if [[ "${RUN_STFAD}" == "1" ]]; then
  for entry in "${DATASETS[@]}"; do
    IFS=: read -r dataset input_c output_c win_size <<< "${entry}"
    data_path="${DATA_ROOT}/${dataset}"
    [[ "${dataset}" == "ST330IR001_CP001" ]] && data_path="${DATA_ROOT}/ST330IR001.CP001"
    for config_entry in "${CONFIGS[@]}"; do
      IFS=: read -r config dgr prior feature score <<< "${config_entry}"
      for seed in ${SEEDS}; do
        ckpt="${CKPT_ROOT}/${dataset}/${config}/seed_${seed}"
        metrics="${RESULT_ROOT}/${dataset}/${config}/seed_${seed}/metrics.json"
        logfile="${LOG_ROOT}/${dataset}_${config}_seed_${seed}.log"
        mkdir -p "${ckpt}" "$(dirname "${logfile}")"
        common=(python main.py --dataset "${dataset}" --data_path "${data_path}" --input_c "${input_c}" --output_c "${output_c}"
          --win_size "${win_size}" --seed "${seed}" --k 3 --temperature 50 --score_mode "${score}"
          --d_model 512 --dropout 0.0 --lr 1e-4 --batch_size "${TRAIN_BATCH_SIZE}"
          --threshold_mode val_percentile --threshold_percentile 95.0 --eval_unit auto
          --window_score_agg mean --window_threshold_mode val_percentile
          --model_save_path "${ckpt}" --result_dir "${RESULT_ROOT}" --experiment_name "${config}"
          --dgr_mode "${dgr}" --prior_fusion "${prior}" --dgr_feature_mode "${feature}")
        if [[ "${RESUME}" != "1" || ! -f "${ckpt}/${dataset}_checkpoint.pth" ]]; then
          CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" run_cmd "${dataset}" "${config}" "${seed}" train "${logfile}" "${common[@]}" --mode train
        else
          log "SKIP train ${dataset}/${config}/seed_${seed}: checkpoint exists"
        fi
        if [[ "${RESUME}" == "1" ]] && metrics_complete "${metrics}" "${dataset}" "${config}" "${seed}"; then
          log "SKIP test ${dataset}/${config}/seed_${seed}: metrics complete"
        else
          test_common=("${common[@]}")
          test_common+=(--batch_size "${TEST_BATCH_SIZE}")
          CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" run_cmd "${dataset}" "${config}" "${seed}" test "${logfile}" "${test_common[@]}" --mode test
        fi
      done
    done
  done
fi

if [[ "${RUN_SCORE_ABLATION}" == "1" ]]; then
  for entry in "${DATASETS[@]}"; do
    IFS=: read -r dataset input_c output_c win_size <<< "${entry}"
    data_path="${DATA_ROOT}/${dataset}"
    [[ "${dataset}" == "ST330IR001_CP001" ]] && data_path="${DATA_ROOT}/ST330IR001.CP001"
    for config_entry in "${CONFIGS[@]}"; do
      IFS=: read -r config dgr prior feature _score <<< "${config_entry}"
      for seed in ${SEEDS}; do
        ckpt="${CKPT_ROOT}/${dataset}/${config}/seed_${seed}"
        ablation_config="${config}_rec_mean"
        metrics="${RESULT_ROOT}/${dataset}/${ablation_config}/seed_${seed}/metrics.json"
        logfile="${LOG_ROOT}/${dataset}_${ablation_config}_seed_${seed}.log"
        mkdir -p "$(dirname "${logfile}")"
        if [[ ! -f "${ckpt}/${dataset}_checkpoint.pth" ]]; then
          log "SKIP score ablation ${dataset}/${ablation_config}/seed_${seed}: checkpoint missing"
          record_failed "${dataset}" "${ablation_config}" "${seed}" "missing_checkpoint" "1"
          continue
        fi
        if [[ "${RESUME}" == "1" ]] && metrics_complete "${metrics}" "${dataset}" "${ablation_config}" "${seed}"; then
          log "SKIP score ablation ${dataset}/${ablation_config}/seed_${seed}: metrics complete"
          continue
        fi
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" run_cmd "${dataset}" "${ablation_config}" "${seed}" test "${logfile}" \
          python main.py --dataset "${dataset}" --data_path "${data_path}" --input_c "${input_c}" --output_c "${output_c}" \
          --win_size "${win_size}" --seed "${seed}" --k 3 --temperature 50 --score_mode rec_mean \
          --d_model 512 --dropout 0.0 --lr 1e-4 --batch_size "${TEST_BATCH_SIZE}" \
          --threshold_mode val_percentile --threshold_percentile 95.0 --eval_unit auto \
          --window_score_agg mean --window_threshold_mode val_percentile \
          --model_save_path "${ckpt}" --result_dir "${RESULT_ROOT}" --experiment_name "${ablation_config}" \
          --dgr_mode "${dgr}" --prior_fusion "${prior}" --dgr_feature_mode "${feature}" --mode test
      done
    done
  done
fi

if [[ "${RUN_BASELINES}" == "1" ]]; then
  for seed in ${SEEDS}; do
    run_cmd "ST330IR001_CP001" "baselines" "${seed}" test "${LOG_ROOT}/baseline_seed_${seed}.log" \
      python baselines/run_baselines.py --data_root "${DATA_ROOT}/ST330IR001.CP001" --dataset ST330IR001_CP001 \
      --seed "${seed}" --result_root "${RESULT_ROOT}" --methods iforest ocsvm autoencoder
  done
fi

python tools/summarize_paper_results.py --result_root "${RESULT_ROOT}" --output_dir "${RESULT_ROOT}" --expected_seeds ${SEEDS}
summary_code=$?
if [[ ${summary_code} -ne 0 ]]; then
  record_failed "ALL" "summary" "NA" "summarize" "${summary_code}"
fi

log "DONE failed_runs=${FAILED}"
if [[ "${AUTO_SHUTDOWN}" == "1" ]]; then
  shutdown -h now
fi
