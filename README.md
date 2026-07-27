# Anomaly-Transformer (Research Extension)
Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

This repository extends the original ICLR 2022 Anomaly Transformer implementation into a broader experimental framework for industrial and spacecraft time-series anomaly detection. In addition to the original Association Discrepancy objective, this branch introduces:

- multiple prior construction modes (`dgr_mode`),
- explicit prior fusion strategies (`prior_fusion`),
- and post-hoc/training-time scoring enhancements for ablation studies.

<p align="center">
<img src=".\pics\structure.png" height="350" alt="Anomaly Transformer" align="center" />
</p>

## Get Started

1. Install Python and dependencies (PyTorch, NumPy, pandas, scikit-learn, matplotlib).
```bash
pip install torch numpy pandas scikit-learn matplotlib
```
2. Prepare datasets under `data/` in `.npy` format.
3. Train and evaluate using either single-run commands or batch scripts.

Example (MSL, baseline E1):
```bash
python main.py --mode train --dataset MSL --data_path data/MSL --input_c 55 --output_c 55 --dgr_mode none --model_save_path checkpoints/E1_MSL
python main.py --mode test  --dataset MSL --data_path data/MSL --input_c 55 --output_c 55 --dgr_mode none --model_save_path checkpoints/E1_MSL
```

## Supported Datasets

Current data loaders support:

- MSL
- SMAP
- SKAB
- HAI
- PSM
- BATADAL
- SMD
- ST330IR001_CP001

The main large-scale scripts in this branch focus on MSL/SMAP/SKAB, while dedicated preprocessing/utilities are provided for HAI and BATADAL.

## Data Preparation

If preprocessed arrays are available, use the following naming convention:

- `data/<DATASET>/<DATASET>_train.npy`
- `data/<DATASET>/<DATASET>_test.npy`
- `data/<DATASET>/<DATASET>_test_label.npy`

For raw-data conversion:

- HAI:
```bash
python scripts/prepare_hai.py --data_dir data/HAI/hai-22.04 --output_dir data/HAI
```
- BATADAL:
```bash
python scripts/prepare_batadal.py --src_dir <raw_csv_dir> --dst_dir data/BATADAL
```
- ST330IR001.CP001:
```bash
python scripts/prepare_st330ir001_cp001.py --raw_root data/ST330IR001.CP001/raw --output_dir data/ST330IR001.CP001
python main.py --mode train --dataset ST330IR001_CP001 --data_path data/ST330IR001.CP001 --win_size 56 --input_c 29 --output_c 29 --eval_unit auto --threshold_mode val_percentile --threshold_percentile 95 --model_save_path checkpoints/E1_ST330IR001_CP001
python main.py --mode test  --dataset ST330IR001_CP001 --data_path data/ST330IR001.CP001 --win_size 56 --input_c 29 --output_c 29 --eval_unit auto --threshold_mode val_percentile --threshold_percentile 95 --model_save_path checkpoints/E1_ST330IR001_CP001
```

## Main Experimental Protocol

This branch organizes experiments into two groups.

### Journal/AutoDL reproducible protocol

Run the journal-oriented multi-seed protocol on AutoDL:
```bash
export DATA_ROOT=/root/autodl-tmp/STFAD/data
export RESULT_ROOT=/root/autodl-tmp/STFAD/results/paper_supplement
export CKPT_ROOT=/root/autodl-tmp/STFAD/checkpoints/paper_supplement
export LOG_ROOT=/root/autodl-tmp/STFAD/logs/paper_supplement
export SEEDS="2024 2025 2026"
export CUDA_DEVICE=0
export TRAIN_BATCH_SIZE=256
export TEST_BATCH_SIZE=256
export RESUME=1
export RUN_STFAD=1
export RUN_SCORE_ABLATION=1
export RUN_BASELINES=1
export AUTO_SHUTDOWN=0

bash scripts/run_paper_supplement.sh 2>&1 | tee "${LOG_ROOT}/tmux_master.log"
```

Summarize saved `metrics.json` files into paper tables:
```bash
python tools/summarize_paper_results.py --result_root results/paper_supplement --output_dir results/paper_supplement --expected_seeds 2024 2025 2026
```

Each STFAD test run writes structured metrics and arrays under:
```text
results/paper_supplement/<dataset>/<experiment_name>/seed_<seed>/
```

The supplement script also writes `summary_long.csv`, `summary_mean_std.csv`,
`run_manifest.json`, `failed_runs.tsv`, per-run logs, and an environment record.

### E/B Series (model/prior design)

Run all E1-E5 and B1-B3 experiments:
```bash
bash scripts/run_all_experiments.sh
```

Definitions:

- E1: Gaussian prior (`dgr_mode=none`)
- E2: Dynamic DGR prior (`dgr_mode=dynamic`)
- E3: Multi-scale dynamic DGR prior (`dgr_mode=multiscale`)
- E4: Static learnable DGR prior (`dgr_mode=static`)
- E5: Sigma-offset prior (`dgr_mode=sigma_offset`)
- B1: Dynamic prior + blend fusion (fixed alpha)
- B2: Dynamic prior + blend fusion (learnable alpha)
- B3: Dynamic prior + entropy-gated fusion

### A/B/C Series (scoring/training ablations)

Run ablations for Direction A/B/C:
```bash
bash run_abc_experiments.sh
```

- A: differential reconstruction auxiliary scoring (`diff_beta`, test-time only)
- B: local z-score post-processing (`score_local_z_win`, test-time only)
- C: differential reconstruction regularization (`lambda_diff`, training-time)

## Key Arguments

Prior construction and fusion:

- `--dgr_mode`: `none | dynamic | multiscale | static | sigma_offset | dynamic_pe`
- `--prior_fusion`: `replace | blend | entropy_gate`
- `--prior_alpha`, `--prior_alpha_learnable`
- `--dgr_input_mode`: `raw | smoothed`
- `--dgr_feature_mode`: `diff | raw`
- `--prior_entropy_tau`, `--prior_entropy_gamma`

Scoring and post-processing:

- `--score_mode`: `combined | rec_only | rec_mean | weighted | chan_var`
- `--score_alpha`
- `--score_smooth_k`
- `--diff_beta`
- `--score_local_z_win`

Training enhancement:

- `--lambda_diff`

## Diagnostics and Figures

Generate the full paper-style figure suite:
```bash
python tools/make_all_figures.py
```
Outputs are saved to `figures/paper/`.

PRC/AUPRC diagnostics (E1 vs E4):
```bash
python diagnostics/plot_prc.py --dataset MSL --e1_ckpt checkpoints/E1_MSL --e4_ckpt checkpoints/E4_MSL
```
Outputs are saved to `diagnostics/`.

## Repository Structure

- `main.py`: argument parser and entry point
- `solver.py`: training, validation, and test pipeline
- `model/`: Anomaly Transformer and prior modules
- `data_factory/data_loader.py`: dataset loaders
- `scripts/`: preprocessing and experiment scripts
- `tools/`: plotting utilities
- `diagnostics/`: analysis scripts
- `checkpoints/`: saved model weights
- `logs/`: experiment logs

## Citation
If you find this repository useful, please cite the original paper:

```bibtex
@inproceedings{
xu2022anomaly,
title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
author={Jiehui Xu and Haixu Wu and Jianmin Wang and Mingsheng Long},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=LzQQ89U1qm_}
}
```

## Contact
For questions regarding the original Anomaly Transformer paper, please contact the original authors.
