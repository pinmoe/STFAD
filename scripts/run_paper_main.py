import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = {
    "MSL": {
        "data_path": "data/MSL",
        "input_c": 55,
        "output_c": 55,
    },
    "SMAP": {
        "data_path": "data/SMAP",
        "input_c": 25,
        "output_c": 25,
    },
    "SKAB": {
        "data_path": "data/SKAB",
        "input_c": 8,
        "output_c": 8,
    },
    "ST330IR001_CP001": {
        "data_path": "data/ST330IR001.CP001",
        "input_c": 29,
        "output_c": 29,
        "win_size": 56,
    },
}


CONFIGS = {
    "E1_gaussian": [
        "--dgr_mode", "none",
        "--score_mode", "combined",
    ],
    "DGR_diff_E2": [
        "--dgr_mode", "dynamic",
        "--dgr_feature_mode", "diff",
        "--prior_fusion", "replace",
        "--score_mode", "combined",
    ],
    "DGR_raw": [
        "--dgr_mode", "dynamic",
        "--dgr_feature_mode", "raw",
        "--prior_fusion", "replace",
        "--score_mode", "combined",
    ],
    "DGR_raw_blend": [
        "--dgr_mode", "dynamic",
        "--dgr_feature_mode", "raw",
        "--prior_fusion", "blend",
        "--prior_alpha", "0.5",
        "--score_mode", "combined",
    ],
    "DGR_raw_recmean": [
        "--dgr_mode", "dynamic",
        "--dgr_feature_mode", "raw",
        "--prior_fusion", "replace",
        "--score_mode", "rec_mean",
    ],
    "B2_learnable_diff": [
        "--dgr_mode", "dynamic",
        "--dgr_feature_mode", "diff",
        "--prior_fusion", "blend",
        "--prior_alpha", "0.5",
        "--prior_alpha_learnable", "true",
        "--score_mode", "combined",
    ],
}


def run(cmd, dry_run):
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run paper main experiments with multiple seeds.")
    parser.add_argument("--datasets", nargs="+", default=["MSL", "SMAP", "SKAB", "ST330IR001_CP001"],
                        choices=sorted(DATASETS.keys()))
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        choices=sorted(CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--result_dir", default="results/paper_main")
    parser.add_argument("--checkpoint_root", default="checkpoints/paper_main")
    parser.add_argument("--threshold_mode", default="val_percentile",
                        choices=["train_percentile", "val_percentile", "val_grid", "oracle_ratio"])
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    main_py = repo_root / "main.py"

    for dataset in args.datasets:
        dataset_args = DATASETS[dataset]
        for config_name in args.configs:
            config_args = CONFIGS[config_name]
            for seed in args.seeds:
                experiment_name = config_name
                checkpoint_path = Path(args.checkpoint_root) / dataset / config_name / f"seed_{seed}"
                common = [
                    args.python, str(main_py),
                    "--dataset", dataset,
                    "--data_path", dataset_args["data_path"],
                    "--input_c", str(dataset_args["input_c"]),
                    "--output_c", str(dataset_args["output_c"]),
                    "--seed", str(seed),
                    "--model_save_path", str(checkpoint_path),
                    "--result_dir", args.result_dir,
                    "--experiment_name", experiment_name,
                    "--threshold_mode", args.threshold_mode,
                    "--threshold_percentile", str(args.threshold_percentile),
                ]
                if "win_size" in dataset_args:
                    common += ["--win_size", str(dataset_args["win_size"])]
                common += config_args

                if not args.skip_train:
                    run(common + ["--mode", "train"], args.dry_run)
                run(common + ["--mode", "test"], args.dry_run)


if __name__ == "__main__":
    main()
