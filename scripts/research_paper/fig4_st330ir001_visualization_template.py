# =============================================================================
#  Figure 4: ST330IR001 anomaly detection visualization template
#
#  This template draws a formal paper-style four-panel time-series figure:
#    (1) Representative raw sensor value
#    (2) E1 baseline anomaly score
#    (3) G2_recmean anomaly score
#    (4) Ground-truth anomaly label band
#
#  Fill either CSV_PATH or the four numpy arrays in load_from_numpy_arrays(),
#  then run from the project root:
#    python scripts/research_paper/fig4_st330ir001_visualization_template.py
#
#  Outputs:
#    pics/research_paper/fig4_st330ir001_visualization.pdf
#    pics/research_paper/fig4_st330ir001_visualization.svg
# =============================================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# --------------------------------------------------------------------------
# Path and font configuration
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "pics" / "research_paper"
DEFAULT_SCORE_DIR = PROJECT_ROOT / "outputs" / "fig4_scores"
DEFAULT_E1_SCORE_PATH = DEFAULT_SCORE_DIR / "E1_ST330IR001_CP001_scores.npz"
DEFAULT_G2_SCORE_PATH = DEFAULT_SCORE_DIR / "G2_ST330IR001_CP001_recmean_scores.npz"
DEFAULT_RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "ST330IR001.CP001"
    / "ST330IR001_CP001_test.npy"
)
DEFAULT_LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "ST330IR001.CP001"
    / "ST330IR001_CP001_test_label.npy"
)

CHINESE_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\STFANGSO.TTF"),
    Path(r"C:\Windows\Fonts\simfang.ttf"),
    PROJECT_ROOT / "pics" / "simsunb.ttf",
]


# --------------------------------------------------------------------------
# User configuration
# --------------------------------------------------------------------------

# Option A: CSV input.
# Replace None with your CSV file path, for example:
# CSV_PATH = r"C:\VSCode\Anomaly-Transformer\data\ST330IR001\fig4_segment.csv"
#
# Required CSV columns:
#   raw_value : representative sensor raw value, shape = (T,)
#   score_e1  : E1 baseline anomaly score, shape = (T,)
#   score_g2  : G2_recmean anomaly score, shape = (T,)
#   label     : ground-truth label, shape = (T,), 0 = normal, 1 = anomaly
CSV_PATH = None

# Option B: numpy-array input.
# If CSV_PATH is None, fill the arrays in load_from_numpy_arrays().

# Select a 200-300 step segment containing obvious anomalies.
# These indices are applied after loading the full arrays.
SEGMENT_LENGTH = 260
START_IDX = None
END_IDX = None

# Thresholds for anomaly-score panels.
# Replace None with your experimental thresholds if available.
# If kept as None, the template uses the 95th percentile of the selected segment
# as a display-only fallback.
THRESHOLD_E1 = None
THRESHOLD_G2 = None


# --------------------------------------------------------------------------
# Style utilities
# --------------------------------------------------------------------------

def setup_publication_style():
    """Set fonts and vector export options for formal paper figures."""
    chinese_font_path = next((p for p in CHINESE_FONT_CANDIDATES if p.exists()), None)
    if chinese_font_path is not None:
        fm.fontManager.addfont(str(chinese_font_path))

    chinese_font = (
        fm.FontProperties(fname=str(chinese_font_path))
        if chinese_font_path is not None
        else fm.FontProperties(family="SimSun")
    )

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "FangSong", "serif"],
            "font.serif": ["Times New Roman", "SimSun", "FangSong"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 1.05,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.95,
            "ytick.major.width": 0.95,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return chinese_font


def style_axis(ax):
    """Apply clean thesis-style axis formatting."""
    ax.tick_params(
        which="major",
        direction="in",
        length=3.3,
        width=0.95,
        labelsize=8.4,
        top=False,
        right=False,
    )
    ax.grid(axis="y", color="#e7e7e7", linewidth=0.48, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_from_csv(csv_path):
    """
    Load ST330IR001 visualization data from CSV.

    The CSV file must contain at least four columns:
      raw_value : shape = (T,), representative sensor raw value
      score_e1  : shape = (T,), anomaly score from E1 baseline
      score_g2  : shape = (T,), anomaly score from G2_recmean
      label     : shape = (T,), 0 = normal, 1 = anomaly

    T is the time length of the full available test sequence or selected file.
    """
    df = pd.read_csv(csv_path)
    required_columns = ["raw_value", "score_e1", "score_g2", "label"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    raw_value = df["raw_value"].to_numpy(dtype=float)
    score_e1 = df["score_e1"].to_numpy(dtype=float)
    score_g2 = df["score_g2"].to_numpy(dtype=float)
    label = df["label"].to_numpy(dtype=int)
    return raw_value, score_e1, score_g2, label


def load_from_numpy_arrays():
    """
    Paste your numpy arrays here when not using CSV input.

    Required arrays:
      raw_value: shape = (T,), dtype float
        Representative sensor channel raw value.

      score_e1: shape = (T,), dtype float
        E1 Gaussian-prior baseline anomaly score.

      score_g2: shape = (T,), dtype float
        G2_recmean anomaly score.

      label: shape = (T,), dtype int
        Ground-truth anomaly label. label = 1 means anomaly,
        label = 0 means normal.

    T is the time length. All four arrays must have the same T.
    """
    raw_value = np.array([], dtype=float)
    score_e1 = np.array([], dtype=float)
    score_g2 = np.array([], dtype=float)
    label = np.array([], dtype=int)
    return raw_value, score_e1, score_g2, label


def load_from_default_exports():
    """Load Figure 4 data exported by solver.py --export_score_path."""
    missing = [
        str(path)
        for path in (DEFAULT_E1_SCORE_PATH, DEFAULT_G2_SCORE_PATH, DEFAULT_RAW_DATA_PATH, DEFAULT_LABEL_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Figure 4 needs exported score files first. Missing:\n  "
            + "\n  ".join(missing)
            + "\n\nRun E1 and G2 test export commands, then rerun this script."
        )

    raw = np.load(DEFAULT_RAW_DATA_PATH)
    labels = np.load(DEFAULT_LABEL_PATH).reshape(-1).astype(int)
    e1 = np.load(DEFAULT_E1_SCORE_PATH)
    g2 = np.load(DEFAULT_G2_SCORE_PATH)

    raw_value = raw.reshape(-1, raw.shape[-1])[:, 0].astype(float)
    score_e1 = np.asarray(e1["score"], dtype=float).reshape(-1)
    score_g2 = np.asarray(g2["score"], dtype=float).reshape(-1)
    label = labels.reshape(-1)
    return raw_value, score_e1, score_g2, label


def validate_arrays(raw_value, score_e1, score_g2, label):
    """Validate shape consistency and label values."""
    arrays = [
        np.asarray(raw_value),
        np.asarray(score_e1),
        np.asarray(score_g2),
        np.asarray(label),
    ]
    lengths = [arr.shape[0] for arr in arrays]
    if any(arr.ndim != 1 for arr in arrays):
        raise ValueError("raw_value, score_e1, score_g2, and label must all be 1-D arrays.")
    if len(set(lengths)) != 1:
        raise ValueError(f"All arrays must have the same length T. Got lengths: {lengths}")
    if lengths[0] == 0:
        raise ValueError(
            "No data provided. Set CSV_PATH or paste arrays in load_from_numpy_arrays()."
        )

    unique_labels = set(np.unique(label).astype(int).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError("label must contain only 0 and 1, where 1 = anomaly and 0 = normal.")


def resolve_segment_indices(label, start_idx, end_idx, segment_length):
    """Use explicit indices or pick a segment around the first anomaly."""
    if start_idx is not None and end_idx is not None:
        return int(start_idx), int(end_idx)

    total_length = len(label)
    anomaly_indices = np.flatnonzero(np.asarray(label, dtype=int) == 1)
    if anomaly_indices.size:
        center = int(anomaly_indices[0])
        start_idx = max(0, center - segment_length // 3)
    else:
        start_idx = 0
    end_idx = min(total_length, start_idx + segment_length)
    start_idx = max(0, end_idx - segment_length)
    return int(start_idx), int(end_idx)


def slice_segment(raw_value, score_e1, score_g2, label, start_idx, end_idx):
    """Slice a 200-300 step segment for visualization."""
    total_length = len(label)
    start_idx, end_idx = resolve_segment_indices(label, start_idx, end_idx, SEGMENT_LENGTH)
    if not (0 <= start_idx < end_idx <= total_length):
        raise ValueError(
            f"Invalid segment [{start_idx}:{end_idx}] for sequence length T={total_length}."
        )

    segment_length = end_idx - start_idx
    if not (200 <= segment_length <= 300):
        print(
            f"Warning: selected segment length is {segment_length}. "
            "The requested figure usually uses 200-300 time steps."
        )

    return (
        np.asarray(raw_value[start_idx:end_idx], dtype=float),
        np.asarray(score_e1[start_idx:end_idx], dtype=float),
        np.asarray(score_g2[start_idx:end_idx], dtype=float),
        np.asarray(label[start_idx:end_idx], dtype=int),
        np.arange(start_idx, end_idx),
    )


# --------------------------------------------------------------------------
# Anomaly interval parsing
# --------------------------------------------------------------------------

def parse_anomaly_intervals(label_segment):
    """
    Parse contiguous anomaly intervals from a binary label segment.

    Returns:
      intervals: list of (start, end) pairs in segment-local indices.
                 The interval is inclusive-exclusive: [start, end).
    """
    binary_label = np.asarray(label_segment, dtype=int)
    padded = np.r_[0, binary_label, 0]
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts, ends))


def apply_anomaly_highlight(ax, time_axis, intervals):
    """Highlight all true-anomaly intervals with light red background spans."""
    for start, end in intervals:
        x0 = time_axis[start]
        x1 = time_axis[end - 1] if end > start else time_axis[start]
        ax.axvspan(x0, x1, color="#f2b6b0", alpha=0.28, linewidth=0, zorder=1)


def resolve_threshold(score_segment, threshold):
    """
    Return a threshold for plotting.

    If threshold is None, use the 95th percentile of the selected segment as a
    display-only fallback. Replace this with your experimental threshold when
    preparing the final paper figure.
    """
    if threshold is None:
        return float(np.percentile(score_segment, 95))
    return float(threshold)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_st330ir001_visualization(
    raw_value,
    score_e1,
    score_g2,
    label,
    start_idx,
    end_idx,
    threshold_e1=None,
    threshold_g2=None,
):
    """Draw and save the four-panel ST330IR001 anomaly detection figure."""
    chinese_font = setup_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_arrays(raw_value, score_e1, score_g2, label)
    raw_seg, e1_seg, g2_seg, label_seg, time_axis = slice_segment(
        raw_value, score_e1, score_g2, label, start_idx, end_idx
    )

    anomaly_intervals = parse_anomaly_intervals(label_seg)
    threshold_e1 = resolve_threshold(e1_seg, threshold_e1)
    threshold_g2 = resolve_threshold(g2_seg, threshold_g2)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16 / 2.54, 12 / 2.54),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={
            "height_ratios": [1.12, 1.12, 1.12, 0.30],
            "hspace": 0.10,
        },
    )

    # Panels (1)-(3): line plots with the same anomaly background highlights.
    for ax in axes[:3]:
        apply_anomaly_highlight(ax, time_axis, anomaly_intervals)
        ax.set_xlim(time_axis[0], time_axis[-1])
        style_axis(ax)

    axes[0].plot(time_axis, raw_seg, color="#666666", linewidth=1.05, zorder=3)
    axes[0].set_ylabel("原始值", fontproperties=chinese_font, fontsize=9.2)
    axes[0].text(
        0.01,
        0.86,
        "（a）代表性传感器通道",
        transform=axes[0].transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        fontproperties=chinese_font,
    )

    axes[1].plot(time_axis, e1_seg, color="#2b4f73", linewidth=1.08, zorder=3)
    axes[1].axhline(
        threshold_e1,
        color="#2b4f73",
        linewidth=0.95,
        linestyle=(0, (4, 2)),
        zorder=2,
    )
    axes[1].set_ylabel("异常分数", fontproperties=chinese_font, fontsize=9.2)
    axes[1].text(
        0.01,
        0.86,
        "（b）E1 基线异常分数",
        transform=axes[1].transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        fontproperties=chinese_font,
    )
    axes[1].text(
        0.985,
        0.82,
        "阈值",
        transform=axes[1].transAxes,
        ha="right",
        va="center",
        fontsize=7.6,
        color="#2b4f73",
        fontproperties=chinese_font,
    )

    axes[2].plot(time_axis, g2_seg, color="#b9552d", linewidth=1.08, zorder=3)
    axes[2].axhline(
        threshold_g2,
        color="#b9552d",
        linewidth=0.95,
        linestyle=(0, (4, 2)),
        zorder=2,
    )
    axes[2].set_ylabel("异常分数", fontproperties=chinese_font, fontsize=9.2)
    axes[2].text(
        0.01,
        0.86,
        "（c）G2_recmean 异常分数",
        transform=axes[2].transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        fontproperties=chinese_font,
    )
    axes[2].text(
        0.985,
        0.82,
        "阈值",
        transform=axes[2].transAxes,
        ha="right",
        va="center",
        fontsize=7.6,
        color="#b9552d",
        fontproperties=chinese_font,
    )

    # Panel (4): ground-truth label band. Green = normal, red = anomaly.
    axes[3].imshow(
        label_seg[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#4f9d69", "#c7524b"]),
        extent=[time_axis[0], time_axis[-1], 0, 1],
        zorder=3,
    )
    axes[3].set_yticks([])
    axes[3].set_ylabel("真实标签", fontproperties=chinese_font, fontsize=9.2)
    axes[3].set_xlabel("时间步 t", fontproperties=chinese_font, fontsize=9.6)
    axes[3].tick_params(
        which="major",
        direction="in",
        length=3.3,
        width=0.95,
        labelsize=8.4,
        top=False,
        right=False,
    )
    for spine in axes[3].spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)

    axes[0].legend(
        handles=[
            Patch(facecolor="#f2b6b0", alpha=0.28, edgecolor="none", label="异常区段"),
            Patch(facecolor="#4f9d69", edgecolor="none", label="正常"),
            Patch(facecolor="#c7524b", edgecolor="none", label="异常"),
        ],
        loc="upper right",
        prop=chinese_font.copy(),
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#7f7f7f",
        fontsize=7.5,
        ncol=3,
        borderpad=0.32,
        handlelength=1.20,
        handletextpad=0.40,
        columnspacing=0.80,
    )

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.090, top=0.975)

    pdf_path = OUTPUT_DIR / "fig4_st330ir001_visualization.pdf"
    svg_path = OUTPUT_DIR / "fig4_st330ir001_visualization.svg"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


def main():
    if CSV_PATH is not None:
        raw_value, score_e1, score_g2, label = load_from_csv(CSV_PATH)
    else:
        try:
            raw_value, score_e1, score_g2, label = load_from_default_exports()
        except FileNotFoundError as exc:
            print(exc)
            raw_value, score_e1, score_g2, label = load_from_numpy_arrays()

    try:
        plot_st330ir001_visualization(
            raw_value=raw_value,
            score_e1=score_e1,
            score_g2=score_g2,
            label=label,
            start_idx=START_IDX,
            end_idx=END_IDX,
            threshold_e1=THRESHOLD_E1,
            threshold_g2=THRESHOLD_G2,
        )
    except ValueError as exc:
        if "No data provided" not in str(exc):
            raise
        raise SystemExit(
            "\n图4模板尚未接入数据，因此未生成图片。\n"
            "请二选一提供数据：\n"
            "  1) 设置 CSV_PATH，CSV 必须包含 raw_value, score_e1, score_g2, label 四列；\n"
            "  2) 在 load_from_numpy_arrays() 中填入四个一维数组：\n"
            "     raw_value=(T,), score_e1=(T,), score_g2=(T,), label=(T,), label 取 0/1。\n\n"
            "提示：本地已有 ST330IR001_CP001_test.npy 和 test_label.npy，"
            "但 E1/G2_recmean 的逐时间步异常分数未在当前仓库中落盘；"
            "需要先导出 score_e1 与 score_g2，或整理为上述 CSV 后再运行。\n"
        ) from None


if __name__ == "__main__":
    main()
