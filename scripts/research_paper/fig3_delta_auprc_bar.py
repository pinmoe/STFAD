# =============================================================================
#  Figure 3: Delta AUPRC bar chart
#
#  Run from the project root:
#    python scripts/research_paper/fig3_delta_auprc_bar.py
#
#  Outputs:
#    pics/research_paper/fig3_delta_auprc_bar.pdf
#    pics/research_paper/fig3_delta_auprc_bar.svg
# =============================================================================

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "pics" / "research_paper"

CHINESE_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\STFANGSO.TTF"),
    Path(r"C:\Windows\Fonts\simfang.ttf"),
    PROJECT_ROOT / "pics" / "simsunb.ttf",
]

DATASETS = ["SKAB", "MSL", "SMAP", "ST330IR001"]
DELTA_AUPRC = np.array([0.0249, 0.0026, 0.0104, 0.0031])
BEST_LABEL_INDICES = [2, 3]


def setup_publication_style():
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


def format_axis(ax, chinese_font):
    y_max = 0.030
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(-0.58, len(DATASETS) - 0.42)
    ax.set_ylabel(r"$\Delta$AUPRC（相对基线 E1）", fontproperties=chinese_font, fontsize=9.6)

    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels(DATASETS, fontsize=8.8)
    ax.set_yticks(np.arange(0.0, y_max + 0.0001, 0.005))

    ax.tick_params(
        which="major",
        direction="in",
        length=3.3,
        width=0.95,
        labelsize=8.5,
        top=False,
        right=False,
    )

    ax.grid(axis="y", color="#e6e6e6", linewidth=0.50, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)


def add_value_labels(ax, bars):
    for bar, value in zip(bars, DELTA_AUPRC):
        x_center = bar.get_x() + bar.get_width() / 2
        y_top = bar.get_height()
        ax.text(
            x_center,
            y_top + 0.00075,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color="#222222",
        )


def add_recommendation_labels(ax, bars, chinese_font):
    for idx in BEST_LABEL_INDICES:
        bar = bars[idx]
        x_center = bar.get_x() + bar.get_width() / 2
        y_top = bar.get_height()
        ax.text(
            x_center,
            y_top + 0.0030,
            "★推荐配置",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#8f2d28",
            fontproperties=chinese_font,
        )


def main():
    chinese_font = setup_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10 / 2.54, 7 / 2.54), constrained_layout=False)

    x = np.arange(len(DATASETS))
    bars = ax.bar(
        x,
        DELTA_AUPRC,
        width=0.60,
        color="#2f6f73",
        alpha=0.88,
        edgecolor="#1d4447",
        linewidth=0.95,
        zorder=3,
    )

    format_axis(ax, chinese_font)
    add_value_labels(ax, bars)
    add_recommendation_labels(ax, bars, chinese_font)

    fig.subplots_adjust(left=0.165, right=0.980, bottom=0.170, top=0.940)

    pdf_path = OUTPUT_DIR / "fig3_delta_auprc_bar.pdf"
    svg_path = OUTPUT_DIR / "fig3_delta_auprc_bar.svg"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
