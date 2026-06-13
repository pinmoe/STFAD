# =============================================================================
#  Figure 2: Precision-Recall scatter plots for SKAB / MSL / SMAP
#
#  Run from the project root:
#    python scripts/research_paper/fig2_precision_recall_scatter.py
#
#  Outputs:
#    pics/research_paper/fig2_precision_recall_scatter.pdf
#    pics/research_paper/fig2_precision_recall_scatter.svg
# =============================================================================

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "pics" / "research_paper"

CHINESE_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\STFANGSO.TTF"),
    Path(r"C:\Windows\Fonts\simfang.ttf"),
    PROJECT_ROOT / "pics" / "simsunb.ttf",
]

DATA = {
    "SKAB": [
        {"name": "E1", "recall": 0.7931, "precision": 0.8070, "kind": "E1"},
        {"name": "E2", "recall": 0.7931, "precision": 0.8413, "kind": "BEST"},
    ],
    "MSL": [
        {"name": "E1", "recall": 0.9445, "precision": 0.9041, "kind": "E1"},
        {"name": "E2", "recall": 0.8454, "precision": 0.9122, "kind": "OTHER"},
        {"name": "BEST", "recall": 0.8542, "precision": 0.8890, "kind": "BEST"},
    ],
    "SMAP": [
        {"name": "E1", "recall": 0.5735, "precision": 0.9239, "kind": "E1"},
        {"name": "E2", "recall": 0.5735, "precision": 0.9285, "kind": "OTHER"},
        {"name": "BEST", "recall": 0.5505, "precision": 0.9230, "kind": "BEST"},
    ],
}

PANEL_TITLES = {
    "SKAB": "(a) SKAB",
    "MSL": "(b) MSL",
    "SMAP": "(c) SMAP",
}

LABEL_OFFSETS = {
    ("SKAB", "E1"): (7, -11),
    ("SKAB", "E2"): (7, 5),
    ("MSL", "E1"): (-25, 8),
    ("MSL", "BEST"): (7, -12),
    ("SMAP", "E1"): (7, 7),
    ("SMAP", "BEST"): (-38, -12),
}


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


def draw_point(ax, point):
    recall = point["recall"]
    precision = point["precision"]
    kind = point["kind"]

    if kind == "E1":
        ax.scatter(
            recall,
            precision,
            s=62,
            marker="o",
            facecolors="white",
            edgecolors="#111111",
            linewidths=1.15,
            zorder=4,
        )
    elif kind == "BEST":
        ax.scatter(
            recall,
            precision,
            s=125,
            marker="*",
            facecolors="#c62828",
            edgecolors="#8e1b1b",
            linewidths=0.85,
            zorder=5,
        )
    else:
        ax.scatter(
            recall,
            precision,
            s=30,
            marker="o",
            facecolors="#8f969e",
            edgecolors="#6b7178",
            linewidths=0.50,
            zorder=3,
        )


def annotate_key_points(ax, dataset_name):
    for point in DATA[dataset_name]:
        if point["kind"] == "E1":
            label = "E1"
        elif point["kind"] == "BEST":
            label = "BEST"
        else:
            continue

        dx, dy = LABEL_OFFSETS[(dataset_name, point["name"])]
        ax.annotate(
            label,
            xy=(point["recall"], point["precision"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.2,
            color="#9e1f1f" if label == "BEST" else "#222222",
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.60,
                "color": "#777777",
                "shrinkA": 1,
                "shrinkB": 3,
            },
            zorder=6,
        )


def format_axis(ax, title):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9.6, pad=5)
    ax.set_xlabel("Recall", fontsize=9.2)

    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(
        which="major",
        direction="in",
        length=3.3,
        width=0.95,
        labelsize=8.4,
        top=False,
        right=False,
    )

    ax.grid(True, color="#e7e7e7", linewidth=0.48, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)


def add_msl_degradation_annotation(ax, chinese_font):
    ellipse = Ellipse(
        xy=(0.850, 0.900),
        width=0.095,
        height=0.080,
        angle=0,
        facecolor="none",
        edgecolor="#7a4a45",
        linewidth=1.05,
        linestyle=(0, (4, 2)),
        zorder=2,
    )
    ax.add_patch(ellipse)

    ax.annotate(
        "B2 崩溃区\n可学习融合退化区",
        xy=(0.850, 0.900),
        xytext=(0.610, 0.755),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=7.7,
        color="#5c403d",
        fontproperties=chinese_font,
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.70,
            "color": "#7a4a45",
            "shrinkA": 2,
            "shrinkB": 4,
        },
        zorder=6,
    )


def build_legend():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#111111",
            markeredgewidth=1.15,
            markersize=5.8,
            label="E1 高斯先验基线",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#c62828",
            markeredgecolor="#8e1b1b",
            markeredgewidth=0.85,
            markersize=9.4,
            label="BEST 推荐配置",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#8f969e",
            markeredgecolor="#6b7178",
            markeredgewidth=0.50,
            markersize=4.5,
            label="其它变体",
        ),
    ]


def main():
    chinese_font = setup_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16 / 2.54, 5.55 / 2.54),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for ax, dataset_name in zip(axes, ["SKAB", "MSL", "SMAP"]):
        for point in DATA[dataset_name]:
            draw_point(ax, point)

        annotate_key_points(ax, dataset_name)
        format_axis(ax, PANEL_TITLES[dataset_name])

        if dataset_name == "MSL":
            add_msl_degradation_annotation(ax, chinese_font)

    axes[0].set_ylabel("Precision", fontsize=9.2)

    legend = axes[-1].legend(
        handles=build_legend(),
        loc="lower right",
        prop=chinese_font.copy(),
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#7f7f7f",
        borderpad=0.36,
        labelspacing=0.35,
        handletextpad=0.45,
        fontsize=7.2,
    )
    legend.get_frame().set_linewidth(0.75)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.89, wspace=0.16)

    pdf_path = OUTPUT_DIR / "fig2_precision_recall_scatter.pdf"
    svg_path = OUTPUT_DIR / "fig2_precision_recall_scatter.svg"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()