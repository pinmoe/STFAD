# =============================================================================
#  Figure 5: Prior selection framework diagram
#
#  Run from the project root:
#    python scripts/research_paper/fig5_prior_selection_framework.py
#
#  Outputs:
#    pics/research_paper/fig5_prior_selection_framework.pdf
#    pics/research_paper/fig5_prior_selection_framework.svg
#    pics/research_paper/fig5_prior_selection_framework.mmd
# =============================================================================

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "pics" / "research_paper"
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\STFANGSO.TTF"),
    Path(r"C:\Windows\Fonts\simfang.ttf"),
    PROJECT_ROOT / "pics" / "simsunb.ttf",
]


MERMAID_CODE = """flowchart LR
    A["目标场景"] -->|点突变| B["点突变型异常"]
    A -->|持续偏移| C["持续偏移型异常"]
    A -->|高异常率| D["高异常率工业监测"]
    B -->|局部差分敏感| E["DGR-diff 评分"]
    C -->|趋势偏移稳健| F["DGR-raw + blend 评分"]
    D -->|重构均值稳定| G["DGR-raw + rec_mean 评分"]

    classDef root fill:#e4f0e6,stroke:#5d8367,color:#111111,stroke-width:1px;
    classDef scene fill:#e7f0f7,stroke:#6f90a8,color:#111111,stroke-width:1px;
    classDef output fill:#f7ead8,stroke:#b69262,color:#111111,stroke-width:1px;
    class A root;
    class B,C,D scene;
    class E,F,G output;
"""


def setup_style() -> fm.FontProperties:
    chinese_font_path = next((path for path in FONT_CANDIDATES if path.exists()), None)
    if chinese_font_path is not None:
        fm.fontManager.addfont(str(chinese_font_path))

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "FangSong"],
            "font.serif": ["Times New Roman", "SimSun", "FangSong"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return fm.FontProperties(fname=str(chinese_font_path)) if chinese_font_path else fm.FontProperties(family="SimSun")


def rounded_box(ax, x, y, w, h, text, fc, ec, chinese_font, fontsize=9.4, weight="normal"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.055",
        linewidth=0.95,
        facecolor=fc,
        edgecolor=ec,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f1f1f",
        fontweight=weight,
        fontproperties=chinese_font,
        zorder=4,
    )


def arrow(ax, start, end, label, chinese_font):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10.5,
        linewidth=0.95,
        color="#555555",
        shrinkA=5,
        shrinkB=5,
        zorder=2,
    )
    ax.add_patch(arr)
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    ax.text(
        mx,
        my + 0.16,
        label,
        ha="center",
        va="center",
        fontsize=7.6,
        color="#555555",
        fontproperties=chinese_font,
        zorder=5,
    )


def main():
    chinese_font = setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mmd_path = OUTPUT_DIR / "fig5_prior_selection_framework.mmd"
    mmd_path.write_text(MERMAID_CODE, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(16 / 2.54, 8.4 / 2.54), constrained_layout=False)
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    ax.add_patch(Rectangle((0.25, 0.32), 11.5, 6.35, facecolor="white", edgecolor="#b7b7b7", linewidth=0.9))
    ax.add_patch(Rectangle((0.25, 6.05), 11.5, 0.62, facecolor="#e4f0e6", edgecolor="#8aaa91", linewidth=0.9))
    ax.text(
        6.0,
        6.36,
        "先验配置选择框架",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        fontproperties=chinese_font,
        color="#1f3a29",
    )

    ax.text(3.75, 5.55, "异常形态识别", ha="center", va="center", fontsize=9.2, fontproperties=chinese_font, color="#31546b")
    ax.text(8.55, 5.55, "推荐评分策略", ha="center", va="center", fontsize=9.2, fontproperties=chinese_font, color="#6f4e26")
    ax.plot([1.75, 10.95], [5.28, 5.28], color="#c9c9c9", linewidth=0.75)

    rounded_box(ax, 0.72, 2.75, 1.72, 1.0, "目标场景", "#e4f0e6", "#6f9278", chinese_font, fontsize=10.2, weight="bold")

    rows = [
        ("01", "点突变型异常", "DGR-diff 评分", "点突变", "局部差分敏感"),
        ("02", "持续偏移型异常", "DGR-raw + blend 评分", "持续偏移", "趋势偏移稳健"),
        ("03", "高异常率工业监测", "DGR-raw + rec_mean 评分", "高异常率", "重构均值稳定"),
    ]
    ys = [4.35, 3.05, 1.75]

    for (idx, scene, output, left_label, right_label), y in zip(rows, ys):
        ax.add_patch(Rectangle((0.55, y - 0.38), 10.9, 0.76, facecolor="#fbfbfb", edgecolor="#dddddd", linewidth=0.65, zorder=0))
        ax.text(2.80, y, idx, ha="center", va="center", fontsize=8.0, color="#6a7f8d", fontproperties=chinese_font)
        rounded_box(ax, 3.15, y - 0.31, 2.35, 0.62, scene, "#e7f0f7", "#7191a6", chinese_font, fontsize=9.0)
        rounded_box(ax, 7.08, y - 0.31, 2.95, 0.62, output, "#f7ead8", "#b69262", chinese_font, fontsize=9.0)
        arrow(ax, (2.45, 3.25), (3.15, y), left_label, chinese_font)
        arrow(ax, (5.50, y), (7.08, y), right_label, chinese_font)

    ax.add_patch(Rectangle((0.55, 0.72), 10.9, 0.48, facecolor="#f6f8f6", edgecolor="#c9d4ca", linewidth=0.7))
    ax.text(
        6.0,
        0.96,
        "依据异常形态与监测场景选择先验配置，服务于工业多变量时序异常检测",
        ha="center",
        va="center",
        fontsize=8.7,
        fontproperties=chinese_font,
        color="#3d4b3f",
    )

    pdf_path = OUTPUT_DIR / "fig5_prior_selection_framework.pdf"
    svg_path = OUTPUT_DIR / "fig5_prior_selection_framework.svg"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {mmd_path}")
    print("\nMermaid code:\n")
    print(MERMAID_CODE)


if __name__ == "__main__":
    main()
