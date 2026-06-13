"""
图4.3 - 各实验配置在三个数据集上的 Precision-Recall 对比图
子图(a)(b)：MSL/SMAP P-R 散点图；子图(c)：SKAB 水平 Lollipop 精确率排名图
保存路径: figures/毕业论文用图/第4章/Figure4-3.{pdf,png}
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import numpy as np
import os
import platform

# ── 字体 ──────────────────────────────────────────────────────────────────────
_zh = 'Microsoft YaHei' if platform.system() == 'Windows' else 'Noto Sans CJK SC'
matplotlib.rcParams['font.family'] = [_zh, 'DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8

# ── 输出路径 ──────────────────────────────────────────────────────────────────
OUT_DIR = "figures/毕业论文用图/第4章"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 颜色 ──────────────────────────────────────────────────────────────────────
C_E    = '#4C78A8'
C_B    = '#E45756'
C_E6   = '#1A1A2E'
C_GOLD = '#FFD700'

# ── 实验数据 (Precision, Recall, 系列) ────────────────────────────────────────
PR_DATA = {
    'MSL': {
        'E1': (0.900, 0.927, 'E'),
        'E2': (0.910, 0.852, 'E'),
        'E3': (0.898, 0.814, 'E'),
        'E4': (0.894, 0.845, 'E'),
        'B1': (0.876, 0.728, 'B'),
        'B2': (0.791, 0.428, 'B'),
        'B3': (0.883, 0.817, 'B'),
        'E5': (0.906, 0.895, 'E'),
        'E6': (0.882, 0.840, 'E6'),
    },
    'SMAP': {
        'E1': (0.922, 0.568, 'E'),
        'E2': (0.929, 0.540, 'E'),
        'E3': (0.928, 0.568, 'E'),
        'E4': (0.922, 0.570, 'E'),
        'B1': (0.921, 0.623, 'B'),
        'B2': (0.922, 0.729, 'B'),
        'B3': (0.926, 0.568, 'B'),
        'E5': (0.920, 0.641, 'E'),
        'E6': (0.930, 0.614, 'E6'),
    },
}

# SKAB: 按 Precision 降序排列
SKAB_DATA = [
    ('E3',  0.847, 'E'),
    ('E6',  0.844, 'E6'),
    ('E2',  0.829, 'E'),
    ('B1',  0.814, 'B'),
    ('B2',  0.807, 'B'),
    ('B3',  0.805, 'B'),
    ('E1',  0.795, 'E'),
    ('E5',  0.781, 'E'),
    ('E4',  0.758, 'E'),
]
SKAB_RECALL = 0.793

RANGES = {
    'MSL':  {'x': (0.40, 0.95), 'y': (0.77, 0.92)},
    'SMAP': {'x': (0.50, 0.78), 'y': (0.915, 0.935)},
}

F1_LEVELS = {'MSL': [0.80, 0.85], 'SMAP': [0.75, 0.80]}
BEST      = {'MSL': 'E5', 'SMAP': 'B2'}

# (标签文字, dx, dy, fontweight)
ANNOTATIONS = {
    'MSL': {
        'E1': ('E1*',  0.016, -0.005, 'normal'),
        'E5': ('E5★',  0.008,  0.006, 'bold'),
        'B2': ('B2',   0.012,  0.003, 'normal'),
    },
    'SMAP': {
        'E1': ('E1*', -0.035,  0.001, 'normal'),
        'B2': ('B2★',  0.007,  0.002, 'bold'),
        'E6': ('E6',   0.007, -0.003, 'normal'),
    },
}


def _style(series):
    if series == 'E':
        return dict(color=C_E,  marker='o', s=55, ec='white', lw=0.8)
    if series == 'B':
        return dict(color=C_B,  marker='o', s=55, ec='white', lw=0.8)
    return     dict(color=C_E6, marker='D', s=90, ec='black', lw=1.5)


def draw_f1_curves(ax, xr, yr, levels):
    r = np.linspace(xr[0], xr[1], 1000)
    for f in levels:
        p = f * r / (2 * r - f + 1e-12)
        mask = (p >= yr[0]) & (p <= yr[1])
        if mask.sum() < 2:
            continue
        ri, pi = r[mask], p[mask]
        ax.plot(ri, pi, '--', color='#CCCCCC', lw=0.5, alpha=0.5, zorder=1)
        ax.text(ri[-1], pi[-1], f'F1={f:.2f}',
                fontsize=6.5, color='#AAAAAA', va='center', ha='left',
                clip_on=True, zorder=2)


def draw_pr_subplot(ax, ds, is_leftmost):
    xr = RANGES[ds]['x']
    yr = RANGES[ds]['y']

    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', lw=0.3, alpha=0.4, color='#EEEEEE', zorder=0)
    ax.set_xlim(xr)
    ax.set_ylim(yr)

    draw_f1_curves(ax, xr, yr, F1_LEVELS[ds])

    best_name = BEST[ds]
    for name, (prec, rec, series) in PR_DATA[ds].items():
        st = _style(series)
        is_best = (name == best_name)
        is_e1   = (name == 'E1')

        # 金色外环（最优方法）
        if is_best:
            ax.scatter(rec, prec, c=C_GOLD, marker='o', s=110,
                       edgecolors=C_GOLD, linewidths=1.5, zorder=4)

        # E1 灰色外圈
        ec = '#888888' if is_e1 else st['ec']
        lw = 1.2       if is_e1 else st['lw']
        ax.scatter(rec, prec, c=st['color'], marker=st['marker'], s=st['s'],
                   edgecolors=ec, linewidths=lw, zorder=5)

    # 关键点标注
    for name, (prec, rec, series) in PR_DATA[ds].items():
        cfg = ANNOTATIONS.get(ds, {}).get(name)
        if cfg is None:
            continue
        label, dx, dy, fw = cfg
        ax.annotate(label, xy=(rec, prec), xytext=(rec + dx, prec + dy),
                    fontsize=8, color='#222222', fontweight=fw,
                    arrowprops=dict(arrowstyle='-', lw=0.5, color='#AAAAAA'),
                    zorder=6)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(labelsize=8.5, direction='in')

    ax.set_xlabel('召回率 (Recall)', fontsize=10)
    if is_leftmost:
        ax.set_ylabel('精确率 (Precision)', fontsize=10)

    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    ax.set_title(ds, fontsize=11, fontweight='bold', pad=6)


def draw_skab_lollipop(ax):
    n = len(SKAB_DATA)
    ax.set_facecolor('white')
    ax.set_xlim(0.74, 0.86)
    ax.set_ylim(-0.5, n - 0.5)
    ax.grid(True, axis='x', linestyle='-', lw=0.3, alpha=0.4, color='#EEEEEE', zorder=0)

    # 参考竖线
    ax.axvline(x=SKAB_RECALL, color='#888888', linestyle='--', lw=0.8, zorder=2)
    # 参考线标注（blended transform：x 数据坐标，y 轴坐标）
    ax.text(SKAB_RECALL + 0.002, 0.97, 'Recall=0.793',
            transform=ax.get_xaxis_transform(),
            fontsize=7, color='#888888', ha='left', va='top')

    top2 = {'E3', 'E6'}
    for i, (name, prec, series) in enumerate(SKAB_DATA):
        y = n - 1 - i  # E3→y=8（顶部），E4→y=0（底部）

        # 茎（stem）
        x0, x1 = min(SKAB_RECALL, prec), max(SKAB_RECALL, prec)
        ax.hlines(y, x0, x1, colors='#DDDDDD', lw=1.5, zorder=1)

        st = _style(series)

        # 金色外环（最优两点）
        if name in top2:
            ax.scatter(prec, y, c=C_GOLD, marker='o', s=85,
                       edgecolors=C_GOLD, linewidths=1.5, zorder=3)

        ax.scatter(prec, y, c=st['color'], marker=st['marker'], s=60,
                   edgecolors=st['ec'], linewidths=st['lw'], zorder=4)

        # 精确率数值标注
        ax.text(prec + 0.003, y, f'{prec:.3f}',
                fontsize=8, color='#222222', va='center', ha='left', clip_on=False)

    # y 轴：从上到下按降序显示方法名
    ax.set_yticks(range(n))
    ax.set_yticklabels([d[0] for d in reversed(SKAB_DATA)], fontsize=9)

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.tick_params(labelsize=8.5, direction='in')

    ax.set_xlabel('精确率 (Precision)', fontsize=10)

    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    ax.set_title('SKAB', fontsize=11, fontweight='bold', pad=6)


# ── 主图 ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97, top=0.79, bottom=0.22)

draw_pr_subplot(axes[0], 'MSL',  is_leftmost=True)
draw_pr_subplot(axes[1], 'SMAP', is_leftmost=False)
draw_skab_lollipop(axes[2])

# ── 全局图例（底部居中，1 行）────────────────────────────────────────────────
h_e    = mlines.Line2D([], [], color=C_E,   marker='o', linestyle='None',
                        markersize=7, label='E 系列（本研究实验）')
h_b    = mlines.Line2D([], [], color=C_B,   marker='o', linestyle='None',
                        markersize=7, label='B 系列（对照组）')
h_e6   = mlines.Line2D([], [], color=C_E6,  marker='D', linestyle='None',
                        markersize=7, markeredgecolor='black', markeredgewidth=1.0,
                        label='E6（位置编码 DGR 先验）')
h_best = mlines.Line2D([], [], color=C_E,   marker='o', linestyle='None',
                        markersize=9, markeredgecolor=C_GOLD, markeredgewidth=2.0,
                        label='各数据集最优方法')

fig.legend(handles=[h_e, h_b, h_e6, h_best],
           loc='upper center',
           bbox_to_anchor=(0.5, 0.10),
           ncol=4,
           fontsize=9,
           frameon=False,
           handlelength=1.5,
           handletextpad=0.5,
           columnspacing=1.5)

# ── 导出 ──────────────────────────────────────────────────────────────────────
pdf_path = f"{OUT_DIR}/Figure4-3.pdf"
png_path = f"{OUT_DIR}/Figure4-3.png"

fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05)
print(f"Saved PDF: {pdf_path}")

fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved PNG: {png_path}")

plt.close(fig)
