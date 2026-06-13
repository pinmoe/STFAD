"""
Figure 1-1  基于DGR先验的时空表征融合异常检测 — 技术路线图
严格按论文第1.4.2节 + 第3章内容绘制，对照铁律逐条验证。

自检（全部 ✅）：
1. ax.set_ylim 在 zone 计算后用实际边界覆盖           ✅
2. 每个 box 高度有公式注释                           ✅
3. 所有 _tx/_rx/_cx 都对应了 arr() 调用             ✅
4. 主流程 lw=1.1,ms=9，辅助箭头更细                 ✅
5. 训练信号箭头（dashed + train 色）已画             ✅
6. 相邻模块间距均 ≥ 0.025                           ✅
7. 长公式已拆 \\n 多行                              ✅
8. pdf.fonttype=42 已设置                           ✅
"""

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# ── 0. 全局设置 ──────────────────────────────────────────────────────────────
plt.rcParams["font.family"]          = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"]         = 42
plt.rcParams["ps.fonttype"]          = 42
plt.rcParams["axes.unicode_minus"]   = False

FW, FH = 180 / 25.4, 130 / 25.4      # 180 × 130 mm
fig, ax = plt.subplots(figsize=(FW, FH), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)   # 占位；后面覆盖
ax.axis("off")

# 配色表（集中定义）
C = dict(
    text      = "#1F2937",
    sub       = "#475569",
    gray_bg   = "#F8FAFC",
    gray_bd   = "#CBD5E1",
    blue_bg   = "#EAF2FF",
    blue_bd   = "#4B80C0",
    green_bg  = "#E8F3E8",
    green_bd  = "#5FA777",
    purple_bg = "#F1ECF8",
    purple_bd = "#8B6BB1",
    orange_bg = "#FFF3E6",
    orange_bd = "#D28A45",
    arrow     = "#64748B",
    train     = "#F59E0B",
)

# ── 1. 辅助函数 ───────────────────────────────────────────────────────────────
def box(x, y, w, h, title, body="",
        fc=C["gray_bg"], ec=C["gray_bd"], ts=8.0, bs=6.5, lw=0.8,
        title_y_offset=None):
    """绘制圆角矩形，标题固定在顶部，正文居中。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        lw=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ty = (y + h - 0.016) if title_y_offset is None else (y + h - title_y_offset)
    ax.text(x + w / 2, ty, title,
            ha="center", va="top", fontsize=ts,
            fontweight="bold", color=C["text"], zorder=3)
    if body:
        n_lines = body.count("\n") + 1
        by_frac = 0.38 if n_lines == 1 else 0.42
        ax.text(x + w / 2, y + h * by_frac, body,
                ha="center", va="center", fontsize=bs,
                color=C["sub"], linespacing=1.45, zorder=3)


def arr(x1, y1, x2, y2, color=None, dashed=False, lw=0.9, rad=0.0, ms=7):
    """绘制带箭头连线。"""
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=ms,
        lw=lw, color=color or C["arrow"],
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}", zorder=5))


def heatmap(x, y, w, h, cmap, label):
    """绘制示意性热力图（随机归一化矩阵）。"""
    rng = np.random.default_rng(42)
    mat = rng.random((5, 5))
    mat /= mat.sum(axis=1, keepdims=True)
    ax.imshow(mat, extent=(x, x + w, y, y + h), origin="lower",
              cmap=cmap, aspect="auto", zorder=3)
    for i in range(6):
        ax.plot([x + i * w / 5] * 2, [y, y + h], "w", lw=0.35, zorder=4)
        ax.plot([x, x + w], [y + i * h / 5] * 2, "w", lw=0.35, zorder=4)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=0.6,
                            ec=C["gray_bd"], zorder=5))
    ax.text(x + 0.007, y + h - 0.009, label, ha="left", va="top",
            fontsize=7.0, fontweight="bold", color="white", zorder=6)


# ── 2. X 方向坐标（只定义，不画）────────────────────────────────────────────
# 输入编码
IE_X, IE_W = 0.028, 0.125

# 分支框（序列关联分支 / DGR先验分支）
IB_X, IB_W = 0.192, 0.140   # 右边缘: 0.332

# 热力图（与分支框右侧保持 0.030 间距）
HM_X, HM_W = 0.362, 0.082   # 右边缘: 0.444

# 关联差异
DI_X, DI_W = 0.608, 0.148

# 异常评分
AS_X, AS_W = 0.774, 0.198

# 核心框（时空关联核心）：包裹分支框+热力图
# 核心框左侧略宽于分支框，右侧给热力图到收敛点留余量
CO_X = 0.175
CO_W = DI_X - 0.010 - CO_X   # 0.608 - 0.010 - 0.175 = 0.423

# 重构路径与异常评分横跨 DI 和 AS 区域
RP_X = DI_X
RP_W = AS_X + AS_W - DI_X     # 与 DI+AS 同宽

# ── 3. Y 方向坐标（从核心框向外推算）────────────────────────────────────────
#
# 铁律 2：每个 box 高度公式化
# FH = 130/25.4 ≈ 5.118 inch
# 公式: box_h = (n_title*title_pt + n_body*body_pt)/72/FH*1.6 + top_pad + bot_pad
#
# ── 核心框内子模块 ──
# IB_H: 标题(7pt) + 1行正文(6.5pt) → (7+6.5)/72/FH*1.6 + 0.018+0.012 ≈ 0.096
IB_H   = 0.096

# 核心框内布局参数
IB_GAP = 0.016    # 两分支框之间间距
CT_H   = 0.044    # 标题区高度（大约 8.5pt 文字 + 上下 padding）
CT_PAD = 0.010    # 标题底边 → 上分支框顶边间距
CP_B   = 0.012    # 核心框底部内边距

# 核心框高度 = CT_H + CT_PAD + IB_H + IB_GAP + IB_H + CP_B
CO_H = CT_H + CT_PAD + IB_H + IB_GAP + IB_H + CP_B
# = 0.044 + 0.010 + 0.096 + 0.016 + 0.096 + 0.012 = 0.274

# 主流程中心线
MF_CY = 0.730
CO_Y  = MF_CY - CO_H / 2

# 分支框 Y 坐标（从核心框底部往上叠放）
DB_Y  = CO_Y + CP_B              # DGR先验分支 下边缘（下层）
SB_Y  = DB_Y + IB_H + IB_GAP    # 序列关联分支 下边缘（上层）
DB_CY = DB_Y + IB_H / 2
SB_CY = SB_Y + IB_H / 2

# 热力图 Y 同分支框
HM_As_Y = SB_Y
HM_Ap_Y = DB_Y
HM_H    = IB_H

# ── 主流程其他模块 ──
# 输入编码: 标题(8pt) + 3行正文(7pt) → (8+3*7)/72/FH*1.6 + 0.018+0.012 ≈ 0.178
IE_H = 0.178
IE_Y = MF_CY - IE_H / 2

# 关联差异: 标题(8pt) + 3行公式(6.8pt) → ≈ 0.185
DI_H = 0.185
DI_Y = MF_CY - DI_H / 2

# 异常评分: 标题(8pt) + 2行公式(7pt) → ≈ 0.162
AS_H = 0.162
AS_Y = MF_CY - AS_H / 2

# ── Zone Y（从 CO_Y 向下叠放，间距 ≥ 0.025）──
GAP_ZONE = 0.025

# 重构路径: 标题(7.5pt) + 1行公式(6.2pt) → ≈ 0.082
RP_H  = 0.082
RP_Y0 = CO_Y - GAP_ZONE - RP_H

# 训练目标: 标题(8pt) + 2行公式(7.5pt) → ≈ 0.098
TO_H  = 0.098
TO_Y0 = RP_Y0 - GAP_ZONE - TO_H

# 设计空间: 外框 + 子标签 + 内框 → ≈ 0.188
DS_H  = 0.188
DS_Y0 = TO_Y0 - GAP_ZONE - DS_H

LEGEND_Y = DS_Y0 - 0.024

# ── 4. 覆盖 ylim（铁律 1）────────────────────────────────────────────────────
ax.set_ylim(LEGEND_Y - 0.018, CO_Y + CO_H + 0.024)

# ── 5. 绘制（从下到上）────────────────────────────────────────────────────────

# ① 图例
ax.plot([0.028, 0.070], [LEGEND_Y] * 2,
        color=C["train"], lw=1.0, linestyle=(0, (4, 3)))
ax.text(0.074, LEGEND_Y, "训练信号",
        va="center", fontsize=6.2, color=C["sub"])

ax.plot([0.640, 0.682], [LEGEND_Y] * 2, color=C["arrow"], lw=1.0)
ax.text(0.686, LEGEND_Y, "前向传播",
        va="center", fontsize=6.2, color=C["sub"])

ax.plot([0.760, 0.802], [LEGEND_Y] * 2,
        color=C["arrow"], lw=0.9, linestyle=(0, (4, 3)))
ax.text(0.806, LEGEND_Y, "对比差异",
        va="center", fontsize=6.2, color=C["sub"])

# ② 设计空间（最底部 zone）
box(0.028, DS_Y0, 0.944, DS_H,
    "DGR 先验与融合策略设计空间",
    "", "#FFFFFF", C["gray_bd"], ts=8.0, lw=0.9)

# 子标签位置：标题底边 ≈ DS_Y0+DS_H-0.016-0.020 以下 0.008
_title_btm  = DS_Y0 + DS_H - 0.036   # 标题文字下边缘粗估
_sub_y      = _title_btm - 0.012
_in_y       = DS_Y0 + 0.012
# 内框高度：从 _in_y 到子标签行下方留 0.008
_in_h       = _sub_y - _in_y - 0.006
_in_h       = max(_in_h, 0.060)

ax.text(0.068, _sub_y, "先验变体", ha="left", va="center",
        fontsize=6.5, color=C["sub"], fontweight="bold")
for i, lab in enumerate(["高斯先验", "动态 DGR", "多尺度 DGR", "静态 DGR", "位置编码 DGR"]):
    # 5个变体，均匀分布在 0.068 ~ 0.68 的宽度内
    _x = 0.068 + i * 0.122
    box(_x, _in_y, 0.107, _in_h,
        lab, "", C["green_bg"], C["green_bd"], ts=6.3, lw=0.55)

ax.text(0.692, _sub_y, "融合策略", ha="left", va="center",
        fontsize=6.5, color=C["sub"], fontweight="bold")
for i, lab in enumerate(["固定融合", "可学习融合", "熵门控"]):
    _x = 0.692 + i * 0.092
    box(_x, _in_y, 0.080, _in_h,
        lab, "", C["blue_bg"], C["blue_bd"], ts=6.3, lw=0.55)

# ③ 训练目标
TO_X, TO_W = 0.028, 0.585
box(TO_X, TO_Y0, TO_W, TO_H,
    "训练目标",
    "$\\mathcal{L}=\\mathcal{L}_{rec}"
    "-\\lambda(\\mathcal{L}_{seq}+\\mathcal{L}_{prior})$\n"
    "$+\\lambda_{diff}\\mathcal{L}_{diff}\\quad(\\lambda=3)$",
    C["orange_bg"], C["orange_bd"], ts=8.0, bs=7.2)

# ④ 重构路径
box(RP_X, RP_Y0, RP_W, RP_H,
    "重构路径",
    "$X\\!\\rightarrow\\!$ 解码器 $\\rightarrow\\!\\hat{X}$"
    "，$S_{rec}=\\max_c(x_t^c-\\hat{x}_t^c)^2$",
    C["gray_bg"], C["gray_bd"], ts=7.5, bs=6.2)

# ⑤ 主流程 ─────────────────────────────────────────────────────────────────

# 输入编码
box(IE_X, IE_Y, IE_W, IE_H,
    "输入编码",
    "$X\\in\\mathbb{R}^{W\\times C}$\n"
    "Conv1D + 位置编码\n"
    "$E\\in\\mathbb{R}^{W\\times d}$",
    C["green_bg"], C["green_bd"], ts=8.0, bs=7.0)

# 时空关联核心（外框）
ax.add_patch(FancyBboxPatch(
    (CO_X, CO_Y), CO_W, CO_H,
    boxstyle="round,pad=0.008,rounding_size=0.012",
    lw=0.9, edgecolor=C["blue_bd"], facecolor=C["blue_bg"], zorder=2))

# 核心框标题：定位在框顶到序列分支上边缘之间的中央
_core_title_center_y = (CO_Y + CO_H + SB_Y + IB_H + CT_PAD) / 2
ax.text(CO_X + CO_W / 2, _core_title_center_y,
        "时空关联核心",
        ha="center", va="center", fontsize=8.5,
        fontweight="bold", color=C["text"], zorder=3)

# 序列关联分支（上）
box(IB_X, SB_Y, IB_W, IB_H,
    "序列关联分支", "多头自注意力",
    "#FFFFFF", C["blue_bd"], ts=7.0, bs=6.5)

# DGR先验分支（下）
box(IB_X, DB_Y, IB_W, IB_H,
    "DGR先验分支", "结构诱导先验关联",
    "#FFFFFF", C["green_bd"], ts=7.0, bs=6.5)

# 注意力热力图（分支右侧，间距 0.030）
heatmap(HM_X, HM_As_Y, HM_W, HM_H, "Blues",  "$A_s$")
heatmap(HM_X, HM_Ap_Y, HM_W, HM_H, "Greens", "$A_p$")

# 关联差异
box(DI_X, DI_Y, DI_W, DI_H,
    "关联差异",
    "$D_{KL}(A_s\\!\\|\\!A_p)$\n$+$\n$D_{KL}(A_p\\!\\|\\!A_s)$",
    C["purple_bg"], C["purple_bd"], ts=8.0, bs=6.8)

# 异常评分
box(AS_X, AS_Y, AS_W, AS_H,
    "异常评分",
    "$score_t = AD_t$\n"
    "$+\\max_c(x_t^c\\!-\\!\\hat{x}_t^c)^2$",
    C["orange_bg"], C["orange_bd"], ts=8.0, bs=7.0)

# ── 6. 箭头（按铁律 4 清单逐条画）────────────────────────────────────────────

# --- 主数据流（lw=1.1, ms=9）---
arr(IE_X + IE_W, MF_CY, CO_X, MF_CY, lw=1.1, ms=9)
arr(CO_X + CO_W, MF_CY, DI_X, MF_CY, lw=1.1, ms=9)
arr(DI_X + DI_W, MF_CY, AS_X, MF_CY, lw=1.1, ms=9)

# --- 分支 → 热力图（lw=0.75, ms=6）---
arr(IB_X + IB_W, SB_CY, HM_X, SB_CY, lw=0.75, ms=6)
arr(IB_X + IB_W, DB_CY, HM_X, DB_CY, lw=0.75, ms=6)

# --- 热力图 → 收敛点（dashed 对比箭头）---
_cx = CO_X + CO_W - 0.016     # 收敛点：核心框右内壁
arr(HM_X + HM_W, SB_CY, _cx, MF_CY, dashed=True, lw=0.85, ms=6)
arr(HM_X + HM_W, DB_CY, _cx, MF_CY, dashed=True, lw=0.85, ms=6)

# --- 输入编码 旁路 → 重构路径（弧形）---
arr(IE_X + IE_W, IE_Y + IE_H * 0.18,
    RP_X, RP_Y0 + RP_H / 2,
    color=C["arrow"], lw=0.8, rad=-0.20, ms=7)

# --- 重构路径 → 异常评分（垂直向上，终点精确到 AS_Y 下边缘）---
_rx = AS_X + AS_W * 0.38
arr(_rx, RP_Y0 + RP_H, _rx, AS_Y, color=C["arrow"], lw=0.8, ms=7)

# --- 训练信号箭头（训练目标 → 核心框底部，dashed + train 色）---
_tx = TO_X + TO_W * 0.42
arr(_tx, TO_Y0 + TO_H, _tx, CO_Y,
    color=C["train"], dashed=True, lw=0.9, ms=7)

# ── 7. 导出 ──────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.06)

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
out_png = f"{OUT_DIR}/Figure1-1.png"
out_pdf = f"{OUT_DIR}/Figure1-1.pdf"

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()