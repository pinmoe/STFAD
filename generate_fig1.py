# /// script
# dependencies = [
#   "python-pptx",
# ]
# ///
import pptx
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

def create_shape(shapes, left, top, width, height, text,
                 bg_color=None, border_color=None, text_color="000000",
                 font_size=11, bold=False, font_name='Microsoft YaHei', is_title=False, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE):
    """创建美观的学术风格形状（支持自定义形状类型，如圆形）"""
    shape = shapes.add_shape(shape_type, Inches(left), Inches(top), Inches(width), Inches(height))

    # 填充背景色
    if bg_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(bg_color)
    else:
        shape.fill.background()

    # 填充边框
    line = shape.line
    if border_color:
        line.color.rgb = RGBColor.from_string(border_color)
        line.width = Pt(1.5)
    else:
        line.fill.background()
        
    # 为内部内容白块启用系统默认的优雅阴影
    if bg_color == "FFFFFF":
        shape.shadow.inherit = True  # 使用 PowerPoint 原生的平滑投影

    # 文本框样式与文字排版
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = font_name
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = RGBColor.from_string(text_color)
    p.alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    
    if is_title:
        p.font.color.rgb = RGBColor.from_string("FFFFFF")
        p.font.size = Pt(14)

    return shape

def add_connector(shapes, start_shape, end_shape, start_conn_idx=3, end_conn_idx=1, elbow=True, arrow=True):
    """在两个形状之间建立一条严格的、不会偏斜的直角或直线连接符"""
    if elbow:
        connector = shapes.add_connector(MSO_CONNECTOR.ELBOW, Inches(0), Inches(0), Inches(0), Inches(0))
    else:
        connector = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0), Inches(0), Inches(0), Inches(0))

    # 连接桩索引必须在 0 - 3 范围内
    connector.begin_connect(start_shape, start_conn_idx)
    connector.end_connect(end_shape, end_conn_idx)
    
    line = connector.line
    line.color.rgb = RGBColor.from_string("595959")
    line.width = Pt(1.5)
    
    if arrow:
        connector.line.end_arrowhead = 2  # 2 代表标准三角形箭头

    return connector

# 16:9 标准高清画板尺寸
prs = pptx.Presentation()
prs.slide_width = Inches(16)
prs.slide_height = Inches(9)

slide = prs.slides.add_slide(prs.slide_layouts[6])
shapes = slide.shapes

# 调色盘配置（提取自您最满意的 Imagen 生成原图）
CLR_BLUE_BG = "DDEBF7"     # 阶段一背景：淡蓝
CLR_PURPLE_BG = "E9E5F2"   # 阶段二背景：淡紫
CLR_RED_BG = "FBE5E1"      # 阶段三背景：淡红
CLR_GREEN_BG = "EBF7E8"    # 先验部分容器背景：浅绿

# --- 1. 三大阶段的宏观背景区域（横向完美分布，决不重叠） ---
create_shape(shapes, 0.5, 0.5, 3.8, 8.0, "", CLR_BLUE_BG)
create_shape(shapes, 4.5, 0.5, 7.0, 8.0, "", CLR_PURPLE_BG)
create_shape(shapes, 11.7, 0.5, 3.8, 8.0, "", CLR_RED_BG)

# --- 2. 阶段标题栏（精美学术渐变感纯色标题） ---
create_shape(shapes, 0.5, 0.6, 3.8, 0.8, "输入与特征编码", bg_color="3B82F6", font_name='Microsoft YaHei', bold=True, is_title=True)
create_shape(shapes, 4.5, 0.6, 7.0, 0.8, "关联建模：E1 / E3 二选一", bg_color="8B5CF6", font_name='Microsoft YaHei', bold=True, is_title=True)
create_shape(shapes, 11.7, 0.6, 3.8, 0.8, "评分与评价", bg_color="EF4444", font_name='Microsoft YaHei', bold=True, is_title=True)

# --- 3. 核心节点坐标参数（经过代数对齐计算，确保完美居中和水平对齐） ---
# 【输入与编码阶段】
b_input = create_shape(shapes, 1.0, 4.0, 1.5, 1.0, "输入窗口\nX: W×C", "FFFFFF")
b_encoder = create_shape(shapes, 2.8, 3.75, 1.4, 1.5, "线性投影与位置编码\nTransformer编码器", "FFFFFF", font_size=9)

# 【关联建模阶段 - 上半部分：重构支线】
b_assoc = create_shape(shapes, 5.0, 2.5, 2.2, 1.0, "序列关联 A_series\n多头自注意力", "FFFFFF")
b_recon = create_shape(shapes, 8.8, 2.5, 2.2, 1.0, "重构序列 x̂\n逐通道重构误差", "FFFFFF")

# 【关联建模阶段 - 下半部分：受控先验绿色子区域】
create_shape(shapes, 5.0, 4.8, 6.0, 3.5, "", CLR_GREEN_BG)
b_e1 = create_shape(shapes, 5.2, 5.2, 2.8, 1.0, "E1：高斯时间先验\nP_gauss\n时间距离 + 核宽度", "FFFFFF", font_size=9)
b_e3 = create_shape(shapes, 5.2, 6.8, 2.8, 1.0, "E3：多尺度差分 DGR\n先验 P_dgr\n本文核心配置", "FFFFFF", font_size=9)
b_switch_text = create_shape(shapes, 3.0, 5.9, 1.2, 0.7, "SWITCH\n二选一", bg_color=None, border_color=None)
b_prior = create_shape(shapes, 8.5, 6.0, 2.3, 1.0, "受控先验 P_st\nE1: P_gauss\nE3: P_dgr (替换)", "FFFFFF", font_size=9)

# 双向 KL 散度汇总汇聚点
b_sum = create_shape(shapes, 7.4, 3.95, 0.7, 0.7, "+", "FFFFFF", shape_type=MSO_SHAPE.OVAL)

# 【评分与评价阶段】
b_diff = create_shape(shapes, 8.0, 4.0, 2.2, 0.7, "关联差异\n双向 KL 散度", "FFFFFF")
b_score = create_shape(shapes, 12.0, 2.5, 2.2, 1.0, "异常分数 s_i\ncombined / rec_mean", "FFFFFF")
b_input_w = create_shape(shapes, 12.8, 4.0, 1.4, 0.7, "输入窗口, W", "FFFFFF")

# 【下发评价任务】
b_public = create_shape(shapes, 12.0, 5.5, 1.5, 1.2, "公开数据集\n逐点评价", "FFFFFF", font_size=10)
b_st330 = create_shape(shapes, 13.8, 5.5, 1.5, 1.2, "ST330IR001\n56步均值→周期评价", "FFFFFF", font_size=9)
b_eval_text = create_shape(shapes, 12.9, 4.8, 1.0, 0.5, "逐点评价", bg_color=None, border_color=None)


# --- 4. 正交折线网络（建立纯直角连接线，自动对齐，保证 0-3 安全连接桩索引） ---
add_connector(shapes, b_input, b_encoder, 3, 1, elbow=False)
add_connector(shapes, b_encoder, b_assoc, 0, 1)
add_connector(shapes, b_encoder, b_recon, 2, 1)
add_connector(shapes, b_encoder, b_e1, 2, 1, elbow=True)
add_connector(shapes, b_encoder, b_e3, 2, 1, elbow=True)

add_connector(shapes, b_e1, b_prior, 3, 1, elbow=True)
add_connector(shapes, b_e3, b_prior, 3, 1, elbow=True)
add_connector(shapes, b_prior, b_sum, 0, 2)

add_connector(shapes, b_assoc, b_sum, 2, 0)
add_connector(shapes, b_sum, b_diff, 3, 1, elbow=False)
add_connector(shapes, b_diff, b_score, 3, 1, elbow=True)
add_connector(shapes, b_recon, b_score, 3, 1, elbow=True)

add_connector(shapes, b_input, b_input_w, 2, 1, elbow=True)
add_connector(shapes, b_score, b_eval_text, 2, 0, elbow=True, arrow=False)
add_connector(shapes, b_eval_text, b_public, 2, 0, elbow=False)
add_connector(shapes, b_eval_text, b_st330, 2, 0, elbow=False)

# 保存最终的文件
prs.save("STFAD_框架图_可编辑版_V3_Final.pptx")
print("🎉 成功生成！请在当前目录下查看 'STFAD_框架图_可编辑版_V3_Final.pptx'")
