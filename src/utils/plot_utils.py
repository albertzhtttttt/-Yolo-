"""
plot_utils.py - 可视化工具模块
路径：src/utils/plot_utils.py

功能：
    - 统一全项目图表风格（字体、颜色、DPI）
    - 提供中文字体自动配置
    - 提供常用绘图辅助函数

使用方式：
    from src.utils.plot_utils import setup_plot_style, PALETTE, save_fig
"""

from __future__ import annotations

import os
import platform
import matplotlib
matplotlib.use("Agg")  # 无头服务器必须在 import pyplot 前设置
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# =============================================================================
# 全局常量
# =============================================================================

# 图表保存 DPI：论文投稿通常要求 ≥300 dpi，这里使用 600 dpi 便于双栏缩放后仍保持清晰。
SAVE_DPI = 600

# 统一调色板：采用 Okabe-Ito 色盲友好配色，避免高饱和商业风配色影响论文观感。
PALETTE = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "teal":    "#56B4E9",
    "gray":    "#4D4D4D",
    "yellow":  "#F0E442",
}

# 论文图中不能只依赖颜色区分曲线/柱子，因此统一准备线型、标记和 hatch 纹理。
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
HATCHES = ["", "//", "\\\\", "xx", "..", "++"]

# 模型对应颜色（对比实验用）
MODEL_COLORS = {
    "YOLOv8":  PALETTE["blue"],
    "YOLOv5":  PALETTE["orange"],
    "U-Net":   PALETTE["green"],
    "FCN":     PALETTE["red"],
}

# 分辨率等级颜色
SCALE_COLORS = {
    100: PALETTE["blue"],
    75:  PALETTE["teal"],
    50:  PALETTE["orange"],
    25:  PALETTE["red"],
}


# =============================================================================
# 中文字体配置
# =============================================================================

def _find_chinese_font() -> str | None:
    """
    自动查找系统中可用的中文字体。

    返回值：
        找到的字体路径，或 None（未找到）
    """
    # 优先字体列表（按优先级排序）
    candidate_fonts = [
        # Linux 常见中文字体
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans CN",
        "AR PL UMing CN",
        # macOS
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        # Windows
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
    ]

    # 获取系统所有字体
    available = {f.name: f.fname for f in fm.fontManager.ttflist}

    for font_name in candidate_fonts:
        if font_name in available:
            return available[font_name]

    # 尝试路径搜索（Linux 服务器常见路径）
    search_paths = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
    ]
    for search_path in search_paths:
        if os.path.isdir(search_path):
            for root, _, files in os.walk(search_path):
                for f in files:
                    if f.endswith((".ttf", ".otf")) and any(
                        kw in f.lower() for kw in ["cjk", "noto", "wqy", "chinese", "zh"]
                    ):
                        return os.path.join(root, f)

    return None


def setup_plot_style(font_size: int = 9) -> None:
    """
    配置全局 matplotlib 绘图风格，面向计算机视觉论文双栏排版。

    参数：
        font_size: 基础字号，默认 9；接近 CVPR 等模板的图注字号，缩放到单栏后仍可读。

    使用方式：
        在任何绘图脚本开头调用一次 setup_plot_style()，随后所有图表继承统一论文风格。
    """
    # 查找中文字体；若没有中文字体，则优先使用论文常见的 serif 字体，保证英文图表观感统一。
    chinese_font_path = _find_chinese_font()

    if chinese_font_path:
        try:
            prop = fm.FontProperties(fname=chinese_font_path)
            font_name = prop.get_name()
            plt.rcParams["font.family"] = ["serif"]
            plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", font_name]
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            print(f"[plot_utils] 已加载中文字体：{font_name}")
        except Exception:
            chinese_font_path = None

    if not chinese_font_path:
        # 服务器没有中文字体时仍使用 serif 英文字体；当前图表标签尽量保持英文以避免乱码。
        print("[plot_utils] 警告：未找到中文字体，中文可能无法正常显示。")
        print("  建议安装：sudo apt install fonts-wqy-microhei")
        plt.rcParams["font.family"] = ["serif"]
        plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    # 解决负号显示问题，并让矢量图保留可编辑文本，便于后续论文排版。
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    # 全局字号与风格：弱化网格和装饰，强调数据线条，确保黑白打印也能区分。
    plt.rcParams.update({
        "font.size":        font_size,
        "axes.titlesize":   font_size + 1,
        "axes.labelsize":   font_size,
        "xtick.labelsize":  font_size - 1,
        "ytick.labelsize":  font_size - 1,
        "legend.fontsize":  font_size - 1,
        "figure.dpi":       150,
        "savefig.dpi":      SAVE_DPI,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#222222",
        "axes.linewidth":   0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         "#D0D0D0",
        "grid.alpha":         0.45,
        "grid.linewidth":     0.45,
        "grid.linestyle":     "--",
        "lines.linewidth":    1.6,
        "lines.markersize":   4.5,
        "patch.linewidth":    0.8,
        "legend.frameon":     False,
        "legend.handlelength": 1.8,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   3,
        "ytick.major.size":   3,
    })


def save_fig(fig: plt.Figure, path: str, dpi: int = SAVE_DPI) -> None:
    """
    保存图表到指定路径，自动创建目录，并同步导出 PDF 矢量版本。

    参数：
        fig:  matplotlib Figure 对象
        path: 输出文件路径（支持 .png / .pdf / .svg）
        dpi:  位图保存分辨率，默认 600，满足论文印刷和缩放需求
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    # 论文排版优先使用矢量图；调用方仍拿 PNG 预览，PDF 供后续插入论文。
    base, ext = os.path.splitext(path)
    if ext.lower() != ".pdf":
        pdf_path = base + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
        print(f"[plot_utils] 已保存：{pdf_path}")

    print(f"[plot_utils] 已保存：{path}")


# =============================================================================
# 常用绘图辅助函数
# =============================================================================

def add_value_labels(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.3f}",
    fontsize: int = 9,
    va: str = "bottom",
    padding: float = 0.005,
) -> None:
    """
    在柱状图每个柱子顶部添加数值标签。

    参数：
        ax:      目标坐标轴
        bars:    ax.bar() 返回的 BarContainer 对象
        fmt:     数值格式化字符串
        fontsize:标签字号
        va:      垂直对齐方式
        padding: 标签与柱顶间距（相对坐标系）
    """
    y_max = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_max * padding,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=fontsize,
        )


def draw_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    ax: plt.Axes | None = None,
    normalize: bool = True,
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
) -> plt.Axes:
    """
    绘制混淆矩阵热力图。

    参数：
        cm:           混淆矩阵（numpy 数组，shape=(n, n)）
        class_names:  类别名称列表
        ax:           目标坐标轴，None 时自动创建
        normalize:    是否归一化为比例
        cmap:         颜色映射
        title:        图标题

    返回值：
        绘制完成的 Axes 对象
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap, vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    # 在格子内写数值
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                format(cm_display[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=10,
            )

    return ax


def draw_pr_curve(
    precisions: list[np.ndarray],
    recalls: list[np.ndarray],
    labels: list[str],
    ap_values: list[float] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Precision-Recall Curve",
) -> plt.Axes:
    """
    绘制 PR 曲线（支持多条曲线对比）。

    参数：
        precisions: 各曲线的 precision 数组列表
        recalls:    各曲线的 recall 数组列表
        labels:     各曲线的图例标签
        ap_values:  各曲线的 AP 值（显示在图例中）
        ax:         目标坐标轴
        title:      图标题

    返回值：
        绘制完成的 Axes 对象
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    colors = list(PALETTE.values())
    for i, (p, r, label) in enumerate(zip(precisions, recalls, labels)):
        legend_label = label
        if ap_values is not None:
            legend_label += f" (AP={ap_values[i]:.3f})"
        ax.plot(
            r, p,
            color=colors[i % len(colors)],
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            label=legend_label,
            linewidth=1.6,
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, pad=4)
    ax.legend(loc="lower left", frameon=False)

    return ax


def draw_loss_curve(
    epochs: list[int],
    losses: dict[str, list[float]],
    ax: plt.Axes | None = None,
    title: str = "Training Loss Curve",
) -> plt.Axes:
    """
    绘制训练 Loss 曲线（支持多个 loss 分量）。

    参数：
        epochs: epoch 列表
        losses: {"box_loss": [...], "cls_loss": [...], "dfl_loss": [...]}
        ax:     目标坐标轴
        title:  图标题

    返回值：
        绘制完成的 Axes 对象
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    loss_colors = {
        "box_loss": PALETTE["blue"],
        "cls_loss": PALETTE["orange"],
        "dfl_loss": PALETTE["green"],
        "val_box_loss": PALETTE["blue"],
        "val_cls_loss": PALETTE["orange"],
        "val_dfl_loss": PALETTE["green"],
    }
    line_styles = {
        "train": "-",
        "val":   "--",
    }

    for loss_name, values in losses.items():
        color = loss_colors.get(loss_name, PALETTE["gray"])
        style = "--" if loss_name.startswith("val") else "-"
        ax.plot(epochs, values, color=color, linestyle=style, label=loss_name, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    return ax
