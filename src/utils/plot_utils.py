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
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# =============================================================================
# 全局常量
# =============================================================================

# 图表保存 DPI（≥300 满足论文要求）
SAVE_DPI = 300

# 统一调色板（色盲友好配色）
PALETTE = {
    "blue":    "#2196F3",
    "orange":  "#FF9800",
    "green":   "#4CAF50",
    "red":     "#F44336",
    "purple":  "#9C27B0",
    "teal":    "#009688",
    "gray":    "#607D8B",
    "yellow":  "#FFC107",
}

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


def setup_plot_style(font_size: int = 12) -> None:
    """
    配置全局 matplotlib 绘图风格，支持中文字体。

    参数：
        font_size: 基础字号，默认 12

    使用方式：
        在任何绘图脚本开头调用一次 setup_plot_style()
    """
    # 查找中文字体
    chinese_font_path = _find_chinese_font()

    if chinese_font_path:
        try:
            prop = fm.FontProperties(fname=chinese_font_path)
            font_name = prop.get_name()
            plt.rcParams["font.family"] = ["sans-serif"]
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            print(f"[plot_utils] 已加载中文字体：{font_name}")
        except Exception:
            chinese_font_path = None

    if not chinese_font_path:
        # 未找到中文字体，使用默认字体（中文可能显示为方块，但不报错）
        print("[plot_utils] 警告：未找到中文字体，中文可能无法正常显示。")
        print("  建议安装：sudo apt install fonts-wqy-microhei")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    # 解决负号显示问题
    plt.rcParams["axes.unicode_minus"] = False

    # 全局字号与风格
    plt.rcParams.update({
        "font.size":        font_size,
        "axes.titlesize":   font_size + 2,
        "axes.labelsize":   font_size,
        "xtick.labelsize":  font_size - 1,
        "ytick.labelsize":  font_size - 1,
        "legend.fontsize":  font_size - 1,
        "figure.dpi":       100,          # 屏幕显示 DPI（保存时另设）
        "savefig.dpi":      SAVE_DPI,     # 保存 DPI
        "savefig.bbox":     "tight",      # 自动裁边
        "axes.spines.top":    False,      # 去掉上边框
        "axes.spines.right":  False,      # 去掉右边框
        "axes.grid":          True,       # 默认显示网格
        "grid.alpha":         0.3,
        "lines.linewidth":    2.0,
        "patch.linewidth":    1.5,
    })


def save_fig(fig: plt.Figure, path: str, dpi: int = SAVE_DPI) -> None:
    """
    保存图表到指定路径，自动创建目录。

    参数：
        fig:  matplotlib Figure 对象
        path: 输出文件路径（支持 .png / .pdf / .svg）
        dpi:  保存分辨率，默认 300
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
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
    title: str = "混淆矩阵",
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
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
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
    title: str = "Precision-Recall 曲线",
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
        ax.plot(r, p, color=colors[i % len(colors)], label=legend_label, linewidth=2)

    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")

    return ax


def draw_loss_curve(
    epochs: list[int],
    losses: dict[str, list[float]],
    ax: plt.Axes | None = None,
    title: str = "训练 Loss 曲线",
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
