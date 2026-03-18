"""
preprocess_satellite.py - 卫星影像预处理脚本
路径：src/data_preparation/preprocess_satellite.py

功能：
    将卫星原始图像（256×256）缩放至 640×640，标注坐标同步更新（YOLO 格式
    归一化坐标不受尺寸变化影响，直接复制即可），并输出目标框数量分布图。

使用方式：
    python src/data_preparation/preprocess_satellite.py \
        --config configs/satellite_config.yaml \
        [--vis]   # 可选：生成可视化统计图

输出：
    data/yolo_dataset_satellite/preprocessed/images/  预处理后图像
    data/yolo_dataset_satellite/preprocessed/labels/  对应标注（直接复制）
    data/yolo_dataset_satellite/preprocessed/stats.png 目标框分布统计图（--vis 时）
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# 将项目根目录加入 sys.path，使 src 包可正常导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.plot_utils import setup_plot_style, save_fig

import matplotlib.pyplot as plt


def load_yolo_labels(label_path: str) -> list[list[float]]:
    """
    读取 YOLO 格式标注文件。

    参数：
        label_path: 标注文件路径（.txt）

    返回值：
        标注列表，每项为 [class_id, cx, cy, w, h]（均为 float）
        若文件为空或不存在，返回空列表
    """
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            boxes.append([float(x) for x in parts])
    return boxes


def save_yolo_labels(boxes: list[list[float]], save_path: str) -> None:
    """
    将 YOLO 格式标注写入文件。

    参数：
        boxes:     标注列表，每项为 [class_id, cx, cy, w, h]
        save_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for box in boxes:
            # class_id 为整数，其余保留 6 位小数
            f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def preprocess_satellite(
    orig_img_dir: str,
    orig_lbl_dir: str,
    out_img_dir: str,
    out_lbl_dir: str,
    target_size: int = 640,
    logger=None,
) -> dict:
    """
    预处理卫星图像：将所有图像缩放至 target_size × target_size。

    YOLO 归一化坐标（cx, cy, w, h 均在 [0,1] 范围内）不随图像尺寸变化，
    因此标注文件直接复制，无需坐标变换。

    参数：
        orig_img_dir: 原始图像目录
        orig_lbl_dir: 原始标注目录
        out_img_dir:  输出图像目录
        out_lbl_dir:  输出标注目录
        target_size:  目标尺寸（默认 640）
        logger:       日志对象

    返回值：
        统计信息字典：
            total_images:   处理的图像总数
            total_boxes:    标注框总数
            box_counts:     每张图像的标注框数列表
            box_sizes:      所有标注框的相对尺寸列表 [(w, h), ...]
            skipped:        跳过的图像数（文件损坏等）
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 支持的图像格式
    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    # 遍历原始图像目录
    img_files = sorted([
        f for f in os.listdir(orig_img_dir)
        if Path(f).suffix.lower() in img_exts
    ])

    if not img_files:
        logger.error(f"未找到图像文件：{orig_img_dir}")
        return {}

    logger.info(f"找到 {len(img_files)} 张卫星图像，目标尺寸：{target_size}×{target_size}")

    stats = {
        "total_images": 0,
        "total_boxes":  0,
        "box_counts":   [],      # 每张图像的框数
        "box_sizes":    [],      # 相对宽高 [(rel_w, rel_h), ...]
        "skipped":      0,
        "orig_sizes":   [],      # 原始图像尺寸列表
    }

    for img_name in img_files:
        img_path = os.path.join(orig_img_dir, img_name)
        stem = Path(img_name).stem
        suffix = Path(img_name).suffix.lower()

        # ── 读取图像 ──────────────────────────────────────────────────────────
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"  无法读取图像，跳过：{img_name}")
            stats["skipped"] += 1
            continue

        orig_h, orig_w = img.shape[:2]
        stats["orig_sizes"].append((orig_w, orig_h))

        # ── 缩放图像（双线性插值）────────────────────────────────────────────
        img_resized = cv2.resize(
            img,
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # ── 保存缩放后图像（统一为 .jpg）────────────────────────────────────
        out_img_name = stem + ".jpg"
        out_img_path = os.path.join(out_img_dir, out_img_name)
        cv2.imwrite(out_img_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── 读取并复制标注（归一化坐标不受缩放影响）────────────────────────
        lbl_path = os.path.join(orig_lbl_dir, stem + ".txt")
        boxes = load_yolo_labels(lbl_path)

        out_lbl_path = os.path.join(out_lbl_dir, stem + ".txt")
        save_yolo_labels(boxes, out_lbl_path)

        # ── 统计信息 ──────────────────────────────────────────────────────────
        stats["total_images"] += 1
        stats["total_boxes"]  += len(boxes)
        stats["box_counts"].append(len(boxes))
        for box in boxes:
            stats["box_sizes"].append((box[3], box[4]))  # (rel_w, rel_h)

        logger.info(f"  [{stats['total_images']:>3d}/{len(img_files)}] {img_name}"
                    f"  {orig_w}×{orig_h} → {target_size}×{target_size}"
                    f"  框数：{len(boxes)}")

    logger.info(
        f"预处理完成：{stats['total_images']} 张图像，"
        f"{stats['total_boxes']} 个标注框，"
        f"跳过 {stats['skipped']} 张"
    )
    return stats


def visualize_stats(stats: dict, out_dir: str) -> None:
    """
    生成数据集统计可视化图表（目标框分布）。

    参数：
        stats:   preprocess_satellite() 返回的统计信息
        out_dir: 图表输出目录
    """
    setup_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("卫星数据集统计", fontsize=14, fontweight="bold")

    # ── 子图1：每张图像的标注框数量分布（直方图）────────────────────────────
    ax = axes[0]
    box_counts = stats["box_counts"]
    ax.hist(box_counts, bins=range(0, max(box_counts) + 2), color="#2196F3",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("标注框数量")
    ax.set_ylabel("图像数量")
    ax.set_title(f"每张图像标注框分布\n总计 {sum(box_counts)} 个框 / {len(box_counts)} 张图")
    ax.set_xticks(range(0, max(box_counts) + 2))

    # ── 子图2：标注框相对宽度分布（直方图）──────────────────────────────────
    ax = axes[1]
    if stats["box_sizes"]:
        box_widths  = [s[0] for s in stats["box_sizes"]]
        box_heights = [s[1] for s in stats["box_sizes"]]
        ax.hist(box_widths,  bins=20, alpha=0.7, color="#2196F3", label="相对宽度")
        ax.hist(box_heights, bins=20, alpha=0.7, color="#FF9800", label="相对高度")
        ax.set_xlabel("相对尺寸（归一化）")
        ax.set_ylabel("频次")
        ax.set_title("标注框相对尺寸分布")
        ax.legend()

    # ── 子图3：标注框宽高散点图 ───────────────────────────────────────────────
    ax = axes[2]
    if stats["box_sizes"]:
        widths  = np.array([s[0] for s in stats["box_sizes"]]) * 640  # 转为像素
        heights = np.array([s[1] for s in stats["box_sizes"]]) * 640
        ax.scatter(widths, heights, alpha=0.6, s=40, color="#4CAF50", edgecolors="white", linewidths=0.5)
        ax.set_xlabel("标注框宽度（像素，640尺寸下）")
        ax.set_ylabel("标注框高度（像素，640尺寸下）")
        ax.set_title("标注框宽高散点图")

    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "satellite_dataset_stats.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="卫星影像预处理脚本")
    parser.add_argument("--config", type=str, default="configs/satellite_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--vis", action="store_true",
                        help="生成统计可视化图表")
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── 初始化日志 ────────────────────────────────────────────────────────────
    logger = get_logger("preprocess_satellite", log_dir="logs")
    logger.info("=" * 60)
    logger.info("卫星影像预处理开始")
    logger.info(f"配置文件：{args.config}")

    # ── 路径配置 ──────────────────────────────────────────────────────────────
    orig_img_dir = cfg["data"]["orig_images"]
    orig_lbl_dir = cfg["data"]["orig_labels"]
    output_dir   = cfg["data"]["output_dir"]
    target_size  = cfg["preprocess"]["target_size"]

    # 预处理后输出到 preprocessed/ 子目录
    out_img_dir = os.path.join(output_dir, "preprocessed", "images")
    out_lbl_dir = os.path.join(output_dir, "preprocessed", "labels")

    # ── 执行预处理 ────────────────────────────────────────────────────────────
    stats = preprocess_satellite(
        orig_img_dir=orig_img_dir,
        orig_lbl_dir=orig_lbl_dir,
        out_img_dir=out_img_dir,
        out_lbl_dir=out_lbl_dir,
        target_size=target_size,
        logger=logger,
    )

    # ── 可视化统计 ────────────────────────────────────────────────────────────
    if args.vis and stats:
        vis_dir = os.path.join(output_dir, "preprocessed")
        visualize_stats(stats, vis_dir)
        logger.info(f"统计图已保存至：{vis_dir}")

    logger.info("=" * 60)
    logger.info("卫星影像预处理完成")


if __name__ == "__main__":
    main()
