"""
preprocess_uav.py - 无人机影像预处理脚本（滑动窗口切片）
路径：src/data_preparation/preprocess_uav.py

功能：
    对高分辨率无人机图像（~1426×1161）进行滑动窗口切片，生成 640×640 子块，
    同步更新 YOLO 格式标注坐标。

切片逻辑：
    1. 以 tile_size=640、stride=320 滑动遍历图像（50% 重叠）
    2. 对每个切片，将原始 YOLO 标注框转换至切片坐标系
    3. 保留可见面积比 ≥ min_bbox_visibility 的标注框
    4. 截断至切片边界后重新归一化
    5. 空白切片（无标注框）按 neg_sample_ratio 随机保留

使用方式：
    python src/data_preparation/preprocess_uav.py \
        --config configs/uav_config.yaml \
        [--vis]

输出：
    data/yolo_dataset_uav/preprocessed/images/  切片图像
    data/yolo_dataset_uav/preprocessed/labels/  切片标注
    data/yolo_dataset_uav/preprocessed/tile_map.txt  切片归属映射
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

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
    """保存 YOLO 格式标注到文件。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for box in boxes:
            f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def clip_boxes_to_tile(
    boxes: list[list[float]],
    img_w: int,
    img_h: int,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_visibility: float = 0.3,
) -> list[list[float]]:
    """
    将原始图像坐标系的 YOLO 标注框裁剪到切片坐标系。

    参数：
        boxes:          原始标注框列表（YOLO 格式，相对于原图归一化）
        img_w, img_h:   原始图像宽高（像素）
        tile_x, tile_y: 切片左上角坐标（像素）
        tile_size:      切片尺寸（像素，正方形）
        min_visibility: 框在切片内可见面积 / 原始面积 的最低比值，低于此值丢弃

    返回值：
        转换后的标注框列表（相对于切片归一化的 YOLO 格式），可能为空列表
    """
    clipped = []
    tile_right  = tile_x + tile_size
    tile_bottom = tile_y + tile_size

    for box in boxes:
        cls_id, cx_rel, cy_rel, w_rel, h_rel = box

        # 1. 归一化坐标 → 原图像素坐标（xyxy 格式）
        cx_px = cx_rel * img_w
        cy_px = cy_rel * img_h
        w_px  = w_rel  * img_w
        h_px  = h_rel  * img_h

        x1 = cx_px - w_px / 2
        y1 = cy_px - h_px / 2
        x2 = cx_px + w_px / 2
        y2 = cy_px + h_px / 2

        orig_area = w_px * h_px
        if orig_area <= 0:
            continue

        # 2. 与切片区域求交集
        ix1 = max(x1, tile_x)
        iy1 = max(y1, tile_y)
        ix2 = min(x2, tile_right)
        iy2 = min(y2, tile_bottom)

        inter_w = ix2 - ix1
        inter_h = iy2 - iy1

        if inter_w <= 0 or inter_h <= 0:
            # 框完全在切片外
            continue

        inter_area = inter_w * inter_h
        visibility = inter_area / orig_area

        if visibility < min_visibility:
            # 可见比例太低，丢弃
            continue

        # 3. 转换为切片坐标系并归一化
        new_cx = (ix1 + ix2) / 2 - tile_x
        new_cy = (iy1 + iy2) / 2 - tile_y
        new_w  = inter_w
        new_h  = inter_h

        # 归一化到 [0, 1]
        new_cx_rel = np.clip(new_cx / tile_size, 0.0, 1.0)
        new_cy_rel = np.clip(new_cy / tile_size, 0.0, 1.0)
        new_w_rel  = np.clip(new_w  / tile_size, 0.0, 1.0)
        new_h_rel  = np.clip(new_h  / tile_size, 0.0, 1.0)

        # 过滤极小框（宽或高 < 2px 等效）
        if new_w_rel < 2 / tile_size or new_h_rel < 2 / tile_size:
            continue

        clipped.append([cls_id, new_cx_rel, new_cy_rel, new_w_rel, new_h_rel])

    return clipped


def slice_image(
    img_path: str,
    lbl_path: str,
    out_img_dir: str,
    out_lbl_dir: str,
    tile_size: int = 640,
    stride: int = 320,
    min_visibility: float = 0.3,
    neg_sample_ratio: float = 0.1,
    rng: Optional[random.Random] = None,
) -> dict:
    """
    对单张无人机图像进行滑动窗口切片并保存切片结果。

    参数：
        img_path:         原始图像路径
        lbl_path:         对应标注文件路径
        out_img_dir:      切片图像输出目录
        out_lbl_dir:      切片标注输出目录
        tile_size:        切片尺寸（像素）
        stride:           滑动步长（像素）
        min_visibility:   标注框最低可见比例
        neg_sample_ratio: 空白切片保留比例
        rng:              随机数生成器（用于负样本采样）

    返回值：
        {
            "total_tiles":    该图像产生的总切片数（包括被丢弃的）
            "pos_tiles":      含标注框的切片数
            "neg_tiles":      保留的空白切片数
            "total_boxes":    保存的标注框总数
            "tile_names":     保存的切片文件名列表
        }
    """
    if rng is None:
        rng = random.Random(42)

    img = cv2.imread(img_path)
    if img is None:
        return {"total_tiles": 0, "pos_tiles": 0, "neg_tiles": 0,
                "total_boxes": 0, "tile_names": []}

    img_h, img_w = img.shape[:2]
    stem = Path(img_path).stem

    # 读取原始标注
    boxes = load_yolo_labels(lbl_path)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    total_tiles = 0
    pos_tiles   = 0
    neg_tiles   = 0
    total_boxes = 0
    tile_names  = []

    # 滑动窗口遍历（从左到右，从上到下）
    y = 0
    row = 0
    while y < img_h:
        # 最后一行：如果余量不足 tile_size，从底部向上对齐
        y_end = min(y + tile_size, img_h)
        y_start = max(0, y_end - tile_size)

        x = 0
        col = 0
        while x < img_w:
            x_end   = min(x + tile_size, img_w)
            x_start = max(0, x_end - tile_size)

            total_tiles += 1

            # ── 裁剪切片 ──────────────────────────────────────────────────────
            tile_img = img[y_start:y_start + tile_size, x_start:x_start + tile_size]

            # 若切片尺寸不足（图像右/下边缘），填充至 tile_size
            pad_h = tile_size - tile_img.shape[0]
            pad_w = tile_size - tile_img.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, pad_h, 0, pad_w,
                    cv2.BORDER_REFLECT_101,
                )

            # ── 坐标变换 ──────────────────────────────────────────────────────
            tile_boxes = clip_boxes_to_tile(
                boxes=boxes,
                img_w=img_w,
                img_h=img_h,
                tile_x=x_start,
                tile_y=y_start,
                tile_size=tile_size,
                min_visibility=min_visibility,
            )

            # ── 决定是否保留该切片 ────────────────────────────────────────────
            if len(tile_boxes) == 0:
                # 空白切片按比例随机保留（负样本）
                if rng.random() > neg_sample_ratio:
                    # 步进到下一个位置
                    x = x_start + stride
                    col += 1
                    if x_start + tile_size >= img_w:
                        break
                    continue
                neg_tiles += 1
            else:
                pos_tiles += 1

            # ── 保存切片 ──────────────────────────────────────────────────────
            tile_name = f"{stem}_r{row:03d}_c{col:03d}"
            tile_img_path = os.path.join(out_img_dir, tile_name + ".jpg")
            tile_lbl_path = os.path.join(out_lbl_dir, tile_name + ".txt")

            cv2.imwrite(tile_img_path, tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            save_yolo_labels(tile_boxes, tile_lbl_path)

            total_boxes += len(tile_boxes)
            tile_names.append(tile_name)

            # 步进
            x = x_start + stride
            col += 1
            if x_start + tile_size >= img_w:
                break

        y = y_start + stride
        row += 1
        if y_start + tile_size >= img_h:
            break

    return {
        "total_tiles": total_tiles,
        "pos_tiles":   pos_tiles,
        "neg_tiles":   neg_tiles,
        "total_boxes": total_boxes,
        "tile_names":  tile_names,
    }


def preprocess_uav(
    orig_img_dir: str,
    orig_lbl_dir: str,
    out_img_dir: str,
    out_lbl_dir: str,
    tile_size: int = 640,
    stride: int = 320,
    min_visibility: float = 0.3,
    neg_sample_ratio: float = 0.1,
    seed: int = 42,
    logger=None,
) -> tuple[dict, dict]:
    """
    批量处理无人机图像集。

    返回值：
        (global_stats, tile_map)
        global_stats: 全局统计信息
        tile_map: {原始图像stem: [切片名列表]} 用于数据集划分时按原图分组
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    img_files = sorted([
        f for f in os.listdir(orig_img_dir)
        if Path(f).suffix.lower() in img_exts
    ])

    if not img_files:
        logger.error(f"未找到图像文件：{orig_img_dir}")
        return {}, {}

    logger.info(
        f"找到 {len(img_files)} 张无人机图像\n"
        f"切片参数：tile_size={tile_size}, stride={stride}, "
        f"min_visibility={min_visibility}, neg_ratio={neg_sample_ratio}"
    )

    rng = random.Random(seed)

    global_stats = {
        "total_images":  0,
        "total_tiles":   0,
        "total_pos":     0,
        "total_neg":     0,
        "total_boxes":   0,
        "tiles_per_img": [],
        "boxes_per_tile": [],
    }
    tile_map = {}   # {orig_stem: [tile_names]}

    for i, img_name in enumerate(img_files):
        img_path = os.path.join(orig_img_dir, img_name)
        stem = Path(img_name).stem
        lbl_path = os.path.join(orig_lbl_dir, stem + ".txt")

        result = slice_image(
            img_path=img_path,
            lbl_path=lbl_path,
            out_img_dir=out_img_dir,
            out_lbl_dir=out_lbl_dir,
            tile_size=tile_size,
            stride=stride,
            min_visibility=min_visibility,
            neg_sample_ratio=neg_sample_ratio,
            rng=rng,
        )

        saved_tiles = len(result["tile_names"])
        global_stats["total_images"] += 1
        global_stats["total_tiles"]  += result["total_tiles"]
        global_stats["total_pos"]    += result["pos_tiles"]
        global_stats["total_neg"]    += result["neg_tiles"]
        global_stats["total_boxes"]  += result["total_boxes"]
        global_stats["tiles_per_img"].append(saved_tiles)

        tile_map[stem] = result["tile_names"]

        logger.info(
            f"  [{i+1:>2d}/{len(img_files)}] {img_name} → "
            f"总切片 {result['total_tiles']}，保留 {saved_tiles}"
            f"（正样本 {result['pos_tiles']}，负样本 {result['neg_tiles']}），"
            f"框数 {result['total_boxes']}"
        )

    logger.info(
        f"\n切片完成：{global_stats['total_images']} 张原图 → "
        f"{global_stats['total_pos'] + global_stats['total_neg']} 个切片"
        f"（正样本 {global_stats['total_pos']}，"
        f"负样本 {global_stats['total_neg']}），"
        f"共 {global_stats['total_boxes']} 个标注框"
    )

    return global_stats, tile_map


def save_tile_map(tile_map: dict, save_path: str) -> None:
    """
    保存切片归属映射文件（供 build_dataset.py 按原图分组）。

    格式：每行 "orig_stem\ttile_name1,tile_name2,..."
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for orig_stem, tile_names in tile_map.items():
            f.write(f"{orig_stem}\t{','.join(tile_names)}\n")


def visualize_stats(stats: dict, tile_map: dict, out_dir: str) -> None:
    """生成无人机切片统计可视化图表。"""
    setup_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("无人机数据集切片统计", fontsize=14, fontweight="bold")

    # ── 子图1：每张原图产生的切片数 ──────────────────────────────────────────
    ax = axes[0]
    tiles_per_img = stats["tiles_per_img"]
    img_indices = range(1, len(tiles_per_img) + 1)
    ax.bar(img_indices, tiles_per_img, color="#2196F3", edgecolor="white")
    ax.axhline(np.mean(tiles_per_img), color="#F44336", linestyle="--",
               linewidth=1.5, label=f"均值={np.mean(tiles_per_img):.1f}")
    ax.set_xlabel("原始图像编号")
    ax.set_ylabel("保留切片数")
    ax.set_title(f"每张原图保留切片数\n总计 {sum(tiles_per_img)} 个切片")
    ax.legend()

    # ── 子图2：正/负样本比例饼图 ─────────────────────────────────────────────
    ax = axes[1]
    pos = stats["total_pos"]
    neg = stats["total_neg"]
    ax.pie(
        [pos, neg],
        labels=[f"正样本\n({pos})", f"负样本\n({neg})"],
        colors=["#4CAF50", "#9E9E9E"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("切片正/负样本比例")

    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "uav_dataset_stats.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="无人机影像切片预处理脚本")
    parser.add_argument("--config", type=str, default="configs/uav_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--vis", action="store_true",
                        help="生成统计可视化图表")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("preprocess_uav", log_dir="logs")
    logger.info("=" * 60)
    logger.info("无人机影像切片预处理开始")

    orig_img_dir = cfg["data"]["orig_images"]
    orig_lbl_dir = cfg["data"]["orig_labels"]
    output_dir   = cfg["data"]["output_dir"]
    pre_cfg      = cfg["preprocess"]

    out_img_dir = os.path.join(output_dir, "preprocessed", "images")
    out_lbl_dir = os.path.join(output_dir, "preprocessed", "labels")

    stats, tile_map = preprocess_uav(
        orig_img_dir=orig_img_dir,
        orig_lbl_dir=orig_lbl_dir,
        out_img_dir=out_img_dir,
        out_lbl_dir=out_lbl_dir,
        tile_size=pre_cfg["target_size"],
        stride=pre_cfg["stride"],
        min_visibility=pre_cfg["min_bbox_visibility"],
        neg_sample_ratio=pre_cfg["neg_sample_ratio"],
        seed=cfg["split"]["seed"],
        logger=logger,
    )

    # 保存切片归属映射（build_dataset.py 按原图分组时使用）
    tile_map_path = os.path.join(output_dir, "preprocessed", "tile_map.txt")
    save_tile_map(tile_map, tile_map_path)
    logger.info(f"切片映射已保存：{tile_map_path}")

    if args.vis and stats:
        visualize_stats(stats, tile_map, os.path.join(output_dir, "preprocessed"))

    logger.info("=" * 60)
    logger.info("无人机影像切片预处理完成")


if __name__ == "__main__":
    main()
