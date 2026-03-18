"""
build_dataset.py - YOLO 数据集构建脚本
路径：src/data_preparation/build_dataset.py

功能：
    将预处理（切片/缩放）后的图像划分为 train / val / test 三个子集，
    生成 YOLO 格式的数据集目录结构和 dataset.yaml 配置文件。

划分原则：
    - 卫星数据：按原始图像划分（34 张原图 → 按比例分配）
    - 无人机数据：按原始图像（非切片）划分，同一原图的全部切片归属同一子集
      （防止相同区域的切片分别出现在 train 和 test，避免数据泄露）

使用方式：
    # 卫星数据集
    python src/data_preparation/build_dataset.py --dataset satellite

    # 无人机数据集
    python src/data_preparation/build_dataset.py --dataset uav

输出：
    data/yolo_dataset_{satellite|uav}/
    ├── images/train/  val/  test/
    ├── labels/train/  val/  test/
    ├── dataset.yaml
    └── split_summary.txt   各子集统计信息
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.plot_utils import setup_plot_style, save_fig

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 工具函数
# =============================================================================

def load_tile_map(tile_map_path: str) -> dict[str, list[str]]:
    """
    读取无人机切片归属映射文件（由 preprocess_uav.py 生成）。

    格式：每行 "orig_stem\ttile_name1,tile_name2,..."

    返回值：
        {orig_stem: [tile_name_list]}
    """
    tile_map = {}
    with open(tile_map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            orig_stem  = parts[0]
            tile_names = parts[1].split(",") if parts[1] else []
            tile_map[orig_stem] = tile_names
    return tile_map


def split_list(items: list, ratios: tuple, seed: int = 42) -> tuple[list, list, list]:
    """
    按比例将列表随机划分为三部分（train / val / test）。

    参数：
        items:  待划分列表
        ratios: (train_ratio, val_ratio, test_ratio)，三者之和应为 1.0
        seed:   随机种子

    返回值：
        (train_items, val_items, test_items)
    """
    items = list(items)
    random.seed(seed)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    # test 获得剩余部分（避免因取整导致丢失样本）
    n_test  = n - n_train - n_val

    train = items[:n_train]
    val   = items[n_train:n_train + n_val]
    test  = items[n_train + n_val:]

    return train, val, test


def copy_files(
    stems: list[str],
    src_img_dir: str,
    src_lbl_dir: str,
    dst_img_dir: str,
    dst_lbl_dir: str,
    logger=None,
) -> int:
    """
    将指定文件名（stem 列表）的图像和标注复制到目标目录。

    参数：
        stems:       文件 stem 列表（不含扩展名）
        src_img_dir: 源图像目录
        src_lbl_dir: 源标注目录
        dst_img_dir: 目标图像目录
        dst_lbl_dir: 目标标注目录
        logger:      日志对象

    返回值：
        成功复制的文件对数
    """
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    count = 0
    img_exts = [".jpg", ".jpeg", ".png"]

    for stem in stems:
        # 找到对应的图像文件（扩展名可能不同）
        img_src = None
        for ext in img_exts:
            candidate = os.path.join(src_img_dir, stem + ext)
            if os.path.isfile(candidate):
                img_src = candidate
                break

        lbl_src = os.path.join(src_lbl_dir, stem + ".txt")

        if img_src is None:
            if logger:
                logger.warning(f"  图像文件不存在，跳过：{stem}")
            continue

        # 复制图像
        img_dst = os.path.join(dst_img_dir, Path(img_src).name)
        shutil.copy2(img_src, img_dst)

        # 复制标注（若无标注文件，创建空文件）
        lbl_dst = os.path.join(dst_lbl_dir, stem + ".txt")
        if os.path.isfile(lbl_src):
            shutil.copy2(lbl_src, lbl_dst)
        else:
            # 创建空标注文件（负样本）
            open(lbl_dst, "w").close()

        count += 1

    return count


def generate_dataset_yaml(
    dataset_dir: str,
    nc: int,
    names: list[str],
    save_path: str,
) -> None:
    """
    生成 YOLOv8 格式的 dataset.yaml 文件。

    参数：
        dataset_dir: 数据集根目录（绝对路径）
        nc:          类别数
        names:       类别名称列表
        save_path:   yaml 输出路径
    """
    content = {
        "path":  os.path.abspath(dataset_dir),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    nc,
        "names": names,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def visualize_split(
    split_info: dict,
    output_path: str,
    dataset_name: str = "",
) -> None:
    """
    生成数据集划分统计可视化图。

    参数：
        split_info: {
            "train": {"images": N, "boxes": M},
            "val":   {"images": N, "boxes": M},
            "test":  {"images": N, "boxes": M},
        }
        output_path: 图表保存路径
        dataset_name: 数据集名称（用于图标题）
    """
    setup_plot_style()

    splits  = ["train", "val", "test"]
    n_imgs  = [split_info[s]["images"] for s in splits]
    n_boxes = [split_info[s]["boxes"]  for s in splits]
    colors  = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{dataset_name} 数据集划分统计", fontsize=14, fontweight="bold")

    # ── 子图1：各子集图像数量 ────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(splits, n_imgs, color=colors, edgecolor="white", linewidth=1.5)
    for bar, n in zip(bars, n_imgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(n), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("子集")
    ax.set_ylabel("图像数量")
    ax.set_title(f"各子集图像数（总计 {sum(n_imgs)} 张）")
    ax.set_ylim(0, max(n_imgs) * 1.15)

    # ── 子图2：各子集标注框数量 ──────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(splits, n_boxes, color=colors, edgecolor="white", linewidth=1.5)
    for bar, n in zip(bars, n_boxes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(n), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("子集")
    ax.set_ylabel("标注框数量")
    ax.set_title(f"各子集标注框数（总计 {sum(n_boxes)} 个）")
    ax.set_ylim(0, max(n_boxes) * 1.15 if max(n_boxes) > 0 else 1)

    plt.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)


def count_boxes_in_dir(lbl_dir: str) -> int:
    """统计标注目录中所有标注框的总数。"""
    total = 0
    if not os.path.isdir(lbl_dir):
        return 0
    for fname in os.listdir(lbl_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(lbl_dir, fname), "r") as f:
                lines = [l.strip() for l in f if l.strip()]
                total += len(lines)
    return total


# =============================================================================
# 卫星数据集构建
# =============================================================================

def build_satellite_dataset(cfg: dict, logger) -> dict:
    """
    构建卫星数据集：按原始图像划分 train/val/test。

    返回值：
        split_info 字典
    """
    pre_dir    = os.path.join(cfg["data"]["output_dir"], "preprocessed")
    src_img    = os.path.join(pre_dir, "images")
    src_lbl    = os.path.join(pre_dir, "labels")
    output_dir = cfg["data"]["output_dir"]

    # 获取所有图像 stem
    img_stems = sorted([
        Path(f).stem for f in os.listdir(src_img)
        if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not img_stems:
        logger.error(f"未找到预处理图像：{src_img}")
        logger.error("请先运行 preprocess_satellite.py")
        return {}

    logger.info(f"卫星图像总数：{len(img_stems)}")

    # 划分
    split_cfg = cfg["split"]
    ratios = (split_cfg["train_ratio"], split_cfg["val_ratio"], split_cfg["test_ratio"])
    train_stems, val_stems, test_stems = split_list(img_stems, ratios, split_cfg["seed"])

    logger.info(f"划分结果：train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)}")

    # 复制文件
    split_info = {}
    for split_name, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        dst_img = os.path.join(output_dir, "images", split_name)
        dst_lbl = os.path.join(output_dir, "labels", split_name)
        n = copy_files(stems, src_img, src_lbl, dst_img, dst_lbl, logger)
        n_boxes = count_boxes_in_dir(dst_lbl)
        split_info[split_name] = {"images": n, "boxes": n_boxes}
        logger.info(f"  {split_name}: {n} 张图像，{n_boxes} 个框")

    return split_info


# =============================================================================
# 无人机数据集构建
# =============================================================================

def build_uav_dataset(cfg: dict, logger) -> dict:
    """
    构建无人机数据集：按原始图像（非切片）划分，防止数据泄露。

    返回值：
        split_info 字典
    """
    pre_dir    = os.path.join(cfg["data"]["output_dir"], "preprocessed")
    src_img    = os.path.join(pre_dir, "images")
    src_lbl    = os.path.join(pre_dir, "labels")
    tile_map_path = os.path.join(pre_dir, "tile_map.txt")
    output_dir = cfg["data"]["output_dir"]

    if not os.path.isfile(tile_map_path):
        logger.error(f"切片映射文件不存在：{tile_map_path}")
        logger.error("请先运行 preprocess_uav.py")
        return {}

    # 读取切片映射（按原图分组）
    tile_map = load_tile_map(tile_map_path)
    orig_stems = sorted(tile_map.keys())
    logger.info(f"无人机原始图像数：{len(orig_stems)}")

    total_tiles = sum(len(v) for v in tile_map.values())
    logger.info(f"切片总数：{total_tiles}")

    # 按原始图像划分
    split_cfg = cfg["split"]
    ratios = (split_cfg["train_ratio"], split_cfg["val_ratio"], split_cfg["test_ratio"])
    train_origs, val_origs, test_origs = split_list(orig_stems, ratios, split_cfg["seed"])

    logger.info(
        f"原图划分：train={len(train_origs)}, val={len(val_origs)}, test={len(test_origs)}"
    )

    # 展开切片 stem 列表
    def expand(origs): return [t for o in origs for t in tile_map.get(o, [])]

    train_tiles = expand(train_origs)
    val_tiles   = expand(val_origs)
    test_tiles  = expand(test_origs)

    logger.info(
        f"切片划分：train={len(train_tiles)}, val={len(val_tiles)}, test={len(test_tiles)}"
    )

    split_info = {}
    for split_name, stems in [("train", train_tiles), ("val", val_tiles), ("test", test_tiles)]:
        dst_img = os.path.join(output_dir, "images", split_name)
        dst_lbl = os.path.join(output_dir, "labels", split_name)
        n = copy_files(stems, src_img, src_lbl, dst_img, dst_lbl, logger)
        n_boxes = count_boxes_in_dir(dst_lbl)
        split_info[split_name] = {"images": n, "boxes": n_boxes}
        logger.info(f"  {split_name}: {n} 张切片，{n_boxes} 个框")

    return split_info


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO 数据集构建脚本")
    parser.add_argument("--dataset", type=str, choices=["satellite", "uav"],
                        required=True, help="数据集类型")
    parser.add_argument("--config", type=str,
                        help="配置文件路径（不指定则自动推断）")
    parser.add_argument("--vis", action="store_true",
                        help="生成划分统计可视化图")
    args = parser.parse_args()

    if args.config is None:
        args.config = f"configs/{args.dataset}_config.yaml"

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger(f"build_dataset_{args.dataset}", log_dir="logs")
    logger.info("=" * 60)
    logger.info(f"YOLO 数据集构建开始：{args.dataset}")

    # 构建数据集
    if args.dataset == "satellite":
        split_info = build_satellite_dataset(cfg, logger)
    else:
        split_info = build_uav_dataset(cfg, logger)

    if not split_info:
        logger.error("构建失败，请检查预处理步骤是否已完成")
        return

    output_dir = cfg["data"]["output_dir"]

    # 生成 dataset.yaml
    yaml_path = cfg["data"]["dataset_yaml"]
    generate_dataset_yaml(
        dataset_dir=output_dir,
        nc=cfg["classes"]["nc"],
        names=cfg["classes"]["names"],
        save_path=yaml_path,
    )
    logger.info(f"dataset.yaml 已生成：{yaml_path}")

    # 保存划分摘要
    summary_path = os.path.join(output_dir, "split_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"数据集：{args.dataset}\n")
        f.write("-" * 40 + "\n")
        total_imgs  = sum(v["images"] for v in split_info.values())
        total_boxes = sum(v["boxes"]  for v in split_info.values())
        for split_name, info in split_info.items():
            f.write(f"{split_name:8s}: {info['images']:5d} 张图像, {info['boxes']:5d} 个标注框\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'总计':8s}: {total_imgs:5d} 张图像, {total_boxes:5d} 个标注框\n")
    logger.info(f"划分摘要已保存：{summary_path}")

    # 可视化
    if args.vis and split_info:
        vis_path = os.path.join(output_dir, "split_statistics.png")
        visualize_split(split_info, vis_path, dataset_name=args.dataset)
        logger.info(f"划分统计图已保存：{vis_path}")

    logger.info("=" * 60)
    logger.info("数据集构建完成！")
    logger.info(f"数据集路径：{os.path.abspath(output_dir)}")
    logger.info(f"dataset.yaml：{os.path.abspath(yaml_path)}")


if __name__ == "__main__":
    main()
