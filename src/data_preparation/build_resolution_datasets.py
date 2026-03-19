"""
build_resolution_datasets.py - 多分辨率数据集构建脚本
路径：src/data_preparation/build_resolution_datasets.py

功能：
    对已有的 YOLO 数据集（train/val/test 图像）进行多分辨率下采样，
    生成4个分辨率级别（100% / 75% / 50% / 25%）的数据集。
    下采样后统一缩放回 640×640 送入模型，YOLO 归一化标注直接复用。

使用方式：
    python src/data_preparation/build_resolution_datasets.py \
        --dataset satellite \
        [--scales 100 75 50 25]

输出目录：
    data/resolution_datasets_{satellite|uav}/
    ├── scale_100/   （原始分辨率，直接复制）
    ├── scale_75/    （75% 下采样后缩放回 640）
    ├── scale_50/    （50% 下采样后缩放回 640）
    └── scale_25/    （25% 下采样后缩放回 640）
    每个子目录结构与原始 YOLO 数据集相同（images/train|val|test + labels/train|val|test）
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="多分辨率数据集构建脚本")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["satellite", "uav"])
    parser.add_argument("--scales",  type=int, nargs="+",
                        default=[100, 75, 50, 25],
                        help="分辨率百分比列表，默认 100 75 50 25")
    parser.add_argument("--config",  type=str, default=None,
                        help="配置文件路径（默认自动推断）")
    return parser.parse_args()


def downsample_image(img: np.ndarray, scale_pct: int, target_size: int = 640) -> np.ndarray:
    """
    对图像进行下采样后缩放回目标尺寸。

    流程：
        1. 将图像缩放至 scale_pct% 大小（双线性插值）
        2. 再缩放回 target_size×target_size（双线性插值）

    参数：
        img:         BGR 图像
        scale_pct:   下采样百分比（100=不变, 50=缩小一半）
        target_size: 最终输出尺寸

    返回值：
        处理后的 BGR 图像，shape=(target_size, target_size, 3)
    """
    if scale_pct == 100:
        # 直接缩放到目标尺寸（如果已经是目标尺寸则不变）
        if img.shape[0] == target_size and img.shape[1] == target_size:
            return img.copy()
        return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    h, w = img.shape[:2]
    # 步骤1：下采样
    new_h = max(1, int(h * scale_pct / 100))
    new_w = max(1, int(w * scale_pct / 100))
    downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 步骤2：缩放回目标尺寸
    result = cv2.resize(downsampled, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return result


def build_scale_dataset(
    src_img_dir: str,
    src_lbl_dir: str,
    dst_img_dir: str,
    dst_lbl_dir: str,
    scale_pct: int,
    target_size: int = 640,
    logger=None,
) -> int:
    """
    对单个子集（train/val/test）构建指定分辨率的数据集。

    参数：
        src_img_dir: 源图像目录
        src_lbl_dir: 源标注目录
        dst_img_dir: 目标图像目录
        dst_lbl_dir: 目标标注目录
        scale_pct:   分辨率百分比
        target_size: 最终图像尺寸

    返回值：
        处理的图像数量
    """
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png"}
    img_files = [f for f in os.listdir(src_img_dir)
                 if Path(f).suffix.lower() in img_exts]

    count = 0
    for img_name in img_files:
        src_img_path = os.path.join(src_img_dir, img_name)
        stem = Path(img_name).stem
        src_lbl_path = os.path.join(src_lbl_dir, stem + ".txt")

        # 处理图像
        img = cv2.imread(src_img_path)
        if img is None:
            if logger:
                logger.warning(f"无法读取图像：{src_img_path}")
            continue

        processed = downsample_image(img, scale_pct, target_size)
        dst_img_path = os.path.join(dst_img_dir, stem + ".jpg")
        cv2.imwrite(dst_img_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 标注直接复制（YOLO 归一化坐标不受分辨率影响）
        dst_lbl_path = os.path.join(dst_lbl_dir, stem + ".txt")
        if os.path.isfile(src_lbl_path):
            shutil.copy2(src_lbl_path, dst_lbl_path)
        else:
            # 无标注文件（负样本）：创建空文件
            open(dst_lbl_path, "w").close()

        count += 1

    return count


def generate_dataset_yaml(
    output_dir: str,
    scale_pct: int,
    dataset: str,
    class_names: list,
) -> str:
    """
    生成 dataset.yaml 文件。

    返回值：yaml 文件路径
    """
    yaml_content = f"""# 多分辨率数据集配置 - {dataset} scale_{scale_pct}
# 由 build_resolution_datasets.py 自动生成

path: {os.path.abspath(output_dir)}
train: images/train
val:   images/val
test:  images/test

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_path


def main():
    args = parse_args()
    logger = get_logger("build_resolution_datasets", log_dir="logs")

    import yaml
    config_path = args.config or f"configs/{args.dataset}_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 源数据集目录（已构建好的 YOLO 数据集）
    src_dataset_dir = cfg["data"]["output_dir"]
    class_names = cfg["classes"]["names"]
    target_size = cfg["train"]["imgsz"]

    # 输出根目录
    output_root = f"data/resolution_datasets_{args.dataset}"
    os.makedirs(output_root, exist_ok=True)

    logger.info(f"构建多分辨率数据集：{args.dataset}")
    logger.info(f"源数据集：{src_dataset_dir}")
    logger.info(f"分辨率级别：{args.scales}")

    splits = ["train", "val", "test"]
    summary = {}

    for scale in args.scales:
        scale_dir = os.path.join(output_root, f"scale_{scale}")
        logger.info(f"\n--- 处理 scale_{scale} ---")

        total = 0
        for split in splits:
            src_img = os.path.join(src_dataset_dir, "images", split)
            src_lbl = os.path.join(src_dataset_dir, "labels", split)
            dst_img = os.path.join(scale_dir, "images", split)
            dst_lbl = os.path.join(scale_dir, "labels", split)

            if not os.path.isdir(src_img):
                logger.warning(f"源目录不存在，跳过：{src_img}")
                continue

            n = build_scale_dataset(src_img, src_lbl, dst_img, dst_lbl,
                                    scale, target_size, logger)
            logger.info(f"  {split}: {n} 张图像")
            total += n

        # 生成 dataset.yaml
        yaml_path = generate_dataset_yaml(scale_dir, scale, args.dataset, class_names)
        logger.info(f"  dataset.yaml: {yaml_path}")
        logger.info(f"  合计: {total} 张图像")
        summary[scale] = total

    # 输出汇总
    logger.info("\n=== 构建完成 ===")
    for scale, total in summary.items():
        logger.info(f"  scale_{scale}: {total} 张图像 → {output_root}/scale_{scale}/")


if __name__ == "__main__":
    main()
