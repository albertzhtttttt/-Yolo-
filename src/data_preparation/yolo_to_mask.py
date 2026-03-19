"""
yolo_to_mask.py - YOLO 标注框转像素掩码脚本
路径：src/data_preparation/yolo_to_mask.py

功能：
    将 YOLO 格式的目标检测标注框（归一化坐标）转换为二值像素掩码图，
    用于 U-Net 和 FCN 等分割模型的训练标签。

    输出：
        目标区域 = 255（白色）
        背景区域 = 0（黑色）

使用方式：
    python src/data_preparation/yolo_to_mask.py --dataset satellite
    python src/data_preparation/yolo_to_mask.py --dataset uav
    python src/data_preparation/yolo_to_mask.py --dataset all

输出目录：
    data/yolo_dataset_{satellite|uav}/masks/{train|val|test}/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 标注框转像素掩码")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["satellite", "uav", "all"])
    parser.add_argument("--config",  type=str, default=None)
    parser.add_argument("--img_size", type=int, default=640,
                        help="掩码图尺寸（与训练图像一致）")
    return parser.parse_args()


def yolo_label_to_mask(
    label_path: str,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    读取 YOLO 格式标注文件，生成二值掩码图。

    参数：
        label_path: YOLO 标注文件路径（.txt）
        img_w, img_h: 图像尺寸

    返回值：
        二值掩码，shape=(img_h, img_w)，dtype=uint8，值为 0 或 255
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if not os.path.isfile(label_path):
        return mask

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = [float(x) for x in parts]
            # 转换为像素坐标
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            # 裁剪到图像范围
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

    return mask


def convert_split(
    img_dir: str,
    lbl_dir: str,
    mask_dir: str,
    img_size: int,
    logger,
) -> int:
    """
    转换单个子集（train/val/test）的所有标注。

    返回值：处理的图像数量
    """
    os.makedirs(mask_dir, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png"}
    img_files = [f for f in os.listdir(img_dir)
                 if Path(f).suffix.lower() in img_exts]

    count = 0
    for img_name in img_files:
        stem = Path(img_name).stem
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        mask_path = os.path.join(mask_dir, stem + ".png")

        # 获取实际图像尺寸
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
        else:
            h = w = img_size

        mask = yolo_label_to_mask(lbl_path, w, h)

        # 如果图像尺寸与目标不符，缩放掩码
        if h != img_size or w != img_size:
            mask = cv2.resize(mask, (img_size, img_size),
                              interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(mask_path, mask)
        count += 1

    return count


def convert_dataset(dataset: str, cfg: dict, img_size: int, logger) -> None:
    """转换单个数据集的所有子集。"""
    dataset_dir = cfg["data"]["output_dir"]
    splits = ["train", "val", "test"]

    logger.info(f"转换 {dataset} 数据集标注为掩码...")
    for split in splits:
        img_dir  = os.path.join(dataset_dir, "images", split)
        lbl_dir  = os.path.join(dataset_dir, "labels", split)
        mask_dir = os.path.join(dataset_dir, "masks", split)

        if not os.path.isdir(img_dir):
            logger.warning(f"目录不存在，跳过：{img_dir}")
            continue

        n = convert_split(img_dir, lbl_dir, mask_dir, img_size, logger)
        logger.info(f"  {split}: {n} 张掩码图 → {mask_dir}")


def main():
    args = parse_args()
    logger = get_logger("yolo_to_mask", log_dir="logs")

    import yaml
    datasets = ["satellite", "uav"] if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        config_path = args.config or f"configs/{dataset}_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        convert_dataset(dataset, cfg, args.img_size, logger)

    logger.info("掩码转换完成！")


if __name__ == "__main__":
    main()
