"""
augment.py - 数据增强脚本
路径：src/data_preparation/augment.py

功能：
    对 train 集执行离线数据增强，生成增强后的图像和标注。
    支持卫星和无人机两种数据集，通过 --dataset 参数切换。

增强操作（所有操作均同步变换 YOLO 标注框）：
    - 水平翻转、垂直翻转
    - 随机旋转（0° / 90° / 180° / 270°）
    - 随机缩放（±20%，保持输出尺寸 640×640，随机裁剪/填充）
    - HSV 色彩抖动（色相/饱和度/亮度）
    - 高斯噪声
    - 随机模糊（均值模糊 / 高斯模糊）
    - Mosaic 拼接（4 图合一）

使用方式：
    python src/data_preparation/augment.py \
        --dataset satellite \
        --config  configs/satellite_config.yaml \
        --input_img  data/yolo_dataset_satellite/images/train \
        --input_lbl  data/yolo_dataset_satellite/labels/train \
        --output_img data/yolo_dataset_satellite/images/train \
        --output_lbl data/yolo_dataset_satellite/labels/train \
        [--dry_run]   # 仅输出统计，不保存文件

注意：
    增强图像保存到与原图相同的 train 目录，文件名带 _aug{N} 后缀，
    不覆盖原始图像。
"""

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


# =============================================================================
# YOLO 标注辅助函数
# =============================================================================

def load_yolo_labels(label_path: str) -> list[list[float]]:
    """读取 YOLO 格式标注。"""
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                boxes.append([float(x) for x in parts])
    return boxes


def save_yolo_labels(boxes: list[list[float]], save_path: str) -> None:
    """保存 YOLO 格式标注。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for box in boxes:
            f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def yolo_to_xyxy(box: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    """
    YOLO 归一化格式 → 像素绝对坐标（xyxy）。

    参数：
        box: [class_id, cx, cy, w_rel, h_rel]
        w, h: 图像宽高

    返回值：
        (x1, y1, x2, y2) 像素坐标（未裁剪）
    """
    _, cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float,
                 w: int, h: int, cls_id: int) -> list[float]:
    """
    像素绝对坐标（xyxy）→ YOLO 归一化格式。

    参数：
        x1, y1, x2, y2: 像素坐标
        w, h: 图像宽高
        cls_id: 类别 ID

    返回值：
        [class_id, cx, cy, w_rel, h_rel]
    """
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [cls_id, cx, cy, bw, bh]


def clip_boxes(boxes: list[list[float]], min_size: float = 0.005) -> list[list[float]]:
    """
    将 YOLO 格式标注框裁剪到 [0,1] 范围，并过滤过小的框。

    参数：
        boxes:    YOLO 格式标注列表
        min_size: 宽或高低于此值（归一化）则丢弃，默认 0.005（约 3px / 640）

    返回值：
        有效标注列表
    """
    valid = []
    for box in boxes:
        cls_id, cx, cy, bw, bh = box
        # 裁剪到图像边界
        x1 = max(0.0, cx - bw / 2)
        y1 = max(0.0, cy - bh / 2)
        x2 = min(1.0, cx + bw / 2)
        y2 = min(1.0, cy + bh / 2)
        new_bw = x2 - x1
        new_bh = y2 - y1
        if new_bw < min_size or new_bh < min_size:
            continue
        valid.append([cls_id, (x1 + x2) / 2, (y1 + y2) / 2, new_bw, new_bh])
    return valid


# =============================================================================
# 单项增强函数
# =============================================================================

def aug_flip_horizontal(img: np.ndarray, boxes: list) -> tuple:
    """水平翻转图像与标注框。"""
    img_out = cv2.flip(img, 1)
    boxes_out = []
    for box in boxes:
        cls_id, cx, cy, bw, bh = box
        boxes_out.append([cls_id, 1.0 - cx, cy, bw, bh])
    return img_out, boxes_out


def aug_flip_vertical(img: np.ndarray, boxes: list) -> tuple:
    """垂直翻转图像与标注框。"""
    img_out = cv2.flip(img, 0)
    boxes_out = []
    for box in boxes:
        cls_id, cx, cy, bw, bh = box
        boxes_out.append([cls_id, cx, 1.0 - cy, bw, bh])
    return img_out, boxes_out


def aug_rotate90(img: np.ndarray, boxes: list, k: int) -> tuple:
    """
    旋转 k×90° 图像与标注框（k∈{1,2,3}）。

    参数：
        k: 逆时针旋转次数（1=90°, 2=180°, 3=270°）
    """
    img_out = np.rot90(img, k)
    h, w = img.shape[:2]
    boxes_out = []
    for box in boxes:
        cls_id, cx, cy, bw, bh = box
        if k == 1:     # 逆时针 90°：(cx,cy) → (cy, 1-cx)
            new_cx, new_cy, new_bw, new_bh = cy, 1.0 - cx, bh, bw
        elif k == 2:   # 180°：(cx,cy) → (1-cx, 1-cy)
            new_cx, new_cy, new_bw, new_bh = 1.0 - cx, 1.0 - cy, bw, bh
        else:          # 逆时针 270°（顺时针90°）：(cx,cy) → (1-cy, cx)
            new_cx, new_cy, new_bw, new_bh = 1.0 - cy, cx, bh, bw
        boxes_out.append([cls_id, new_cx, new_cy, new_bw, new_bh])
    return img_out, boxes_out


def aug_scale_crop(
    img: np.ndarray,
    boxes: list,
    scale_range: tuple = (0.8, 1.2),
    out_size: int = 640,
    rng: Optional[random.Random] = None,
) -> tuple:
    """
    随机缩放并裁剪/填充到 out_size×out_size。

    缩放 > 1：图像放大后随机裁剪
    缩放 < 1：图像缩小后四周随机填充
    """
    if rng is None:
        rng = random.Random()

    scale = rng.uniform(*scale_range)
    h, w = img.shape[:2]

    new_w = int(w * scale)
    new_h = int(h * scale)

    img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 确定裁剪/填充区域（相对于 out_size × out_size 的画布）
    if scale >= 1.0:
        # 放大：随机裁剪回 out_size
        max_x = max(0, new_w - out_size)
        max_y = max(0, new_h - out_size)
        off_x = rng.randint(0, max_x)
        off_y = rng.randint(0, max_y)
        img_out = img_scaled[off_y:off_y + out_size, off_x:off_x + out_size]
        # 更新坐标：减去偏移量
        ratio_x = new_w / out_size  # 放大后坐标比例（相对于原始归一化）
        ratio_y = new_h / out_size
        boxes_out = []
        for box in boxes:
            cls_id, cx, cy, bw, bh = box
            # 转为放大后图像的绝对像素坐标
            cx_px = cx * w * scale
            cy_px = cy * h * scale
            bw_px = bw * w * scale
            bh_px = bh * h * scale
            # 减去裁剪偏移
            new_cx_px = cx_px - off_x
            new_cy_px = cy_px - off_y
            # 归一化到 out_size
            new_box = xyxy_to_yolo(
                new_cx_px - bw_px / 2, new_cy_px - bh_px / 2,
                new_cx_px + bw_px / 2, new_cy_px + bh_px / 2,
                out_size, out_size, int(cls_id)
            )
            boxes_out.append(new_box)
        # 若 img_out 不足 out_size（边缘），填充
        if img_out.shape[0] < out_size or img_out.shape[1] < out_size:
            pad_h = out_size - img_out.shape[0]
            pad_w = out_size - img_out.shape[1]
            img_out = cv2.copyMakeBorder(
                img_out, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
    else:
        # 缩小：四周随机填充到 out_size
        pad_total_x = out_size - new_w
        pad_total_y = out_size - new_h
        pad_left = rng.randint(0, pad_total_x)
        pad_top  = rng.randint(0, pad_total_y)
        pad_right  = pad_total_x - pad_left
        pad_bottom = pad_total_y - pad_top

        img_out = cv2.copyMakeBorder(
            img_scaled, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101
        )

        boxes_out = []
        for box in boxes:
            cls_id, cx, cy, bw, bh = box
            # 缩放后坐标
            new_cx_px = cx * w * scale + pad_left
            new_cy_px = cy * h * scale + pad_top
            new_bw_px = bw * w * scale
            new_bh_px = bh * h * scale
            new_box = xyxy_to_yolo(
                new_cx_px - new_bw_px / 2, new_cy_px - new_bh_px / 2,
                new_cx_px + new_bw_px / 2, new_cy_px + new_bh_px / 2,
                out_size, out_size, int(cls_id)
            )
            boxes_out.append(new_box)

    return img_out, clip_boxes(boxes_out)


def aug_hsv(
    img: np.ndarray,
    hue: float = 0.015,
    sat: float = 0.7,
    val: float = 0.4,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """
    HSV 色彩空间随机抖动（标注框不受影响）。

    参数：
        hue: 色相抖动范围（相对于 H 通道 [0,179]）
        sat: 饱和度抖动范围（乘法因子范围）
        val: 亮度抖动范围（乘法因子范围）
    """
    if rng is None:
        rng = random.Random()

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    dh = rng.uniform(-hue, hue) * 179
    ds = rng.uniform(1 - sat, 1 + sat)
    dv = rng.uniform(1 - val, 1 + val)

    img_hsv[..., 0] = (img_hsv[..., 0] + dh) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * ds, 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * dv, 0, 255)

    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_gaussian_noise(img: np.ndarray, std: float = 15.0, rng=None) -> np.ndarray:
    """添加高斯噪声（标注框不受影响）。"""
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    img_out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img_out


def aug_blur(
    img: np.ndarray,
    kernel_range: tuple = (3, 7),
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """
    随机模糊（均值模糊或高斯模糊，随机选一种）。

    参数：
        kernel_range: 核大小范围（取奇数）
    """
    if rng is None:
        rng = random.Random()

    # 随机选奇数核大小
    ksize = rng.choice([k for k in range(kernel_range[0], kernel_range[1] + 1, 2)])

    if rng.random() < 0.5:
        return cv2.blur(img, (ksize, ksize))
    else:
        return cv2.GaussianBlur(img, (ksize, ksize), 0)


def aug_mosaic(
    imgs: list[np.ndarray],
    boxes_list: list[list],
    out_size: int = 640,
    rng: Optional[random.Random] = None,
) -> tuple:
    """
    Mosaic 拼接：将 4 张图像随机拼接为一张，同步变换标注框。

    参数：
        imgs:       4 张图像（numpy 数组列表，形状可不同）
        boxes_list: 对应的 4 个标注框列表（YOLO 格式）
        out_size:   输出图像尺寸（正方形）
        rng:        随机数生成器

    返回值：
        (mosaic_img, mosaic_boxes)
    """
    if rng is None:
        rng = random.Random()

    assert len(imgs) == 4 and len(boxes_list) == 4

    # 随机中心点（在 [out_size*0.3, out_size*0.7] 范围内，避免某个区域过小）
    cx = rng.randint(int(out_size * 0.3), int(out_size * 0.7))
    cy = rng.randint(int(out_size * 0.3), int(out_size * 0.7))

    mosaic_img = np.full((out_size, out_size, 3), 114, dtype=np.uint8)  # 灰色背景
    mosaic_boxes = []

    # 4个子区域：左上、右上、左下、右下
    regions = [
        (0,  0,  cx,       cy,       0,  0),   # 左上：放置 imgs[0] 的右下部分
        (cx, 0,  out_size, cy,       cx, 0),   # 右上：放置 imgs[1] 的左下部分
        (0,  cy, cx,       out_size, 0,  cy),  # 左下：放置 imgs[2] 的右上部分
        (cx, cy, out_size, out_size, cx, cy),  # 右下：放置 imgs[3] 的左上部分
    ]

    for idx, (x1r, y1r, x2r, y2r, x_off, y_off) in enumerate(regions):
        region_w = x2r - x1r
        region_h = y2r - y1r

        img = imgs[idx]
        orig_h, orig_w = img.shape[:2]

        # 将子图缩放至恰好覆盖该区域
        img_scaled = cv2.resize(img, (region_w, region_h), interpolation=cv2.INTER_LINEAR)
        mosaic_img[y1r:y2r, x1r:x2r] = img_scaled

        # 变换标注框到 mosaic 坐标系
        scale_x = region_w / orig_w
        scale_y = region_h / orig_h

        for box in boxes_list[idx]:
            cls_id, cx_rel, cy_rel, bw_rel, bh_rel = box
            # 子图内像素坐标
            cx_sub = cx_rel * region_w
            cy_sub = cy_rel * region_h
            bw_sub = bw_rel * region_w
            bh_sub = bh_rel * region_h
            # mosaic 全图像素坐标
            cx_mos = cx_sub + x1r
            cy_mos = cy_sub + y1r
            # 归一化
            new_box = xyxy_to_yolo(
                cx_mos - bw_sub / 2, cy_mos - bh_sub / 2,
                cx_mos + bw_sub / 2, cy_mos + bh_sub / 2,
                out_size, out_size, int(cls_id)
            )
            mosaic_boxes.append(new_box)

    return mosaic_img, clip_boxes(mosaic_boxes)


# =============================================================================
# 主增强流程
# =============================================================================

def augment_dataset(
    in_img_dir: str,
    in_lbl_dir: str,
    out_img_dir: str,
    out_lbl_dir: str,
    aug_cfg: dict,
    out_size: int = 640,
    seed: int = 42,
    dry_run: bool = False,
    logger=None,
) -> dict:
    """
    批量执行数据增强，保存增强图像和标注。

    参数：
        in_img_dir:  输入图像目录
        in_lbl_dir:  输入标注目录
        out_img_dir: 输出图像目录（增强结果，与原图同目录时使用 _aug{N} 后缀）
        out_lbl_dir: 输出标注目录
        aug_cfg:     增强配置字典（来自 yaml 的 augmentation 节）
        out_size:    输出图像尺寸
        seed:        随机种子
        dry_run:     True 时仅统计，不保存文件
        logger:      日志对象

    返回值：
        统计信息字典
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    rng = random.Random(seed)
    np.random.seed(seed)

    img_exts = {".jpg", ".jpeg", ".png"}
    img_files = sorted([
        f for f in os.listdir(in_img_dir)
        if Path(f).suffix.lower() in img_exts
    ])

    if not img_files:
        logger.error(f"未找到图像：{in_img_dir}")
        return {}

    logger.info(f"开始增强：{len(img_files)} 张图像，每张生成 {aug_cfg['augment_factor']} 张增强图")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    total_generated = 0
    augment_factor = aug_cfg["augment_factor"]
    mosaic_prob = aug_cfg.get("mosaic_prob", 0.5)

    # 预加载所有图像（Mosaic 需要随机取 4 张）
    all_imgs   = []
    all_boxes  = []
    all_stems  = []
    for img_name in img_files:
        stem = Path(img_name).stem
        img  = cv2.imread(os.path.join(in_img_dir, img_name))
        if img is None:
            continue
        boxes = load_yolo_labels(os.path.join(in_lbl_dir, stem + ".txt"))
        # 统一 resize 到 out_size（若尺寸不一致）
        if img.shape[0] != out_size or img.shape[1] != out_size:
            img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        all_imgs.append(img)
        all_boxes.append(boxes)
        all_stems.append(stem)

    n = len(all_imgs)

    for i in range(n):
        img   = all_imgs[i]
        boxes = all_boxes[i]
        stem  = all_stems[i]

        for aug_idx in range(augment_factor):
            aug_img   = img.copy()
            aug_boxes = [b[:] for b in boxes]

            # ── 决定是否使用 Mosaic（4图合一）────────────────────────────────
            use_mosaic = aug_cfg.get("mosaic", True) and rng.random() < mosaic_prob and n >= 4

            if use_mosaic:
                # 随机选取另外 3 张图
                idxs = [i] + rng.sample([j for j in range(n) if j != i], 3)
                m_imgs  = [all_imgs[j]  for j in idxs]
                m_boxes = [all_boxes[j] for j in idxs]
                aug_img, aug_boxes = aug_mosaic(m_imgs, m_boxes, out_size, rng)
            else:
                # ── 随机旋转 ──────────────────────────────────────────────────
                if aug_cfg.get("rotate_90", True):
                    k = rng.choice([0, 1, 2, 3])
                    if k > 0:
                        aug_img, aug_boxes = aug_rotate90(aug_img, aug_boxes, k)

                # ── 水平翻转 ──────────────────────────────────────────────────
                if aug_cfg.get("flip_horizontal", True) and rng.random() < 0.5:
                    aug_img, aug_boxes = aug_flip_horizontal(aug_img, aug_boxes)

                # ── 垂直翻转 ──────────────────────────────────────────────────
                if aug_cfg.get("flip_vertical", True) and rng.random() < 0.5:
                    aug_img, aug_boxes = aug_flip_vertical(aug_img, aug_boxes)

                # ── 随机缩放裁剪 ──────────────────────────────────────────────
                scale_range = aug_cfg.get("scale_range", [0.8, 1.2])
                if rng.random() < 0.5:
                    aug_img, aug_boxes = aug_scale_crop(
                        aug_img, aug_boxes, tuple(scale_range), out_size, rng
                    )

            # ── HSV 色彩抖动 ──────────────────────────────────────────────────
            aug_img = aug_hsv(
                aug_img,
                hue=aug_cfg.get("hsv_hue", 0.015),
                sat=aug_cfg.get("hsv_saturation", 0.7),
                val=aug_cfg.get("hsv_value", 0.4),
                rng=rng,
            )

            # ── 高斯噪声 ──────────────────────────────────────────────────────
            if aug_cfg.get("gaussian_noise", True) and rng.random() < 0.5:
                aug_img = aug_gaussian_noise(aug_img, std=aug_cfg.get("gaussian_noise_std", 15))

            # ── 随机模糊 ──────────────────────────────────────────────────────
            if aug_cfg.get("random_blur", True) and rng.random() < 0.3:
                kr = aug_cfg.get("blur_kernel_range", [3, 7])
                aug_img = aug_blur(aug_img, tuple(kr), rng)

            # ── 过滤无效框 ────────────────────────────────────────────────────
            aug_boxes = clip_boxes(aug_boxes)

            # ── 保存 ──────────────────────────────────────────────────────────
            if not dry_run:
                out_name = f"{stem}_aug{aug_idx:03d}"
                out_img_path = os.path.join(out_img_dir, out_name + ".jpg")
                out_lbl_path = os.path.join(out_lbl_dir, out_name + ".txt")
                cv2.imwrite(out_img_path, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                save_yolo_labels(aug_boxes, out_lbl_path)

            total_generated += 1

        if (i + 1) % 10 == 0 or (i + 1) == n:
            logger.info(f"  [{i+1:>3d}/{n}] {all_stems[i]} 增强完成，"
                        f"当前共生成 {total_generated} 张")

    logger.info(f"增强完成：原始 {n} 张 → 增强 {total_generated} 张")
    return {
        "original_count": n,
        "augmented_count": total_generated,
        "total_count": n + total_generated,
    }


def main():
    parser = argparse.ArgumentParser(description="数据增强脚本")
    parser.add_argument("--dataset", type=str, choices=["satellite", "uav"],
                        required=True, help="数据集类型")
    parser.add_argument("--config", type=str,
                        help="配置文件路径（不指定则自动从 --dataset 推断）")
    parser.add_argument("--input_img",  type=str, help="输入图像目录（覆盖配置文件）")
    parser.add_argument("--input_lbl",  type=str, help="输入标注目录（覆盖配置文件）")
    parser.add_argument("--output_img", type=str, help="输出图像目录（覆盖配置文件）")
    parser.add_argument("--output_lbl", type=str, help="输出标注目录（覆盖配置文件）")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅统计，不保存文件")
    args = parser.parse_args()

    # 自动推断配置文件
    if args.config is None:
        args.config = f"configs/{args.dataset}_config.yaml"

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger(f"augment_{args.dataset}", log_dir="logs")
    logger.info("=" * 60)
    logger.info(f"数据增强开始：{args.dataset}")

    output_dir = cfg["data"]["output_dir"]

    # 默认对 train 集执行增强
    in_img_dir  = args.input_img  or os.path.join(output_dir, "images", "train")
    in_lbl_dir  = args.input_lbl  or os.path.join(output_dir, "labels", "train")
    out_img_dir = args.output_img or in_img_dir   # 默认保存到同一目录
    out_lbl_dir = args.output_lbl or in_lbl_dir

    stats = augment_dataset(
        in_img_dir=in_img_dir,
        in_lbl_dir=in_lbl_dir,
        out_img_dir=out_img_dir,
        out_lbl_dir=out_lbl_dir,
        aug_cfg=cfg["augmentation"],
        out_size=cfg["preprocess"]["target_size"],
        seed=cfg["split"]["seed"],
        dry_run=args.dry_run,
        logger=logger,
    )

    logger.info(f"结果统计：{stats}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
