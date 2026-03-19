"""
train_unet.py - U-Net 二值分割训练脚本
路径：src/train/train_unet.py

功能：
    使用 U-Net 对林窗进行二值分割，后处理提取检测框用于与 YOLOv8 对比。
    损失函数：BCE + Dice Loss
    后处理：连通域分析提取实例边界框

使用方式：
    python src/train/train_unet.py \
        --dataset satellite \
        [--epochs 100] [--batch 8] [--lr 0.001] [--device 0]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="U-Net 训练脚本")
    parser.add_argument("--dataset",  type=str, required=True,
                        choices=["satellite", "uav"])
    parser.add_argument("--config",   type=str, default=None)
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--lr",       type=float, default=0.001)
    parser.add_argument("--device",   type=str, default="0")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--resume",   type=str, default=None,
                        help="断点续训权重路径")
    return parser.parse_args()


# =============================================================================
# U-Net 模型定义
# =============================================================================

def double_conv(in_ch, out_ch):
    """U-Net 双卷积块：Conv→BN→ReLU→Conv→BN→ReLU"""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(object):
    """
    标准 U-Net（4层下采样）。
    输入：(B, 3, H, W)
    输出：(B, 1, H, W)，sigmoid 激活后为概率图
    """
    def __new__(cls, in_channels=3, out_channels=1, features=None):
        import torch
        import torch.nn as nn

        if features is None:
            features = [64, 128, 256, 512]

        class _UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.downs = nn.ModuleList()
                self.ups   = nn.ModuleList()
                self.pool  = nn.MaxPool2d(2, 2)

                # 下采样路径
                in_ch = in_channels
                for f in features:
                    self.downs.append(double_conv(in_ch, f))
                    in_ch = f

                # 瓶颈层
                self.bottleneck = double_conv(features[-1], features[-1] * 2)

                # 上采样路径
                for f in reversed(features):
                    self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
                    self.ups.append(double_conv(f * 2, f))

                # 输出层
                self.final = nn.Conv2d(features[0], out_channels, 1)

            def forward(self, x):
                import torch
                skip_connections = []
                for down in self.downs:
                    x = down(x)
                    skip_connections.append(x)
                    x = self.pool(x)

                x = self.bottleneck(x)
                skip_connections = skip_connections[::-1]

                for i in range(0, len(self.ups), 2):
                    x = self.ups[i](x)
                    skip = skip_connections[i // 2]
                    # 处理尺寸不匹配
                    if x.shape != skip.shape:
                        x = torch.nn.functional.interpolate(
                            x, size=skip.shape[2:], mode="bilinear", align_corners=False
                        )
                    x = torch.cat([skip, x], dim=1)
                    x = self.ups[i + 1](x)

                return torch.sigmoid(self.final(x))

        return _UNet()


# =============================================================================
# 损失函数：BCE + Dice
# =============================================================================

def bce_dice_loss(pred, target, bce_weight=0.5):
    """
    BCE + Dice 混合损失。

    参数：
        pred:   模型输出（sigmoid 后），shape=(B, 1, H, W)
        target: 二值掩码，shape=(B, 1, H, W)，值为 0 或 1
        bce_weight: BCE 损失权重（Dice 权重 = 1 - bce_weight）
    """
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy(pred, target)

    # Dice Loss
    smooth = 1e-5
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return bce_weight * bce + (1 - bce_weight) * dice


# =============================================================================
# 数据集
# =============================================================================

class SegDataset(object):
    """
    分割数据集：读取图像和对应的二值掩码。
    """
    def __new__(cls, img_dir, mask_dir, img_size=640, augment=False):
        import torch
        from torch.utils.data import Dataset

        class _SegDataset(Dataset):
            def __init__(self):
                self.img_dir   = img_dir
                self.mask_dir  = mask_dir
                self.img_size  = img_size
                self.augment   = augment

                img_exts = {".jpg", ".jpeg", ".png"}
                self.img_files = sorted([
                    f for f in os.listdir(img_dir)
                    if Path(f).suffix.lower() in img_exts
                ])

            def __len__(self):
                return len(self.img_files)

            def __getitem__(self, idx):
                img_name = self.img_files[idx]
                stem = Path(img_name).stem

                img = cv2.imread(os.path.join(self.img_dir, img_name))
                if img is None:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                mask_path = os.path.join(self.mask_dir, stem + ".png")
                if os.path.isfile(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                    mask = cv2.resize(mask, (self.img_size, self.img_size),
                                      interpolation=cv2.INTER_NEAREST)
                else:
                    mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

                # 简单数据增强（仅训练集）
                if self.augment:
                    if np.random.rand() > 0.5:
                        img  = cv2.flip(img, 1)
                        mask = cv2.flip(mask, 1)
                    if np.random.rand() > 0.5:
                        img  = cv2.flip(img, 0)
                        mask = cv2.flip(mask, 0)

                # 转换为 tensor
                img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

                return img_t, mask_t

        return _SegDataset()


# =============================================================================
# 后处理：连通域分析提取检测框
# =============================================================================

def mask_to_boxes(mask: np.ndarray, min_area: int = 100) -> list:
    """
    对二值掩码进行连通域分析，提取各实例的边界框。

    参数：
        mask:     二值掩码（0/255），shape=(H, W)
        min_area: 最小连通域面积（过滤噪声）

    返回值：
        [(x1, y1, x2, y2), ...]（像素坐标）
    """
    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    boxes = []
    for i in range(1, num_labels):  # 跳过背景（label=0）
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, x + w, y + h))

    return boxes


# =============================================================================
# 训练主函数
# =============================================================================

def train(args, cfg, logger):
    import torch
    from torch.utils.data import DataLoader

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备：{device}")

    dataset_dir = cfg["data"]["output_dir"]
    img_size    = args.img_size

    # 数据集
    train_ds = SegDataset(
        os.path.join(dataset_dir, "images", "train"),
        os.path.join(dataset_dir, "masks",  "train"),
        img_size, augment=True,
    )
    val_ds = SegDataset(
        os.path.join(dataset_dir, "images", "val"),
        os.path.join(dataset_dir, "masks",  "val"),
        img_size, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    logger.info(f"训练集：{len(train_ds)} 张，验证集：{len(val_ds)} 张")

    # 模型
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # 断点续训
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"从 epoch {start_epoch} 续训：{args.resume}")

    # 输出目录
    save_dir = f"runs/{args.dataset}/unet/{args.dataset}_unet"
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(start_epoch, args.epochs):
        # ── 训练 ──────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for imgs, masks in train_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss  = bce_dice_loss(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # ── 验证 ──────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)
                preds = model(imgs)
                loss  = bce_dice_loss(preds, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()

        logger.info(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.6f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(save_dir, "best.pt"))

        # 每10轮保存一次
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(save_dir, f"epoch_{epoch+1}.pt"))

    # 保存训练曲线数据
    import csv
    with open(os.path.join(save_dir, "train_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([i + 1, tl, vl])

    logger.info(f"训练完成！最优 val_loss={best_val_loss:.4f}")
    logger.info(f"权重保存至：{save_dir}/best.pt")
    return os.path.join(save_dir, "best.pt")


def main():
    args = parse_args()
    logger = get_logger(f"train_unet_{args.dataset}", log_dir="logs")

    import yaml
    config_path = args.config or f"configs/{args.dataset}_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info(f"U-Net 训练启动：{args.dataset.upper()}")
    logger.info(f"epochs={args.epochs}, batch={args.batch}, lr={args.lr}")

    # 检查掩码是否已生成
    mask_dir = os.path.join(cfg["data"]["output_dir"], "masks", "train")
    if not os.path.isdir(mask_dir) or not os.listdir(mask_dir):
        logger.info("掩码目录不存在，先运行 yolo_to_mask.py...")
        import subprocess
        ret = subprocess.run(
            [sys.executable, "src/data_preparation/yolo_to_mask.py",
             "--dataset", args.dataset],
            capture_output=False,
        )
        if ret.returncode != 0:
            logger.error("掩码生成失败，退出")
            sys.exit(1)

    train(args, cfg, logger)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
