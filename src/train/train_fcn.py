"""
train_fcn.py - FCN 全卷积网络二值分割训练脚本
路径：src/train/train_fcn.py

功能：
    使用 FCN-8s（基于 VGG16 backbone）对林窗进行二值分割，
    后处理提取检测框用于与 YOLOv8 对比。
    损失函数：BCE + Dice Loss（与 U-Net 一致）

使用方式：
    python src/train/train_fcn.py \
        --dataset satellite \
        [--epochs 100] [--batch 8] [--lr 0.001] [--device 0]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.train.train_unet import bce_dice_loss, SegDataset, mask_to_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="FCN 训练脚本")
    parser.add_argument("--dataset",  type=str, required=True,
                        choices=["satellite", "uav"])
    parser.add_argument("--config",   type=str, default=None)
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--lr",       type=float, default=0.001)
    parser.add_argument("--device",   type=str, default="0")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--resume",   type=str, default=None)
    return parser.parse_args()


# =============================================================================
# FCN-8s 模型定义（基于 torchvision VGG16）
# =============================================================================

class FCN8s(object):
    """
    FCN-8s：使用 VGG16 作为 backbone，输出与输入同尺寸的分割图。
    输入：(B, 3, H, W)
    输出：(B, 1, H, W)，sigmoid 激活后为概率图
    """
    def __new__(cls, pretrained=True):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        try:
            from torchvision.models import vgg16, VGG16_Weights
            backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            from torchvision.models import vgg16
            backbone = vgg16(pretrained=pretrained)

        class _FCN8s(nn.Module):
            def __init__(self):
                super().__init__()
                features = backbone.features

                # VGG16 特征提取分段
                self.pool3 = features[:17]   # 输出 stride=8
                self.pool4 = features[17:24] # 输出 stride=16
                self.pool5 = features[24:]   # 输出 stride=32

                # 1×1 卷积将通道数降至1
                self.score_pool3 = nn.Conv2d(256, 1, 1)
                self.score_pool4 = nn.Conv2d(512, 1, 1)
                self.score_pool5 = nn.Conv2d(512, 1, 1)

                # 上采样层
                self.upscore2  = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
                self.upscore4  = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
                self.upscore8  = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)

                # 初始化上采样为双线性插值
                self._init_bilinear()

            def _init_bilinear(self):
                import torch
                for m in [self.upscore2, self.upscore4, self.upscore8]:
                    f = m.weight.data.shape[2]
                    c = (2 * f - 1 - f % 2) / (2.0 * f)
                    for i in range(f):
                        for j in range(f):
                            m.weight.data[0, 0, i, j] = (1 - abs(i / f - c)) * (1 - abs(j / f - c))

            def forward(self, x):
                h, w = x.shape[2], x.shape[3]

                p3 = self.pool3(x)
                p4 = self.pool4(p3)
                p5 = self.pool5(p4)

                s5 = self.score_pool5(p5)
                s5_up = self.upscore2(s5)

                s4 = self.score_pool4(p4)
                s4 = F.interpolate(s4, size=s5_up.shape[2:], mode="bilinear", align_corners=False)
                fuse4 = s4 + s5_up

                fuse4_up = self.upscore4(fuse4)
                s3 = self.score_pool3(p3)
                s3 = F.interpolate(s3, size=fuse4_up.shape[2:], mode="bilinear", align_corners=False)
                fuse3 = s3 + fuse4_up

                out = self.upscore8(fuse3)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
                return torch.sigmoid(out)

        return _FCN8s()


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

    model = FCN8s(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"从 epoch {start_epoch} 续训：{args.resume}")

    save_dir = f"runs/{args.dataset}/fcn/{args.dataset}_fcn"
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(start_epoch, args.epochs):
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
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(save_dir, "best.pt"))

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(save_dir, f"epoch_{epoch+1}.pt"))

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
    logger = get_logger(f"train_fcn_{args.dataset}", log_dir="logs")

    import yaml
    config_path = args.config or f"configs/{args.dataset}_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info(f"FCN 训练启动：{args.dataset.upper()}")

    # 检查掩码
    mask_dir = os.path.join(cfg["data"]["output_dir"], "masks", "train")
    if not os.path.isdir(mask_dir) or not os.listdir(mask_dir):
        logger.info("掩码目录不存在，先运行 yolo_to_mask.py...")
        import subprocess
        subprocess.run(
            [sys.executable, "src/data_preparation/yolo_to_mask.py",
             "--dataset", args.dataset],
        )

    train(args, cfg, logger)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
