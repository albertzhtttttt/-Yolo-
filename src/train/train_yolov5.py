"""
train_yolov5.py - YOLOv5 训练脚本（对比实验）
路径：src/train/train_yolov5.py

功能：
    使用 YOLOv5 在相同数据集上训练，与 YOLOv8 进行对比。
    通过 ultralytics 包调用 YOLOv5（需安装 yolov5 包）。
    超参数与 YOLOv8 实验对齐。

使用方式：
    python src/train/train_yolov5.py \
        --dataset satellite \
        [--model yolov5n] [--epochs 150] [--device 0]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 训练脚本（对比实验）")
    parser.add_argument("--dataset",  type=str, required=True,
                        choices=["satellite", "uav"])
    parser.add_argument("--config",   type=str, default=None)
    parser.add_argument("--model",    type=str, default="yolov5n",
                        choices=["yolov5n", "yolov5s", "yolov5m", "yolov5l"])
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--batch",    type=int, default=None)
    parser.add_argument("--device",   type=str, default=None)
    parser.add_argument("--resume",   type=str, default=None)
    return parser.parse_args()


def _auto_select_gpu() -> str:
    """自动选择显存最空闲的 GPU（与 train_yolov8.py 一致）。"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            free_mem = [int(x.strip()) for x in result.stdout.strip().split("\n")]
            best_gpu = int(np.argmax(free_mem))
            return str(best_gpu)
    except Exception:
        pass
    return "0"


def main():
    args = parse_args()
    logger = get_logger(f"train_yolov5_{args.dataset}", log_dir="logs")

    import yaml
    import numpy as np

    config_path = args.config or f"configs/{args.dataset}_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    epochs  = args.epochs  or train_cfg["epochs"]
    batch   = args.batch   or train_cfg["batch"]
    device  = args.device  or _auto_select_gpu()
    dataset_yaml = cfg["data"]["dataset_yaml"]

    exp_name = f"{args.dataset}_{args.model}"
    save_dir = f"runs/{args.dataset}/yolov5"

    logger.info("=" * 60)
    logger.info(f"YOLOv5 训练启动：{args.dataset.upper()}")
    logger.info(f"模型：{args.model}，epochs={epochs}，batch={batch}，device={device}")

    # 尝试使用 ultralytics YOLOv5 接口
    try:
        from ultralytics import YOLO
        # ultralytics 支持 yolov5*.pt 格式
        model_pt = f"{args.model}.pt"
        model = YOLO(model_pt)

        # 关闭内置增强（与 YOLOv8 实验一致）
        augment_off = dict(
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            degrees=0.0, translate=0.0, scale=0.0,
            shear=0.0, perspective=0.0,
            flipud=0.0, fliplr=0.0,
            mosaic=0.0, mixup=0.0, copy_paste=0.0,
        )

        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=train_cfg["imgsz"],
            device=device,
            project=save_dir,
            name=exp_name,
            pretrained=True,
            optimizer=train_cfg["optimizer"],
            lr0=train_cfg["lr0"],
            lrf=train_cfg["lrf"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
            warmup_epochs=train_cfg["warmup_epochs"],
            patience=train_cfg["patience"],
            seed=train_cfg["seed"],
            workers=train_cfg["workers"],
            resume=args.resume or False,
            **augment_off,
        )

        best_pt = Path(save_dir) / exp_name / "weights" / "best.pt"
        logger.info(f"训练完成！权重：{best_pt}")

    except Exception as e:
        logger.error(f"YOLOv5 训练失败：{e}")
        logger.error("请确认已安装 ultralytics 并支持 YOLOv5 模型")
        sys.exit(1)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
