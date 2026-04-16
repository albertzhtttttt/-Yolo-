"""
run_train_pipeline.py - P2 训练全流程一键脚本
路径：scripts/run_train_pipeline.py

功能：
    按顺序执行：训练 → 评估 → 可视化
    支持卫星和无人机两个数据集，可分别或同时运行。

使用方式：
    # 训练并评估（默认 yolov8n，读配置文件参数）
    python scripts/run_train_pipeline.py --dataset satellite
    python scripts/run_train_pipeline.py --dataset uav
    python scripts/run_train_pipeline.py --dataset all

    # 指定模型规格和轮数
    python scripts/run_train_pipeline.py --dataset satellite --model yolov8s --epochs 200

    # 仅评估（跳过训练，指定已有权重）
    python scripts/run_train_pipeline.py --dataset satellite --eval_only \
        --weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger


def run_step(cmd: list, step_name: str, logger) -> bool:
    logger.info(f"▶ {step_name}")
    logger.info(f"  命令：{' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logger.error(f"✗ {step_name} 失败（返回码 {result.returncode}）")
        return False
    logger.info(f"✓ {step_name} 完成\n")
    return True


def find_best_weights(dataset: str, config_path: str | None = None) -> str | None:
    """
    自动查找最新实验的 best.pt 路径。

    参数：
        dataset:     数据集名称，仅在未显式传入配置文件时用于推断默认配置路径。
        config_path: 可选训练配置路径；用于 satellite CBAM 这类与 baseline 分离的专用配置。

    返回值：
        best.pt 路径字符串，或 None（未找到）
    """
    import yaml

    resolved_config_path = Path(config_path) if config_path else PROJECT_ROOT / f"configs/{dataset}_config.yaml"
    with open(resolved_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    save_dir = PROJECT_ROOT / cfg["train"]["save_dir"]
    exp_name = cfg["train"]["project_name"]

    if not save_dir.exists():
        return None

    # 找所有匹配的实验目录（按修改时间排序，取最新）
    exp_dirs = sorted(save_dir.glob(f"{exp_name}*"), key=os.path.getmtime, reverse=True)
    for exp_dir in exp_dirs:
        best_pt = exp_dir / "weights" / "best.pt"
        if best_pt.exists():
            return str(best_pt)
    return None


def run_pipeline(dataset: str, args, logger) -> bool:
    python = sys.executable

    logger.info("=" * 60)
    logger.info(f"开始训练流程：{dataset.upper()}")
    logger.info("=" * 60)

    weights = args.weights

    # ── Step 1：训练 ──────────────────────────────────────────────────────────
    if not args.eval_only:
        train_cmd = [
            python, "src/train/train_yolov8.py",
            "--dataset", dataset,
        ]
        if args.config:  train_cmd += ["--config",  args.config]
        if args.model:   train_cmd += ["--model",   args.model]
        if args.epochs:  train_cmd += ["--epochs",  str(args.epochs)]
        if args.batch:   train_cmd += ["--batch",   str(args.batch)]
        if args.lr0:     train_cmd += ["--lr0",     str(args.lr0)]
        if args.resume:  train_cmd += ["--resume",  args.resume]

        ok = run_step(train_cmd, f"[{dataset}] Step 1: YOLOv8 训练", logger)
        if not ok:
            return False

        # 自动查找刚训练好的 best.pt
        weights = find_best_weights(dataset, config_path=args.config)
        if weights is None:
            logger.error("未找到 best.pt，请手动指定 --weights 路径")
            return False
        logger.info(f"自动找到最优权重：{weights}")
    else:
        if weights is None:
            logger.error("--eval_only 模式需要指定 --weights 参数")
            return False

    # ── Step 2：评估 ──────────────────────────────────────────────────────────
    eval_cmd = [
        python, "src/evaluate/evaluate.py",
        "--dataset", dataset,
        "--weights", weights,
        "--conf",    str(args.conf),
        "--iou",     str(args.iou),
        "--vis_n",   str(args.vis_n),
    ]
    ok = run_step(eval_cmd, f"[{dataset}] Step 2: 模型评估与可视化", logger)
    if not ok:
        return False

    logger.info(f"✓✓ {dataset.upper()} 训练流程全部完成！\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="P2 训练全流程一键脚本")
    parser.add_argument("--dataset",   type=str, choices=["satellite", "uav", "all"],
                        default="all")
    # 训练参数（可覆盖配置文件）
    parser.add_argument("--config",    type=str,   help="训练配置文件路径，默认按数据集自动推断")
    parser.add_argument("--model",     type=str,   help="模型规格，如 yolov8n / yolov8s / yolov8m")
    parser.add_argument("--epochs",    type=int,   help="训练轮数")
    parser.add_argument("--batch",     type=int,   help="批大小")
    parser.add_argument("--lr0",       type=float, help="初始学习率")
    parser.add_argument("--resume",    type=str,   help="断点续训权重路径")
    # 评估参数
    parser.add_argument("--conf",      type=float, default=0.25)
    parser.add_argument("--iou",       type=float, default=0.45)
    parser.add_argument("--vis_n",     type=int,   default=20, help="可视化预测图数量")
    # 仅评估模式
    parser.add_argument("--eval_only", action="store_true", help="跳过训练，仅评估")
    parser.add_argument("--weights",   type=str,   help="--eval_only 时指定权重路径")
    args = parser.parse_args()

    logger = get_logger("run_train_pipeline", log_dir="logs")
    logger.info("=" * 60)
    logger.info("P2 训练流程启动")
    logger.info("=" * 60)

    datasets = ["satellite", "uav"] if args.dataset == "all" else [args.dataset]

    # 注意：两个数据集串行执行（单卡 P100 无法同时跑两个训练）
    all_ok = True
    for ds in datasets:
        ok = run_pipeline(ds, args, logger)
        if not ok:
            logger.error(f"❌ {ds} 流程出错")
            all_ok = False

    if all_ok:
        logger.info("=" * 60)
        logger.info("✓✓✓ 全部训练与评估完成！")
        logger.info("    结果目录：results/phase1_training/")
        logger.info("=" * 60)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
