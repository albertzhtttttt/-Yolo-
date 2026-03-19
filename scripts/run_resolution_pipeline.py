"""
run_resolution_pipeline.py - 分辨率实验一键脚本
路径：scripts/run_resolution_pipeline.py

功能：
    1. 构建4个分辨率级别的数据集（100%/75%/50%/25%）
    2. 对每个分辨率级别依次训练 YOLOv8 模型（GPU串行）
    3. 评估各模型，汇总指标

使用方式：
    python scripts/run_resolution_pipeline.py --dataset satellite
    python scripts/run_resolution_pipeline.py --dataset uav
    python scripts/run_resolution_pipeline.py --dataset all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="分辨率实验一键脚本")
    parser.add_argument("--dataset",  type=str, required=True,
                        choices=["satellite", "uav", "all"])
    parser.add_argument("--scales",   type=int, nargs="+",
                        default=[100, 75, 50, 25])
    parser.add_argument("--model",    type=str, default=None,
                        help="YOLOv8 模型规格（默认从配置文件读取）")
    parser.add_argument("--epochs",   type=int, default=None,
                        help="训练轮数（默认从配置文件读取）")
    parser.add_argument("--skip_build", action="store_true",
                        help="跳过数据集构建（已构建时使用）")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅评估，不训练")
    parser.add_argument("--device",   type=str, default=None,
                        help="指定 GPU 编号（默认自动选择最空闲）")
    return parser.parse_args()


def run_cmd(cmd: list, logger, desc: str = "") -> int:
    """运行子进程命令，实时输出日志。"""
    logger.info(f"执行：{' '.join(cmd)}")
    if desc:
        logger.info(f"  ({desc})")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"  | {line}")
    proc.wait()
    return proc.returncode


def run_resolution_experiment(dataset: str, scales: list, args, logger) -> dict:
    """
    对单个数据集运行完整的分辨率实验。

    返回值：
        {scale: {"map50": float, "precision": float, "recall": float, "f1": float}, ...}
    """
    import yaml as _yaml
    config_path = f"configs/{dataset}_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = _yaml.safe_load(f)

    model_name = args.model or cfg["train"]["model"]
    epochs = args.epochs or cfg["train"]["epochs"]
    output_root = f"data/resolution_datasets_{dataset}"

    results = {}

    # ── 步骤1：构建多分辨率数据集 ─────────────────────────────────────────────
    if not args.skip_build and not args.eval_only:
        logger.info(f"\n{'='*50}")
        logger.info(f"步骤1：构建 {dataset} 多分辨率数据集")
        ret = run_cmd(
            [sys.executable, "src/data_preparation/build_resolution_datasets.py",
             "--dataset", dataset, "--scales"] + [str(s) for s in scales],
            logger, "构建多分辨率数据集",
        )
        if ret != 0:
            logger.error("数据集构建失败，退出")
            return results

    # ── 步骤2：逐分辨率训练 ───────────────────────────────────────────────────
    for scale in scales:
        scale_dir = os.path.join(output_root, f"scale_{scale}")
        dataset_yaml = os.path.join(scale_dir, "dataset.yaml")

        if not os.path.isfile(dataset_yaml):
            logger.warning(f"dataset.yaml 不存在，跳过 scale_{scale}：{dataset_yaml}")
            continue

        exp_name = f"{dataset}_scale{scale}_{model_name}"
        save_dir = f"runs/{dataset}/resolution"

        logger.info(f"\n{'='*50}")
        logger.info(f"步骤2：训练 {dataset} scale_{scale}（{model_name}, {epochs} epochs）")

        if not args.eval_only:
            cmd = [sys.executable, "src/train/train_yolov8.py",
                   "--dataset", dataset,
                   "--model", model_name,
                   "--epochs", str(epochs),
                   "--data", dataset_yaml,
                   "--name", exp_name,
                   "--save_dir", save_dir]
            if args.device:
                cmd += ["--device", args.device]
            ret = run_cmd(cmd, logger, f"训练 scale_{scale}")
            if ret != 0:
                logger.error(f"scale_{scale} 训练失败，跳过评估")
                continue

        # ── 步骤3：评估 ───────────────────────────────────────────────────────
        # 查找 best.pt
        weights_path = Path(save_dir) / exp_name / "weights" / "best.pt"
        if not weights_path.exists():
            # 尝试查找最新的实验目录
            candidates = sorted(
                Path(save_dir).glob(f"{exp_name}*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if candidates:
                weights_path = candidates[0]
            else:
                logger.warning(f"找不到 best.pt，跳过 scale_{scale} 评估")
                continue

        out_dir = f"results/phase2_resolution/{dataset}/scale_{scale}"
        logger.info(f"评估 scale_{scale}，权重：{weights_path}")

        ret = run_cmd(
            [sys.executable, "src/evaluate/evaluate.py",
             "--dataset", dataset,
             "--weights", str(weights_path),
             "--config", config_path],
            logger, f"评估 scale_{scale}",
        )

        # 读取评估结果
        metrics_csv = os.path.join(out_dir, "metrics.csv")
        if os.path.isfile(metrics_csv):
            try:
                import pandas as pd
                df = pd.read_csv(metrics_csv)
                results[scale] = {
                    "map50":     float(df["map50"].iloc[0]),
                    "precision": float(df["precision"].iloc[0]),
                    "recall":    float(df["recall"].iloc[0]),
                    "f1":        float(df["f1"].iloc[0]),
                }
                logger.info(f"  scale_{scale} mAP@0.5 = {results[scale]['map50']:.4f}")
            except Exception as e:
                logger.warning(f"读取 metrics.csv 失败：{e}")

    return results


def save_resolution_summary(all_results: dict, output_dir: str) -> None:
    """保存多分辨率实验汇总 CSV。"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "metrics_by_scale.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("dataset,scale,map50,precision,recall,f1\n")
        for dataset, scale_results in all_results.items():
            for scale, metrics in scale_results.items():
                f.write(f"{dataset},{scale},"
                        f"{metrics['map50']:.4f},{metrics['precision']:.4f},"
                        f"{metrics['recall']:.4f},{metrics['f1']:.4f}\n")
    print(f"[run_resolution_pipeline] 汇总 CSV 已保存：{out_path}")


def main():
    args = parse_args()
    logger = get_logger("run_resolution_pipeline", log_dir="logs")

    datasets = ["satellite", "uav"] if args.dataset == "all" else [args.dataset]
    all_results = {}

    for dataset in datasets:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# 数据集：{dataset.upper()}")
        logger.info(f"{'#'*60}")
        results = run_resolution_experiment(dataset, args.scales, args, logger)
        all_results[dataset] = results

    # 保存汇总
    if any(all_results.values()):
        save_resolution_summary(all_results, "results/phase2_resolution")

        # 打印汇总表
        logger.info("\n=== 分辨率实验汇总 ===")
        logger.info(f"{'数据集':12s} {'分辨率':8s} {'mAP@0.5':10s} {'Precision':10s} {'Recall':10s} {'F1':8s}")
        logger.info("-" * 60)
        for dataset, scale_results in all_results.items():
            for scale in sorted(scale_results.keys(), reverse=True):
                m = scale_results[scale]
                logger.info(f"{dataset:12s} {scale:>6}%   "
                            f"{m['map50']:8.4f}   {m['precision']:8.4f}   "
                            f"{m['recall']:8.4f}   {m['f1']:6.4f}")

    logger.info("\n分辨率实验完成！")


if __name__ == "__main__":
    main()
