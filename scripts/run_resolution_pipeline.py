"""
run_resolution_pipeline.py - 分辨率实验一键脚本
路径：scripts/run_resolution_pipeline.py

功能：
    1. 构建 6 个分辨率级别的数据集（100%/50%/25%/12.5%/6.25%/3.125%）
    2. 对每个分辨率级别依次训练 YOLOv8 模型（GPU 串行）
    3. 评估各模型，汇总指标，并覆盖写回 results/phase2_resolution 下的旧结果

使用方式：
    python scripts/run_resolution_pipeline.py --dataset satellite
    python scripts/run_resolution_pipeline.py --dataset uav
    python scripts/run_resolution_pipeline.py --dataset all
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
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
    parser.add_argument("--scales",   type=float, nargs="+",
                        default=[100, 50, 25, 12.5, 6.25, 3.125],
                        help="分辨率百分比列表，默认 100 50 25 12.5 6.25 3.125")
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


def format_scale(scale: float) -> str:
    """将分辨率比例格式化为稳定字符串，供目录名、实验名和日志统一复用。"""
    if math.isclose(scale, round(scale)):
        return str(int(round(scale)))
    return str(scale).rstrip("0").rstrip(".")


def format_model_tag(model_name: str) -> str:
    """将模型配置或权重名转换为可安全用于实验目录的短标识。"""
    model_path = Path(model_name)
    model_tag = model_path.stem if model_path.suffix else model_name
    return model_tag.replace("/", "_").replace("\\", "_").replace(" ", "_")


def remove_path_if_exists(path: Path, logger, desc: str) -> None:
    """在覆盖旧结果前清理目录，确保本次实验输出不会与历史文件混杂。"""
    if path.exists():
        logger.info(f"清理旧{desc}：{path}")
        shutil.rmtree(path)


def cleanup_obsolete_scale_results(results_root: Path, scales: list[float], logger) -> None:
    """删除本轮实验不再使用的 scale_* 旧结果目录，例如旧版 scale_75。"""
    if not results_root.exists():
        return
    expected_dirs = {f"scale_{format_scale(scale)}" for scale in scales}
    for child in results_root.iterdir():
        if child.is_dir() and child.name.startswith("scale_") and child.name not in expected_dirs:
            remove_path_if_exists(child, logger, "过期评估输出目录")


def find_latest_weights(save_dir: Path, exp_name: str) -> Path | None:
    """查找指定实验名前缀下最新的 best.pt，兼容 Ultralytics 自动追加数字后缀。"""
    candidates = sorted(
        save_dir.glob(f"{exp_name}*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


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


def run_resolution_experiment(dataset: str, scales: list[float], args, logger) -> dict:
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
    model_tag = format_model_tag(model_name)
    epochs = args.epochs or cfg["train"]["epochs"]
    output_root = f"data/resolution_datasets_{dataset}"
    save_dir = Path(f"runs/{dataset}/resolution")
    results_root = Path("results") / "phase2_resolution" / dataset
    cleanup_obsolete_scale_results(results_root, scales, logger)

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
        scale_name = format_scale(scale)
        scale_dir = os.path.join(output_root, f"scale_{scale_name}")
        dataset_yaml = os.path.join(scale_dir, "dataset.yaml")

        if not os.path.isfile(dataset_yaml):
            logger.warning(f"dataset.yaml 不存在，跳过 scale_{scale_name}：{dataset_yaml}")
            continue

        exp_name = f"{dataset}_scale{scale_name}_{model_tag}"

        logger.info(f"\n{'='*50}")
        logger.info(f"步骤2：训练 {dataset} scale_{scale_name}（{model_name}, {epochs} epochs）")

        if not args.eval_only:
            cmd = [sys.executable, "src/train/train_yolov8.py",
                   "--dataset", dataset,
                   "--model", model_name,
                   "--epochs", str(epochs),
                   "--data", dataset_yaml,
                   "--name", exp_name,
                   "--save_dir", str(save_dir)]
            if args.device:
                cmd += ["--device", args.device]
            ret = run_cmd(cmd, logger, f"训练 scale_{scale_name}")
            if ret != 0:
                logger.error(f"scale_{scale_name} 训练失败，跳过评估")
                continue

        # ── 步骤3：评估 ───────────────────────────────────────────────────────
        weights_path = find_latest_weights(save_dir, exp_name)
        if weights_path is None:
            logger.warning(f"找不到 best.pt，跳过 scale_{scale_name} 评估：{save_dir}/{exp_name}*/weights/best.pt")
            continue

        out_dir = results_root / f"scale_{scale_name}"
        remove_path_if_exists(out_dir, logger, "评估输出目录")
        logger.info(f"评估 scale_{scale_name}，权重：{weights_path}")

        ret = run_cmd(
            [sys.executable, "src/evaluate/evaluate.py",
             "--dataset", dataset,
             "--weights", str(weights_path),
             "--config", config_path,
             "--data", dataset_yaml,
             "--output_dir", str(out_dir)],
            logger, f"评估 scale_{scale_name}",
        )
        if ret != 0:
            logger.error(f"scale_{scale_name} 评估失败，跳过指标汇总")
            continue

        # 读取评估结果
        metrics_csv = out_dir / "metrics.csv"
        if metrics_csv.is_file():
            try:
                import pandas as pd
                df = pd.read_csv(metrics_csv)
                results[scale] = {
                    "scale_name": scale_name,
                    "map50":     float(df["map50"].iloc[0]),
                    "precision": float(df["precision"].iloc[0]),
                    "recall":    float(df["recall"].iloc[0]),
                    "f1":        float(df["f1"].iloc[0]),
                }
                logger.info(f"  scale_{scale_name} mAP@0.5 = {results[scale]['map50']:.4f}")
            except Exception as e:
                logger.warning(f"读取 metrics.csv 失败：{e}")

    return results


def load_existing_phase2_results(results_root: Path, scales: list[float], logger) -> dict:
    """
    从现有 results/phase2_resolution/{dataset}/scale_*/metrics.csv 回读指标。

    设计原因：
        satellite 与 uav 常常分开跑在不同 GPU 上，如果单次运行结束后直接覆盖
        根目录 metrics_by_scale.csv，就会把另一数据集已经完成的结果顶掉。
        因此这里在写总汇总前先回读已有结果，并且只并入“档位完整”的数据集，
        避免把旧版 75% 等历史实验结果混入新的 6 档实验汇总。

    参数：
        results_root: results/phase2_resolution 根目录
        scales:       当前实验要求的分辨率档位
        logger:       日志对象

    返回值：
        {dataset: {scale: {scale_name, map50, precision, recall, f1}}}
    """
    merged_results = {}
    expected_scales = {float(scale) for scale in scales}

    if not results_root.exists():
        return merged_results

    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name not in {"satellite", "uav"}:
            continue

        dataset_results = {}
        for scale_dir in sorted(dataset_dir.iterdir()):
            if not scale_dir.is_dir() or not scale_dir.name.startswith("scale_"):
                continue

            metrics_csv = scale_dir / "metrics.csv"
            if not metrics_csv.is_file():
                continue

            scale_name = scale_dir.name.replace("scale_", "", 1)
            try:
                scale_value = float(scale_name)
                with open(metrics_csv, "r", encoding="utf-8-sig", newline="") as f:
                    row = next(csv.DictReader(f), None)
                if row is None:
                    continue

                dataset_results[scale_value] = {
                    "scale_name": scale_name,
                    "map50": float(row["map50"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                }
            except Exception as e:
                logger.warning(f"读取历史汇总指标失败，已跳过：{metrics_csv} ({e})")

        if expected_scales.issubset(dataset_results.keys()):
            merged_results[dataset_dir.name] = {
                scale: dataset_results[scale]
                for scale in scales
                if scale in dataset_results
            }
        elif dataset_results:
            logger.info(
                f"检测到 {dataset_dir.name} 历史结果，但当前 6 档结果不完整，暂不并入根目录汇总"
            )

    return merged_results


def save_resolution_summary(all_results: dict, output_dir: str) -> None:
    """保存多分辨率实验汇总 CSV，并覆盖旧的汇总结果。"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "metrics_by_scale.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("dataset,scale,map50,precision,recall,f1\n")
        for dataset, scale_results in all_results.items():
            for scale in sorted(scale_results.keys(), reverse=True):
                metrics = scale_results[scale]
                f.write(f"{dataset},{metrics['scale_name']},"
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
    summary_results = load_existing_phase2_results(Path("results") / "phase2_resolution", args.scales, logger)
    expected_scales = {float(scale) for scale in args.scales}
    for dataset, scale_results in all_results.items():
        if scale_results:
            current_scales = {float(scale) for scale in scale_results.keys()}
            if expected_scales.issubset(current_scales):
                # 当前运行结果优先级更高；若本轮刚跑完某个数据集，应覆盖历史缓存结果。
                # 但只有当前 6 档完整时才写入根汇总，避免某一档评估失败后把已有完整结果覆盖成半成品。
                summary_results[dataset] = scale_results
            else:
                missing_scales = sorted(expected_scales - current_scales, reverse=True)
                logger.warning(
                    f"{dataset} 当前运行结果不完整，缺少档位 {missing_scales}，暂不覆盖根目录汇总"
                )

    if any(summary_results.values()):
        save_resolution_summary(summary_results, "results/phase2_resolution")
        plot_ret = run_cmd(
            [sys.executable, "src/visualize/plot_resolution_analysis.py",
             "--metrics_csv", "results/phase2_resolution/metrics_by_scale.csv",
             "--output_dir", "results/phase2_resolution"],
            logger, "更新分辨率分析可视化",
        )
        if plot_ret != 0:
            logger.warning("分辨率分析可视化更新失败，请稍后单独运行绘图脚本")

        # 打印汇总表
        logger.info("\n=== 分辨率实验汇总 ===")
        logger.info(f"{'数据集':12s} {'分辨率':10s} {'mAP@0.5':10s} {'Precision':10s} {'Recall':10s} {'F1':8s}")
        logger.info("-" * 66)
        for dataset, scale_results in summary_results.items():
            for scale in sorted(scale_results.keys(), reverse=True):
                m = scale_results[scale]
                logger.info(f"{dataset:12s} {m['scale_name']:>8s}%   "
                            f"{m['map50']:8.4f}   {m['precision']:8.4f}   "
                            f"{m['recall']:8.4f}   {m['f1']:6.4f}")

    logger.info("\n分辨率实验完成！")


if __name__ == "__main__":
    main()
