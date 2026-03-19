"""
plot_resolution_analysis.py - 分辨率影响分析可视化脚本
路径：src/visualize/plot_resolution_analysis.py

功能：
    读取分辨率实验结果，生成：
    1. 各指标（mAP/Precision/Recall/F1）随分辨率变化折线图（卫星与无人机双线并排）
    2. 不同分辨率下同一区域检测结果对比图（4列横向拼图）

使用方式：
    python src/visualize/plot_resolution_analysis.py \
        --metrics_csv results/phase2_resolution/metrics_by_scale.csv \
        [--output_dir results/phase2_resolution]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot_utils import setup_plot_style, save_fig, PALETTE, SCALE_COLORS
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="分辨率影响分析可视化")
    parser.add_argument("--metrics_csv", type=str,
                        default="results/phase2_resolution/metrics_by_scale.csv")
    parser.add_argument("--output_dir",  type=str,
                        default="results/phase2_resolution")
    return parser.parse_args()


def load_metrics(csv_path: str) -> dict:
    """
    读取分辨率实验指标 CSV。

    返回值：
        {dataset: {scale: {metric: value, ...}, ...}, ...}
    """
    results = {}
    if not os.path.isfile(csv_path):
        print(f"[plot_resolution] 文件不存在：{csv_path}")
        return results

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0].strip().split(",")
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        dataset = row["dataset"]
        scale = int(row["scale"])
        if dataset not in results:
            results[dataset] = {}
        results[dataset][scale] = {
            "map50":     float(row["map50"]),
            "precision": float(row["precision"]),
            "recall":    float(row["recall"]),
            "f1":        float(row["f1"]),
        }
    return results


def plot_metrics_by_scale(results: dict, output_dir: str) -> None:
    """
    绘制各指标随分辨率变化的折线图。
    卫星和无人机数据集并排显示（2行×4列）。
    """
    setup_plot_style()

    metrics_to_plot = [
        ("map50",     "mAP@0.5",   PALETTE["blue"]),
        ("precision", "Precision", PALETTE["orange"]),
        ("recall",    "Recall",    PALETTE["green"]),
        ("f1",        "F1 Score",  PALETTE["purple"]),
    ]

    datasets = sorted(results.keys())
    n_datasets = len(datasets)
    n_metrics = len(metrics_to_plot)

    if n_datasets == 0:
        print("[plot_resolution] 无数据，跳过绘图")
        return

    fig, axes = plt.subplots(n_datasets, n_metrics,
                             figsize=(n_metrics * 4, n_datasets * 3.5))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle("Detection Performance vs. Image Resolution", fontsize=14, fontweight="bold")

    for row_idx, dataset in enumerate(datasets):
        scale_data = results[dataset]
        scales = sorted(scale_data.keys(), reverse=True)  # 100, 75, 50, 25
        x = [s for s in scales]

        for col_idx, (metric_key, metric_label, color) in enumerate(metrics_to_plot):
            ax = axes[row_idx][col_idx]
            y = [scale_data[s][metric_key] for s in scales]

            ax.plot(x, y, color=color, marker="o", linewidth=2, markersize=6)

            # 标注数值
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.3f}", (xi, yi),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8)

            ax.set_xlabel("Resolution (%)")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{dataset.upper()} - {metric_label}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{s}%" for s in x])
            ax.set_ylim(0, 1.1)
            ax.invert_xaxis()  # 从高分辨率到低分辨率

    plt.tight_layout()
    out_path = os.path.join(output_dir, "resolution_analysis.png")
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"[plot_resolution] 分辨率分析图已保存：{out_path}")


def plot_scale_comparison_bar(results: dict, output_dir: str) -> None:
    """
    绘制各分辨率 mAP@0.5 对比柱状图（卫星和无人机并排）。
    """
    setup_plot_style()

    datasets = sorted(results.keys())
    if not datasets:
        return

    # 获取所有分辨率级别
    all_scales = sorted(set(
        s for d in datasets for s in results[d].keys()
    ), reverse=True)

    x = np.arange(len(all_scales))
    width = 0.35
    dataset_colors = [PALETTE["blue"], PALETTE["orange"]]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, dataset in enumerate(datasets):
        map50_values = [results[dataset].get(s, {}).get("map50", 0) for s in all_scales]
        offset = (i - len(datasets) / 2 + 0.5) * width
        bars = ax.bar(x + offset, map50_values, width,
                      label=dataset.upper(), color=dataset_colors[i % len(dataset_colors)],
                      edgecolor="white", linewidth=0.8)
        # 标注数值
        for bar, val in zip(bars, map50_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Resolution (%)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("mAP@0.5 vs. Resolution")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}%" for s in all_scales])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.axhline(0.70, color="red", linestyle="--", alpha=0.5, label="Threshold (0.70)")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "resolution_map50_bar.png")
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"[plot_resolution] mAP@0.5 对比图已保存：{out_path}")


def main():
    args = parse_args()
    logger = get_logger("plot_resolution_analysis", log_dir="logs")

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取指标数据
    results = load_metrics(args.metrics_csv)
    if not results:
        logger.error(f"无法读取指标数据：{args.metrics_csv}")
        logger.info("请先运行 scripts/run_resolution_pipeline.py 完成训练和评估")
        sys.exit(1)

    logger.info(f"读取到 {len(results)} 个数据集的分辨率实验结果")
    for dataset, scale_data in results.items():
        logger.info(f"  {dataset}: {sorted(scale_data.keys(), reverse=True)}")

    # 绘制折线图
    plot_metrics_by_scale(results, args.output_dir)

    # 绘制柱状图
    plot_scale_comparison_bar(results, args.output_dir)

    logger.info(f"分辨率分析可视化完成，输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
