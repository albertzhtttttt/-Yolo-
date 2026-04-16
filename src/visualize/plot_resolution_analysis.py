"""
plot_resolution_analysis.py - 分辨率影响分析可视化脚本
路径：src/visualize/plot_resolution_analysis.py

功能：
    读取分辨率实验结果，生成：
    1. 各指标（mAP/Precision/Recall/F1）随分辨率变化折线图（卫星与无人机双线并排）
    2. 各分辨率 mAP@0.5 对比柱状图

说明：
    当前默认支持 6 个分辨率级别：100% / 50% / 25% / 12.5% / 6.25% / 3.125%，
    但若后续继续扩展新比例，脚本仍会自动读取并绘图。

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

from src.utils.plot_utils import setup_plot_style, save_fig, PALETTE, MARKERS
from src.utils.logger import get_logger

# 当前分辨率实验默认展示 6 个等级，统一按从低到高排列，
# 便于横向阅读并与更新后的实验设置保持一致。
DEFAULT_SCALE_ORDER = [3.125, 6.25, 12.5, 25.0, 50.0, 100.0]


def format_scale(scale: float) -> str:
    """将比例格式化为更适合坐标轴与图例展示的字符串。"""
    if float(scale).is_integer():
        return str(int(scale))
    return str(scale).rstrip("0").rstrip(".")


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
        scale = float(row["scale"])
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
    按数据集分别绘制各指标随分辨率变化的折线图。

    说明：
        论文结果展示中 UAV 与 satellite 不再放入同一张图，避免两个数据源在
        同一版面中相互干扰；每个数据集单独导出一张 2×2 指标图。
    """
    setup_plot_style()

    metrics_to_plot = [
        ("map50",     "mAP@0.5",      PALETTE["blue"]),
        ("precision", "Precision",    PALETTE["orange"]),
        ("recall",    "Recall",       PALETTE["green"]),
        ("f1",        "F1 score",     PALETTE["red"]),
    ]

    datasets = sorted(results.keys())
    if not datasets:
        print("[plot_resolution] 无数据，跳过绘图")
        return

    dataset_labels = {
        "satellite": "Satellite",
        "uav": "UAV",
    }

    for dataset in datasets:
        scale_data = results[dataset]
        # 每张图只使用当前数据集实际存在的分辨率比例，避免另一个数据集的缺失档位影响坐标轴。
        observed_scales = sorted(scale_data.keys())
        scale_order = [scale for scale in DEFAULT_SCALE_ORDER if scale in observed_scales]
        scale_order.extend(scale for scale in observed_scales if scale not in scale_order)

        fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.7), sharex=True, sharey=True)
        axes = axes.reshape(-1)

        x = np.arange(len(scale_order))
        for ax, (metric_key, metric_label, color) in zip(axes, metrics_to_plot):
            values = [scale_data[scale][metric_key] for scale in scale_order]
            ax.plot(
                x,
                values,
                color=color,
                marker=MARKERS[0],
                linewidth=1.6,
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.0,
            )
            ax.set_title(metric_label, pad=4)
            ax.set_xticks(x)
            ax.set_xticklabels([format_scale(scale) for scale in scale_order], rotation=20)
            ax.set_ylim(0, 1.02)

        for ax in axes[::2]:
            ax.set_ylabel("Score")
        for ax in axes[-2:]:
            ax.set_xlabel("Resolution scale (%)")

        fig.suptitle(dataset_labels.get(dataset, dataset.upper()), y=0.995, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(output_dir, f"resolution_analysis_{dataset}.png")
        save_fig(fig, out_path)
        plt.close(fig)
        print(f"[plot_resolution] 分辨率分析图已保存：{out_path}")


def plot_scale_comparison_bar(results: dict, output_dir: str) -> None:
    """
    按数据集分别绘制各分辨率 mAP@0.5 柱状图。

    说明：
        每张图只展示一个数据集，不再用图例区分 UAV / satellite，避免图例压到柱体
        或不同数据源混排造成阅读负担。
    """
    setup_plot_style()

    datasets = sorted(results.keys())
    if not datasets:
        return

    dataset_labels = {"satellite": "Satellite", "uav": "UAV"}
    dataset_colors = {"satellite": PALETTE["blue"], "uav": PALETTE["orange"]}

    for dataset in datasets:
        observed_scales = sorted(results[dataset].keys())
        all_scales = [scale for scale in DEFAULT_SCALE_ORDER if scale in observed_scales]
        all_scales.extend(scale for scale in observed_scales if scale not in all_scales)

        x = np.arange(len(all_scales))
        map50_values = [results[dataset][scale]["map50"] for scale in all_scales]

        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        bars = ax.bar(
            x,
            map50_values,
            0.48,
            color=dataset_colors.get(dataset, PALETTE["gray"]),
            edgecolor="#222222",
            linewidth=0.7,
            alpha=0.92,
        )
        for bar, val in zip(bars, map50_values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.018,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_title(dataset_labels.get(dataset, dataset.upper()), pad=4)
        ax.set_xlabel("Resolution scale (%)")
        ax.set_ylabel("mAP@0.5")
        ax.set_xticks(x)
        ax.set_xticklabels([format_scale(scale) for scale in all_scales], rotation=20)
        ax.set_ylim(0, 1.08)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"resolution_map50_bar_{dataset}.png")
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
        logger.info(f"  {dataset}: {[format_scale(scale) for scale in sorted(scale_data.keys(), reverse=True)]}")

    # 绘制折线图
    plot_metrics_by_scale(results, args.output_dir)

    # 绘制柱状图
    plot_scale_comparison_bar(results, args.output_dir)

    logger.info(f"分辨率分析可视化完成，输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
