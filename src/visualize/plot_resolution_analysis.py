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

from src.utils.plot_utils import setup_plot_style, save_fig, PALETTE, LINE_STYLES, MARKERS, HATCHES
from src.utils.logger import get_logger

# 论文正文当前固定展示 4 个分辨率等级，统一按从低到高排列，
# 便于横向阅读并与论文表格中的 25/50/75/100 顺序保持一致。
DEFAULT_SCALE_ORDER = [25, 50, 75, 100]


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
    论文版式采用 2×2 子图，每个子图同时展示卫星与无人机，便于横向比较和单栏/双栏排版。
    """
    setup_plot_style()

    metrics_to_plot = [
        ("map50",     "mAP@0.5"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1",        "F1 score"),
    ]

    datasets = sorted(results.keys())
    if not datasets:
        print("[plot_resolution] 无数据，跳过绘图")
        return

    dataset_colors = {
        "satellite": PALETTE["blue"],
        "uav": PALETTE["orange"],
    }
    dataset_labels = {
        "satellite": "Satellite",
        "uav": "UAV",
    }
    # 优先沿用论文当前采用的 25→100 横向顺序；如果后续扩展了新比例，
    # 则自动把额外比例追加到末尾，避免脚本因新实验比例而失效。
    observed_scales = sorted(set(scale for scale_data in results.values() for scale in scale_data.keys()))
    scale_order = [scale for scale in DEFAULT_SCALE_ORDER if scale in observed_scales]
    scale_order.extend(scale for scale in observed_scales if scale not in scale_order)

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.9), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        for idx, dataset in enumerate(datasets):
            scale_data = results[dataset]
            scales = [scale for scale in scale_order if scale in scale_data]
            values = [scale_data[s][metric_key] for s in scales]
            ax.plot(
                scales,
                values,
                color=dataset_colors.get(dataset, PALETTE["gray"]),
                linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
                marker=MARKERS[idx % len(MARKERS)],
                linewidth=1.6,
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=dataset_labels.get(dataset, dataset.upper()),
            )

        ax.set_title(metric_label, pad=4)
        ax.set_xticks(scale_order)
        ax.set_xticklabels([str(scale) for scale in scale_order])
        ax.set_ylim(0, 1.02)

    for ax in axes[::2]:
        ax.set_ylabel("Score")
    for ax in axes[-2:]:
        ax.set_xlabel("Resolution scale (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.00))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(output_dir, "resolution_analysis.png")
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"[plot_resolution] 分辨率分析图已保存：{out_path}")


def plot_scale_comparison_bar(results: dict, output_dir: str) -> None:
    """
    绘制各分辨率 mAP@0.5 对比柱状图（卫星和无人机并排）。
    使用 hatch 纹理辅助区分，避免论文黑白打印时信息丢失。
    """
    setup_plot_style()

    datasets = sorted(results.keys())
    if not datasets:
        return

    # 柱状图与折线图保持同一分辨率顺序，避免论文排版时读者在两张图之间来回切换。
    observed_scales = sorted(set(s for d in datasets for s in results[d].keys()))
    all_scales = [scale for scale in DEFAULT_SCALE_ORDER if scale in observed_scales]
    all_scales.extend(scale for scale in observed_scales if scale not in all_scales)

    x = np.arange(len(all_scales))
    width = 0.32
    dataset_colors = [PALETTE["blue"], PALETTE["orange"]]
    dataset_labels = {"satellite": "Satellite", "uav": "UAV"}

    fig, ax = plt.subplots(figsize=(5.6, 3.2))

    for i, dataset in enumerate(datasets):
        map50_values = [results[dataset].get(s, {}).get("map50", 0) for s in all_scales]
        offset = (i - len(datasets) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            map50_values,
            width,
            label=dataset_labels.get(dataset, dataset.upper()),
            color=dataset_colors[i % len(dataset_colors)],
            edgecolor="#222222",
            linewidth=0.7,
            hatch=HATCHES[i % len(HATCHES)],
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

    ax.set_xlabel("Resolution scale (%)")
    ax.set_ylabel("mAP@0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in all_scales])
    ax.set_ylim(0, 1.06)
    ax.legend(loc="upper center", ncol=len(datasets), bbox_to_anchor=(0.5, 1.16), frameon=False)

    fig.tight_layout()
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
