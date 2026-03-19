"""
plot_comparison.py - 多模型对比可视化脚本
路径：src/visualize/plot_comparison.py

功能：
    读取 compare_models.py 生成的 metrics_summary.csv，生成：
    1. 各模型指标对比柱状图（卫星/无人机分两组）
    2. 速度 vs 精度散点图（气泡大小 = 参数量）
    3. 综合性能雷达图（精度/召回/速度/轻量化）

使用方式：
    python src/visualize/plot_comparison.py \
        --output_dir results/phase3_comparison
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

from src.utils.plot_utils import setup_plot_style, save_fig, PALETTE, MODEL_COLORS
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="多模型对比可视化")
    parser.add_argument("--output_dir", type=str,
                        default="results/phase3_comparison")
    return parser.parse_args()


def load_metrics(output_dir: str) -> dict:
    """
    读取所有 metrics_summary_*.csv 文件。

    返回值：
        {dataset: {model: {metric: value, ...}, ...}, ...}
    """
    results = {}
    for csv_file in Path(output_dir).glob("metrics_summary_*.csv"):
        dataset = csv_file.stem.replace("metrics_summary_", "")
        results[dataset] = {}
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header = lines[0].strip().split(",")
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            model = row["model"]
            results[dataset][model] = {
                "precision": float(row.get("precision", 0)),
                "recall":    float(row.get("recall",    0)),
                "map50":     float(row.get("map50",     0)),
                "f1":        float(row.get("f1",        0)),
                "fps":       float(row.get("fps",       0)),
                "params_M":  float(row.get("params_M",  0)),
            }
    return results


def plot_metrics_bar(results: dict, output_dir: str) -> None:
    """
    绘制各模型指标对比柱状图（卫星/无人机分两组）。
    """
    setup_plot_style()

    metrics_to_plot = [
        ("map50",     "mAP@0.5"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1",        "F1 Score"),
    ]

    datasets = sorted(results.keys())
    if not datasets:
        return

    # 获取所有模型名称
    all_models = []
    for d in datasets:
        for m in results[d]:
            if m not in all_models:
                all_models.append(m)

    model_color_list = [MODEL_COLORS.get(m, PALETTE["gray"]) for m in all_models]

    n_metrics = len(metrics_to_plot)
    n_datasets = len(datasets)

    fig, axes = plt.subplots(n_datasets, n_metrics,
                             figsize=(n_metrics * 4, n_datasets * 4))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")

    x = np.arange(len(all_models))
    width = 0.6

    for row_idx, dataset in enumerate(datasets):
        for col_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
            ax = axes[row_idx][col_idx]
            values = [results[dataset].get(m, {}).get(metric_key, 0) for m in all_models]

            bars = ax.bar(x, values, width, color=model_color_list,
                          edgecolor="white", linewidth=0.8)

            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(all_models, rotation=15, ha="right")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{dataset.upper()} - {metric_label}")
            ax.set_ylim(0, 1.15)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "comparison_bar.png")
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"[plot_comparison] 对比柱状图已保存：{out_path}")


def plot_speed_accuracy(results: dict, output_dir: str) -> None:
    """
    绘制速度 vs 精度散点图（气泡大小 = 参数量）。
    """
    setup_plot_style()

    datasets = sorted(results.keys())
    if not datasets:
        return

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(len(datasets) * 6, 5))
    if len(datasets) == 1:
        axes = [axes]

    fig.suptitle("Speed vs. Accuracy", fontsize=14, fontweight="bold")

    for ax, dataset in zip(axes, datasets):
        for model_name, m in results[dataset].items():
            fps    = m.get("fps", 0)
            map50  = m.get("map50", 0)
            params = m.get("params_M", 1)
            color  = MODEL_COLORS.get(model_name, PALETTE["gray"])

            # 气泡大小与参数量成正比
            size = max(100, params * 50)
            ax.scatter(fps, map50, s=size, color=color, alpha=0.8,
                       edgecolors="white", linewidth=1.5, label=model_name, zorder=5)
            ax.annotate(model_name, (fps, map50),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=9)

        ax.set_xlabel("FPS (frames/second)")
        ax.set_ylabel("mAP@0.5")
        ax.set_title(f"{dataset.upper()}")
        ax.set_ylim(0, 1.1)
        ax.axhline(0.70, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.text(ax.get_xlim()[0], 0.71, "Threshold 0.70",
                color="red", fontsize=8, alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "speed_accuracy.png")
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"[plot_comparison] 速度-精度散点图已保存：{out_path}")


def plot_radar_chart(results: dict, output_dir: str) -> None:
    """
    绘制综合性能雷达图（4个维度：精度/召回/速度/轻量化）。
    """
    setup_plot_style()

    datasets = sorted(results.keys())
    if not datasets:
        return

    # 雷达图维度
    dimensions = ["mAP@0.5", "Recall", "Speed\n(norm)", "Lightweight\n(norm)"]
    n_dim = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(len(datasets) * 5, 5),
                             subplot_kw=dict(polar=True))
    if len(datasets) == 1:
        axes = [axes]

    fig.suptitle("Comprehensive Performance Radar", fontsize=14, fontweight="bold")

    for ax, dataset in zip(axes, datasets):
        model_data = results[dataset]
        if not model_data:
            continue

        # 归一化 FPS 和参数量
        all_fps    = [m.get("fps", 0) for m in model_data.values()]
        all_params = [m.get("params_M", 1) for m in model_data.values()]
        max_fps    = max(all_fps) if max(all_fps) > 0 else 1
        max_params = max(all_params) if max(all_params) > 0 else 1

        for model_name, m in model_data.items():
            map50  = m.get("map50", 0)
            recall = m.get("recall", 0)
            fps_norm    = m.get("fps", 0) / max_fps
            # 轻量化：参数量越少越好，归一化后取反
            lightweight = 1 - m.get("params_M", 0) / max_params

            values = [map50, recall, fps_norm, lightweight]
            values += values[:1]  # 闭合

            color = MODEL_COLORS.get(model_name, PALETTE["gray"])
            ax.plot(angles, values, color=color, linewidth=2, label=model_name)
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"{dataset.upper()}", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "radar_chart.png")
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"[plot_comparison] 雷达图已保存：{out_path}")


def main():
    args = parse_args()
    logger = get_logger("plot_comparison", log_dir="logs")

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_metrics(args.output_dir)
    if not results:
        logger.error(f"未找到 metrics_summary_*.csv 文件：{args.output_dir}")
        logger.info("请先运行 src/evaluate/compare_models.py 生成评估结果")
        sys.exit(1)

    logger.info(f"读取到 {len(results)} 个数据集的对比结果")

    plot_metrics_bar(results, args.output_dir)
    plot_speed_accuracy(results, args.output_dir)
    plot_radar_chart(results, args.output_dir)

    logger.info(f"对比可视化完成，输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
