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

from src.utils.plot_utils import setup_plot_style, save_fig, PALETTE, MODEL_COLORS, HATCHES, MARKERS
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
    绘制各模型指标对比柱状图（卫星/无人机分组）。
    论文版式以指标为子图、数据集为组，减少重复标题并提升双栏排版可读性。
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
        return

    all_models = []
    for dataset in datasets:
        for model_name in results[dataset]:
            if model_name not in all_models:
                all_models.append(model_name)

    dataset_labels = {"satellite": "Satellite", "uav": "UAV"}
    x = np.arange(len(datasets))
    width = 0.16 if len(all_models) >= 4 else 0.22

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey=True)
    axes = axes.reshape(-1)

    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        for idx, model_name in enumerate(all_models):
            values = [results[dataset].get(model_name, {}).get(metric_key, 0) for dataset in datasets]
            offset = (idx - (len(all_models) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=model_name,
                color=MODEL_COLORS.get(model_name, PALETTE["gray"]),
                edgecolor="#222222",
                linewidth=0.6,
                hatch=HATCHES[idx % len(HATCHES)],
                alpha=0.92,
            )
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.018,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=90 if len(all_models) > 3 else 0,
                    )

        ax.set_title(metric_label, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([dataset_labels.get(d, d.upper()) for d in datasets])
        ax.set_ylim(0, 1.06)

    for ax in axes[::2]:
        ax.set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(all_models), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(output_dir, "comparison_bar.png")
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"[plot_comparison] 对比柱状图已保存：{out_path}")


def plot_speed_accuracy(results: dict, output_dir: str) -> None:
    """
    绘制速度 vs 精度散点图（气泡大小 = 参数量）。
    采用同一坐标系比较全部数据集，marker 形状区分数据集，颜色区分模型。
    """
    setup_plot_style()

    datasets = sorted(results.keys())
    if not datasets:
        return

    dataset_labels = {"satellite": "Satellite", "uav": "UAV"}
    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    all_fps = []
    for dataset in datasets:
        for model_name, metrics in results[dataset].items():
            fps = metrics.get("fps", 0)
            map50 = metrics.get("map50", 0)
            params = metrics.get("params_M", 1)
            all_fps.append(fps)
            size = 35 + max(params, 0.5) * 28
            ax.scatter(
                fps,
                map50,
                s=size,
                color=MODEL_COLORS.get(model_name, PALETTE["gray"]),
                marker=MARKERS[datasets.index(dataset) % len(MARKERS)],
                edgecolors="#222222",
                linewidth=0.6,
                alpha=0.9,
                zorder=5,
            )
            ax.annotate(
                f"{model_name}-{dataset_labels.get(dataset, dataset).split()[0]}",
                (fps, map50),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=6.8,
            )

    ax.set_xlabel("Inference speed (FPS)")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0, 1.03)
    if all_fps:
        ax.set_xlim(left=max(0, min(all_fps) * 0.85), right=max(all_fps) * 1.15)

    # 单独构造简洁图例，避免散点数量过多导致 legend 拥挤。
    model_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS.get(m, PALETTE["gray"]),
                   markeredgecolor="#222222", markersize=6, label=m)
        for m in sorted({m for d in datasets for m in results[d]})
    ]
    dataset_handles = [
        plt.Line2D([0], [0], marker=MARKERS[i % len(MARKERS)], color="#222222", linestyle="none",
                   markerfacecolor="white", markersize=6, label=dataset_labels.get(d, d.upper()))
        for i, d in enumerate(datasets)
    ]
    first_legend = ax.legend(handles=model_handles, loc="lower right", frameon=False, title="Model", title_fontsize=8)
    ax.add_artist(first_legend)
    ax.legend(handles=dataset_handles, loc="lower left", frameon=False, title="Dataset", title_fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "speed_accuracy.png")
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"[plot_comparison] 速度-精度散点图已保存：{out_path}")


def plot_radar_chart(results: dict, output_dir: str) -> None:
    """
    绘制综合性能雷达图（4个维度：精度/召回/速度/轻量化）。
    保留雷达图用于综合展示，但压缩留白并弱化填充，避免论文中显得花哨。
    """
    setup_plot_style()

    datasets = sorted(results.keys())
    if not datasets:
        return

    dimensions = ["mAP@0.5", "Recall", "Speed", "Compact"]
    n_dim = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]
    dataset_labels = {"satellite": "Satellite", "uav": "UAV"}

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(3.4 * len(datasets), 3.4),
        subplot_kw=dict(polar=True),
    )
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        model_data = results[dataset]
        if not model_data:
            continue

        all_fps = [m.get("fps", 0) for m in model_data.values()]
        all_params = [m.get("params_M", 1) for m in model_data.values()]
        max_fps = max(all_fps) if max(all_fps) > 0 else 1
        max_params = max(all_params) if max(all_params) > 0 else 1

        for idx, (model_name, metrics) in enumerate(model_data.items()):
            map50 = metrics.get("map50", 0)
            recall = metrics.get("recall", 0)
            fps_norm = metrics.get("fps", 0) / max_fps
            compact = 1 - metrics.get("params_M", 0) / max_params
            values = [map50, recall, fps_norm, compact]
            values += values[:1]

            color = MODEL_COLORS.get(model_name, PALETTE["gray"])
            ax.plot(angles, values, color=color, linewidth=1.4, label=model_name)
            ax.fill(angles, values, color=color, alpha=0.06)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7)
        ax.set_title(dataset_labels.get(dataset, dataset.upper()), pad=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=7)

    fig.tight_layout()
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
