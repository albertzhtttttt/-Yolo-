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

from src.utils.plot_utils import setup_plot_style, save_fig, PALETTE, MODEL_COLORS, MARKERS
from src.utils.logger import get_logger

# 论文中的模型展示顺序固定为检测模型在前、分割模型在后，
# 这样柱状图、散点图和雷达图都能维持一致的阅读习惯。
MODEL_ORDER = ["YOLOv8", "YOLOv5", "U-Net", "FCN"]


def get_model_names(results: dict) -> list[str]:
    """按论文展示顺序返回当前结果中实际出现过的模型名称。"""
    observed = []
    for dataset_results in results.values():
        for model_name in dataset_results:
            if model_name not in observed:
                observed.append(model_name)

    ordered = [model_name for model_name in MODEL_ORDER if model_name in observed]
    ordered.extend(model_name for model_name in observed if model_name not in ordered)
    return ordered


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
    按数据集分别绘制各模型指标对比柱状图。

    说明：
        UAV 与 satellite 单独成图，柱状图不再用数据集分组，避免不同数据源混在
        同一张图中；模型图例放在图外顶部，避免遮挡柱体与数值标注。
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

    dataset_labels = {"satellite": "Satellite", "uav": "UAV"}

    for dataset in datasets:
        model_data = results[dataset]
        model_names = get_model_names({dataset: model_data})
        x = np.arange(len(model_names))

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharey=True)
        axes = axes.reshape(-1)

        for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
            values = [model_data.get(model_name, {}).get(metric_key, 0) for model_name in model_names]
            colors = [MODEL_COLORS.get(model_name, PALETTE["gray"]) for model_name in model_names]
            bars = ax.bar(
                x,
                values,
                0.56,
                color=colors,
                edgecolor="#222222",
                linewidth=0.6,
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
                    )

            ax.set_title(metric_label, pad=4)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=20, ha="right")
            ax.set_ylim(0, 1.08)

        for ax in axes[::2]:
            ax.set_ylabel("Score")

        legend_handles = [
            plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=MODEL_COLORS.get(model_name, PALETTE["gray"]),
                       markeredgecolor="#222222", markersize=6, label=model_name)
            for model_name in model_names
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=len(model_names),
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )
        fig.suptitle(dataset_labels.get(dataset, dataset.upper()), y=1.05, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        out_path = os.path.join(output_dir, f"comparison_bar_{dataset}.png")
        save_fig(fig, out_path)
        plt.close(fig)
        print(f"[plot_comparison] 对比柱状图已保存：{out_path}")


def plot_speed_accuracy(results: dict, output_dir: str) -> None:
    """
    按数据集分别绘制速度 vs 精度散点图（气泡大小 = 参数量）。

    说明：
        每张图只展示一个数据集，图例统一放在图外顶部，避免落在散点或标注上。
    """
    setup_plot_style()

    datasets = sorted(results.keys())
    if not datasets:
        return

    dataset_labels = {"satellite": "Satellite", "uav": "UAV"}

    for dataset in datasets:
        model_data = results[dataset]
        model_names = get_model_names({dataset: model_data})
        fig, ax = plt.subplots(figsize=(5.4, 3.5))

        all_fps = []
        for model_name in model_names:
            metrics = model_data.get(model_name)
            if not metrics:
                continue
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
                marker=MARKERS[0],
                edgecolors="#222222",
                linewidth=0.6,
                alpha=0.9,
                zorder=5,
            )
            ax.annotate(
                model_name,
                (fps, map50),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=6.8,
            )

        ax.set_title(dataset_labels.get(dataset, dataset.upper()), pad=4)
        ax.set_xlabel("Inference speed (FPS)")
        ax.set_ylabel("mAP@0.5")
        ax.set_ylim(0, 1.03)
        if all_fps:
            ax.set_xlim(left=max(0, min(all_fps) * 0.85), right=max(all_fps) * 1.18)

        model_handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS.get(model_name, PALETTE["gray"]),
                       markeredgecolor="#222222", markersize=6, label=model_name)
            for model_name in model_names
        ]
        fig.legend(
            handles=model_handles,
            loc="upper center",
            ncol=len(model_names),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out_path = os.path.join(output_dir, f"speed_accuracy_{dataset}.png")
        save_fig(fig, out_path)
        plt.close(fig)
        print(f"[plot_comparison] 速度-精度散点图已保存：{out_path}")


def plot_radar_chart(results: dict, output_dir: str) -> None:
    """
    按数据集分别绘制综合性能雷达图（4个维度：精度/召回/速度/轻量化）。

    说明：
        每个数据集单独导出，图例放在图外底部，避免遮挡雷达图主体。
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

    for dataset in datasets:
        model_data = results[dataset]
        if not model_data:
            continue

        fig, ax = plt.subplots(figsize=(3.8, 3.9), subplot_kw=dict(polar=True))
        all_fps = [model_metrics.get("fps", 0) for model_metrics in model_data.values()]
        all_params = [model_metrics.get("params_M", 1) for model_metrics in model_data.values()]
        max_fps = max(all_fps) if max(all_fps) > 0 else 1
        max_params = max(all_params) if max(all_params) > 0 else 1

        for model_name in get_model_names({dataset: model_data}):
            metrics = model_data.get(model_name)
            if not metrics:
                continue
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

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        out_path = os.path.join(output_dir, f"radar_chart_{dataset}.png")
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
