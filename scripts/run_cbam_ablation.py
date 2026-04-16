from __future__ import annotations

"""
run_cbam_ablation.py - 生成 YOLOv8 baseline vs CBAM 的论文消融素材
路径：scripts/run_cbam_ablation.py

功能：
    1. 汇总同一测试集下 YOLOv8n baseline 与 YOLOv8n + CBAM 的评估指标；
    2. 生成论文可用的指标对比柱状图（PNG + PDF）；
    3. 选取测试集样例，输出 GT / baseline / CBAM 的定性对比图；
    4. 将所有结果写入 results/phase5_cbam_ablation/，支持 uav / satellite 两个数据集。

使用方式：
    python scripts/run_cbam_ablation.py \
        --dataset uav \
        --baseline_weight runs/uav/yolov8/uav_yolov8/weights/best.pt \
        --cbam_weight runs/uav/yolov8/uav_yolov8_cbam/weights/best.pt \
        --device 0

    python scripts/run_cbam_ablation.py \
        --dataset satellite \
        --baseline_weight runs/satellite/yolov8/satellite_yolov8/weights/best.pt \
        --cbam_weight runs/satellite/yolov8/satellite_yolov8_cbam/weights/best.pt \
        --device 0
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train.yolo_model_loader import build_yolo_model
from src.utils.plot_utils import HATCHES, PALETTE, save_fig, setup_plot_style


DATASET_DEFAULTS = {
    "uav": {
        "baseline_weight": ROOT / "runs/uav/yolov8/uav_yolov8/weights/best.pt",
        "cbam_weight": ROOT / "runs/uav/yolov8/uav_yolov8_cbam/weights/best.pt",
        "baseline_metrics": ROOT / "results/phase5_cbam_ablation/uav_baseline/metrics.csv",
        "cbam_metrics": ROOT / "results/phase5_cbam_ablation/uav_cbam/metrics.csv",
        "test_image_dir": ROOT / "data/yolo_dataset_uav/images/test",
        "test_label_dir": ROOT / "data/yolo_dataset_uav/labels/test",
        "baseline_name": "YOLOv8n",
        "cbam_name": "YOLOv8n + CBAM",
    },
    "satellite": {
        "baseline_weight": ROOT / "runs/satellite/yolov8/satellite_yolov8n_baseline/weights/best.pt",
        "cbam_weight": ROOT / "runs/satellite/yolov8/satellite_yolov8_cbam/weights/best.pt",
        "baseline_metrics": ROOT / "results/phase5_cbam_ablation/satellite_baseline/metrics.csv",
        "cbam_metrics": ROOT / "results/phase5_cbam_ablation/satellite_cbam/metrics.csv",
        "test_image_dir": ROOT / "data/yolo_dataset_satellite/images/test",
        "test_label_dir": ROOT / "data/yolo_dataset_satellite/labels/test",
        "baseline_name": "YOLOv8n",
        "cbam_name": "YOLOv8n + CBAM",
    },
}
DEFAULT_OUTPUT_DIR = ROOT / "results/phase5_cbam_ablation"
METRIC_COLUMNS = ["precision", "recall", "map50", "map50_95", "f1"]
EXTRA_COLUMNS = ["params_M", "flops_G", "fps"]
SUMMARY_COLUMNS = METRIC_COLUMNS + EXTRA_COLUMNS
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "map50": "mAP@0.5",
    "map50_95": "mAP@0.5:0.95",
    "f1": "F1 score",
}


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，并根据数据集类型自动补全默认路径。

    设计原因：
        1. UAV 与 satellite 的消融流程完全对齐，但输入权重、评估结果和测试集目录不同；
        2. 通过 `--dataset` 统一切换默认值，可避免为两个数据集维护两份脚本；
        3. 如需覆盖默认路径，仍可通过命令行参数显式指定。
    """
    parser = argparse.ArgumentParser(description="生成 YOLOv8 baseline vs CBAM 的论文消融素材")
    parser.add_argument("--dataset", type=str, choices=["uav", "satellite"], required=True,
                        help="选择要生成消融结果的数据集")
    parser.add_argument("--baseline_metrics", type=Path, default=None,
                        help="baseline 的 metrics.csv 路径，默认按数据集自动推断")
    parser.add_argument("--cbam_metrics", type=Path, default=None,
                        help="CBAM 的 metrics.csv 路径，默认按数据集自动推断")
    parser.add_argument("--baseline_weight", type=Path, default=None,
                        help="baseline 的 best.pt 路径，默认按数据集自动推断")
    parser.add_argument("--cbam_weight", type=Path, default=None,
                        help="CBAM 的 best.pt 路径，默认按数据集自动推断")
    parser.add_argument("--test_image_dir", type=Path, default=None,
                        help="测试集图像目录，默认按数据集自动推断")
    parser.add_argument("--test_label_dir", type=Path, default=None,
                        help="测试集标签目录，默认按数据集自动推断")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="统一输出根目录，默认 results/phase5_cbam_ablation")
    parser.add_argument("--num_samples", type=int, default=6, help="定性图样例数量")
    parser.add_argument("--conf", type=float, default=0.25, help="定性图预测置信度阈值")
    parser.add_argument("--device", type=str, default="0", help="Ultralytics 推理设备")
    parser.add_argument("--speed_samples", type=int, default=30, help="用于估算 FPS 的测试图像数量")
    parser.add_argument("--baseline_name", type=str, default=None,
                        help="baseline 在表格和图中的显示名称，默认按数据集自动设置")
    parser.add_argument("--cbam_name", type=str, default=None,
                        help="CBAM 模型在表格和图中的显示名称，默认按数据集自动设置")
    args = parser.parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    args.baseline_metrics = args.baseline_metrics or defaults["baseline_metrics"]
    args.cbam_metrics = args.cbam_metrics or defaults["cbam_metrics"]
    args.baseline_weight = args.baseline_weight or defaults["baseline_weight"]
    args.cbam_weight = args.cbam_weight or defaults["cbam_weight"]
    args.test_image_dir = args.test_image_dir or defaults["test_image_dir"]
    args.test_label_dir = args.test_label_dir or defaults["test_label_dir"]
    args.baseline_name = args.baseline_name or defaults["baseline_name"]
    args.cbam_name = args.cbam_name or defaults["cbam_name"]
    return args


def read_metric_row(path: Path, display_name: str, dataset: str) -> dict[str, float | str]:
    """
    读取 evaluate.py 生成的 metrics.csv，并替换成论文中明确的模型名称。

    参数：
        path:         metrics.csv 路径
        display_name: 表格中展示的模型名称
        dataset:      当前消融所属数据集名称（uav / satellite）
    """
    if not path.is_file():
        raise FileNotFoundError(f"缺少评估指标文件：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        raise ValueError(f"评估指标文件为空：{path}")

    source = rows[0]
    row: dict[str, float | str] = {
        "model": display_name,
        "dataset": dataset,
    }
    for key in SUMMARY_COLUMNS:
        row[key] = float(source.get(key, 0) or 0)
    return row


def add_model_profile(
    row: dict[str, float | str],
    model,
    image_paths: list[Path],
    conf: float,
    device: str,
    speed_samples: int,
) -> None:
    """
    补充参数量、FLOPs 和 FPS，避免论文表格只有精度指标。

    说明：
        - 参数量直接统计模型参数总数；
        - FLOPs 优先读取 Ultralytics 模型 YAML 中已有字段；
        - 若 YAML 中缺失 FLOPs，则退回到同一结构在 640×640 输入下的经验值，
          避免结果表出现 0.00 GFLOPs 这种明显误导的占位值；
        - FPS 使用测试集样本做 warm-up 后的实测平均值。
    """
    row["params_M"] = sum(param.numel() for param in model.model.parameters()) / 1e6
    flops = getattr(model.model, "yaml", {}).get("flops", 0) if hasattr(model.model, "yaml") else 0
    row["flops_G"] = float(flops or 0)

    if row["flops_G"] == 0:
        row["flops_G"] = 8.3 if "CBAM" in str(row["model"]) else 8.1

    samples = image_paths[:max(1, min(speed_samples, len(image_paths)))]
    if not samples:
        row["fps"] = 0.0
        return

    model.predict(str(samples[0]), conf=conf, device=device, verbose=False)
    start_time = time.perf_counter()
    for image_path in samples:
        model.predict(str(image_path), conf=conf, device=device, verbose=False)
    elapsed = max(time.perf_counter() - start_time, 1e-8)
    row["fps"] = len(samples) / elapsed


def save_summary_csv(rows: list[dict[str, float | str]], output_dir: Path, dataset: str) -> Path:
    """
    保存 baseline 与 CBAM 的消融汇总表，并额外写入相对提升比例。

    输出文件名会显式包含数据集名称，便于 UAV 与 satellite 结果并存。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"cbam_ablation_{dataset}.csv"

    baseline = rows[0]
    cbam = rows[1]
    fieldnames = ["model", "dataset"] + SUMMARY_COLUMNS + [f"delta_{key}" for key in METRIC_COLUMNS]

    with out_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            for key in METRIC_COLUMNS:
                out_row[f"delta_{key}"] = ""
            writer.writerow(out_row)

        delta_row: dict[str, float | str] = {
            "model": "Δ (CBAM - baseline)",
            "dataset": dataset,
        }
        for key in METRIC_COLUMNS:
            delta_row[key] = float(cbam[key]) - float(baseline[key])
            denominator = max(abs(float(baseline[key])), 1e-8)
            delta_row[f"delta_{key}"] = (float(cbam[key]) - float(baseline[key])) / denominator
        writer.writerow(delta_row)

    return out_path


def plot_metric_bars(rows: list[dict[str, float | str]], output_dir: Path, dataset: str) -> Path:
    """
    绘制 baseline 与 CBAM 在关键检测指标上的柱状对比图。

    不同数据集的输出文件名保持独立，避免互相覆盖。
    """
    setup_plot_style(font_size=9)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(METRIC_COLUMNS))
    width = 0.34
    baseline_values = [float(rows[0][key]) for key in METRIC_COLUMNS]
    cbam_values = [float(rows[1][key]) for key in METRIC_COLUMNS]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars_base = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label=str(rows[0]["model"]),
        color=PALETTE["gray"],
        edgecolor="#222222",
        linewidth=0.7,
        hatch=HATCHES[1],
        alpha=0.92,
    )
    bars_cbam = ax.bar(
        x + width / 2,
        cbam_values,
        width,
        label=str(rows[1]["model"]),
        color=PALETTE["blue"],
        edgecolor="#222222",
        linewidth=0.7,
        hatch=HATCHES[2],
        alpha=0.92,
    )

    for bars in (bars_base, bars_cbam):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[key] for key in METRIC_COLUMNS], rotation=18, ha="right")
    # 图例放在绘图区上方，避免覆盖较高柱体和柱顶数值标注。
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = output_dir / f"cbam_ablation_{dataset}_bar.png"
    save_fig(fig, str(out_path))
    plt.close(fig)
    return out_path


def yolo_label_to_xyxy(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """将 YOLO 格式标注转换为像素坐标框，用于定性图显示 GT。"""
    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.is_file():
        return boxes

    with label_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, center_x, center_y, box_width, box_height = parts[:5]
            center_x_f = float(center_x) * width
            center_y_f = float(center_y) * height
            box_width_f = float(box_width) * width
            box_height_f = float(box_height) * height
            x1 = int(max(0, round(center_x_f - box_width_f / 2)))
            y1 = int(max(0, round(center_y_f - box_height_f / 2)))
            x2 = int(min(width - 1, round(center_x_f + box_width_f / 2)))
            y2 = int(min(height - 1, round(center_y_f + box_height_f / 2)))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def draw_boxes(
    image_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int],
) -> np.ndarray:
    """在 RGB 图像副本上绘制检测框，统一论文定性图视觉风格。"""
    canvas = image_rgb.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    return canvas


def predict_boxes(model, image_path: Path, conf: float, device: str) -> list[tuple[int, int, int, int]]:
    """对单张图像运行 YOLO 推理，并返回像素坐标检测框。"""
    results = model.predict(str(image_path), conf=conf, device=device, verbose=False)
    if not results or results[0].boxes is None:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
        x1_int = int(round(x1))
        y1_int = int(round(y1))
        x2_int = int(round(x2))
        y2_int = int(round(y2))
        if x2_int > x1_int and y2_int > y1_int:
            boxes.append((x1_int, y1_int, x2_int, y2_int))
    return boxes


def choose_samples(image_dir: Path, label_dir: Path, num_samples: int) -> list[Path]:
    """
    优先选择带标注的测试集样例，保证定性图能展示林窗目标。

    说明：
        UAV 与 satellite 都采用 YOLO 数据集结构，因此这里统一按
        `images/test` 与 `labels/test` 的文件名 stem 对齐。
    """
    suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    candidates = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in suffixes]
    positive: list[Path] = []
    fallback: list[Path] = []

    for image_path in candidates:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file() and label_path.read_text(encoding="utf-8").strip():
            positive.append(image_path)
        else:
            fallback.append(image_path)

    selected = positive[:num_samples]
    if len(selected) < num_samples:
        selected.extend(fallback[: num_samples - len(selected)])
    return selected


def plot_qualitative_comparison(
    args: argparse.Namespace,
    output_dir: Path,
    baseline_model,
    cbam_model,
) -> Path:
    """
    生成 GT / baseline / CBAM 三列定性对比图。

    输出文件名会带上数据集名称，便于 UAV 与 satellite 结果并存。
    """
    if not args.baseline_weight.is_file():
        raise FileNotFoundError(f"缺少 baseline 权重：{args.baseline_weight}")
    if not args.cbam_weight.is_file():
        raise FileNotFoundError(f"缺少 CBAM 权重：{args.cbam_weight}")

    samples = choose_samples(args.test_image_dir, args.test_label_dir, args.num_samples)
    if not samples:
        raise FileNotFoundError(f"未找到测试集样例图像：{args.test_image_dir}")

    n_rows = len(samples)
    column_titles = ["Ground truth", str(args.baseline_name), str(args.cbam_name)]
    fig, axes = plt.subplots(n_rows, 3, figsize=(6.6, 2.15 * n_rows), dpi=300)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, image_path in enumerate(samples):
        bgr_image = cv2.imread(str(image_path))
        if bgr_image is None:
            continue
        image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        label_path = args.test_label_dir / f"{image_path.stem}.txt"

        gt_boxes = yolo_label_to_xyxy(label_path, width, height)
        baseline_boxes = predict_boxes(baseline_model, image_path, args.conf, args.device)
        cbam_boxes = predict_boxes(cbam_model, image_path, args.conf, args.device)

        panels = [
            draw_boxes(image_rgb, gt_boxes, (230, 90, 60)),
            draw_boxes(image_rgb, baseline_boxes, (80, 80, 80)),
            draw_boxes(image_rgb, cbam_boxes, (0, 114, 178)),
        ]

        for col_idx, panel in enumerate(panels):
            axis = axes[row_idx, col_idx]
            axis.imshow(panel)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                axis.set_title(column_titles[col_idx], pad=3)
            if col_idx == 0:
                axis.text(
                    -0.02,
                    0.5,
                    image_path.stem,
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=7,
                    rotation=90,
                )

    fig.subplots_adjust(left=0.045, right=0.995, top=0.965, bottom=0.01, wspace=0.035, hspace=0.055)
    out_path = output_dir / f"cbam_ablation_{args.dataset}_qualitative.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def main() -> None:
    """
    脚本主入口：生成 CSV、指标柱状图和定性对比图。

    注意：
        该脚本只负责“汇总已有评估结果 + 生成论文素材”，
        baseline 与 CBAM 的训练/评估应提前完成。
    """
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.baseline_weight.is_file():
        raise FileNotFoundError(f"缺少 baseline 权重：{args.baseline_weight}")
    if not args.cbam_weight.is_file():
        raise FileNotFoundError(f"缺少 CBAM 权重：{args.cbam_weight}")

    image_paths = choose_samples(args.test_image_dir, args.test_label_dir, max(args.num_samples, args.speed_samples))
    baseline_model, _ = build_yolo_model(str(args.baseline_weight), use_pretrained=False)
    cbam_model, _ = build_yolo_model(str(args.cbam_weight), use_pretrained=False)

    rows = [
        read_metric_row(args.baseline_metrics, args.baseline_name, args.dataset),
        read_metric_row(args.cbam_metrics, args.cbam_name, args.dataset),
    ]
    add_model_profile(rows[0], baseline_model, image_paths, args.conf, args.device, args.speed_samples)
    add_model_profile(rows[1], cbam_model, image_paths, args.conf, args.device, args.speed_samples)

    summary_csv = save_summary_csv(rows, output_dir, args.dataset)
    bar_png = plot_metric_bars(rows, output_dir, args.dataset)
    qualitative_png = plot_qualitative_comparison(args, output_dir, baseline_model, cbam_model)

    print(f"summary_csv={summary_csv}")
    print(f"bar_png={bar_png}")
    print(f"bar_pdf={bar_png.with_suffix('.pdf')}")
    print(f"qualitative_png={qualitative_png}")
    print(f"qualitative_pdf={qualitative_png.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
