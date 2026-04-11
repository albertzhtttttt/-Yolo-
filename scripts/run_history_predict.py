"""
run_history_predict.py - 历史时序卫星影像批量预测脚本
路径：scripts/run_history_predict.py

功能：
    1. 扫描 data/predict/history/ 下按日期组织的历史卫星影像目录
    2. 复用 predict_geotiff.py 对每个时间节点执行大图滑窗预测
    3. 汇总各年份检测框数量、影像尺寸与输出模式
    4. 生成时序汇总 CSV 与年度检测框数量柱状图

使用方式：
    python scripts/run_history_predict.py \
        --weights runs/satellite/yolov8/satellite_yolov86/weights/best.pt

    python scripts/run_history_predict.py \
        --weights runs/satellite/yolov8/satellite_yolov86/weights/best.pt \
        --years 2024-01-01 2025-01-01 2026-01-01 \
        --device 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict.predict_geotiff import predict_geotiff
from src.utils.logger import get_logger
from src.utils.plot_utils import PALETTE, save_fig, setup_plot_style


DEFAULT_HISTORY_ROOT = "data/predict/history"
DEFAULT_OUTPUT_ROOT = "results/phase4_temporal/predictions"
DEFAULT_SUMMARY_ROOT = "results/phase4_temporal/summaries"


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="历史时序卫星影像批量预测脚本")
    parser.add_argument("--weights", type=str, required=True, help="卫星模型权重路径")
    parser.add_argument("--history_root", type=str, default=DEFAULT_HISTORY_ROOT,
                        help="历史预测数据根目录")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT,
                        help="逐年预测输出根目录")
    parser.add_argument("--summary_root", type=str, default=DEFAULT_SUMMARY_ROOT,
                        help="汇总结果输出目录")
    parser.add_argument("--years", type=str, nargs="+", default=None,
                        help="仅预测指定年份目录，例如 2024-01-01 2025-01-01")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "geo", "tiles"],
                        help="输出模式，默认 auto")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="单切片 NMS IoU 阈值")
    parser.add_argument("--global_iou", type=float, default=0.5, help="全图 NMS IoU 阈值")
    parser.add_argument("--tile_size", type=int, default=640, help="切片尺寸")
    parser.add_argument("--stride", type=int, default=320, help="滑窗步长")
    parser.add_argument("--batch_size", type=int, default=8, help="批量推理切片数")
    parser.add_argument("--device", type=str, default="0", help="GPU 编号")
    return parser.parse_args()


def find_history_items(history_root: Path, selected_years: list[str] | None) -> list[dict]:
    """
    扫描历史目录，返回每个年份的影像与辅助文件路径。

    返回值：
        [
            {
                "year": "2021-01-01",
                "tif": Path(...),
                "tfw": Path(...),
                "prj": Path(...),
            },
            ...
        ]
    """
    items = []
    target_years = set(selected_years) if selected_years else None

    for year_dir in sorted([p for p in history_root.iterdir() if p.is_dir()]):
        if target_years is not None and year_dir.name not in target_years:
            continue

        level_dir = year_dir / "Level18"
        if not level_dir.is_dir():
            continue

        tif = next(level_dir.glob("*.tif"), None)
        tfw = next(level_dir.glob("*.tfw"), None)
        prj = next(level_dir.glob("*.prj"), None)

        if tif is None:
            continue

        items.append({
            "year": year_dir.name,
            "tif": tif,
            "tfw": tfw,
            "prj": prj,
        })

    return items


def read_summary_row(summary_csv: Path) -> dict:
    """读取单个年份预测目录下的 summary.csv。"""
    if not summary_csv.is_file():
        return {
            "location": "",
            "n_detections": 0,
            "output_mode": "missing",
            "tif_width": 0,
            "tif_height": 0,
            "n_tiles": 0,
        }

    with open(summary_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        return {
            "location": "",
            "n_detections": 0,
            "output_mode": "invalid",
            "tif_width": 0,
            "tif_height": 0,
            "n_tiles": 0,
        }

    values = lines[1].split(",")
    return {
        "location": values[0],
        "n_detections": int(values[1]),
        "output_mode": values[2],
        "tif_width": int(values[3]),
        "tif_height": int(values[4]),
        "n_tiles": int(values[5]),
    }


def save_temporal_summary(results: list[dict], summary_root: Path) -> Path:
    """保存逐年预测汇总 CSV。"""
    summary_root.mkdir(parents=True, exist_ok=True)
    out_path = summary_root / "temporal_summary.csv"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("year,n_detections,output_mode,tif_width,tif_height,n_tiles,output_dir\n")
        for item in results:
            f.write(
                f"{item['year']},{item['n_detections']},{item['output_mode']},"
                f"{item['tif_width']},{item['tif_height']},{item['n_tiles']},"
                f"{item['output_dir'].as_posix()}\n"
            )

    return out_path


def plot_temporal_counts(results: list[dict], summary_root: Path) -> Path:
    """绘制逐年检测框数量柱状图。"""
    summary_root.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    years = [item["year"] for item in results]
    counts = [item["n_detections"] for item in results]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(years, counts, color=PALETTE["teal"], edgecolor="white", linewidth=0.8)

    y_offset = max(counts) * 0.01 if counts else 1
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Detections")
    ax.set_title("Temporal Detection Count by Year")
    ax.set_ylim(0, max(counts) * 1.15 if counts else 10)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_path = summary_root / "temporal_detection_counts.png"
    save_fig(fig, str(out_path))
    plt.close(fig)
    return out_path


def main():
    """脚本主入口。"""
    args = parse_args()
    logger = get_logger("run_history_predict", log_dir="logs")

    history_root = Path(args.history_root)
    output_root = Path(args.output_root)
    summary_root = Path(args.summary_root)

    if not history_root.is_dir():
        logger.error(f"历史数据目录不存在：{history_root}")
        sys.exit(1)

    if not Path(args.weights).is_file():
        logger.error(f"模型权重不存在：{args.weights}")
        sys.exit(1)

    history_items = find_history_items(history_root, args.years)
    if not history_items:
        logger.error("未找到可预测的历史影像，请检查 history_root 或 years 参数")
        sys.exit(1)

    logger.info(f"共找到 {len(history_items)} 个历史时相待预测")
    logger.info(f"使用权重：{args.weights}")
    logger.info(
        f"统一参数：tile_size={args.tile_size}, stride={args.stride}, "
        f"conf={args.conf_thres}, iou={args.iou_thres}, global_iou={args.global_iou}, "
        f"batch={args.batch_size}, device={args.device}"
    )

    temporal_results = []

    for item in history_items:
        year = item["year"]
        tif_path = item["tif"]
        tfw_path = item["tfw"]
        output_dir = output_root / year

        logger.info(f"{'=' * 60}")
        logger.info(f"开始预测历史时相：{year}")
        logger.info(f"影像路径：{tif_path.as_posix()}")

        result = predict_geotiff(
            model_path=args.weights,
            tif_path=str(tif_path),
            output_dir=str(output_dir),
            tfw_path=str(tfw_path) if tfw_path else None,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            global_iou_thres=args.global_iou,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size,
            device=args.device,
            mode=args.mode,
            save_annotated_tiles=True,
            logger=logger,
        )

        summary_row = read_summary_row(output_dir / "summary.csv")
        temporal_results.append({
            "year": year,
            "n_detections": summary_row["n_detections"],
            "output_mode": summary_row["output_mode"],
            "tif_width": summary_row["tif_width"],
            "tif_height": summary_row["tif_height"],
            "n_tiles": summary_row["n_tiles"],
            "output_dir": output_dir,
            "result_mode": result["output_mode"],
        })

        logger.info(
            f"历史时相 {year} 预测完成："
            f"{summary_row['n_detections']} 个检测框，输出模式={summary_row['output_mode']}"
        )

    summary_csv = save_temporal_summary(temporal_results, summary_root)
    summary_png = plot_temporal_counts(temporal_results, summary_root)

    logger.info("=" * 60)
    logger.info("历史时序预测全部完成")
    logger.info(f"时序汇总表：{summary_csv.as_posix()}")
    logger.info(f"时序统计图：{summary_png.as_posix()}")

    total_detections = sum(item["n_detections"] for item in temporal_results)
    logger.info(f"6期合计检测框数：{total_detections}")
    for item in temporal_results:
        logger.info(
            f"  {item['year']:12s}  {item['n_detections']:5d} 个检测框  "
            f"[{item['output_mode']}]"
        )


if __name__ == "__main__":
    main()
