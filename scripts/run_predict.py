"""
run_predict.py - 批量大图预测一键脚本
路径：scripts/run_predict.py

功能：
    读取 configs/predict_config.yaml，对所有6个TIF位置批量运行预测，
    汇总各位置检测框数量，生成统计图表。

使用方式：
    python scripts/run_predict.py \
        --dataset satellite \
        --weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt \
        [--mode auto] [--conf_thres 0.25] [--iou_thres 0.45]

    python scripts/run_predict.py \
        --dataset uav \
        --weights runs/uav/yolov8/uav_yolov8/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.plot_utils import setup_plot_style, save_fig, PALETTE, HATCHES
from src.predict.predict_geotiff import predict_geotiff


def parse_args():
    parser = argparse.ArgumentParser(description="批量大图预测脚本")
    parser.add_argument("--dataset",    type=str, required=True,
                        choices=["satellite", "uav", "all"])
    parser.add_argument("--weights",    type=str, default=None,
                        help="模型权重路径（--dataset all 时忽略，自动查找）")
    parser.add_argument("--sat_weights", type=str, default=None,
                        help="卫星模型权重（--dataset all 时使用）")
    parser.add_argument("--uav_weights", type=str, default=None,
                        help="无人机模型权重（--dataset all 时使用）")
    parser.add_argument("--config",     type=str, default="configs/predict_config.yaml")
    parser.add_argument("--mode",       type=str, default="auto",
                        choices=["auto", "geo", "tiles"])
    parser.add_argument("--conf_thres", type=float, default=None)
    parser.add_argument("--iou_thres",  type=float, default=None)
    parser.add_argument("--device",     type=str,   default=None)
    return parser.parse_args()


def find_best_weights(dataset: str) -> str:
    """自动查找最新训练的 best.pt 权重文件。"""
    search_dir = Path(f"runs/{dataset}/yolov8")
    if not search_dir.exists():
        return ""
    candidates = sorted(search_dir.glob("*/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else ""


def run_dataset_predict(
    dataset: str,
    weights: str,
    cfg: dict,
    logger,
    mode: str,
    conf_thres: float,
    iou_thres: float,
    device: str,
) -> list:
    """
    对单个数据集（satellite 或 uav）运行全部6个位置的预测。

    返回值：
        [{"name": str, "n_detections": int, "output_mode": str}, ...]
    """
    data_cfg = cfg["data"]
    output_root = data_cfg["output_root"]
    tiling = cfg["tiling"]
    inference = cfg["inference"]
    output_cfg = cfg["output"]

    # 参数优先级：命令行 > 配置文件
    conf = conf_thres if conf_thres is not None else inference["conf_thres"]
    iou  = iou_thres  if iou_thres  is not None else inference["iou_thres"]
    dev  = device     if device     is not None else inference["device"]

    results = []
    locations_map = data_cfg.get("locations", {})
    if isinstance(locations_map, dict):
        locations = locations_map.get(dataset, [])
    else:
        # 向后兼容旧版配置：locations 直接是列表时，沿用旧行为。
        locations = locations_map

    logger.info(f"=== {dataset.upper()} 模型预测开始，共 {len(locations)} 个位置 ===")
    logger.info(f"权重：{weights}")

    for loc in locations:
        name = loc["name"]
        tif_path = loc["tif"]
        tfw_path = loc.get("tfw")
        out_dir  = os.path.join(output_root, dataset, name)

        logger.info(f"--- 位置：{name} ---")

        if not os.path.isfile(tif_path):
            logger.warning(f"TIF 文件不存在，跳过：{tif_path}")
            results.append({"name": name, "n_detections": 0, "output_mode": "skipped"})
            continue

        result = predict_geotiff(
            model_path=weights,
            tif_path=tif_path,
            output_dir=out_dir,
            tfw_path=tfw_path,
            conf_thres=conf,
            iou_thres=iou,
            global_iou_thres=inference["global_iou_thres"],
            tile_size=tiling["tile_size"],
            stride=tiling["stride"],
            batch_size=inference["batch_size"],
            device=dev,
            mode=mode,
            save_annotated_tiles=output_cfg.get("save_annotated_tiles", True),
            logger=logger,
        )
        results.append({
            "name": name,
            "n_detections": result["n_detections"],
            "output_mode": result["output_mode"],
        })

    return results


def plot_summary(results: list, dataset: str, output_root: str) -> None:
    """
    绘制各位置检测框数量柱状图，保存到 output_root/{dataset}/summary.png。
    """
    setup_plot_style()

    names = [f"Site{i+1}" for i in range(len(results))]
    counts = [r["n_detections"] for r in results]

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bars = ax.bar(names, counts, color=PALETTE["blue"], edgecolor="#222222", linewidth=0.7, hatch=HATCHES[1], alpha=0.92)

    # 在柱顶标注数值，字号按论文图表控制，避免插入双栏后拥挤。
    y_offset = max(counts) * 0.012 if counts else 1
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            str(count),
            ha="center", va="bottom", fontsize=7,
        )

    ax.set_xlabel("Location")
    ax.set_ylabel("Number of detections")
    ax.set_ylim(0, max(counts) * 1.12 if counts else 10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    fig.tight_layout()

    out_path = os.path.join(output_root, dataset, "summary_bar.png")
    save_fig(fig, out_path)
    plt.close(fig)


def save_summary_csv(results: list, dataset: str, output_root: str) -> None:
    """保存汇总 CSV。"""
    out_path = os.path.join(output_root, dataset, "summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("location,n_detections,output_mode\n")
        for r in results:
            f.write(f"{r['name']},{r['n_detections']},{r['output_mode']}\n")
    print(f"[run_predict] 汇总 CSV 已保存：{out_path}")


def main():
    args = parse_args()
    logger = get_logger("run_predict", log_dir="logs")

    # 读取配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_root = cfg["data"]["output_root"]

    # 确定要运行的数据集列表
    if args.dataset == "all":
        datasets = ["satellite", "uav"]
        weights_map = {
            "satellite": args.sat_weights or find_best_weights("satellite"),
            "uav":       args.uav_weights or find_best_weights("uav"),
        }
    else:
        datasets = [args.dataset]
        weights_map = {
            args.dataset: args.weights or find_best_weights(args.dataset)
        }

    all_results = {}
    for dataset in datasets:
        weights = weights_map[dataset]
        if not weights or not os.path.isfile(weights):
            logger.error(f"找不到 {dataset} 模型权重：{weights}")
            logger.error("请通过 --weights 或 --sat_weights/--uav_weights 指定权重路径")
            continue

        results = run_dataset_predict(
            dataset=dataset,
            weights=weights,
            cfg=cfg,
            logger=logger,
            mode=args.mode,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            device=args.device,
        )
        all_results[dataset] = results

        # 保存汇总
        save_summary_csv(results, dataset, output_root)
        plot_summary(results, dataset, output_root)

        # 打印汇总
        logger.info(f"\n{'='*50}")
        logger.info(f"{dataset.upper()} 预测汇总：")
        total = 0
        for r in results:
            logger.info(f"  {r['name']:15s}  {r['n_detections']:5d} 个检测框  [{r['output_mode']}]")
            total += r["n_detections"]
        logger.info(f"  {'合计':15s}  {total:5d} 个检测框")
        logger.info(f"{'='*50}")

    logger.info("全部预测完成！")


if __name__ == "__main__":
    main()
