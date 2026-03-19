"""
compare_models.py - 多模型统一对比评估脚本
路径：src/evaluate/compare_models.py

功能：
    在相同测试集上评估所有模型（YOLOv8/YOLOv5/U-Net/FCN），
    计算并汇总：Precision / Recall / mAP@0.5 / F1 / FPS / Params / FLOPs

使用方式：
    python src/evaluate/compare_models.py \
        --dataset satellite \
        [--yolov8_weights runs/satellite/yolov8/.../best.pt] \
        [--yolov5_weights runs/satellite/yolov5/.../best.pt] \
        [--unet_weights   runs/satellite/unet/.../best.pt] \
        [--fcn_weights    runs/satellite/fcn/.../best.pt]

输出：
    results/phase3_comparison/
    ├── metrics_summary.csv
    └── （由 plot_comparison.py 生成图表）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.train.train_unet import UNet, mask_to_boxes
from src.train.train_fcn import FCN8s


def parse_args():
    parser = argparse.ArgumentParser(description="多模型对比评估")
    parser.add_argument("--dataset",       type=str, required=True,
                        choices=["satellite", "uav"])
    parser.add_argument("--config",        type=str, default=None)
    parser.add_argument("--yolov8_weights", type=str, default=None)
    parser.add_argument("--yolov5_weights", type=str, default=None)
    parser.add_argument("--unet_weights",   type=str, default=None)
    parser.add_argument("--fcn_weights",    type=str, default=None)
    parser.add_argument("--conf",          type=float, default=0.25)
    parser.add_argument("--iou",           type=float, default=0.45)
    parser.add_argument("--device",        type=str,   default="0")
    parser.add_argument("--output_dir",    type=str,
                        default="results/phase3_comparison")
    return parser.parse_args()


# =============================================================================
# IoU 计算工具
# =============================================================================

def compute_iou(box1, box2):
    """计算两个框的 IoU。box 格式：(x1, y1, x2, y2)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


def compute_detection_metrics(
    pred_boxes_list: list,
    gt_boxes_list: list,
    iou_thres: float = 0.5,
) -> dict:
    """
    计算检测指标（Precision / Recall / F1）。

    参数：
        pred_boxes_list: 每张图的预测框列表 [[(x1,y1,x2,y2,score), ...], ...]
        gt_boxes_list:   每张图的 GT 框列表 [[(x1,y1,x2,y2), ...], ...]
        iou_thres:       IoU 匹配阈值

    返回值：
        {"precision": float, "recall": float, "f1": float, "ap": float}
    """
    tp_total = 0
    fp_total = 0
    fn_total = 0

    all_scores = []
    all_tp_flags = []
    total_gt = sum(len(gt) for gt in gt_boxes_list)

    for pred_boxes, gt_boxes in zip(pred_boxes_list, gt_boxes_list):
        # 按置信度降序排列
        pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
        matched_gt = set()

        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred[:4], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thres and best_gt_idx >= 0:
                tp_total += 1
                matched_gt.add(best_gt_idx)
                all_tp_flags.append(1)
            else:
                fp_total += 1
                all_tp_flags.append(0)
            all_scores.append(pred[4])

        fn_total += len(gt_boxes) - len(matched_gt)

    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall    = tp_total / (total_gt + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # 近似 AP（按置信度排序的 PR 曲线面积）
    if all_scores:
        sorted_idx = np.argsort(all_scores)[::-1]
        tp_flags = np.array(all_tp_flags)[sorted_idx]
        cum_tp = np.cumsum(tp_flags)
        cum_fp = np.cumsum(1 - tp_flags)
        prec = cum_tp / (cum_tp + cum_fp + 1e-8)
        rec  = cum_tp / (total_gt + 1e-8)
        ap = float(np.trapz(prec[::-1], rec[::-1]))
    else:
        ap = 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "map50": ap}


# =============================================================================
# 模型推理接口
# =============================================================================

def eval_yolo_model(weights: str, dataset_yaml: str, conf: float, iou: float,
                    device: str, logger) -> dict:
    """评估 YOLO 模型（YOLOv8 或 YOLOv5）。"""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics 未安装")
        return {}

    model = YOLO(weights)
    results = model.val(
        data=dataset_yaml,
        split="test",
        conf=conf,
        iou=iou,
        verbose=False,
    )
    m = results.results_dict
    p  = float(m.get("metrics/precision(B)", 0))
    r  = float(m.get("metrics/recall(B)",    0))
    m50   = float(m.get("metrics/mAP50(B)",    0))
    m5095 = float(m.get("metrics/mAP50-95(B)", 0))
    f1 = 2 * p * r / (p + r + 1e-8)

    # 测速（推理 100 张）
    fps = measure_fps_yolo(model, device)

    # 参数量和 FLOPs
    params, flops = get_model_complexity_yolo(model)

    return {
        "precision": p, "recall": r, "map50": m50, "map50_95": m5095,
        "f1": f1, "fps": fps, "params_M": params, "flops_G": flops,
    }


def measure_fps_yolo(model, device: str, n_warmup: int = 10, n_test: int = 100) -> float:
    """测量 YOLO 模型推理速度（FPS）。"""
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(n_warmup):
        model.predict(dummy, device=device, verbose=False)
    t0 = time.time()
    for _ in range(n_test):
        model.predict(dummy, device=device, verbose=False)
    elapsed = time.time() - t0
    return n_test / elapsed


def get_model_complexity_yolo(model) -> tuple:
    """获取 YOLO 模型参数量（M）和 FLOPs（G）。"""
    try:
        info = model.info(verbose=False)
        # ultralytics 返回 (layers, params, gradients, flops)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            params_M = info[1] / 1e6
            flops_G  = info[3] / 1e9
            return round(params_M, 2), round(flops_G, 2)
    except Exception:
        pass
    return 0.0, 0.0


def eval_seg_model(
    model_class,
    weights: str,
    test_img_dir: str,
    test_lbl_dir: str,
    img_size: int,
    device_str: str,
    conf_thres: float,
    logger,
) -> dict:
    """
    评估分割模型（U-Net / FCN）。
    将分割掩码后处理为检测框，计算检测指标。
    """
    import torch

    device = torch.device(f"cuda:{device_str}" if torch.cuda.is_available() else "cpu")

    model = model_class().to(device)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img_exts = {".jpg", ".jpeg", ".png"}
    img_files = sorted([f for f in os.listdir(test_img_dir)
                        if Path(f).suffix.lower() in img_exts])

    pred_boxes_list = []
    gt_boxes_list   = []

    # 测速
    fps_times = []

    with torch.no_grad():
        for img_name in img_files:
            stem = Path(img_name).stem
            img_path = os.path.join(test_img_dir, img_name)
            lbl_path = os.path.join(test_lbl_dir, stem + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            img_resized = cv2.resize(img, (img_size, img_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(
                img_rgb.transpose(2, 0, 1)
            ).float().unsqueeze(0).to(device) / 255.0

            t0 = time.time()
            pred = model(img_t)
            fps_times.append(time.time() - t0)

            # 后处理：掩码 → 检测框
            mask = (pred[0, 0].cpu().numpy() > conf_thres).astype(np.uint8) * 255
            boxes_local = mask_to_boxes(mask, min_area=50)

            # 转换为原图坐标（带虚拟置信度=1.0）
            scale_x = w / img_size
            scale_y = h / img_size
            pred_boxes = [
                (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, 1.0)
                for x1, y1, x2, y2 in boxes_local
            ]
            pred_boxes_list.append(pred_boxes)

            # 读取 GT 框
            gt_boxes = []
            if os.path.isfile(lbl_path):
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, cx, cy, bw, bh = [float(x) for x in parts]
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w
                            y2 = (cy + bh / 2) * h
                            gt_boxes.append((x1, y1, x2, y2))
            gt_boxes_list.append(gt_boxes)

    metrics = compute_detection_metrics(pred_boxes_list, gt_boxes_list)
    fps = len(fps_times) / sum(fps_times) if fps_times else 0.0

    # 参数量
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    metrics.update({"fps": fps, "params_M": round(params_M, 2), "flops_G": 0.0, "map50_95": 0.0})
    return metrics


# =============================================================================
# 主函数
# =============================================================================

def main():
    args = parse_args()
    logger = get_logger("compare_models", log_dir="logs")

    import yaml
    config_path = args.config or f"configs/{args.dataset}_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_yaml = cfg["data"]["dataset_yaml"]
    test_img_dir = os.path.join(cfg["data"]["output_dir"], "images", "test")
    test_lbl_dir = os.path.join(cfg["data"]["output_dir"], "labels", "test")
    img_size     = cfg["train"]["imgsz"]

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {}

    # ── YOLOv8 ────────────────────────────────────────────────────────────────
    if args.yolov8_weights and os.path.isfile(args.yolov8_weights):
        logger.info("评估 YOLOv8...")
        m = eval_yolo_model(args.yolov8_weights, dataset_yaml,
                            args.conf, args.iou, args.device, logger)
        all_metrics["YOLOv8"] = m
        logger.info(f"  YOLOv8: mAP@0.5={m.get('map50', 0):.4f}, FPS={m.get('fps', 0):.1f}")

    # ── YOLOv5 ────────────────────────────────────────────────────────────────
    if args.yolov5_weights and os.path.isfile(args.yolov5_weights):
        logger.info("评估 YOLOv5...")
        m = eval_yolo_model(args.yolov5_weights, dataset_yaml,
                            args.conf, args.iou, args.device, logger)
        all_metrics["YOLOv5"] = m
        logger.info(f"  YOLOv5: mAP@0.5={m.get('map50', 0):.4f}, FPS={m.get('fps', 0):.1f}")

    # ── U-Net ─────────────────────────────────────────────────────────────────
    if args.unet_weights and os.path.isfile(args.unet_weights):
        logger.info("评估 U-Net...")
        m = eval_seg_model(
            lambda: UNet(in_channels=3, out_channels=1),
            args.unet_weights, test_img_dir, test_lbl_dir,
            img_size, args.device, args.conf, logger,
        )
        all_metrics["U-Net"] = m
        logger.info(f"  U-Net: mAP@0.5={m.get('map50', 0):.4f}, FPS={m.get('fps', 0):.1f}")

    # ── FCN ───────────────────────────────────────────────────────────────────
    if args.fcn_weights and os.path.isfile(args.fcn_weights):
        logger.info("评估 FCN...")
        m = eval_seg_model(
            lambda: FCN8s(pretrained=False),
            args.fcn_weights, test_img_dir, test_lbl_dir,
            img_size, args.device, args.conf, logger,
        )
        all_metrics["FCN"] = m
        logger.info(f"  FCN: mAP@0.5={m.get('map50', 0):.4f}, FPS={m.get('fps', 0):.1f}")

    if not all_metrics:
        logger.error("没有可评估的模型，请通过 --yolov8_weights 等参数指定权重路径")
        sys.exit(1)

    # ── 保存汇总 CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, f"metrics_summary_{args.dataset}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,dataset,precision,recall,map50,map50_95,f1,fps,params_M,flops_G\n")
        for model_name, m in all_metrics.items():
            f.write(f"{model_name},{args.dataset},"
                    f"{m.get('precision', 0):.4f},{m.get('recall', 0):.4f},"
                    f"{m.get('map50', 0):.4f},{m.get('map50_95', 0):.4f},"
                    f"{m.get('f1', 0):.4f},{m.get('fps', 0):.1f},"
                    f"{m.get('params_M', 0):.2f},{m.get('flops_G', 0):.2f}\n")

    logger.info(f"\n汇总 CSV 已保存：{csv_path}")

    # 打印汇总表
    logger.info("\n=== 模型对比汇总 ===")
    logger.info(f"{'模型':10s} {'mAP@0.5':10s} {'Precision':10s} {'Recall':10s} "
                f"{'F1':8s} {'FPS':8s} {'Params(M)':10s}")
    logger.info("-" * 70)
    for model_name, m in all_metrics.items():
        logger.info(f"{model_name:10s} {m.get('map50', 0):8.4f}   "
                    f"{m.get('precision', 0):8.4f}   {m.get('recall', 0):8.4f}   "
                    f"{m.get('f1', 0):6.4f}   {m.get('fps', 0):6.1f}   "
                    f"{m.get('params_M', 0):8.2f}")


if __name__ == "__main__":
    main()
