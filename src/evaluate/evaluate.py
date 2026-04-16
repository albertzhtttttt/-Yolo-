"""
evaluate.py - 模型评估脚本
路径：src/evaluate/evaluate.py

功能：
    在 test 集上对训练好的 YOLOv8 模型进行完整评估，输出：
    - Precision / Recall / mAP@0.5 / mAP@0.5:0.95 / F1
    - 混淆矩阵
    - PR 曲线
    - F1-Confidence 曲线
    - 测试集预测结果可视化（随机抽取 N 张）
    - 训练过程曲线（从 results.csv 读取）
    - 所有指标保存到 metrics.csv

使用方式：
    python src/evaluate/evaluate.py \
        --dataset satellite \
        --weights  runs/satellite/yolov8/satellite_yolov8/weights/best.pt \
        [--conf 0.25] [--iou 0.45] [--vis_n 20]

输出目录：
    results/phase1_training/{satellite|uav}/
    ├── metrics.csv              数值指标汇总
    ├── training_curves.png      训练过程曲线
    ├── confusion_matrix.png     混淆矩阵
    ├── pr_curve.png             PR 曲线
    ├── f1_curve.png             F1-Confidence 曲线
    └── predictions/             测试集预测可视化（随机 N 张）
        ├── pred_001.png
        └── ...
"""

from __future__ import annotations

# 远端实验室服务器当前使用 Python 3.8，这里延迟解析类型注解，
# 以兼容 tuple[str, str] 等现代写法，避免分辨率实验评估阶段在导入时直接报错。
import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")  # 无头服务器必须在 import pyplot 前设置

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.plot_utils import (
    setup_plot_style, save_fig, PALETTE,
    draw_confusion_matrix, draw_pr_curve, draw_loss_curve,
)
from src.train.yolo_model_loader import build_yolo_model

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 模型评估脚本")
    parser.add_argument("--dataset",  type=str, choices=["satellite", "uav"], required=True)
    parser.add_argument("--weights",  type=str, required=True, help="模型权重路径（best.pt）")
    parser.add_argument("--config",   type=str, help="配置文件路径（默认自动推断）")
    parser.add_argument("--data",     type=str, default=None,
                        help="覆盖 dataset.yaml 路径（用于分辨率实验等场景）")
    parser.add_argument("--conf",     type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou",      type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--vis_n",    type=int,   default=20,   help="可视化预测结果的图像数量")
    parser.add_argument("--results_csv", type=str, help="训练 results.csv 路径（用于绘制训练曲线）")
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="覆盖输出目录（默认 results/phase1_training/{dataset}）")
    return parser.parse_args()


# =============================================================================
# 训练曲线可视化
# =============================================================================

def plot_training_curves(results_csv: str, out_dir: str) -> None:
    """
    从 Ultralytics 生成的 results.csv 绘制训练过程曲线。

    参数：
        results_csv: results.csv 文件路径
        out_dir:     图表输出目录
    """
    try:
        import pandas as pd
    except ImportError:
        print("[evaluate] pandas 未安装，跳过训练曲线绘制")
        return

    if not os.path.isfile(results_csv):
        print(f"[evaluate] results.csv 不存在：{results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()   # 去除列名空格
    epochs = df["epoch"].tolist() if "epoch" in df.columns else list(range(len(df)))

    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axes = axes.reshape(-1)

    # ── 列名映射（Ultralytics 8.x 格式）──────────────────────────────────────
    # 论文版训练曲线只保留最能说明收敛过程的 4 个面板，减少 2×3 大图在论文中的留白。
    col_map = {
        "train/box_loss":       ("Train box loss", PALETTE["blue"],   axes[0]),
        "val/box_loss":         ("Val box loss",   PALETTE["red"],    axes[1]),
        "metrics/mAP50(B)":     ("mAP@0.5",        PALETTE["green"],  axes[2]),
        "metrics/mAP50-95(B)":  ("mAP@0.5:0.95",   PALETTE["orange"], axes[3]),
    }

    for col, (title, color, ax) in col_map.items():
        if col in df.columns:
            ax.plot(epochs, df[col].tolist(), color=color, linewidth=1.5)
            if "mAP" in col:
                best_idx = df[col].idxmax()
                ax.scatter(
                    epochs[best_idx],
                    df[col][best_idx],
                    color=PALETTE["red"],
                    edgecolor="#222222",
                    linewidth=0.5,
                    s=22,
                    zorder=5,
                    label=f"best={df[col][best_idx]:.3f}",
                )
                # 图例只说明最优点数值，放在子图外侧避免遮挡曲线。
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
            ax.set_xlabel("Epoch")
            ax.set_title(title, pad=4)
        else:
            ax.set_title(f"{title} (N/A)", pad=4)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

    for ax in axes[:2]:
        ax.set_ylabel("Loss")
    for ax in axes[2:]:
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.02)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, os.path.join(out_dir, "training_curves.png"))
    plt.close(fig)
    print(f"[evaluate] 训练曲线已保存：{out_dir}/training_curves.png")


# =============================================================================
# 测试集评估
# =============================================================================

def run_val(model, dataset_yaml: str, conf: float, iou: float, logger) -> dict:
    """
    在 test 集上运行 YOLO 验证，返回指标字典。

    参数：
        model:        YOLO 模型对象
        dataset_yaml: dataset.yaml 路径
        conf:         置信度阈值
        iou:          NMS IoU 阈值
        logger:       日志对象

    返回值：
        {
            "precision": float,
            "recall":    float,
            "map50":     float,
            "map50_95":  float,
            "f1":        float,
        }
    """
    logger.info(f"在 test 集上评估（conf={conf}, iou={iou}）...")

    # 远端服务器在多次重复 val() 时，DataLoader 多进程析构偶发触发
    # "can only test a child process"，这里统一改为 workers=0，
    # 保证分辨率实验长链路评估稳定完成。
    results = model.val(
        data=dataset_yaml,
        split="test",
        conf=conf,
        iou=iou,
        verbose=True,
        workers=0,
    )

    # 从 results 对象提取指标
    metrics = results.results_dict
    p  = float(metrics.get("metrics/precision(B)", 0))
    r  = float(metrics.get("metrics/recall(B)",    0))
    m50   = float(metrics.get("metrics/mAP50(B)",    0))
    m5095 = float(metrics.get("metrics/mAP50-95(B)", 0))
    f1 = 2 * p * r / (p + r + 1e-8)

    logger.info(f"  Precision  = {p:.4f}")
    logger.info(f"  Recall     = {r:.4f}")
    logger.info(f"  mAP@0.5    = {m50:.4f}")
    logger.info(f"  mAP@0.5:95 = {m5095:.4f}")
    logger.info(f"  F1         = {f1:.4f}")

    return {"precision": p, "recall": r, "map50": m50, "map50_95": m5095, "f1": f1}


def save_metrics_csv(metrics: dict, dataset: str, model_name: str, out_path: str) -> None:
    """
    将评估指标保存为 CSV 文件。

    参数：
        metrics:    指标字典
        dataset:    数据集名称
        model_name: 模型名称
        out_path:   输出 CSV 路径
    """
    try:
        import pandas as pd
        row = {"dataset": dataset, "model": model_name, **metrics}
        df = pd.DataFrame([row])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[evaluate] 指标已保存：{out_path}")
    except ImportError:
        # pandas 不可用时手动写 CSV
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            headers = ["dataset", "model"] + list(metrics.keys())
            f.write(",".join(headers) + "\n")
            values = [dataset, model_name] + [str(v) for v in metrics.values()]
            f.write(",".join(values) + "\n")


# =============================================================================
# 混淆矩阵可视化
# =============================================================================

def plot_confusion_matrix_from_val(model, dataset_yaml: str, conf: float,
                                   iou: float, out_dir: str, class_names: list) -> None:
    """
    运行验证并绘制混淆矩阵。

    Ultralytics 在 val() 时会自动生成混淆矩阵数据，
    这里直接读取其生成的图像并重新用统一风格绘制。
    """
    setup_plot_style()

    # Ultralytics 会在 val 输出目录生成 confusion_matrix.png
    # 我们重新调用 val 并指定 save_dir，然后读取其混淆矩阵数据
    results = model.val(
        data=dataset_yaml,
        split="test",
        conf=conf,
        iou=iou,
        plots=True,
        save_dir=os.path.join(out_dir, "_tmp_val"),
        workers=0,
    )

    # 尝试从 results 对象获取混淆矩阵数组
    cm = None
    if hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
        cm_obj = results.confusion_matrix
        if hasattr(cm_obj, "matrix"):
            cm = cm_obj.matrix.astype(int)

    if cm is not None:
        # 用统一风格重新绘制
        fig, ax = plt.subplots(figsize=(6, 5))
        # 添加背景类（Ultralytics 混淆矩阵含 background 行/列）
        display_names = class_names + ["Background"]
        if cm.shape[0] == len(display_names):
            draw_confusion_matrix(cm, display_names, ax=ax, normalize=True,
                                  title="Confusion Matrix (Normalized)")
        else:
            draw_confusion_matrix(cm, class_names, ax=ax, normalize=True,
                                  title="Confusion Matrix (Normalized)")
        plt.tight_layout()
        save_fig(fig, os.path.join(out_dir, "confusion_matrix.png"))
        plt.close(fig)
        print(f"[evaluate] 混淆矩阵已保存：{out_dir}/confusion_matrix.png")
    else:
        # 回退：直接复制 Ultralytics 生成的图
        tmp_cm = os.path.join(out_dir, "_tmp_val", "confusion_matrix_normalized.png")
        if os.path.isfile(tmp_cm):
            import shutil
            shutil.copy2(tmp_cm, os.path.join(out_dir, "confusion_matrix.png"))
            print(f"[evaluate] 混淆矩阵（Ultralytics 原图）已复制：{out_dir}/confusion_matrix.png")


# =============================================================================
# PR 曲线 & F1 曲线
# =============================================================================

def plot_pr_f1_curves(model, dataset_yaml: str, conf_range: np.ndarray,
                      iou: float, out_dir: str, class_names: list) -> None:
    """
    在多个置信度阈值下评估，绘制 PR 曲线和 F1-Confidence 曲线。

    参数：
        model:       YOLO 模型对象
        dataset_yaml:dataset.yaml 路径
        conf_range:  置信度阈值数组（如 np.linspace(0.01, 0.99, 50)）
        iou:         NMS IoU 阈值
        out_dir:     输出目录
        class_names: 类别名称列表
    """
    setup_plot_style()

    precisions = []
    recalls    = []
    f1_scores  = []

    print(f"[evaluate] 计算 PR/F1 曲线（{len(conf_range)} 个置信度阈值）...")
    for conf in conf_range:
        results = model.val(
            data=dataset_yaml,
            split="test",
            conf=float(conf),
            iou=iou,
            verbose=False,
            plots=False,
            workers=0,
        )
        m = results.results_dict
        p  = float(m.get("metrics/precision(B)", 0))
        r  = float(m.get("metrics/recall(B)",    0))
        f1 = 2 * p * r / (p + r + 1e-8)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    # ── PR 曲线 ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ap = np.trapz(precisions[::-1], recalls[::-1])   # 近似 AP
    draw_pr_curve(
        [np.array(precisions)],
        [np.array(recalls)],
        labels=[class_names[0] if class_names else "gap"],
        ap_values=[ap],
        ax=ax,
        title="Precision-Recall Curve",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, os.path.join(out_dir, "pr_curve.png"))
    plt.close(fig)

    # ── F1-Confidence 曲线 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(conf_range, f1_scores, color=PALETTE["blue"], linewidth=2)
    best_idx = int(np.argmax(f1_scores))
    ax.scatter(conf_range[best_idx], f1_scores[best_idx],
               color="red", s=80, zorder=5,
               label=f"Best F1={f1_scores[best_idx]:.4f} @ conf={conf_range[best_idx]:.2f}")
    ax.axvline(conf_range[best_idx], color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1-Confidence Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    # 图例移到图外，避免覆盖最佳阈值附近的曲线。
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, os.path.join(out_dir, "f1_curve.png"))
    plt.close(fig)

    print(f"[evaluate] PR 曲线和 F1 曲线已保存至：{out_dir}")


# =============================================================================
# 测试集预测可视化
# =============================================================================

def visualize_predictions(
    model,
    test_img_dir: str,
    test_lbl_dir: str,
    out_dir: str,
    conf: float,
    n: int = 20,
    seed: int = 42,
) -> None:
    """
    随机抽取 n 张测试图像，绘制预测框（绿色）和 GT 框（红色）对比图。

    参数：
        model:        YOLO 模型对象
        test_img_dir: 测试集图像目录
        test_lbl_dir: 测试集标注目录（GT）
        out_dir:      输出目录
        conf:         置信度阈值
        n:            可视化图像数量
        seed:         随机种子
    """
    setup_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png"}
    img_files = sorted([
        f for f in os.listdir(test_img_dir)
        if Path(f).suffix.lower() in img_exts
    ])

    if not img_files:
        print(f"[evaluate] 测试集图像目录为空：{test_img_dir}")
        return

    random.seed(seed)
    selected = random.sample(img_files, min(n, len(img_files)))

    for idx, img_name in enumerate(selected):
        img_path = os.path.join(test_img_dir, img_name)
        stem     = Path(img_name).stem
        lbl_path = os.path.join(test_lbl_dir, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # ── 模型推理 ──────────────────────────────────────────────────────────
        results = model.predict(img_path, conf=conf, verbose=False)
        pred_boxes = []
        if results and len(results[0].boxes):
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                pred_boxes.append((x1, y1, x2, y2, score))

        # ── 读取 GT 框 ────────────────────────────────────────────────────────
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

        # ── 绘图 ──────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img_rgb)

        # GT 框（红色虚线）
        for (x1, y1, x2, y2) in gt_boxes:
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)

        # 预测框（绿色实线 + 置信度标签）
        for (x1, y1, x2, y2, score) in pred_boxes:
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="#00E676", facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f"{score:.2f}",
                    color="#00E676", fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

        # 图例
        legend_elements = [
            mpatches.Patch(edgecolor="red",    facecolor="none", linestyle="--", label=f"GT ({len(gt_boxes)})"),
            mpatches.Patch(edgecolor="#00E676", facecolor="none", label=f"Pred ({len(pred_boxes)})"),
        ]
        # 图例放在图像外侧上方，避免遮挡真实目标或预测框。
        ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=9, frameon=False)
        ax.set_title(f"{img_name}", fontsize=10)
        ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_fig(fig, os.path.join(out_dir, f"pred_{idx+1:03d}_{stem}.png"))
        plt.close(fig)

    print(f"[evaluate] 预测可视化已保存：{out_dir}（共 {len(selected)} 张）")


def resolve_test_dirs(dataset_yaml: str, cfg: dict) -> tuple[str, str]:
    """
    根据 dataset.yaml 解析测试集图像目录与标签目录。

    参数：
        dataset_yaml: 当前评估使用的 YOLO dataset.yaml，可指向分辨率实验数据集
        cfg:          原始配置文件内容，用作兼容回退

    返回值：
        (test_img_dir, test_lbl_dir)，供预测可视化读取对应分辨率的测试图像与标签。
    """
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f) or {}

    base_path = Path(dataset_cfg.get("path") or Path(dataset_yaml).parent)
    test_entry = dataset_cfg.get("test", "images/test")

    if isinstance(test_entry, list):
        # 当前项目 dataset.yaml 使用字符串路径；若未来改成列表，则回退到原配置路径，避免误判。
        test_img_dir = Path(cfg["data"]["output_dir"]) / "images" / "test"
    else:
        test_img_dir = Path(test_entry)
        if not test_img_dir.is_absolute():
            test_img_dir = base_path / test_img_dir

    parts = list(test_img_dir.parts)
    if "images" in parts:
        # YOLO 数据集采用 images/test 与 labels/test 平行目录结构，替换最后一个 images 片段即可。
        images_idx = len(parts) - 1 - parts[::-1].index("images")
        parts[images_idx] = "labels"
        test_lbl_dir = Path(*parts)
    else:
        test_lbl_dir = Path(cfg["data"]["output_dir"]) / "labels" / "test"

    return str(test_img_dir), str(test_lbl_dir)


# =============================================================================
# 主函数
# =============================================================================

def main():
    args = parse_args()

    config_path = args.config or f"configs/{args.dataset}_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger(f"evaluate_{args.dataset}", log_dir="logs")
    logger.info("=" * 60)
    logger.info(f"模型评估启动：{args.dataset.upper()}")
    logger.info(f"权重文件：{args.weights}")

    # 输出目录
    out_dir = args.output_dir if args.output_dir else os.path.join("results", "phase1_training", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # 导入 YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("未找到 ultralytics，请运行：pip install ultralytics==8.2.87")
        sys.exit(1)

    model, _ = build_yolo_model(
        model_name=args.weights,
        use_pretrained=False,
        logger=logger,
    )
    dataset_yaml  = args.data or cfg["data"]["dataset_yaml"]
    class_names   = cfg["classes"]["names"]
    logger.info(f"数据集配置：{dataset_yaml}")

    # ── 1. 训练曲线 ───────────────────────────────────────────────────────────
    # 自动查找 results.csv（从权重路径推断）
    results_csv = args.results_csv
    if results_csv is None:
        weights_dir = Path(args.weights).parent.parent   # .../exp/weights/best.pt → .../exp/
        candidate   = weights_dir / "results.csv"
        if candidate.exists():
            results_csv = str(candidate)
            logger.info(f"自动找到 results.csv：{results_csv}")

    if results_csv:
        plot_training_curves(results_csv, out_dir)

    # ── 2. 测试集指标 ─────────────────────────────────────────────────────────
    metrics = run_val(model, dataset_yaml, args.conf, args.iou, logger)

    model_name = Path(args.weights).parent.parent.name   # 实验目录名
    save_metrics_csv(
        metrics, args.dataset, model_name,
        os.path.join(out_dir, "metrics.csv"),
    )

    # ── 3. 混淆矩阵 ───────────────────────────────────────────────────────────
    plot_confusion_matrix_from_val(
        model, dataset_yaml, args.conf, args.iou, out_dir, class_names
    )

    # ── 4. PR 曲线 & F1 曲线 ─────────────────────────────────────────────────
    conf_range = np.linspace(0.01, 0.95, 40)
    plot_pr_f1_curves(model, dataset_yaml, conf_range, args.iou, out_dir, class_names)

    # ── 5. 测试集预测可视化 ───────────────────────────────────────────────────
    test_img_dir, test_lbl_dir = resolve_test_dirs(dataset_yaml, cfg)
    pred_out_dir = os.path.join(out_dir, "predictions")

    visualize_predictions(
        model=model,
        test_img_dir=test_img_dir,
        test_lbl_dir=test_lbl_dir,
        out_dir=pred_out_dir,
        conf=args.conf,
        n=args.vis_n,
    )

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("评估完成！结果汇总：")
    logger.info(f"  Precision  = {metrics['precision']:.4f}")
    logger.info(f"  Recall     = {metrics['recall']:.4f}")
    logger.info(f"  mAP@0.5    = {metrics['map50']:.4f}")
    logger.info(f"  mAP@0.5:95 = {metrics['map50_95']:.4f}")
    logger.info(f"  F1         = {metrics['f1']:.4f}")
    logger.info(f"输出目录：{os.path.abspath(out_dir)}")

    # 达标判断
    if metrics["map50"] >= 0.70:
        logger.info("✓ mAP@0.5 ≥ 0.70，模型达标！可进入 P5/P6 阶段。")
    else:
        logger.warning(f"✗ mAP@0.5 = {metrics['map50']:.4f} < 0.70，建议调整超参数后重新训练。")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
