"""
train_yolov8.py - YOLOv8 模型训练脚本
路径：src/train/train_yolov8.py

功能：
    基于 Ultralytics YOLOv8 框架，对卫星或无人机数据集进行目标检测训练。
    支持从配置文件读取超参数，支持命令行覆盖关键参数，便于迭代调参。

使用方式：
    # 基础训练（读取配置文件默认参数）
    python src/train/train_yolov8.py --dataset satellite
    python src/train/train_yolov8.py --dataset uav

    # 覆盖关键参数
    python src/train/train_yolov8.py --dataset satellite --model yolov8s --epochs 200 --batch 32

    # 从断点继续训练
    python src/train/train_yolov8.py --dataset satellite --resume runs/satellite/yolov8/exp1/weights/last.pt

输出：
    runs/{satellite|uav}/yolov8/{exp_name}/
    ├── weights/
    │   ├── best.pt     最优权重（val mAP 最高）
    │   └── last.pt     最后一轮权重
    ├── results.csv     每 epoch 的指标记录
    ├── args.yaml       本次训练的完整参数记录
    └── ...             Ultralytics 自动生成的其他文件
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger


def _auto_select_gpu() -> str:
    """返回显存最空闲的 GPU 编号，失败时返回 '0'。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        free = [int(x.strip()) for x in out.strip().splitlines()]
        return str(free.index(max(free)))
    except Exception:
        return "0"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本")

    # 必选参数
    parser.add_argument("--dataset", type=str, choices=["satellite", "uav"],
                        required=True, help="数据集类型")

    # 可选：覆盖配置文件中的参数
    parser.add_argument("--config",  type=str, help="配置文件路径（默认自动推断）")
    parser.add_argument("--model",   type=str, help="模型规格，如 yolov8n / yolov8s / yolov8m")
    parser.add_argument("--epochs",  type=int, help="训练轮数")
    parser.add_argument("--batch",   type=int, help="批大小")
    parser.add_argument("--imgsz",   type=int, help="输入图像尺寸（默认 640）")
    parser.add_argument("--lr0",     type=float, help="初始学习率")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "AdamW", "Adam"],
                        help="优化器")
    parser.add_argument("--device",  type=str, help="GPU 编号，如 0 或 0,1（多卡）")
    parser.add_argument("--workers", type=int, help="DataLoader 工作进程数")
    parser.add_argument("--patience", type=int, help="早停耐心值")
    parser.add_argument("--name",    type=str, help="实验名称（用于区分多次运行）")
    parser.add_argument("--resume",  type=str, help="从指定权重文件断点续训")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="不使用预训练权重（从头训练）")

    return parser.parse_args()


def build_train_args(cfg: dict, args) -> dict:
    """
    合并配置文件参数与命令行参数，命令行参数优先级更高。

    参数：
        cfg:  从 yaml 加载的配置字典
        args: argparse 解析结果

    返回值：
        传递给 YOLO.train() 的参数字典
    """
    t = cfg["train"]

    # 基础参数（来自配置文件）
    train_args = {
        "data":           cfg["data"]["dataset_yaml"],
        "imgsz":          t["imgsz"],
        "epochs":         t["epochs"],
        "batch":          t["batch"],
        "lr0":            t["lr0"],
        "lrf":            t["lrf"],
        "momentum":       t["momentum"],
        "weight_decay":   t["weight_decay"],
        "warmup_epochs":  t["warmup_epochs"],
        "optimizer":      t["optimizer"],
        "patience":       t["patience"],
        "seed":           t["seed"],
        "device":         t["device"],
        "workers":        t["workers"],
        "project":        t["save_dir"],
        "name":           t["project_name"],
        "exist_ok":       False,   # 同名实验自动加编号
        "verbose":        True,
        # 关闭 Ultralytics 内置增强（我们已做离线增强）
        "augment":        False,
        "hsv_h":          0.0,
        "hsv_s":          0.0,
        "hsv_v":          0.0,
        "degrees":        0.0,
        "translate":      0.0,
        "scale":          0.0,
        "shear":          0.0,
        "perspective":    0.0,
        "flipud":         0.0,
        "fliplr":         0.0,
        "mosaic":         0.0,
        "mixup":          0.0,
        "copy_paste":     0.0,
    }

    # 命令行参数覆盖
    if args.epochs:   train_args["epochs"]    = args.epochs
    if args.batch:    train_args["batch"]     = args.batch
    if args.imgsz:    train_args["imgsz"]     = args.imgsz
    if args.lr0:      train_args["lr0"]       = args.lr0
    if args.optimizer: train_args["optimizer"] = args.optimizer
    if args.device:   train_args["device"]    = args.device
    else:
        train_args["device"] = _auto_select_gpu()
    if args.workers:  train_args["workers"]   = args.workers
    if args.patience: train_args["patience"]  = args.patience
    if args.name:     train_args["name"]      = args.name

    return train_args


def main():
    args = parse_args()

    # ── 加载配置文件 ──────────────────────────────────────────────────────────
    config_path = args.config or f"configs/{args.dataset}_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── 初始化日志 ────────────────────────────────────────────────────────────
    logger = get_logger(f"train_yolov8_{args.dataset}", log_dir="logs")
    logger.info("=" * 60)
    logger.info(f"YOLOv8 训练启动：{args.dataset.upper()}")
    logger.info(f"配置文件：{config_path}")

    # ── 确定模型规格 ──────────────────────────────────────────────────────────
    model_name = args.model or cfg["train"]["model"]
    logger.info(f"模型规格：{model_name}")

    # ── 检查 dataset.yaml 是否存在 ────────────────────────────────────────────
    dataset_yaml = cfg["data"]["dataset_yaml"]
    if not os.path.isfile(dataset_yaml):
        logger.error(f"dataset.yaml 不存在：{dataset_yaml}")
        logger.error("请先运行 P1 数据准备流程：python scripts/run_data_pipeline.py")
        sys.exit(1)

    # ── 导入 Ultralytics（延迟导入，避免未安装时报错影响其他模块）────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("未找到 ultralytics，请运行：pip install ultralytics==8.2.87")
        sys.exit(1)

    # ── 构建训练参数 ──────────────────────────────────────────────────────────
    train_args = build_train_args(cfg, args)
    if not args.device:
        logger.info(f"自动选择 GPU：{train_args['device']}（显存最空闲）")

    logger.info("训练参数：")
    for k, v in train_args.items():
        logger.info(f"  {k:20s} = {v}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    if args.resume:
        # 断点续训：直接加载 last.pt
        logger.info(f"断点续训：{args.resume}")
        model = YOLO(args.resume)
        train_args["resume"] = True
    else:
        if args.no_pretrain:
            # 从头训练：加载模型结构 yaml（不含预训练权重）
            model_yaml = model_name + ".yaml"
            logger.info(f"从头训练（无预训练权重）：{model_yaml}")
            model = YOLO(model_yaml)
        else:
            # 迁移学习：加载 COCO 预训练权重
            model_pt = model_name + ".pt"
            logger.info(f"迁移学习（COCO 预训练）：{model_pt}")
            model = YOLO(model_pt)

    # ── 开始训练 ──────────────────────────────────────────────────────────────
    logger.info("开始训练...")
    results = model.train(**train_args)

    # ── 训练完成，输出关键指标 ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("训练完成！")

    # 找到本次实验的输出目录
    save_dir = Path(train_args["project"]) / train_args["name"]
    # Ultralytics 会在同名时自动加数字后缀，找最新的
    parent = Path(train_args["project"])
    if parent.exists():
        exp_dirs = sorted(parent.glob(f"{train_args['name']}*"), key=os.path.getmtime)
        if exp_dirs:
            save_dir = exp_dirs[-1]

    logger.info(f"实验目录：{save_dir.resolve()}")
    best_pt = save_dir / "weights" / "best.pt"
    if best_pt.exists():
        logger.info(f"最优权重：{best_pt.resolve()}")

    # 读取并打印最终指标
    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            # 找 val mAP50 最高的行
            map_col = [c for c in df.columns if "mAP50" in c and "95" not in c]
            if map_col:
                best_row = df.loc[df[map_col[0]].idxmax()]
                logger.info(f"最优 epoch：{int(best_row.get('epoch', -1)) + 1}")
                logger.info(f"  mAP@0.5    = {best_row[map_col[0]]:.4f}")
                map95_col = [c for c in df.columns if "mAP50-95" in c]
                if map95_col:
                    logger.info(f"  mAP@0.5:95 = {best_row[map95_col[0]]:.4f}")
        except Exception as e:
            logger.warning(f"读取 results.csv 失败：{e}")

    logger.info("=" * 60)
    logger.info(f"下一步：运行评估脚本")
    logger.info(f"  python src/evaluate/evaluate.py --dataset {args.dataset} --weights {best_pt}")


if __name__ == "__main__":
    main()
