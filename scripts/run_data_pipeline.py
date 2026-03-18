#!/usr/bin/env python3
"""
run_data_pipeline.py - P1 数据准备全流程一键运行脚本
路径：scripts/run_data_pipeline.py

功能：
    按顺序执行卫星和无人机数据集的完整数据准备流程：
        Step 1: 预处理（缩放 / 切片）
        Step 2: 数据集划分（train / val / test）
        Step 3: 数据增强（仅对 train 集）

    卫星和无人机两个流程相互独立，可分别运行或一起运行。

使用方式：
    # 运行全部（卫星 + 无人机）
    python scripts/run_data_pipeline.py

    # 仅运行卫星
    python scripts/run_data_pipeline.py --dataset satellite

    # 仅运行无人机
    python scripts/run_data_pipeline.py --dataset uav

    # 跳过增强（仅做预处理和划分）
    python scripts/run_data_pipeline.py --no_augment

    # 生成可视化统计图
    python scripts/run_data_pipeline.py --vis

注意：
    - 首次运行前请确保已安装所有依赖（bash scripts/install_env.sh）
    - 原始数据位于 data/orig_data/{satellite,uav}/
    - 运行过程日志保存到 logs/ 目录
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger


def run_step(cmd: list[str], step_name: str, logger) -> bool:
    """
    执行一个子命令并记录日志。

    参数：
        cmd:       命令列表（subprocess 格式）
        step_name: 步骤名称（用于日志）
        logger:    日志对象

    返回值：
        True 表示成功，False 表示失败
    """
    logger.info(f"▶ {step_name}")
    logger.info(f"  命令：{' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,   # 子进程输出直接打印到控制台
    )

    if result.returncode != 0:
        logger.error(f"✗ {step_name} 失败（返回码 {result.returncode}）")
        return False

    logger.info(f"✓ {step_name} 完成\n")
    return True


def run_pipeline(dataset: str, vis: bool, no_augment: bool, logger) -> bool:
    """
    执行单个数据集的完整流程。

    参数：
        dataset:    "satellite" 或 "uav"
        vis:        是否生成可视化统计
        no_augment: True 时跳过增强
        logger:     日志对象

    返回值：
        True 表示全部成功
    """
    logger.info("=" * 60)
    logger.info(f"开始处理数据集：{dataset.upper()}")
    logger.info("=" * 60)

    python = sys.executable  # 使用当前 Python 解释器

    # ── Step 1：预处理 ────────────────────────────────────────────────────────
    if dataset == "satellite":
        preprocess_cmd = [
            python, "src/data_preparation/preprocess_satellite.py",
            "--config", f"configs/{dataset}_config.yaml",
        ]
    else:
        preprocess_cmd = [
            python, "src/data_preparation/preprocess_uav.py",
            "--config", f"configs/{dataset}_config.yaml",
        ]
    if vis:
        preprocess_cmd.append("--vis")

    ok = run_step(preprocess_cmd, f"[{dataset}] Step 1: 图像预处理", logger)
    if not ok:
        return False

    # ── Step 2：数据集划分 ────────────────────────────────────────────────────
    build_cmd = [
        python, "src/data_preparation/build_dataset.py",
        "--dataset", dataset,
        "--config", f"configs/{dataset}_config.yaml",
    ]
    if vis:
        build_cmd.append("--vis")

    ok = run_step(build_cmd, f"[{dataset}] Step 2: 数据集划分 (train/val/test)", logger)
    if not ok:
        return False

    # ── Step 3：数据增强（仅 train 集）───────────────────────────────────────
    if not no_augment:
        augment_cmd = [
            python, "src/data_preparation/augment.py",
            "--dataset", dataset,
            "--config", f"configs/{dataset}_config.yaml",
        ]
        ok = run_step(augment_cmd, f"[{dataset}] Step 3: 数据增强 (train only)", logger)
        if not ok:
            return False
    else:
        logger.info(f"[{dataset}] Step 3: 数据增强（已跳过，--no_augment）\n")

    logger.info(f"✓✓ {dataset.upper()} 数据准备全部完成！\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="P1 数据准备全流程一键脚本")
    parser.add_argument("--dataset", type=str, choices=["satellite", "uav", "all"],
                        default="all", help="运行哪个数据集（默认 all）")
    parser.add_argument("--vis",        action="store_true", help="生成可视化统计图")
    parser.add_argument("--no_augment", action="store_true", help="跳过数据增强步骤")
    args = parser.parse_args()

    logger = get_logger("run_data_pipeline", log_dir="logs")
    logger.info("=" * 60)
    logger.info("P1 数据准备全流程启动")
    logger.info(f"dataset={args.dataset}, vis={args.vis}, no_augment={args.no_augment}")
    logger.info("=" * 60)

    datasets = ["satellite", "uav"] if args.dataset == "all" else [args.dataset]

    all_ok = True
    for ds in datasets:
        ok = run_pipeline(ds, args.vis, args.no_augment, logger)
        if not ok:
            logger.error(f"❌ {ds} 流程出错，请检查日志后重试")
            all_ok = False

    if all_ok:
        logger.info("=" * 60)
        logger.info("✓✓✓ 全部数据准备完成！可以开始 P2 模型训练。")
        logger.info("    下一步：python src/train/train_yolov8.py --dataset satellite")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ 部分流程失败，请修复后重新运行")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
