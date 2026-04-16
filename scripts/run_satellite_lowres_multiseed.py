"""
run_satellite_lowres_multiseed.py - Satellite 低分辨率多 seed 复现实验脚本
路径：scripts/run_satellite_lowres_multiseed.py

功能：
    1. 针对 satellite 低分辨率档位 12.5% / 6.25% / 3.125% 运行多 seed 训练
    2. 每个 scale × seed 独立评估，避免单次训练随机性支配结论
    3. 聚合 mean/std 指标，并生成论文风格误差棒图

使用方式：
    python scripts/run_satellite_lowres_multiseed.py --device 0
    python scripts/run_satellite_lowres_multiseed.py --device 0 --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger


DEFAULT_SCALES = [12.5, 6.25, 3.125]
DEFAULT_SEEDS = [42, 43, 44]
METRIC_KEYS = ["map50", "precision", "recall", "f1"]
METRIC_LABELS = {
    "map50": "mAP@0.5",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 score",
}


def parse_args():
    """解析低分辨率多 seed 复现实验参数。"""
    parser = argparse.ArgumentParser(description="Satellite 低分辨率多 seed 复现实验脚本")
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES,
                        help="低分辨率百分比列表，默认 12.5 6.25 3.125")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="训练随机种子列表，默认 42 43 44")
    parser.add_argument("--model", type=str, default=None,
                        help="YOLOv8 模型规格（默认从 satellite 配置文件读取）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（默认从 satellite 配置文件读取）")
    parser.add_argument("--device", type=str, default=None,
                        help="指定 GPU 编号（建议远端显式传入 0/1/2）")
    parser.add_argument("--skip_train", action="store_true",
                        help="跳过训练，直接查找已有权重并重新评估/汇总")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅评估与汇总，等价于 --skip_train")
    return parser.parse_args()


def format_scale(scale: float) -> str:
    """将分辨率比例格式化为稳定字符串，用于目录名、实验名与 CSV。"""
    if math.isclose(scale, round(scale)):
        return str(int(round(scale)))
    return str(scale).rstrip("0").rstrip(".")


def format_model_tag(model_name: str) -> str:
    """将模型配置或权重名转换为可安全用于实验目录的短标识。"""
    model_path = Path(model_name)
    model_tag = model_path.stem if model_path.suffix else model_name
    return model_tag.replace("/", "_").replace("\\", "_").replace(" ", "_")


def run_cmd(cmd: list[str], logger, desc: str = "") -> int:
    """运行子进程命令并实时写入统一日志。"""
    logger.info(f"执行：{' '.join(cmd)}")
    if desc:
        logger.info(f"  ({desc})")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"  | {line}")
    proc.wait()
    return proc.returncode


def find_latest_weights(save_dir: Path, exp_name: str) -> Path | None:
    """查找指定实验名前缀下最新的 best.pt，兼容 Ultralytics 自动追加数字后缀。"""
    candidates = sorted(
        save_dir.glob(f"{exp_name}*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def reset_dir(path: Path, logger, desc: str) -> None:
    """覆盖单个 seed 的评估输出目录，避免同名历史文件混入本轮结果。"""
    if path.exists():
        logger.info(f"清理旧{desc}：{path}")
        shutil.rmtree(path)


def read_metrics(metrics_csv: Path) -> dict[str, float] | None:
    """读取 evaluate.py 生成的 metrics.csv，并返回聚合所需指标。"""
    if not metrics_csv.is_file():
        return None

    with open(metrics_csv, "r", encoding="utf-8-sig", newline="") as f:
        row = next(csv.DictReader(f), None)
    if row is None:
        return None

    try:
        return {metric: float(row[metric]) for metric in METRIC_KEYS}
    except (KeyError, TypeError, ValueError):
        return None


def run_single_experiment(
    scale: float,
    seed: int,
    model_name: str,
    model_tag: str,
    epochs: int,
    config_path: Path,
    save_dir: Path,
    results_root: Path,
    args,
    logger,
) -> dict[str, float] | None:
    """运行单个 scale × seed 的训练与评估，并返回该组合的指标。"""
    scale_name = format_scale(scale)
    dataset_yaml = Path("data") / "resolution_datasets_satellite" / f"scale_{scale_name}" / "dataset.yaml"
    if not dataset_yaml.is_file():
        logger.error(f"dataset.yaml 不存在，跳过 scale_{scale_name} seed_{seed}：{dataset_yaml}")
        return None

    exp_name = f"satellite_lowres_scale{scale_name}_seed{seed}_{model_tag}"
    out_dir = results_root / f"scale_{scale_name}" / f"seed_{seed}"

    logger.info("\n" + "=" * 60)
    logger.info(f"低分辨率多 seed 实验：scale_{scale_name}, seed_{seed}")

    if not args.skip_train and not args.eval_only:
        train_cmd = [
            sys.executable,
            "src/train/train_yolov8.py",
            "--dataset", "satellite",
            "--config", str(config_path),
            "--model", model_name,
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--data", str(dataset_yaml),
            "--name", exp_name,
            "--save_dir", str(save_dir),
        ]
        if args.device:
            train_cmd += ["--device", args.device]
        ret = run_cmd(train_cmd, logger, f"训练 scale_{scale_name} seed_{seed}")
        if ret != 0:
            logger.error(f"scale_{scale_name} seed_{seed} 训练失败，跳过评估")
            return None

    weights_path = find_latest_weights(save_dir, exp_name)
    if weights_path is None:
        logger.warning(f"找不到 best.pt，跳过 scale_{scale_name} seed_{seed}：{save_dir}/{exp_name}*/weights/best.pt")
        return None

    reset_dir(out_dir, logger, "评估输出目录")
    eval_cmd = [
        sys.executable,
        "src/evaluate/evaluate.py",
        "--dataset", "satellite",
        "--weights", str(weights_path),
        "--config", str(config_path),
        "--data", str(dataset_yaml),
        "--output_dir", str(out_dir),
        "--vis_n", "0",
    ]
    ret = run_cmd(eval_cmd, logger, f"评估 scale_{scale_name} seed_{seed}")
    if ret != 0:
        logger.error(f"scale_{scale_name} seed_{seed} 评估失败，跳过指标汇总")
        return None

    metrics = read_metrics(out_dir / "metrics.csv")
    if metrics is None:
        logger.warning(f"读取 metrics.csv 失败：{out_dir / 'metrics.csv'}")
        return None

    logger.info(f"scale_{scale_name} seed_{seed} mAP@0.5 = {metrics['map50']:.4f}")
    return metrics


def save_metrics_by_seed(rows: list[dict], output_path: Path) -> None:
    """保存每个 scale × seed 的原始指标明细，便于回溯异常 seed。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scale", "seed", "map50", "precision", "recall", "f1"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "scale": row["scale_name"],
                "seed": row["seed"],
                "map50": f"{row['map50']:.4f}",
                "precision": f"{row['precision']:.4f}",
                "recall": f"{row['recall']:.4f}",
                "f1": f"{row['f1']:.4f}",
            })


def summarize_metrics(rows: list[dict], scales: list[float], seeds: list[int], logger) -> list[dict]:
    """按 scale 聚合多 seed 指标，仅保留 seed 完整的档位。"""
    summary_rows = []
    expected_seeds = set(seeds)

    for scale in scales:
        scale_name = format_scale(scale)
        scale_rows = [row for row in rows if math.isclose(float(row["scale"]), float(scale))]
        available_seeds = {int(row["seed"]) for row in scale_rows}
        if available_seeds != expected_seeds:
            missing = sorted(expected_seeds - available_seeds)
            logger.warning(f"scale_{scale_name} 结果不完整，缺少 seeds={missing}，不写入统计汇总")
            continue

        summary = {"scale": scale, "scale_name": scale_name, "n": len(scale_rows)}
        for metric in METRIC_KEYS:
            values = [float(row[metric]) for row in scale_rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    return summary_rows


def save_metrics_summary(summary_rows: list[dict], output_path: Path) -> None:
    """保存每个分辨率档位的 mean/std 汇总表。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scale", "n"]
    for metric in METRIC_KEYS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            out_row = {"scale": row["scale_name"], "n": row["n"]}
            for metric in METRIC_KEYS:
                out_row[f"{metric}_mean"] = f"{row[f'{metric}_mean']:.4f}"
                out_row[f"{metric}_std"] = f"{row[f'{metric}_std']:.4f}"
            writer.writerow(out_row)


def plot_map50_errorbar(summary_rows: list[dict], output_dir: Path) -> None:
    """绘制 mAP@0.5 的 mean ± std 误差棒图，用于解释低分辨率端稳定性。"""
    if not summary_rows:
        return

    # 绘图依赖延迟导入，避免本地只查看 --help 时被 Matplotlib / NumPy ABI 环境问题阻断。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.plot_utils import MARKERS, PALETTE, save_fig, setup_plot_style

    setup_plot_style()
    scales = [row["scale"] for row in summary_rows]
    labels = [row["scale_name"] for row in summary_rows]
    means = [row["map50_mean"] for row in summary_rows]
    stds = [row["map50_std"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.errorbar(
        scales,
        means,
        yerr=stds,
        color=PALETTE["blue"],
        marker=MARKERS[0],
        markerfacecolor="white",
        markeredgewidth=1.0,
        capsize=4,
        linewidth=1.6,
        label="Mean ± std",
    )
    ax.set_xlabel("Resolution scale (%)")
    ax.set_ylabel("mAP@0.5")
    ax.set_xticks(scales)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    # 图例放到绘图区上方，避免遮挡低分辨率端误差棒。
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, str(output_dir / "lowres_multiseed_map50_errorbar.png"))
    plt.close(fig)


def plot_metrics_errorbar(summary_rows: list[dict], output_dir: Path) -> None:
    """绘制四个核心指标的 mean ± std 子图，辅助判断异常拐点是否稳定。"""
    if not summary_rows:
        return

    # 绘图依赖延迟导入，保证脚本参数解析和非绘图路径能在轻量环境中运行。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.plot_utils import MARKERS, PALETTE, save_fig, setup_plot_style

    setup_plot_style()
    scales = [row["scale"] for row in summary_rows]
    labels = [row["scale_name"] for row in summary_rows]
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.8), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for ax, metric, color in zip(axes, METRIC_KEYS, colors):
        means = [row[f"{metric}_mean"] for row in summary_rows]
        stds = [row[f"{metric}_std"] for row in summary_rows]
        ax.errorbar(
            scales,
            means,
            yerr=stds,
            color=color,
            marker=MARKERS[0],
            markerfacecolor="white",
            markeredgewidth=1.0,
            capsize=3,
            linewidth=1.5,
        )
        ax.set_title(METRIC_LABELS[metric], pad=4)
        ax.set_xticks(scales)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.02)

    for ax in axes[::2]:
        ax.set_ylabel("Score")
    for ax in axes[-2:]:
        ax.set_xlabel("Resolution scale (%)")

    fig.tight_layout()
    save_fig(fig, str(output_dir / "lowres_multiseed_metrics.png"))
    plt.close(fig)


def main() -> None:
    """执行 satellite 低分辨率多 seed 训练、评估、聚合与绘图。"""
    args = parse_args()
    logger = get_logger("run_satellite_lowres_multiseed", log_dir="logs")

    config_path = Path("configs/satellite_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg["train"]["model"]
    model_tag = format_model_tag(model_name)
    epochs = args.epochs or cfg["train"]["epochs"]
    save_dir = Path("runs/satellite/resolution_multiseed")
    results_root = Path("results/phase2_resolution/satellite_lowres_multiseed")

    logger.info("=" * 60)
    logger.info("Satellite 低分辨率多 seed 复现实验启动")
    logger.info(f"scales: {[format_scale(scale) for scale in args.scales]}")
    logger.info(f"seeds: {args.seeds}")
    logger.info(f"model: {model_name}, epochs: {epochs}")

    rows = []
    for scale in args.scales:
        for seed in args.seeds:
            metrics = run_single_experiment(
                scale=scale,
                seed=seed,
                model_name=model_name,
                model_tag=model_tag,
                epochs=epochs,
                config_path=config_path,
                save_dir=save_dir,
                results_root=results_root,
                args=args,
                logger=logger,
            )
            if metrics is not None:
                rows.append({
                    "scale": scale,
                    "scale_name": format_scale(scale),
                    "seed": seed,
                    **metrics,
                })

    if not rows:
        logger.error("没有可汇总的多 seed 指标，实验结束")
        sys.exit(1)

    save_metrics_by_seed(rows, results_root / "metrics_by_seed.csv")
    summary_rows = summarize_metrics(rows, args.scales, args.seeds, logger)
    if not summary_rows:
        logger.error("没有完整 scale 的多 seed 指标，未生成统计图")
        sys.exit(1)

    save_metrics_summary(summary_rows, results_root / "metrics_seed_summary.csv")
    plot_map50_errorbar(summary_rows, results_root)
    plot_metrics_errorbar(summary_rows, results_root)

    logger.info("\n=== Satellite 低分辨率多 seed 汇总 ===")
    logger.info(f"{'分辨率':10s} {'n':>3s} {'mAP@0.5 mean±std':>20s} {'F1 mean±std':>18s}")
    logger.info("-" * 58)
    for row in summary_rows:
        logger.info(
            f"{row['scale_name']:>8s}% {row['n']:3d} "
            f"{row['map50_mean']:.4f}±{row['map50_std']:.4f} "
            f"{row['f1_mean']:.4f}±{row['f1_std']:.4f}"
        )

    logger.info(f"多 seed 复现实验完成，输出目录：{results_root}")


if __name__ == "__main__":
    main()
