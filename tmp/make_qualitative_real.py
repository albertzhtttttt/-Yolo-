from __future__ import annotations

"""
make_qualitative_real.py - 使用真实模型输出生成论文式定性对比图
路径：tmp/make_qualitative_real.py

功能：
    1. 读取用户指定的 satellite / UAV 原始样例图像；
    2. 分别调用 FCN / U-Net / YOLOv8(ours) 做真实推理；
    3. 将分割输出与检测框结果统一转成二值可视化面板；
    4. 分别导出 satellite 与 UAV 的 3×4 论文式定性对比图。

说明：
    - FCN、U-Net 直接显示真实预测掩码；
    - YOLOv8 为检测模型，为了与分割面板统一排版，这里将预测框栅格化为二值面板；
    - 按用户要求移除圆圈等额外强调元素，保持版面更干净；
    - 所有输出统一保存在 tmp/qualitative_real/ 下，便于本地查看与后续替换。
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.train_fcn import FCN8s
from src.train.train_unet import UNet
from ultralytics import YOLO

OUT_DIR = ROOT / "tmp" / "qualitative_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 用户明确指定的代表性样图。这里直接使用 orig_data 下的原图与原始 YOLO 标注，
# 这样导出的定性图更接近论文里展示“真实场景样例”的常见做法，避免只展示 test split 的缩放版本。
DATASETS = {
    "satellite": {
        "rows": [
            ("data/orig_data/satellite/images/1_orig.jpg", "data/orig_data/satellite/labels/1_orig.txt"),
            ("data/orig_data/satellite/images/12_orig.jpg", "data/orig_data/satellite/labels/12_orig.txt"),
            ("data/orig_data/satellite/images/22_orig.jpg", "data/orig_data/satellite/labels/22_orig.txt"),
        ],
        "weights": {
            "fcn": ROOT / "runs/satellite/fcn/satellite_fcn/best.pt",
            "unet": ROOT / "runs/satellite/unet/satellite_unet/best.pt",
        },
        # YOLOv8 按候选顺序自动回退，优先使用当前环境内效果更好的实验权重。
        "yolo_candidates": [
            ROOT / "runs/satellite/yolov8/satellite_yolov86/weights/best.pt",
            ROOT / "runs/satellite/yolov8/satellite_yolov85/weights/best.pt",
            ROOT / "runs/satellite/yolov8/satellite_yolov84/weights/best.pt",
            ROOT / "runs/satellite/yolov8/satellite_yolov8/weights/best.pt",
            ROOT / "runs/satellite/yolov8/satellite_cv_fold3_yolov8_gap_cbam_p2_n/weights/best.pt",
        ],
    },
    "uav": {
        "rows": [
            ("data/orig_data/uav/images/3_orig.jpg", "data/orig_data/uav/labels/3_orig.txt"),
            ("data/orig_data/uav/images/6_orig.jpg", "data/orig_data/uav/labels/6_orig.txt"),
            ("data/orig_data/uav/images/23_orig.jpg", "data/orig_data/uav/labels/23_orig.txt"),
        ],
        "weights": {
            "fcn": ROOT / "runs/uav/fcn/uav_fcn/best.pt",
            "unet": ROOT / "runs/uav/unet/uav_unet/best.pt",
        },
        # UAV 侧优先使用已经验证更强的 CBAM 版本，其次回退到普通 YOLOv8。
        "yolo_candidates": [
            ROOT / "runs/uav/yolov8/uav_yolov8_cbam/weights/best.pt",
            ROOT / "runs/uav/yolov8/uav_yolov8/weights/best.pt",
        ],
    },
}

IMG_SIZE = 640
SEG_THRESHOLD = 0.40
YOLO_CONF = 0.25


def load_image(path: Path) -> np.ndarray:
    """读取图像并转换为 RGB，便于 matplotlib 直接显示。"""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_tensor(image_rgb: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    """将原图缩放到训练尺寸并转换为模型推理所需张量。"""
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return tensor.to(device)


def load_segmentation_model(model_name: str, weight_path: Path, device: torch.device) -> torch.nn.Module:
    """按模型名恢复分割模型，并加载训练得到的 best.pt 权重。"""
    if model_name == "fcn":
        model = FCN8s(pretrained=False).to(device)
    elif model_name == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError(f"未知分割模型：{model_name}")

    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict_segmentation_mask(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    img_size: int,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    """对单张图运行分割模型，并将预测掩码恢复到原图尺寸。"""
    height, width = image_rgb.shape[:2]
    image_t = prepare_tensor(image_rgb, img_size, device)

    with torch.no_grad():
        pred = model(image_t)[0, 0].detach().cpu().numpy()

    mask = (pred >= threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def predict_yolo_mask(
    model: YOLO,
    image_path: Path,
    shape: tuple[int, int],
    conf: float,
    device_arg: str | int,
) -> np.ndarray:
    """对检测框结果做栅格化，得到与分割面板统一风格的二值图。"""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    results = model.predict(str(image_path), conf=conf, device=device_arg, verbose=False)
    if not results:
        return mask

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return mask

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
        x1 = int(max(0, min(width, x1)))
        y1 = int(max(0, min(height, y1)))
        x2 = int(max(0, min(width, x2)))
        y2 = int(max(0, min(height, y2)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def panel_label(ax, text: str) -> None:
    """在最后一行底部添加论文常见的列标注。"""
    ax.text(
        0.5,
        -0.12,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        family="serif",
    )


def resolve_yolo_weight(candidates: list[Path]) -> Path:
    """按候选顺序选择一个当前环境中真实存在的 YOLOv8 权重。"""
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "缺少 YOLOv8 权重文件：\n" + "\n".join(str(path) for path in candidates)
    )


def validate_dataset_assets(dataset_name: str, dataset_cfg: dict) -> None:
    """在真正推理前先检查图片、标签和权重是否齐全，减少中途失败。"""
    missing = []

    for image_rel, label_rel in dataset_cfg["rows"]:
        image_path = ROOT / image_rel
        label_path = ROOT / label_rel
        if not image_path.is_file():
            missing.append(str(image_path))
        if not label_path.is_file():
            missing.append(str(label_path))

    for weight_path in dataset_cfg["weights"].values():
        if not weight_path.is_file():
            missing.append(str(weight_path))

    if missing:
        raise FileNotFoundError(
            f"数据集 {dataset_name} 缺少必要文件：\n" + "\n".join(missing)
        )


def render_dataset_figure(dataset_name: str, dataset_cfg: dict) -> tuple[Path, Path]:
    """为单个数据集生成一张 3×4 的真实定性对比图。"""
    validate_dataset_assets(dataset_name, dataset_cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo_device = 0 if torch.cuda.is_available() else "cpu"
    yolo_weight = resolve_yolo_weight(dataset_cfg["yolo_candidates"])

    # 分割模型按训练时结构恢复；这里提前加载一次，避免对每张图重复初始化。
    fcn = load_segmentation_model("fcn", dataset_cfg["weights"]["fcn"], device)
    unet = load_segmentation_model("unet", dataset_cfg["weights"]["unet"], device)
    ours = YOLO(str(yolo_weight))

    rows = dataset_cfg["rows"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(6.4, 4.8), dpi=300)
    column_labels = ["(a) Input image", "(b) FCN", "(c) U-Net", "(d) Ours"]

    # 当只有一行时，matplotlib 返回一维 axes；这里统一成二维，避免索引分支。
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (img_rel, _lbl_rel) in enumerate(rows):
        image_path = ROOT / img_rel
        image = load_image(image_path)

        fcn_mask = predict_segmentation_mask(fcn, image, IMG_SIZE, SEG_THRESHOLD, device)
        unet_mask = predict_segmentation_mask(unet, image, IMG_SIZE, SEG_THRESHOLD, device)
        ours_mask = predict_yolo_mask(ours, image_path, image.shape[:2], YOLO_CONF, yolo_device)

        panels = [image, fcn_mask, unet_mask, ours_mask]
        cmaps = [None, "gray", "gray", "gray"]

        for col_idx, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[row_idx, col_idx]
            if cmap is None:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == len(rows) - 1:
                panel_label(ax, column_labels[col_idx])

    fig.subplots_adjust(left=0.03, right=0.995, top=0.99, bottom=0.12, wspace=0.04, hspace=0.06)

    out_png = OUT_DIR / f"qualitative_comparison_{dataset_name}.png"
    out_pdf = OUT_DIR / f"qualitative_comparison_{dataset_name}.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    """按数据集逐一导出真实定性对比图。"""
    for dataset_name, dataset_cfg in DATASETS.items():
        out_png, out_pdf = render_dataset_figure(dataset_name, dataset_cfg)
        print(out_png)
        print(out_pdf)


if __name__ == "__main__":
    main()
