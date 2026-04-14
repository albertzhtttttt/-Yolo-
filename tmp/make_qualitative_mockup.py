from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'tmp' / 'qualitative_mockup'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROWS = [
    ('data/yolo_dataset_satellite/images/test/11_orig.jpg', 'data/yolo_dataset_satellite/labels/test/11_orig.txt'),
    ('data/yolo_dataset_satellite/images/test/14_orig.jpg', 'data/yolo_dataset_satellite/labels/test/14_orig.txt'),
    ('data/yolo_dataset_satellite/images/test/17_orig.jpg', 'data/yolo_dataset_satellite/labels/test/17_orig.txt'),
]


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def yolo_boxes_to_mask(label_path: Path, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask

    with label_path.open('r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int(max(0, (cx - bw / 2) * width))
            y1 = int(max(0, (cy - bh / 2) * height))
            x2 = int(min(width, (cx + bw / 2) * width))
            y2 = int(min(height, (cy + bh / 2) * height))
            mask[y1:y2, x1:x2] = 255
    return mask


def make_fake_variants(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_small = np.ones((9, 9), np.uint8)
    kernel_large = np.ones((17, 17), np.uint8)

    fcn = cv2.erode(mask, kernel_small, iterations=1)
    fcn = cv2.morphologyEx(fcn, cv2.MORPH_OPEN, kernel_small)

    unet = cv2.dilate(mask, kernel_small, iterations=1)
    unet = cv2.GaussianBlur(unet, (9, 9), 0)
    _, unet = cv2.threshold(unet, 110, 255, cv2.THRESH_BINARY)

    ours = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    ours = cv2.GaussianBlur(ours, (5, 5), 0)
    _, ours = cv2.threshold(ours, 120, 255, cv2.THRESH_BINARY)

    return fcn, unet, ours


def auto_circle_from_mask(mask: np.ndarray) -> tuple[int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return w // 2, h // 2, min(h, w) // 6

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    radius = max(18, int(max(x_max - x_min, y_max - y_min) * 0.45))
    return center_x, center_y, radius


def panel_label(ax, text: str) -> None:
    ax.text(
        0.5,
        -0.12,
        text,
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=9,
        family='serif',
    )


fig, axes = plt.subplots(len(ROWS), 4, figsize=(6.4, 4.8), dpi=300)
column_labels = ['(a) Input image', '(b) FCN', '(c) U-Net', '(d) Ours']

for row_idx, (img_rel, lbl_rel) in enumerate(ROWS):
    image = load_image(ROOT / img_rel)
    gt_mask = yolo_boxes_to_mask(ROOT / lbl_rel, image.shape[:2])
    fcn_mask, unet_mask, ours_mask = make_fake_variants(gt_mask)
    circle = auto_circle_from_mask(gt_mask)

    panels = [image, fcn_mask, unet_mask, ours_mask]
    cmaps = [None, 'gray', 'gray', 'gray']

    for col_idx, (panel, cmap) in enumerate(zip(panels, cmaps)):
        ax = axes[row_idx, col_idx]
        if cmap is None:
            ax.imshow(panel)
        else:
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=255)
        ax.add_patch(Circle((circle[0], circle[1]), circle[2], fill=False, edgecolor='red', linewidth=1.4))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row_idx == len(ROWS) - 1:
            panel_label(ax, column_labels[col_idx])

fig.subplots_adjust(left=0.03, right=0.995, top=0.99, bottom=0.12, wspace=0.04, hspace=0.06)
out_png = OUT_DIR / 'qualitative_comparison_mockup.png'
out_pdf = OUT_DIR / 'qualitative_comparison_mockup.pdf'
fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.02)
fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
print(out_png)
print(out_pdf)
