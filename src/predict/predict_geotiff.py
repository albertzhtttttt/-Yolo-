"""
predict_geotiff.py - GeoTIFF 大图预测脚本
路径：src/predict/predict_geotiff.py

功能：
    对单张 GeoTIFF 大图进行滑动窗口推理，输出检测结果。
    方案A（优先）：读取 .tfw 地理变换参数，将像素坐标转为地理坐标，输出 KML/KMZ + SHP
    方案B（备用）：切片保存为 JPEG，逐片推理，收集含检测结果的切片图像

使用方式：
    python src/predict/predict_geotiff.py \
        --model_path runs/satellite/yolov8/satellite_yolov8/weights/best.pt \
        --tif_path   data/predict/位置1/Level20/pos1_now.tif \
        --tfw_path   data/predict/位置1/Level20/位置1_now.tfw \
        --output_dir results/phase1_predict/satellite/位置1 \
        [--conf_thres 0.25] [--iou_thres 0.45] [--mode auto]
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import zipfile
import struct
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger


# =============================================================================
# 地理变换工具
# =============================================================================

def read_tfw(tfw_path: str) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    读取 .tfw 世界文件，返回仿射变换参数。

    .tfw 格式（6行）：
        行1: x方向像素分辨率（东向为正）
        行2: 旋转参数（通常为0）
        行3: 旋转参数（通常为0）
        行4: y方向像素分辨率（北向为负）
        行5: 左上角像素中心的X坐标（经度）
        行6: 左上角像素中心的Y坐标（纬度）

    返回值：(pixel_size_x, rot1, rot2, pixel_size_y, origin_x, origin_y)
    """
    if not os.path.isfile(tfw_path):
        return None
    try:
        with open(tfw_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 6:
            return None
        return tuple(float(x) for x in lines[:6])
    except Exception:
        return None


def pixel_to_geo(px: float, py: float,
                 tfw: Tuple[float, float, float, float, float, float]) -> Tuple[float, float]:
    """
    将像素坐标转换为地理坐标（WGS84 经纬度）。

    参数：
        px, py: 像素坐标（列, 行），以左上角为原点
        tfw:    (.tfw 仿射参数)

    返回值：(longitude, latitude)
    """
    pixel_size_x, rot1, rot2, pixel_size_y, origin_x, origin_y = tfw
    lon = origin_x + px * pixel_size_x + py * rot1
    lat = origin_y + px * rot2 + py * pixel_size_y
    return lon, lat


def bbox_pixel_to_geo(
    x1: float, y1: float, x2: float, y2: float,
    tfw: Tuple[float, float, float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    将像素坐标的检测框转换为地理坐标框（min_lon, min_lat, max_lon, max_lat）。
    """
    corners = [
        pixel_to_geo(x1, y1, tfw),
        pixel_to_geo(x2, y1, tfw),
        pixel_to_geo(x1, y2, tfw),
        pixel_to_geo(x2, y2, tfw),
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return min(lons), min(lats), max(lons), max(lats)


# =============================================================================
# 图像尺寸查询（不加载全图）
# =============================================================================

def get_tif_size(tif_path: str) -> Tuple[int, int]:
    """
    获取 GeoTIFF 图像尺寸（宽, 高），不加载全图数据。
    优先使用 rasterio，回退到 OpenCV。
    """
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            return src.width, src.height
    except Exception:
        pass
    img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img.shape[1], img.shape[0]
    return 0, 0


def read_tile_rasterio(tif_path: str, col: int, row: int, tile_size: int) -> Optional[np.ndarray]:
    """
    使用 rasterio 窗口读取单个切片（不加载全图）。

    参数：
        tif_path:  GeoTIFF 路径
        col, row:  切片左上角像素坐标
        tile_size: 切片尺寸

    返回值：BGR uint8 图像，shape=(tile_size, tile_size, 3)
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(tif_path) as src:
        win = Window(col, row, tile_size, tile_size)
        n_bands = src.count

        if n_bands >= 3:
            r = src.read(1, window=win)
            g = src.read(2, window=win)
            b = src.read(3, window=win)
        elif n_bands == 1:
            gray = src.read(1, window=win)
            r = g = b = gray
        else:
            r = src.read(1, window=win)
            g = src.read(2, window=win) if n_bands >= 2 else r.copy()
            b = r.copy()

    # 填充到 tile_size（边缘切片可能不足）
    def pad_band(band):
        if band.shape[0] < tile_size or band.shape[1] < tile_size:
            padded = np.zeros((tile_size, tile_size), dtype=band.dtype)
            padded[:band.shape[0], :band.shape[1]] = band
            return padded
        return band

    r, g, b = pad_band(r), pad_band(g), pad_band(b)

    # 转换为 uint8
    def to_uint8(band):
        if band.dtype == np.uint8:
            return band
        lo, hi = band.min(), band.max()
        if hi > lo:
            return ((band.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        return np.zeros_like(band, dtype=np.uint8)

    r, g, b = to_uint8(r), to_uint8(g), to_uint8(b)
    return cv2.merge([b, g, r])


def read_tile_opencv(tif_path: str, col: int, row: int, tile_size: int,
                     img_cache: dict) -> Optional[np.ndarray]:
    """
    使用 OpenCV 读取切片（需要全图缓存，仅用于小图回退）。
    img_cache 用于避免重复读取全图。
    """
    if tif_path not in img_cache:
        img = cv2.imread(tif_path, cv2.IMREAD_COLOR)
        img_cache[tif_path] = img
    img = img_cache[tif_path]
    if img is None:
        return None
    h, w = img.shape[:2]
    tile = img[row:row + tile_size, col:col + tile_size]
    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        padded[:tile.shape[0], :tile.shape[1]] = tile
        tile = padded
    return tile


# =============================================================================
# 滑动窗口坐标生成（不加载图像）
# =============================================================================

def sliding_window_tiles(
    img: np.ndarray,
    tile_size: int = 640,
    stride: int = 320,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    对大图进行滑动窗口切片（用于小图或已加载的图像）。

    参数：
        img:       BGR 图像
        tile_size: 切片尺寸（正方形）
        stride:    滑动步长

    返回值：
        [(row_start, col_start, tile_img), ...]
    """
    h, w = img.shape[:2]
    tiles = []

    row_starts = list(range(0, h - tile_size, stride))
    if not row_starts or row_starts[-1] + tile_size < h:
        row_starts.append(max(0, h - tile_size))

    col_starts = list(range(0, w - tile_size, stride))
    if not col_starts or col_starts[-1] + tile_size < w:
        col_starts.append(max(0, w - tile_size))

    for r in row_starts:
        for c in col_starts:
            tile = img[r:r + tile_size, c:c + tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            tiles.append((r, c, tile))

    return tiles


def get_tile_coords(img_w: int, img_h: int,
                    tile_size: int = 640, stride: int = 320) -> List[Tuple[int, int]]:
    """
    生成滑动窗口切片的起始坐标列表（不加载图像）。

    返回值：[(row_start, col_start), ...]
    """
    row_starts = list(range(0, img_h - tile_size, stride))
    if not row_starts or row_starts[-1] + tile_size < img_h:
        row_starts.append(max(0, img_h - tile_size))

    col_starts = list(range(0, img_w - tile_size, stride))
    if not col_starts or col_starts[-1] + tile_size < img_w:
        col_starts.append(max(0, img_w - tile_size))

    return [(r, c) for r in row_starts for c in col_starts]


# =============================================================================
# 跨切片全局 NMS
# =============================================================================

def global_nms(
    detections: List[Tuple[float, float, float, float, float]],
    iou_thres: float = 0.5,
) -> List[Tuple[float, float, float, float, float]]:
    """
    对全图所有检测框执行 NMS 去重。

    参数：
        detections: [(x1, y1, x2, y2, score), ...]（像素坐标）
        iou_thres:  IoU 阈值

    返回值：
        去重后的检测框列表
    """
    if not detections:
        return []

    boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
    scores = np.array([d[4] for d in detections], dtype=np.float32)

    # 计算面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # 计算与其余框的 IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]


# =============================================================================
# 方案A：输出 KML / KMZ / SHP
# =============================================================================

def write_kml(
    detections_geo: List[Tuple[float, float, float, float, float]],
    location_name: str,
    out_path: str,
) -> None:
    """
    将地理坐标检测框写入 KML 文件。

    参数：
        detections_geo: [(min_lon, min_lat, max_lon, max_lat, score), ...]
        location_name:  位置名称（用于 KML 文档名）
        out_path:       输出 KML 文件路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        f'  <Document>',
        f'    <name>Mangrove Gap Detections - {location_name}</name>',
        '    <Style id="gapStyle">',
        '      <LineStyle><color>ff00ff00</color><width>2</width></LineStyle>',
        '      <PolyStyle><color>4000ff00</color></PolyStyle>',
        '    </Style>',
    ]

    for idx, (min_lon, min_lat, max_lon, max_lat, score) in enumerate(detections_geo):
        lines += [
            '    <Placemark>',
            f'      <name>gap_{idx+1:04d}</name>',
            f'      <description>confidence: {score:.4f}</description>',
            '      <styleUrl>#gapStyle</styleUrl>',
            '      <Polygon>',
            '        <outerBoundaryIs><LinearRing><coordinates>',
            f'          {min_lon},{min_lat},0',
            f'          {max_lon},{min_lat},0',
            f'          {max_lon},{max_lat},0',
            f'          {min_lon},{max_lat},0',
            f'          {min_lon},{min_lat},0',
            '        </coordinates></LinearRing></outerBoundaryIs>',
            '      </Polygon>',
            '    </Placemark>',
        ]

    lines += ['  </Document>', '</kml>']

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_kmz(kml_path: str, kmz_path: str) -> None:
    """将 KML 文件压缩为 KMZ。"""
    with zipfile.ZipFile(kmz_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(kml_path, arcname="doc.kml")


def write_shp(
    detections_geo: List[Tuple[float, float, float, float, float]],
    out_path: str,
) -> None:
    """
    将地理坐标检测框写入 Shapefile（.shp + .dbf + .shx + .prj）。

    使用纯 Python 实现，不依赖 pyshp/shapefile 库。
    每个检测框写为一个 Polygon 要素。
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    base = out_path.replace(".shp", "")

    # ── SHP 文件（几何数据）──────────────────────────────────────────────────
    # Shapefile 格式：文件头(100字节) + 记录
    # Shape type 5 = Polygon

    records_shp = []
    for min_lon, min_lat, max_lon, max_lat, score in detections_geo:
        # 矩形多边形（5个点，首尾相同）
        coords = [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ]
        # 内容长度（以16位字为单位）：4(type) + 32(bbox) + 4(numparts) + 4(numpoints) + 4(parts[0]) + 5*16(points)
        content_len = (4 + 32 + 4 + 4 + 4 + len(coords) * 16) // 2
        records_shp.append((coords, min_lon, min_lat, max_lon, max_lat, content_len))

    # 计算文件长度（字节）
    file_len_bytes = 100  # 文件头
    for rec in records_shp:
        file_len_bytes += 8 + rec[5] * 2  # 记录头(8) + 内容

    file_len_words = file_len_bytes // 2

    # 全局 bbox
    all_lons = [r[1] for r in records_shp] + [r[3] for r in records_shp]
    all_lats = [r[2] for r in records_shp] + [r[4] for r in records_shp]
    bbox_xmin = min(all_lons) if all_lons else 0.0
    bbox_ymin = min(all_lats) if all_lats else 0.0
    bbox_xmax = max(all_lons) if all_lons else 0.0
    bbox_ymax = max(all_lats) if all_lats else 0.0

    with open(base + ".shp", "wb") as f:
        # 文件头（大端）
        f.write(struct.pack(">i", 9994))          # 文件码
        f.write(b"\x00" * 20)                     # 未使用
        f.write(struct.pack(">i", file_len_words)) # 文件长度（16位字）
        # 文件头（小端）
        f.write(struct.pack("<i", 1000))           # 版本
        f.write(struct.pack("<i", 5))              # Shape type: Polygon
        f.write(struct.pack("<d", bbox_xmin))
        f.write(struct.pack("<d", bbox_ymin))
        f.write(struct.pack("<d", bbox_xmax))
        f.write(struct.pack("<d", bbox_ymax))
        f.write(struct.pack("<d", 0.0))            # Zmin
        f.write(struct.pack("<d", 0.0))            # Zmax
        f.write(struct.pack("<d", 0.0))            # Mmin
        f.write(struct.pack("<d", 0.0))            # Mmax

        for rec_idx, (coords, xmin, ymin, xmax, ymax, content_len) in enumerate(records_shp):
            # 记录头（大端）
            f.write(struct.pack(">i", rec_idx + 1))   # 记录号（从1开始）
            f.write(struct.pack(">i", content_len))    # 内容长度（16位字）
            # 记录内容（小端）
            f.write(struct.pack("<i", 5))              # Shape type: Polygon
            f.write(struct.pack("<d", xmin))
            f.write(struct.pack("<d", ymin))
            f.write(struct.pack("<d", xmax))
            f.write(struct.pack("<d", ymax))
            f.write(struct.pack("<i", 1))              # NumParts
            f.write(struct.pack("<i", len(coords)))    # NumPoints
            f.write(struct.pack("<i", 0))              # Parts[0]
            for lon, lat in coords:
                f.write(struct.pack("<d", lon))
                f.write(struct.pack("<d", lat))

    # ── SHX 文件（索引）──────────────────────────────────────────────────────
    shx_len_words = (100 + len(records_shp) * 8) // 2
    with open(base + ".shx", "wb") as f:
        f.write(struct.pack(">i", 9994))
        f.write(b"\x00" * 20)
        f.write(struct.pack(">i", shx_len_words))
        f.write(struct.pack("<i", 1000))
        f.write(struct.pack("<i", 5))
        f.write(struct.pack("<d", bbox_xmin))
        f.write(struct.pack("<d", bbox_ymin))
        f.write(struct.pack("<d", bbox_xmax))
        f.write(struct.pack("<d", bbox_ymax))
        f.write(struct.pack("<d", 0.0))
        f.write(struct.pack("<d", 0.0))
        f.write(struct.pack("<d", 0.0))
        f.write(struct.pack("<d", 0.0))

        offset_words = 50  # 文件头 100 字节 = 50 个16位字
        for _, _, _, _, _, content_len in records_shp:
            f.write(struct.pack(">i", offset_words))
            f.write(struct.pack(">i", content_len))
            offset_words += 4 + content_len  # 记录头(4字) + 内容

    # ── DBF 文件（属性表）────────────────────────────────────────────────────
    # 字段：FID(N,10,0), SCORE(N,10,6)
    with open(base + ".dbf", "wb") as f:
        n_records = len(records_shp)
        n_fields = 2
        header_size = 32 + n_fields * 32 + 1
        record_size = 1 + 10 + 10  # 删除标志 + FID + SCORE

        f.write(struct.pack("B", 3))           # 版本
        f.write(bytes([26, 3, 19]))            # 年月日（占位）
        f.write(struct.pack("<i", n_records))
        f.write(struct.pack("<H", header_size))
        f.write(struct.pack("<H", record_size))
        f.write(b"\x00" * 20)                 # 保留

        # 字段描述符：FID
        f.write(b"FID\x00\x00\x00\x00\x00\x00\x00\x00")
        f.write(b"N")
        f.write(b"\x00" * 4)
        f.write(struct.pack("B", 10))          # 字段长度
        f.write(struct.pack("B", 0))           # 小数位
        f.write(b"\x00" * 14)

        # 字段描述符：SCORE
        f.write(b"SCORE\x00\x00\x00\x00\x00\x00")
        f.write(b"N")
        f.write(b"\x00" * 4)
        f.write(struct.pack("B", 10))
        f.write(struct.pack("B", 6))
        f.write(b"\x00" * 14)

        f.write(b"\r")  # 头部终止符

        for rec_idx, (_, _, _, _, score, _) in enumerate(
            [(r[0], r[1], r[2], r[3], detections_geo[i][4], r[5])
             for i, r in enumerate(records_shp)]
        ):
            f.write(b" ")  # 未删除标志
            f.write(f"{rec_idx+1:>10}".encode("ascii"))
            f.write(f"{score:>10.6f}".encode("ascii"))

    # ── PRJ 文件（坐标系）────────────────────────────────────────────────────
    wgs84_wkt = (
        'GEOGCS["GCS_WGS_1984",'
        'DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
        'PRIMEM["Greenwich",0.0],'
        'UNIT["Degree",0.0174532925199433]]'
    )
    with open(base + ".prj", "w") as f:
        f.write(wgs84_wkt)


# =============================================================================
# 方案B：输出带标注的切片图像
# =============================================================================

def save_annotated_tile(
    tile: np.ndarray,
    boxes: List[Tuple[float, float, float, float, float]],
    out_path: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.6,
) -> None:
    """
    在切片图像上绘制检测框并保存。

    参数：
        tile:       BGR 切片图像
        boxes:      [(x1, y1, x2, y2, score), ...]（相对于切片的像素坐标）
        out_path:   输出图像路径
        color:      框颜色（BGR）
        font_scale: 置信度标签字号
    """
    vis = tile.copy()
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{score:.2f}"
        cv2.putText(vis, label, (int(x1), max(int(y1) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, vis)


# =============================================================================
# 主预测函数
# =============================================================================

def predict_geotiff(
    model_path: str,
    tif_path: str,
    output_dir: str,
    tfw_path: Optional[str] = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    global_iou_thres: float = 0.5,
    tile_size: int = 640,
    stride: int = 320,
    batch_size: int = 8,
    device: str = "0",
    mode: str = "auto",
    save_annotated_tiles: bool = True,
    logger=None,
) -> dict:
    """
    对单张 GeoTIFF 大图进行滑动窗口推理。

    参数：
        model_path:       模型权重路径
        tif_path:         GeoTIFF 图像路径
        output_dir:       输出目录
        tfw_path:         .tfw 世界文件路径（方案A需要）
        conf_thres:       置信度阈值
        iou_thres:        单切片 NMS IoU 阈值
        global_iou_thres: 全图 NMS IoU 阈值
        tile_size:        切片尺寸
        stride:           滑动步长
        batch_size:       批量推理切片数
        device:           GPU 编号
        mode:             "auto" / "geo" / "tiles"
        save_annotated_tiles: 是否保存带标注的切片（方案B）
        logger:           日志对象

    返回值：
        {"n_detections": int, "output_mode": str, "output_files": [...]}
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    os.makedirs(output_dir, exist_ok=True)
    location_name = Path(tif_path).stem

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("未找到 ultralytics，请运行：pip install ultralytics==8.2.87")
        sys.exit(1)

    logger.info(f"加载模型：{model_path}")
    model = YOLO(model_path)

    # ── 获取图像尺寸（不加载全图）────────────────────────────────────────────
    logger.info(f"读取图像信息：{tif_path}")
    w, h = get_tif_size(tif_path)
    if w == 0 or h == 0:
        logger.error(f"无法获取图像尺寸：{tif_path}")
        return {"n_detections": 0, "output_mode": "failed", "output_files": []}
    logger.info(f"图像尺寸：{w}×{h}（{w*h/1e6:.1f} MP）")

    # ── 读取地理变换参数 ──────────────────────────────────────────────────────
    tfw = None
    if tfw_path:
        tfw = read_tfw(tfw_path)
        if tfw:
            logger.info(f"地理变换参数：origin=({tfw[4]:.6f}, {tfw[5]:.6f}), "
                        f"pixel_size=({tfw[0]:.8f}, {tfw[3]:.8f})")
        else:
            logger.warning(f"无法读取 .tfw 文件：{tfw_path}")

    # 判断输出模式
    use_geo = (mode == "geo") or (mode == "auto" and tfw is not None)
    if mode == "geo" and tfw is None:
        logger.error("mode=geo 但未提供有效的 .tfw 文件，退出")
        return {"n_detections": 0, "output_mode": "failed", "output_files": []}

    # ── 生成切片坐标（不加载图像）────────────────────────────────────────────
    logger.info(f"生成切片坐标（tile_size={tile_size}, stride={stride}）...")
    tile_coords = get_tile_coords(w, h, tile_size, stride)
    logger.info(f"共 {len(tile_coords)} 个切片")

    # 检测 rasterio 是否可用（决定读取方式）
    use_rasterio = False
    try:
        import rasterio
        use_rasterio = True
        logger.info("使用 rasterio 窗口读取（内存高效）")
    except ImportError:
        logger.warning("rasterio 不可用，回退到 OpenCV 全图读取")

    # OpenCV 回退时的全图缓存
    img_cache = {}

    # ── 批量推理 ──────────────────────────────────────────────────────────────
    all_detections = []  # [(x1_global, y1_global, x2_global, y2_global, score), ...]
    tile_detections = {}  # {(row, col): [(x1_local, y1_local, x2_local, y2_local, score), ...]}

    logger.info(f"开始推理（batch_size={batch_size}）...")
    for batch_start in range(0, len(tile_coords), batch_size):
        batch_coords = tile_coords[batch_start:batch_start + batch_size]

        # 读取当前批次的切片图像
        batch_imgs = []
        for row, col in batch_coords:
            if use_rasterio:
                tile = read_tile_rasterio(tif_path, col, row, tile_size)
            else:
                tile = read_tile_opencv(tif_path, col, row, tile_size, img_cache)
            if tile is None:
                tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            batch_imgs.append(tile)

        results = model.predict(
            batch_imgs,
            conf=conf_thres,
            iou=iou_thres,
            imgsz=tile_size,
            device=device,
            verbose=False,
        )

        for (row, col), result in zip(batch_coords, results):
            local_boxes = []
            if result.boxes and len(result.boxes):
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    # 转换为全图像素坐标
                    all_detections.append((
                        x1 + col, y1 + row,
                        x2 + col, y2 + row,
                        score,
                    ))
                    local_boxes.append((x1, y1, x2, y2, score))
            tile_detections[(row, col)] = local_boxes

        processed = batch_start + len(batch_coords)
        if processed % (batch_size * 10) == 0 or processed == len(tile_coords):
            logger.info(f"  已处理 {processed}/{len(tile_coords)} 个切片，"
                        f"当前检测框数：{len(all_detections)}")

    # 释放 OpenCV 全图缓存
    img_cache.clear()

    logger.info(f"推理完成，NMS前检测框数：{len(all_detections)}")

    # ── 全图 NMS ──────────────────────────────────────────────────────────────
    all_detections = global_nms(all_detections, global_iou_thres)
    logger.info(f"全图 NMS 后检测框数：{len(all_detections)}")

    output_files = []

    # ── 方案A：输出 KML/SHP ───────────────────────────────────────────────────
    if use_geo and tfw is not None:
        logger.info("方案A：转换地理坐标，输出 KML/SHP...")
        detections_geo = []
        for x1, y1, x2, y2, score in all_detections:
            min_lon, min_lat, max_lon, max_lat = bbox_pixel_to_geo(x1, y1, x2, y2, tfw)
            detections_geo.append((min_lon, min_lat, max_lon, max_lat, score))

        kml_path = os.path.join(output_dir, "detections.kml")
        kmz_path = os.path.join(output_dir, "detections.kmz")
        shp_path = os.path.join(output_dir, "detections.shp")

        write_kml(detections_geo, location_name, kml_path)
        write_kmz(kml_path, kmz_path)
        write_shp(detections_geo, shp_path)

        output_files += [kml_path, kmz_path, shp_path]
        logger.info(f"KML/KMZ/SHP 已保存至：{output_dir}")
        output_mode = "geo"

    else:
        # ── 方案B：保存带标注的切片 ───────────────────────────────────────────
        logger.info("方案B：保存带检测框的切片图像...")
        tiles_dir = os.path.join(output_dir, "tiles")
        saved_count = 0
        for (row, col), local_boxes in tile_detections.items():
            if not local_boxes:
                continue
            # 重新读取该切片用于可视化
            if use_rasterio:
                tile_img = read_tile_rasterio(tif_path, col, row, tile_size)
            else:
                tile_img = read_tile_opencv(tif_path, col, row, tile_size, img_cache)
            if tile_img is None:
                continue
            tile_path = os.path.join(tiles_dir, f"tile_r{row:05d}_c{col:05d}.jpg")
            save_annotated_tile(tile_img, local_boxes, tile_path)
            output_files.append(tile_path)
            saved_count += 1
        img_cache.clear()
        logger.info(f"已保存 {saved_count} 张含检测结果的切片至：{tiles_dir}")
        output_mode = "tiles"

    # ── 保存 summary.csv ──────────────────────────────────────────────────────
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("location,n_detections,output_mode,tif_width,tif_height,n_tiles\n")
        f.write(f"{location_name},{len(all_detections)},{output_mode},{w},{h},{len(tile_coords)}\n")
    output_files.append(summary_path)

    # ── 保存检测框像素坐标 CSV ────────────────────────────────────────────────
    boxes_csv = os.path.join(output_dir, "detections_pixel.csv")
    with open(boxes_csv, "w", encoding="utf-8") as f:
        f.write("x1,y1,x2,y2,score\n")
        for x1, y1, x2, y2, score in all_detections:
            f.write(f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{score:.4f}\n")
    output_files.append(boxes_csv)

    logger.info(f"预测完成：{location_name}，共 {len(all_detections)} 个检测框")
    return {
        "n_detections": len(all_detections),
        "output_mode": output_mode,
        "output_files": output_files,
    }


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GeoTIFF 大图预测脚本")
    parser.add_argument("--model_path",  type=str, required=True, help="模型权重路径")
    parser.add_argument("--tif_path",    type=str, required=True, help="GeoTIFF 图像路径")
    parser.add_argument("--output_dir",  type=str, required=True, help="输出目录")
    parser.add_argument("--tfw_path",    type=str, default=None,  help=".tfw 世界文件路径")
    parser.add_argument("--conf_thres",  type=float, default=0.25)
    parser.add_argument("--iou_thres",   type=float, default=0.45)
    parser.add_argument("--global_iou",  type=float, default=0.5,  help="全图 NMS IoU 阈值")
    parser.add_argument("--tile_size",   type=int,   default=640)
    parser.add_argument("--stride",      type=int,   default=320)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--device",      type=str,   default="0")
    parser.add_argument("--mode",        type=str,   default="auto",
                        choices=["auto", "geo", "tiles"])
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("predict_geotiff", log_dir="logs")

    result = predict_geotiff(
        model_path=args.model_path,
        tif_path=args.tif_path,
        output_dir=args.output_dir,
        tfw_path=args.tfw_path,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        global_iou_thres=args.global_iou,
        tile_size=args.tile_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=args.device,
        mode=args.mode,
        logger=logger,
    )

    logger.info(f"检测框总数：{result['n_detections']}")
    logger.info(f"输出模式：{result['output_mode']}")
    logger.info(f"输出文件：{result['output_files']}")


if __name__ == "__main__":
    main()
