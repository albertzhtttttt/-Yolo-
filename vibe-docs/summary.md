# 项目总结报告

**项目名称：** 基于 YOLO 模型的高分辨率遥感影像红树林林窗探测
**完成日期：** 2026-03-21
**服务器：** lfy@172.31.226.112，GPU: Tesla P100 ×6

---

## 一、项目概述

本项目针对红树林林窗（Forest Gap）目标，基于高分辨率遥感影像（卫星 + 无人机），构建了完整的目标检测与分析流程，涵盖数据准备、模型训练、大图预测、分辨率影响分析和多模型对比实验。

---

## 二、各阶段完成情况

| 阶段 | 内容 | 状态 |
|------|------|------|
| P0 | 环境配置、目录结构、配置文件 | ✅ 完成 |
| P1 | 数据预处理、增强、数据集构建 | ✅ 完成 |
| P2 | YOLOv8 训练（satellite + UAV） | ✅ 完成 |
| P3 | 评估可视化（指标/曲线/混淆矩阵） | ✅ 完成 |
| P4 | GeoTIFF 大图预测（6个位置） | ✅ 完成 |
| P5 | 分辨率影响分析（4个分辨率级别） | ✅ 完成 |
| P6 | 多模型对比实验（YOLOv5/U-Net/FCN） | ✅ 完成 |

---

## 三、核心模型性能

### YOLOv8 基准模型（P2/P3）

| 数据集 | mAP@0.5 | Precision | Recall | F1 |
|--------|---------|-----------|--------|----|
| satellite | 0.418 | 0.920 | 0.222 | 0.358 |
| uav | 0.974 | 0.974 | 0.950 | 0.962 |

当前仅 UAV 数据集通过 mAP@0.5 ≥ 0.70 关卡；satellite 基线结果偏低，后续仍需继续调参或补充实验说明。

---

## 四、P4：大图预测结果

使用滑动窗口（640×640，步长 320）对 6 个 TIF 位置进行推理，rasterio 窗口读取避免 OOM，全局 NMS 去重后输出 KML/KMZ/SHP。

| 数据集 | 总检测框 |
|--------|----------|
| satellite | 23,712 |
| uav | 47,196 |

各位置检测框数量（satellite）：Site1=2704, Site2_1=1409, Site2_2=4046, Site2_3=3941, Site3_1=7785, Site3_2=3827

各位置检测框数量（uav）：Site1=13036, Site2_1=3407, Site2_2=7721, Site2_3=9594, Site3_1=7616, Site3_2=5822

---

## 五、P5：分辨率影响分析

对原始图像进行 100%/75%/50%/25% 下采样后缩放回 640×640，分别训练 YOLOv8n（150 epochs）。

**主要发现：**
- Satellite：100% 分辨率 mAP=0.667 最高，75% 下降明显（0.446），50% 有所回升（0.645），说明中等分辨率损失对卫星图像影响较大
- UAV：各分辨率 mAP 均保持在 0.877 以上，对分辨率降低更鲁棒，50% 时甚至略高于 100%（0.957 vs 0.950）
- 当前 `results/phase2_resolution/metrics_by_scale.csv` 仅汇总了 UAV 行；satellite 结果已在 `results/phase2_resolution/satellite/scale_*/metrics.csv` 中生成，文档按各子目录实际指标汇总

| 数据集 | 100% | 75% | 50% | 25% |
|--------|------|-----|-----|-----|
| satellite mAP@0.5 | 0.667 | 0.446 | 0.645 | 0.556 |
| uav mAP@0.5 | 0.950 | 0.877 | 0.957 | 0.897 |

---

## 六、P6：多模型对比实验

在相同测试集上对比 YOLOv8 / YOLOv5 / U-Net / FCN 四种模型。

### Satellite

| 模型 | mAP@0.5 | F1 | FPS | Params(M) |
|------|---------|-----|-----|-----------|
| YOLOv8 | 0.418 | 0.358 | 106.2 | — |
| YOLOv5 | **0.630** | 0.613 | 107.0 | — |
| U-Net  | 0.196 | 0.533 | 119.5 | 31.0 |
| FCN    | 0.064 | 0.333 | 186.7 | 14.7 |

### UAV

| 模型 | mAP@0.5 | F1 | FPS | Params(M) |
|------|---------|-----|-----|-----------|
| YOLOv8 | 0.974 | 0.962 | 112.9 | — |
| YOLOv5 | **0.986** | 0.986 | 87.4 | — |
| U-Net  | 0.210 | 0.430 | 155.1 | 31.0 |
| FCN    | 0.267 | 0.497 | 337.6 | 14.7 |

**结论：**
- 检测模型（YOLOv8/YOLOv5）整体仍显著优于分割模型（U-Net/FCN）
- 在当前结果下，YOLOv5 在 satellite 与 UAV 两个测试集上的 mAP@0.5 都高于 YOLOv8
- 分割模型 FPS 更高，但精度明显不足，更适合作为补充性对比而非主方案
- FCN 参数量最少、推理速度最快，但检测精度仍最低；U-Net 在 satellite 上的 F1 高于 FCN，但 mAP@0.5 仍明显落后于检测模型

---

## 七、关键文件索引

### 模型权重

| 模型 | 路径 | mAP@0.5 |
|------|------|---------|
| YOLOv8 satellite | `runs/satellite/yolov8/satellite_yolov8/weights/best.pt` | 0.418 |
| YOLOv8 uav | `runs/uav/yolov8/uav_yolov8/weights/best.pt` | 0.974 |
| YOLOv5 satellite | `runs/satellite/yolov5/satellite_yolov5n/weights/best.pt` | 0.630 |
| YOLOv5 uav | `runs/uav/yolov5/uav_yolov5n/weights/best.pt` | 0.986 |
| U-Net satellite | `runs/satellite/unet/satellite_unet/best.pt` | 0.196 |
| U-Net uav | `runs/uav/unet/uav_unet/best.pt` | 0.210 |
| FCN satellite | `runs/satellite/fcn/satellite_fcn/best.pt` | 0.064 |
| FCN uav | `runs/uav/fcn/uav_fcn/best.pt` | 0.267 |

### 结果目录

| 阶段 | 路径 |
|------|------|
| P3 评估结果 | `results/phase1_training/{satellite\|uav}/` |
| P4 预测结果 | `results/phase1_predict/{satellite\|uav}/{位置名}/` |
| P5 分辨率分析 | `results/phase2_resolution/` |
| P6 对比实验 | `results/phase3_comparison/` |

---

## 八、技术要点

1. **大图 OOM 问题**：使用 rasterio 窗口读取（windowed reading），避免将 28928×51200 像素图像全部加载入内存
2. **并行训练**：P5/P6 充分利用服务器 6 块 Tesla P100，通过 `--device` 参数分配 GPU，最多同时运行 6 个训练任务
3. **分割模型评估**：U-Net/FCN 输出二值掩码，通过连通域分析转为检测框后计算 P/R/F1，mAP 计算方式与检测模型不同，不可直接比较
4. **图表英文化**：所有 matplotlib 图表标签均使用英文，避免服务器无中文字体导致的渲染问题
