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
| P5 | 分辨率影响分析（6个分辨率级别 + satellite 低分辨率多 seed 复现实验） | ✅ 完成 |
| P6 | 多模型对比实验（YOLOv5/U-Net/FCN） | ✅ 完成 |
| P7 | YOLOv8 CBAM 消融实验与论文图补充（UAV + satellite） | ✅ 完成 |

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

对原始图像进行 `100% / 50% / 25% / 12.5% / 6.25% / 3.125%` 下采样后统一缩放回 `640×640`，分别训练 YOLOv8n（150 epochs）。针对 `satellite` 在极低分辨率端出现的异常跳动，又补充执行了 `12.5% / 6.25% / 3.125%` 的 3 seed 复现实验。

**主实验结果（单次运行，mAP@0.5）：**

| 数据集 | 100% | 50% | 25% | 12.5% | 6.25% | 3.125% |
|--------|------|-----|-----|-------|------|--------|
| satellite | 0.6668 | 0.5837 | 0.6667 | 0.0000 | 0.6111 | 0.0000 |
| uav | 0.9795 | 0.9890 | 0.9812 | 0.9855 | 0.9344 | 0.9160 |

**Satellite 低分辨率多 seed 复现实验：**

| Scale | seed 42 | seed 43 | seed 44 | mAP@0.5 mean ± std |
|-------|---------|---------|---------|--------------------|
| 12.5% | 0.0000 | 0.4186 | 0.0000 | 0.1395 ± 0.2417 |
| 6.25% | 0.6111 | 0.0000 | 0.0000 | 0.2037 ± 0.3528 |
| 3.125% | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |

**主要发现：**
- UAV 在 6 个分辨率档位上都保持较高精度，说明其对分辨率退化更鲁棒。
- Satellite 在 `12.5%` 以下进入明显不稳定区，单次结果会出现 `0 → 0.6111 → 0` 的非单调跳动。
- 多 seed 结果表明，这种跳动并不是稳定规律：`6.25%` 并未稳定高于 `12.5%`，而是个别 seed 偶发命中少量目标。
- `3.125%` 在 3 个 seed 上全部为 `0.0000`，说明该档位下卫星林窗检测已基本失效。
- 结合 `satellite` test 集仅 `6` 张图、`9` 个目标，更合理的解释是“低分辨率端训练随机性 + 小样本评估”共同放大了单次指标波动，而不是实验串档。

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
| YOLOv8 baseline | 0.974 | 0.962 | 112.9 | — |
| YOLOv5 | **0.986** | 0.986 | 87.4 | — |
| U-Net  | 0.210 | 0.430 | 155.1 | 31.0 |
| FCN    | 0.267 | 0.497 | 337.6 | 14.7 |

**结论：**
- 检测模型（YOLOv8/YOLOv5）整体仍显著优于分割模型（U-Net/FCN）
- 在当前对比实验结果下，YOLOv5 在 satellite 与 UAV 两个测试集上的 mAP@0.5 都高于 YOLOv8 baseline
- 补充的 UAV CBAM 消融实验表明，YOLOv8n + CBAM 相对 baseline 将 mAP@0.5 从 0.974 提升到 0.987，将 mAP@0.5:0.95 从 0.709 提升到 0.801
- 卫星数据集的对等 CBAM 消融表明，YOLOv8n + CBAM 相对同规模 baseline 将 Recall 从 0.333 提升到 0.556，mAP@0.5 从 0.584 提升到 0.605，F1 从 0.496 提升到 0.657，但 mAP@0.5:0.95 从 0.398 下降到 0.193，说明其收益主要来自召回增强而非高 IoU 定位质量提升
- 分割模型 FPS 更高，但精度明显不足，更适合作为补充性对比而非主方案
- FCN 参数量最少、推理速度最快，但检测精度仍最低；U-Net 在 satellite 上的 F1 高于 FCN，但 mAP@0.5 仍明显落后于检测模型

---

## 七、关键文件索引

### 模型权重

| 模型 | 路径 | mAP@0.5 |
|------|------|---------|
| YOLOv8 satellite baseline | `runs/satellite/yolov8/satellite_yolov8n_baseline/weights/best.pt` | 0.584 |
| YOLOv8 satellite + CBAM | `runs/satellite/yolov8/satellite_yolov8_cbam/weights/best.pt` | 0.605 |
| YOLOv8 uav baseline | `runs/uav/yolov8/uav_yolov8/weights/best.pt` | 0.974 |
| YOLOv8 uav + CBAM | `runs/uav/yolov8/uav_yolov8_cbam/weights/best.pt` | 0.987 |
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
| P7 CBAM 消融 | `results/phase5_cbam_ablation/` |

---

## 八、技术要点

1. **大图 OOM 问题**：使用 rasterio 窗口读取（windowed reading），避免将 28928×51200 像素图像全部加载入内存
2. **并行训练**：P5/P6 充分利用服务器 6 块 Tesla P100，通过 `--device` 参数分配 GPU，最多同时运行 6 个训练任务
3. **分割模型评估**：U-Net/FCN 输出二值掩码，通过连通域分析转为检测框后计算 P/R/F1，mAP 计算方式与检测模型不同，不可直接比较
4. **图表英文化**：所有 matplotlib 图表标签均使用英文，避免服务器无中文字体导致的渲染问题
5. **补充消融实验**：`scripts/run_cbam_ablation.py` 现已统一支持 UAV 与 satellite 两个数据集的 baseline vs CBAM 汇总，结果统一写入 `results/phase5_cbam_ablation/`，便于论文直接引用
