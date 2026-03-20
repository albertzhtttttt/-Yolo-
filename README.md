# 基于 YOLO 模型的高分辨率遥感影像红树林林窗探测

基于 YOLOv8 的红树林林窗（Forest Gap）目标检测系统，支持卫星与无人机两类高分辨率遥感影像，涵盖数据准备、模型训练、大图预测、分辨率影响分析和多模型对比实验。

---

## 环境要求

- Python 3.8+
- CUDA 12.1（Tesla P100 验证）
- 主要依赖：ultralytics 8.2.87、rasterio、torch 2.4.1

```bash
bash scripts/install_env.sh
```

---

## 项目结构

```
.
├── configs/                    # 配置文件
│   ├── satellite_config.yaml   # 卫星数据集参数
│   ├── uav_config.yaml         # 无人机数据集参数
│   └── predict_config.yaml     # 大图预测参数（6个TIF位置）
├── scripts/                    # 一键运行脚本
│   ├── run_data_pipeline.py    # P1 数据准备流水线
│   ├── run_train_pipeline.py   # P2/P3 训练+评估流水线
│   ├── run_predict.py          # P4 批量大图预测
│   └── run_resolution_pipeline.py  # P5 分辨率实验流水线
├── src/
│   ├── data_preparation/       # 数据预处理、增强、数据集构建
│   ├── train/                  # 训练脚本（YOLOv8/YOLOv5/U-Net/FCN）
│   ├── evaluate/               # 评估脚本
│   ├── predict/                # 大图预测
│   └── visualize/              # 可视化
├── data/                       # 数据集（不纳入版本控制）
├── runs/                       # 训练输出（不纳入版本控制）
├── results/                    # 实验结果（不纳入版本控制）
└── vibe-docs/                  # 项目文档
    ├── requirements.md         # 需求文档
    ├── TODO.md                 # 执行计划
    ├── deliverables.md         # 交付记录
    └── summary.md              # 总结报告
```

---

## 快速开始

### P1：数据准备

```bash
python scripts/run_data_pipeline.py --dataset all
```

### P2/P3：训练与评估

```bash
# 后台训练（satellite + uav）
bash scripts/train_bg.sh --dataset all

# 仅评估
python src/evaluate/evaluate.py --dataset satellite \
    --weights runs/satellite/yolov8/satellite_yolov83/weights/best.pt
```

### P4：大图预测（6个TIF位置）

```bash
python scripts/run_predict.py --dataset satellite \
    --weights runs/satellite/yolov8/satellite_yolov83/weights/best.pt

python scripts/run_predict.py --dataset uav \
    --weights runs/uav/yolov8/uav_yolov82/weights/best.pt
```

输出：`results/phase1_predict/{satellite|uav}/{位置名}/detections.kml/.shp`

### P5：分辨率影响分析

```bash
# 卫星（GPU 0），无人机（GPU 1）并行
python scripts/run_resolution_pipeline.py --dataset satellite --device 0
python scripts/run_resolution_pipeline.py --dataset uav --device 1
```

### P6：多模型对比实验

```bash
# 生成分割标签
python src/data_preparation/yolo_to_mask.py --dataset all

# 训练各模型（分配不同GPU）
python src/train/train_yolov5.py --dataset satellite --device 0
python src/train/train_unet.py   --dataset satellite --device 1 --batch 2
python src/train/train_fcn.py    --dataset satellite --device 2

# 对比评估
python src/evaluate/compare_models.py --dataset satellite \
    --yolov8_weights runs/satellite/yolov8/satellite_yolov83/weights/best.pt \
    --yolov5_weights runs/satellite/yolov5/satellite_yolov5n/weights/best.pt \
    --unet_weights   runs/satellite/unet/satellite_unet/best.pt \
    --fcn_weights    runs/satellite/fcn/satellite_fcn/best.pt
```

---

## 主要结果

### P2/P3：YOLOv8 基准模型

| 数据集 | mAP@0.5 | Precision | Recall | F1 | 状态 |
|--------|---------|-----------|--------|----|------|
| satellite | 0.778 | 1.000 | 0.556 | 0.714 | ✓ 达标 |
| uav | 0.950 | 1.000 | 0.900 | 0.947 | ✓ 达标 |

### P4：大图预测检测框统计

| 数据集 | Site1 | Site2_1 | Site2_2 | Site2_3 | Site3_1 | Site3_2 | 合计 |
|--------|-------|---------|---------|---------|---------|---------|------|
| satellite | 3006 | 1278 | 2987 | 3454 | 6069 | 3263 | **20,057** |
| uav | 7022 | 2036 | 6864 | 9639 | 5245 | 5467 | **36,273** |

### P5：分辨率影响分析（mAP@0.5）

| 数据集 | 100% | 75% | 50% | 25% |
|--------|------|-----|-----|-----|
| satellite | 0.667 | 0.446 | 0.645 | 0.556 |
| uav | 0.950 | 0.877 | **0.957** | 0.897 |

### P6：多模型对比

**Satellite**

| 模型 | mAP@0.5 | Precision | Recall | F1 | FPS |
|------|---------|-----------|--------|----|-----|
| YOLOv8 | **0.778** | 1.000 | 0.556 | 0.714 | 106 |
| YOLOv5 | 0.630 | 0.987 | 0.444 | 0.613 | 86 |
| U-Net | — | 0.600 | 0.333 | 0.429 | 108 |
| FCN | — | 0.182 | 0.444 | 0.258 | 171 |

**UAV**

| 模型 | mAP@0.5 | Precision | Recall | F1 | FPS |
|------|---------|-----------|--------|----|-----|
| YOLOv8 | 0.950 | 1.000 | 0.900 | 0.947 | 114 |
| YOLOv5 | **0.986** | 1.000 | 0.972 | 0.986 | 104 |
| U-Net | — | 0.357 | 1.000 | 0.526 | 248 |
| FCN | — | 0.283 | 0.975 | 0.438 | 419 |

> U-Net/FCN 为分割模型，mAP 通过 mask→bbox 后处理计算，与检测模型不可直接比较。

---

## 最优模型权重

| 模型 | 路径 | mAP@0.5 |
|------|------|---------|
| YOLOv8 satellite | `runs/satellite/yolov8/satellite_yolov83/weights/best.pt` | 0.778 |
| YOLOv8 uav | `runs/uav/yolov8/uav_yolov82/weights/best.pt` | 0.950 |
| YOLOv5 satellite | `runs/satellite/yolov5/satellite_yolov5n/weights/best.pt` | 0.630 |
| YOLOv5 uav | `runs/uav/yolov5/uav_yolov5n/weights/best.pt` | 0.986 |
| U-Net satellite | `runs/satellite/unet/satellite_unet/best.pt` | — |
| U-Net uav | `runs/uav/unet/uav_unet/best.pt` | — |
| FCN satellite | `runs/satellite/fcn/satellite_fcn/best.pt` | — |
| FCN uav | `runs/uav/fcn/uav_fcn/best.pt` | — |
