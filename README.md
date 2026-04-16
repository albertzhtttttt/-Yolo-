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
│   ├── satellite_config.yaml   # 卫星 baseline 数据集参数
│   ├── satellite_cbam_config.yaml # 卫星 CBAM 训练参数
│   ├── uav_config.yaml         # 无人机数据集参数
│   └── predict_config.yaml     # 大图预测参数（按 satellite/uav 拆分位置）
├── scripts/                    # 一键运行脚本
│   ├── run_data_pipeline.py    # P1 数据准备流水线
│   ├── run_train_pipeline.py   # P2/P3 训练+评估流水线
│   ├── run_predict.py          # P4 批量大图预测
│   ├── run_resolution_pipeline.py  # P5 分辨率实验流水线
│   ├── run_satellite_lowres_multiseed.py # Satellite 低分辨率多 seed 复现实验
│   ├── run_history_predict.py  # 历史时序批量预测
│   └── run_cbam_ablation.py    # UAV / Satellite CBAM 消融实验素材生成
├── src/
│   ├── data_preparation/       # 数据预处理、增强、数据集构建
│   ├── train/                  # 训练脚本（YOLOv8/YOLOv5/U-Net/FCN）
│   ├── evaluate/               # 评估脚本
│   ├── predict/                # 大图预测
│   └── visualize/              # 可视化
├── data/                       # 数据集（不纳入版本控制）
├── runs/                       # 训练输出（不纳入版本控制）
├── tmp/                        # 临时脚本、论文拼图草稿与中间结果
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
    --weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt
```

### P4：大图预测（按数据集拆分位置，UAV 当前为福田保护区全域）

```bash
python scripts/run_predict.py --dataset satellite \
    --weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt

python scripts/run_predict.py --dataset uav \
    --weights runs/uav/yolov8/uav_yolov8_cbam/weights/best.pt
```

输出：`results/phase1_predict/{satellite|uav}/{位置名}/detections.kml/.shp`

说明：
- UAV 当前预测位置为 `福田保护区全域/shidian.tif`
- 预测阶段优先使用 GeoTIFF 内嵌空间参考，避免误用旧的外部 `.tfw/.prj`

### P5：分辨率影响分析

```bash
# 卫星（GPU 0），无人机（GPU 1）并行
python scripts/run_resolution_pipeline.py --dataset satellite --device 0
python scripts/run_resolution_pipeline.py --dataset uav --device 1

# 卫星低分辨率多 seed 复现实验
python scripts/run_satellite_lowres_multiseed.py --device 0 --seeds 42 43 44
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
    --yolov8_weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt \
    --yolov5_weights runs/satellite/yolov5/satellite_yolov5n/weights/best.pt \
    --unet_weights   runs/satellite/unet/satellite_unet/best.pt \
    --fcn_weights    runs/satellite/fcn/satellite_fcn/best.pt
```

### 补充实验：YOLOv8 CBAM 消融（UAV / Satellite）

```bash
# UAV
python scripts/run_cbam_ablation.py \
    --dataset uav \
    --baseline_weight runs/uav/yolov8/uav_yolov8/weights/best.pt \
    --cbam_weight runs/uav/yolov8/uav_yolov8_cbam/weights/best.pt \
    --device 0

# Satellite
python src/train/train_yolov8.py --dataset satellite \
    --config configs/satellite_cbam_config.yaml --device 0
python src/train/train_yolov8.py --dataset satellite \
    --name satellite_yolov8n_baseline --device 0
python src/evaluate/evaluate.py --dataset satellite \
    --weights runs/satellite/yolov8/satellite_yolov8n_baseline/weights/best.pt \
    --output_dir results/phase5_cbam_ablation/satellite_baseline
python src/evaluate/evaluate.py --dataset satellite \
    --config configs/satellite_cbam_config.yaml \
    --weights runs/satellite/yolov8/satellite_yolov8_cbam/weights/best.pt \
    --output_dir results/phase5_cbam_ablation/satellite_cbam
python scripts/run_cbam_ablation.py \
    --dataset satellite \
    --baseline_weight runs/satellite/yolov8/satellite_yolov8n_baseline/weights/best.pt \
    --cbam_weight runs/satellite/yolov8/satellite_yolov8_cbam/weights/best.pt \
    --device 0
```

输出：`results/phase5_cbam_ablation/`

说明：
- UAV 定量结果：`cbam_ablation_uav.csv`
- Satellite 定量结果：`cbam_ablation_satellite.csv`
- Satellite 消融统一使用同规模 `YOLOv8n baseline`（`satellite_yolov8n_baseline`）对比 `YOLOv8n + CBAM`，避免与更大模型基线混用。
- 论文柱状图：`cbam_ablation_{uav|satellite}_bar.png/.pdf`
- 定性对比图：`cbam_ablation_{uav|satellite}_qualitative.png/.pdf`

---

## 主要结果

### P2/P3：YOLOv8 基准模型

| 数据集 | mAP@0.5 | Precision | Recall | F1 | 状态 |
|--------|---------|-----------|--------|----|------|
| satellite | 0.418 | 0.920 | 0.222 | 0.358 | 需改进 |
| uav | 0.974 | 0.974 | 0.950 | 0.962 | ✓ 达标 |

### P4：大图预测检测框统计

| 数据集 | 位置 | 检测框数量 | 说明 |
|--------|------|------------|------|
| satellite | 位置1~位置3_2 | 23,712 | 卫星 6 个位置汇总 |
| uav | 福田保护区全域 | 20,056 | `shidian.tif` 重建后的正式结果 |

### P5：分辨率影响分析（mAP@0.5）

| 数据集 | 100% | 50% | 25% | 12.5% | 6.25% | 3.125% |
|--------|------|-----|-----|-------|------|--------|
| satellite | 0.6668 | 0.5837 | 0.6667 | 0.0000 | 0.6111 | 0.0000 |
| uav | 0.9795 | 0.9890 | 0.9812 | 0.9855 | 0.9344 | 0.9160 |

**Satellite 低分辨率多 seed 复现实验（3 seeds）**

| Scale | seed 42 | seed 43 | seed 44 | mAP@0.5 mean ± std |
|-------|---------|---------|---------|--------------------|
| 12.5% | 0.0000 | 0.4186 | 0.0000 | 0.1395 ± 0.2417 |
| 6.25% | 0.6111 | 0.0000 | 0.0000 | 0.2037 ± 0.3528 |
| 3.125% | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |

结论：
- `satellite` 在极低分辨率端的 `0 → 0.6111 → 0` 不是稳定规律，更像低分辨率端训练随机性与小测试集共同导致的高波动现象。
- `6.25%` 并未稳定高于 `12.5%`，单次高值主要来自某个 seed 偶发命中少量目标。
- `3.125%` 在 3 个 seed 上均为 `0.0000`，说明该档位下卫星林窗检测已基本失效。

### P6：多模型对比

**Satellite**

| 模型 | mAP@0.5 | Precision | Recall | F1 | FPS |
|------|---------|-----------|--------|----|-----|
| YOLOv8 | 0.418 | 0.920 | 0.222 | 0.358 | 106.2 |
| YOLOv5 | **0.630** | 0.987 | 0.444 | 0.613 | 107.0 |
| U-Net | 0.196 | 0.667 | 0.444 | 0.533 | 119.5 |
| FCN | 0.064 | 0.333 | 0.333 | 0.333 | 186.7 |

**UAV**

| 模型 | mAP@0.5 | Precision | Recall | F1 | FPS |
|------|---------|-----------|--------|----|-----|
| YOLOv8 | 0.974 | 0.974 | 0.950 | 0.962 | 112.9 |
| YOLOv5 | **0.986** | 1.000 | 0.972 | 0.986 | 87.4 |
| U-Net | 0.210 | 0.274 | 1.000 | 0.430 | 155.1 |
| FCN | 0.267 | 0.333 | 0.975 | 0.497 | 337.6 |

> U-Net/FCN 为分割模型，mAP 通过 mask→bbox 后处理近似计算，仅可作为同一后处理流程下的参考值，不宜与检测模型直接等价比较。

### 补充实验：CBAM 消融（UAV / Satellite）

**UAV**

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | F1 | Params(M) | FLOPs(G) | FPS |
|------|-----------|--------|---------|--------------|----|-----------|----------|-----|
| YOLOv8n | 0.974 | 0.950 | 0.974 | 0.709 | 0.962 | 3.01 | 8.1 | 58.4 |
| YOLOv8n + CBAM | **1.000** | **0.964** | **0.987** | **0.801** | **0.982** | 3.20 | 8.3 | 46.9 |

**Satellite**

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | F1 | Params(M) | FLOPs(G) | FPS |
|------|-----------|--------|---------|--------------|----|-----------|----------|-----|
| YOLOv8n | **0.971** | 0.333 | 0.584 | **0.398** | 0.496 | 3.01 | 8.1 | 47.1 |
| YOLOv8n + CBAM | 0.804 | **0.556** | **0.605** | 0.193 | **0.657** | 3.20 | 8.3 | 56.1 |

结论：
- CBAM 在 UAV 测试集上带来稳定增益，其中 `mAP@0.5:0.95` 绝对提升 `0.092`，对论文更有说服力。
- 在 Satellite 测试集上，CBAM 将 `Recall` 从 `0.333` 提升到 `0.556`，`mAP@0.5` 从 `0.584` 小幅提升到 `0.605`，`F1` 从 `0.496` 提升到 `0.657`；但 `Precision` 与 `mAP@0.5:0.95` 下降，说明其收益主要体现在更积极地召回目标，而高 IoU 下的定位稳定性仍需进一步优化。
- 对应图表与汇总表位于 `results/phase5_cbam_ablation/`。

---

## 最优模型权重

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
