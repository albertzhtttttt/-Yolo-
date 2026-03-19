# 交付记录

---

## 2026-03-18 | P0 + P1：环境配置与数据准备

### P0：环境与配置

| 文件 | 说明 |
|------|------|
| `scripts/install_env.sh` | 服务器一键安装所有依赖（固定版本号） |
| `scripts/init_dirs.sh` | 创建完整项目目录结构 |
| `configs/satellite_config.yaml` | 卫星数据集所有参数（路径/预处理/增强/训练/评估） |
| `configs/uav_config.yaml` | 无人机数据集所有参数 |
| `configs/predict_config.yaml` | 大图预测参数（6个TIF位置、切片参数、输出格式） |
| `src/utils/plot_utils.py` | 统一图表风格（所有标签使用英文）、PR曲线/混淆矩阵绘图函数，DPI≥300 |
| `src/utils/logger.py` | 统一日志（同时输出控制台+文件，带时间戳） |

### P1：数据准备

| 文件 | 说明 |
|------|------|
| `src/data_preparation/preprocess_satellite.py` | 卫星图像 256→640 缩放，标注直接复制（归一化坐标不变） |
| `src/data_preparation/preprocess_uav.py` | 无人机滑动窗口切片（640×640，步长320），坐标变换+框截断+负样本采样 |
| `src/data_preparation/augment.py` | 所有增强操作（翻转/旋转/缩放/HSV/噪声/模糊/Mosaic），标注框同步变换 |
| `src/data_preparation/build_dataset.py` | 按原图划分 70/15/15，无人机按原图组分组防泄露，生成 dataset.yaml |
| `scripts/run_data_pipeline.py` | 一键运行全流程（预处理→划分→增强），支持 `--dataset satellite/uav/all` |

### 服务器运行命令

```bash
# 1. 安装依赖
bash scripts/install_env.sh

# 2. 创建目录结构
bash scripts/init_dirs.sh

# 3. 一键执行 P1 全流程（含可视化统计图）
python scripts/run_data_pipeline.py --dataset all --vis
```

---

## 2026-03-18 | P2 + P3：YOLOv8 训练与评估

### P2：训练脚本

| 文件 | 说明 |
|------|------|
| `src/train/train_yolov8.py` | YOLOv8 训练主脚本，支持 `--dataset satellite/uav`，命令行覆盖超参数，断点续训，关闭 Ultralytics 内置增强（使用离线增强数据） |

### P3：评估与可视化脚本

| 文件 | 说明 |
|------|------|
| `src/evaluate/evaluate.py` | 完整评估脚本：test 集指标（P/R/mAP/F1）、训练曲线、混淆矩阵、PR 曲线、F1-Confidence 曲线、测试集预测可视化（GT vs 预测框对比）；所有图表标签使用英文 |

### 一键入口

| 文件 | 说明 |
|------|------|
| `scripts/run_train_pipeline.py` | 训练→评估全流程一键脚本，自动查找 best.pt，支持 `--eval_only` 仅评估模式 |
| `scripts/train_bg.sh` | 后台运行训练流程，日志写入 `logs/train_bg_<timestamp>.log` |

### 服务器运行命令

```bash
# 完整流程（训练 + 评估，两个数据集串行，后台运行）
bash scripts/train_bg.sh --dataset all

# 指定更大模型和更多轮数
bash scripts/train_bg.sh --dataset satellite --model yolov8s --epochs 200

# 训练完成后仅重新评估（不重新训练）
bash scripts/train_bg.sh --dataset satellite --eval_only \
    --weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt

# 实时查看日志
tail -f logs/train_bg_<timestamp>.log
```

### 评估输出目录

```
results/phase1_training/{satellite|uav}/
├── metrics.csv           数值指标（P/R/mAP50/mAP50-95/F1）
├── training_curves.png   训练过程曲线（Loss + mAP）
├── confusion_matrix.png  混淆矩阵（归一化）
├── pr_curve.png          Precision-Recall 曲线
├── f1_curve.png          F1-Confidence 曲线
└── predictions/          测试集预测可视化（GT红色虚线 vs 预测绿色实线）
```

---

## 2026-03-19 | 验收结果 & 下一步执行计划

### P2/P3 验收结果（关卡已通过）

| 数据集 | mAP@0.5 | Precision | Recall | F1 | 状态 |
|--------|---------|-----------|--------|----|------|
| satellite | 0.778 | — | — | — | ✓ 达标 |
| uav | 0.950 | — | — | — | ✓ 达标 |

两个模型均通过 mAP@0.5 ≥ 0.70 关卡，P3/P4/P5/P6 可同时推进。

---

## 下一步：P4 + P5 + P6 并行执行

### P4：GeoTIFF 大图预测（最高优先级）

| 文件 | 说明 |
|------|------|
| `src/predict/predict_geotiff.py` | 滑动窗口推理大图，Plan A：rasterio读取+坐标转换→KML/SHP；Plan B：切片JPEG收集 |
| `scripts/run_predict.py` | 一键对6个TIF位置批量预测，支持 `--dataset satellite/uav/all` |

#### 服务器运行命令

```bash
# 卫星模型预测全部6个位置
python scripts/run_predict.py --dataset satellite \
    --weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt

# 无人机模型预测全部6个位置
python scripts/run_predict.py --dataset uav \
    --weights runs/uav/yolov8/uav_yolov8/weights/best.pt
```

#### 输出目录

```
results/phase1_predict/{satellite|uav}/{位置名}/
├── detections.kml        检测框（KML格式，Plan A）
├── detections.shp        检测框（SHP格式，Plan A）
├── summary.csv           各位置检测框数量统计
└── tiles/                含检测结果的切片图（Plan B备用）
```

---

### P5：分辨率影响分析

| 文件 | 说明 |
|------|------|
| `src/data_preparation/build_resolution_datasets.py` | 对原始图像下采样至100%/75%/50%/25%后统一缩放回640×640 |
| `scripts/run_resolution_pipeline.py` | 一键构建4个分辨率数据集并串行训练 |
| `src/visualize/plot_resolution_analysis.py` | 各指标随分辨率变化折线图、检测结果对比拼图 |

#### 服务器运行命令

```bash
# 构建多分辨率数据集并训练（GPU串行，耗时较长）
bash scripts/run_resolution_pipeline.sh --dataset satellite
bash scripts/run_resolution_pipeline.sh --dataset uav
```

#### 输出目录

```
results/phase2_resolution/
├── metrics_by_scale.csv          各分辨率指标汇总
├── resolution_analysis.png       指标随分辨率变化折线图
└── detection_comparison.png      4分辨率检测结果横向对比
```

---

### P6：对比实验（YOLOv5 / U-Net / FCN）

| 文件 | 说明 |
|------|------|
| `src/data_preparation/yolo_to_mask.py` | YOLO框→二值掩码，为U-Net/FCN生成分割标签 |
| `src/train/train_yolov5.py` | YOLOv5训练脚本，对齐超参数 |
| `src/train/train_unet.py` | U-Net二值分割，BCE+Dice Loss，后处理提取检测框 |
| `src/train/train_fcn.py` | FCN分割，同U-Net后处理策略 |
| `src/evaluate/compare_models.py` | 统一评估：P/R/mAP/F1/FPS/Params/FLOPs |
| `src/visualize/plot_comparison.py` | 对比柱状图、速度-精度散点图、雷达图 |

#### 输出目录

```
results/phase3_comparison/
├── metrics_summary.csv           所有模型指标汇总表
├── comparison_bar.png            各模型指标对比柱状图
├── speed_accuracy.png            速度 vs 精度散点图
└── radar_chart.png               综合性能雷达图
```

---

### 执行优先级建议

1. **P4 优先**：验证实际预测价值，输出KML/SHP可直接用于论文
2. **P5 并行**：分辨率实验耗时长，尽早启动
3. **P6 最后**：对比实验依赖P4/P5结论，且训练量最大

---

## 2026-03-19 | P4：GeoTIFF 大图预测（完成）

### 交付文件

| 文件 | 说明 |
|------|------|
| `src/predict/predict_geotiff.py` | 滑动窗口推理（640×640，步长320），rasterio 窗口读取避免 OOM，全局 NMS 去重，输出 KML/KMZ/SHP |
| `scripts/run_predict.py` | 批量对6个TIF位置预测，生成各位置检测框统计图 |
| `src/predict/__init__.py` | 模块初始化 |

### 预测结果

| 数据集 | 位置数 | 总检测框 | 输出格式 |
|--------|--------|----------|----------|
| satellite | 6 | 20,057 | KML / KMZ / SHP |
| uav | 6 | 36,273 | KML / KMZ / SHP |

各位置检测框数量（satellite）：Site1=3006, Site2_1=1278, Site2_2=2987, Site2_3=3454, Site3_1=6069, Site3_2=3263

各位置检测框数量（uav）：Site1=7022, Site2_1=2036, Site2_2=6864, Site2_3=9639, Site3_1=5245, Site3_2=5467

### 输出目录

```
results/phase1_predict/{satellite|uav}/{位置名}/
├── detections.kml / detections.kmz
├── detections.shp / detections.dbf / detections.shx
└── summary_bar.png
```

---

## 2026-03-19 | P5：分辨率影响分析（完成）

### 交付文件

| 文件 | 说明 |
|------|------|
| `src/data_preparation/build_resolution_datasets.py` | 构建100%/75%/50%/25%四个分辨率数据集 |
| `scripts/run_resolution_pipeline.py` | 一键构建→训练→评估流水线，支持 `--device` 指定GPU |
| `src/visualize/plot_resolution_analysis.py` | 分辨率影响折线图与柱状图 |

### 实验结果

**Satellite mAP@0.5 vs 分辨率：**

| Scale | mAP@0.5 | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| 100% | 0.667 | 1.000 | 0.439 | 0.610 |
| 75%  | 0.446 | 0.746 | 0.333 | 0.461 |
| 50%  | 0.645 | 0.800 | 0.444 | 0.571 |
| 25%  | 0.556 | 0.750 | 0.333 | 0.462 |

**UAV mAP@0.5 vs 分辨率：**

| Scale | mAP@0.5 | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| 100% | 0.950 | 1.000 | 0.900 | 0.947 |
| 75%  | 0.877 | 0.945 | 0.775 | 0.852 |
| 50%  | 0.957 | 0.949 | 0.925 | 0.937 |
| 25%  | 0.897 | 0.982 | 0.800 | 0.882 |

### 输出目录

```
results/phase2_resolution/
├── metrics_by_scale.csv
├── resolution_analysis.png
└── resolution_map50_bar.png
```

---

## 2026-03-20 | P6：对比实验（YOLOv5 / U-Net / FCN）（完成）

### 交付文件

| 文件 | 说明 |
|------|------|
| `src/data_preparation/yolo_to_mask.py` | YOLO bbox 转二值掩码，为分割模型生成标签 |
| `src/train/train_yolov5.py` | YOLOv5n 训练，对齐超参数，150 epochs |
| `src/train/train_unet.py` | U-Net（4层编解码），BCE+Dice Loss，100 epochs，batch=2 |
| `src/train/train_fcn.py` | FCN-8s（VGG16 backbone），BCE+Dice Loss，100 epochs |
| `src/evaluate/compare_models.py` | 统一评估：P/R/mAP/F1/FPS/Params |
| `src/visualize/plot_comparison.py` | 对比柱状图、速度-精度散点图、雷达图 |

### 对比结果

**Satellite：**

| 模型 | mAP@0.5 | Precision | Recall | F1 | FPS |
|------|---------|-----------|--------|----|-----|
| YOLOv8 | **0.778** | 1.000 | 0.556 | 0.714 | 106 |
| YOLOv5 | 0.630 | 0.987 | 0.444 | 0.613 | 86 |
| U-Net  | — | 0.600 | 0.333 | 0.429 | 108 |
| FCN    | — | 0.182 | 0.444 | 0.258 | 171 |

**UAV：**

| 模型 | mAP@0.5 | Precision | Recall | F1 | FPS |
|------|---------|-----------|--------|----|-----|
| YOLOv8 | 0.950 | 1.000 | 0.900 | 0.947 | 114 |
| YOLOv5 | **0.986** | 1.000 | 0.972 | 0.986 | 104 |
| U-Net  | — | 0.357 | 1.000 | 0.526 | 248 |
| FCN    | — | 0.283 | 0.975 | 0.438 | 419 |

> 注：U-Net/FCN 为分割模型，mAP 通过 mask→bbox 后处理计算，与检测模型不可直接比较。

### 输出目录

```
results/phase3_comparison/
├── metrics_summary_satellite.csv
├── metrics_summary_uav.csv
├── comparison_bar.png
├── speed_accuracy.png
└── radar_chart.png
```

---
