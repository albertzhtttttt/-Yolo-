# 交付记录

> 说明：当前仓库中的论文风格可视化结果，以 `tmp/visual_review_paper_style_20260413/images/results/` 为人工确认后的最终版本，并已同步覆盖到正式 `results/` 目录。

---

## 2026-03-18 | P0 + P1：环境配置与数据准备

### P0：环境与配置

| 文件 | 说明 |
|------|------|
| `scripts/install_env.sh` | 服务器一键安装所有依赖（固定版本号） |
| `scripts/init_dirs.sh` | 创建完整项目目录结构 |
| `configs/satellite_config.yaml` | 卫星数据集所有参数（路径/预处理/增强/训练/评估） |
| `configs/uav_config.yaml` | 无人机数据集所有参数 |
| `configs/predict_config.yaml` | 大图预测参数（按 satellite/uav 拆分位置、切片参数、输出格式） |
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
| `src/train/train_yolov8.py` | YOLOv8 训练主脚本，支持 `--dataset satellite/uav`，命令行覆盖超参数，断点续训，关闭 Ultralytics 内置增强（使用离线增强数据）；支持加载 CBAM 自定义结构 |
| `src/train/yolo_model_loader.py` | 自定义 YOLO 模型加载器，负责注册 CBAM 模块并兼容自定义 YAML / 权重加载 |
| `configs/yolov8_gap_cbam_p2.yaml` | UAV 使用的 CBAM 自定义 YOLOv8 结构配置 |

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

### P2/P3 最新评估结果

| 数据集 | mAP@0.5 | Precision | Recall | F1 | 状态 |
|--------|---------|-----------|--------|----|------|
| satellite | 0.418 | 0.920 | 0.222 | 0.358 | 需改进 |
| uav | 0.974 | 0.974 | 0.950 | 0.962 | ✓ 达标 |

当前仅 UAV 数据集通过 mAP@0.5 ≥ 0.70 关卡；satellite 结果低于早期文档记录，后续总结需以最新评估输出为准。

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
    --weights runs/uav/yolov8/uav_yolov8_cbam/weights/best.pt
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
├── metrics_summary_satellite.csv 卫星数据集模型指标汇总表
├── metrics_summary_uav.csv       无人机数据集模型指标汇总表
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
| satellite | 6 | 23,712 | KML / KMZ / SHP |
| uav | 6 | 47,196 | KML / KMZ / SHP |

各位置检测框数量（satellite）：Site1=2704, Site2_1=1409, Site2_2=4046, Site2_3=3941, Site3_1=7785, Site3_2=3827

各位置检测框数量（uav）：Site1=13036, Site2_1=3407, Site2_2=7721, Site2_3=9594, Site3_1=7616, Site3_2=5822

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
| `src/data_preparation/build_resolution_datasets.py` | 构建 `100% / 50% / 25% / 12.5% / 6.25% / 3.125%` 六个分辨率数据集 |
| `scripts/run_resolution_pipeline.py` | 一键构建→训练→评估流水线，支持 `--device` 指定 GPU |
| `src/visualize/plot_resolution_analysis.py` | 六档分辨率影响折线图与柱状图 |

### 实验结果

**主实验（单次运行，mAP@0.5）**

| 数据集 | 100% | 50% | 25% | 12.5% | 6.25% | 3.125% |
|--------|------|-----|-----|-------|------|--------|
| satellite | 0.6668 | 0.5837 | 0.6667 | 0.0000 | 0.6111 | 0.0000 |
| uav | 0.9795 | 0.9890 | 0.9812 | 0.9855 | 0.9344 | 0.9160 |

**结论补充**

- UAV 在 6 个分辨率档位上都保持较高精度，对分辨率退化更鲁棒。
- Satellite 在 `12.5%` 以下进入明显不稳定区，单次结果会出现 `0 → 0.6111 → 0` 的非单调跳动。
- 该异常已在同日追加的 `satellite_lowres_multiseed` 实验中进一步验证，不宜直接把单次 `6.25%` 高值解释为稳定收益。

### 输出目录

```
results/phase2_resolution/
├── metrics_by_scale.csv
├── satellite/scale_{100,50,25,12.5,6.25,3.125}/metrics.csv
├── uav/scale_{100,50,25,12.5,6.25,3.125}/metrics.csv
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
| YOLOv8 | 0.418 | 0.920 | 0.222 | 0.358 | 106.2 |
| YOLOv5 | **0.630** | 0.987 | 0.444 | 0.613 | 107.0 |
| U-Net  | 0.196 | 0.667 | 0.444 | 0.533 | 119.5 |
| FCN    | 0.064 | 0.333 | 0.333 | 0.333 | 186.7 |

**UAV：**

| 模型 | mAP@0.5 | Precision | Recall | F1 | FPS |
|------|---------|-----------|--------|----|-----|
| YOLOv8 | 0.974 | 0.974 | 0.950 | 0.962 | 112.9 |
| YOLOv5 | **0.986** | 1.000 | 0.972 | 0.986 | 87.4 |
| U-Net  | 0.210 | 0.274 | 1.000 | 0.430 | 155.1 |
| FCN    | 0.267 | 0.333 | 0.975 | 0.497 | 337.6 |

> 注：U-Net/FCN 为分割模型，mAP 通过 mask→bbox 后处理近似计算，仅可作为同一后处理流程下的参考值，不宜与检测模型直接等价比较。

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

## 2026-04-15 | P5：Satellite 低分辨率多 seed 复现实验（完成）

### 交付文件

| 文件 | 说明 |
|------|------|
| `src/train/train_yolov8.py` | 新增 `--seed` 参数，允许在不改数据划分的前提下覆盖训练随机种子 |
| `scripts/run_satellite_lowres_multiseed.py` | 低分辨率 `12.5% / 6.25% / 3.125%` 的多 seed 训练、评估、汇总与绘图脚本 |
| `results/phase2_resolution/satellite_lowres_multiseed/metrics_by_seed.csv` | 9 组 `scale × seed` 原始指标明细 |
| `results/phase2_resolution/satellite_lowres_multiseed/metrics_seed_summary.csv` | 3 个低分辨率档位的 mean/std 汇总表 |
| `results/phase2_resolution/satellite_lowres_multiseed/lowres_multiseed_map50_errorbar.png` | `mAP@0.5 mean ± std` 误差棒图 |
| `results/phase2_resolution/satellite_lowres_multiseed/lowres_multiseed_metrics.png` | `mAP@0.5 / Precision / Recall / F1` 多指标误差棒图 |

### 关键结果

| Scale | seed 42 | seed 43 | seed 44 | mAP@0.5 mean ± std |
|-------|---------|---------|---------|--------------------|
| 12.5% | 0.0000 | 0.4186 | 0.0000 | 0.1395 ± 0.2417 |
| 6.25% | 0.6111 | 0.0000 | 0.0000 | 0.2037 ± 0.3528 |
| 3.125% | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |

### 实验结论

- 先前 `satellite` 在极低分辨率端出现的 `0 → 0.6111 → 0`，在 3 个 seed 下并未表现为稳定规律，更接近低分辨率端的高波动现象。
- `6.25%` 的单次高值并不稳健：只有 `seed=42` 命中了少量目标，`seed=43/44` 都回到 `0.0000`。
- `3.125%` 在 3 个 seed 上均为 `0.0000`，说明该档位下卫星林窗检测已基本失效。
- 结合 `satellite` test 集仅 `6` 张图、`9` 个目标，这组结果更支持“低分辨率 + 小样本评估导致单次结果不稳定”的解释，而不是实验串档。

---

## 2026-04-15 | P7：YOLOv8 CBAM 消融实验扩展到卫星数据集（完成）

### 交付文件

| 文件 | 说明 |
|------|------|
| `configs/satellite_cbam_config.yaml` | 卫星数据集专用的 YOLOv8n + CBAM 训练配置，保持与 baseline 相同的数据与增强设置 |
| `scripts/run_cbam_ablation.py` | 正式的 YOLOv8 baseline vs CBAM 消融实验素材生成脚本，现同时支持 UAV 与 satellite |
| `scripts/run_train_pipeline.py` | 新增 `--config` 参数，支持用专用配置文件运行卫星 CBAM 训练流水线 |
| `results/phase5_cbam_ablation/cbam_ablation_uav.csv` | UAV baseline 与 CBAM 的正式消融汇总表 |
| `results/phase5_cbam_ablation/cbam_ablation_satellite.csv` | 卫星 baseline 与 CBAM 的正式消融汇总表 |
| `results/phase5_cbam_ablation/cbam_ablation_uav_bar.png` | UAV CBAM 消融定量柱状图 |
| `results/phase5_cbam_ablation/cbam_ablation_satellite_bar.png` | 卫星 CBAM 消融定量柱状图 |
| `results/phase5_cbam_ablation/cbam_ablation_uav_qualitative.png` | UAV 的 GT / baseline / CBAM 定性对比图 |
| `results/phase5_cbam_ablation/cbam_ablation_satellite_qualitative.png` | 卫星的 GT / baseline / CBAM 定性对比图 |

### 关键结果

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

### 实验结论

- UAV 数据集上，CBAM 带来稳定增益，尤其 `mAP@0.5:0.95` 从 `0.709` 提升到 `0.801`，适合作为论文中“注意力机制有效”的主证据。
- Satellite 数据集上，CBAM 将 `Recall` 从 `0.333` 提升到 `0.556`，`mAP@0.5` 从 `0.584` 提升到 `0.605`，`F1` 从 `0.496` 提升到 `0.657`，说明其主要收益来自减少漏检。
- 但 Satellite 上 `Precision` 与 `mAP@0.5:0.95` 同时下降，表明高 IoU 下的定位稳定性仍弱于 baseline，论文表述宜突出“召回增强”而非“全面提升”。

### 本次同步说明

- `tmp/visual_review_paper_style_20260413/images/results/` 中人工确认后的论文风格图，已覆盖同步到正式 `results/` 目录。
- `results/phase2_resolution/` 与 `results/phase3_comparison/` 下的论文图现以当前手工确认版本为准。
- `results/phase5_cbam_ablation/` 作为正式补充实验目录，后续论文引用统一从该目录取图取表。
- 卫星消融已改为严格对等设置：baseline 使用同规模 `YOLOv8n` 权重 `runs/satellite/yolov8/satellite_yolov8n_baseline/weights/best.pt`，避免与更大规模基线混用导致结论失真。

---
