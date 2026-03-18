# 执行计划 TODO

**基于：** `vibe-docs/requirements.md` v1.2
**更新日期：** 2026-03-18

---

## 整体依赖关系图

```
 [P0 环境与配置] ────────────────────────────────────────────────────────────────►
        │
        ▼
 [P1 数据准备]
   ┌─────────────────────────┬─────────────────────────┐
   │  P1-SAT 卫星数据处理     │  P1-UAV 无人机数据处理   │  ← 【并行】
   └─────────────────────────┴─────────────────────────┘
        │                               │
        ▼                               ▼
 [P2 YOLOv8 训练（迭代）]
   ┌─────────────────────────┬─────────────────────────┐
   │  P2-SAT 训练 model_sat  │  P2-UAV 训练 model_uav  │  ← 【并行，GPU资源允许时】
   └─────────────────────────┴─────────────────────────┘
        │                               │
        ▼                               ▼
   ████████████████████████████████████████████
   █  关卡：两个模型均达标（mAP@0.5 ≥ 0.70）  █
   ████████████████████████████████████████████
        │
        ├────────────────┬────────────────┬────────────────┐
        ▼                ▼                ▼                ▼
   [P3 评估可视化]  [P4 大图预测]  [P5 分辨率分析]  [P6 对比实验]
   (并行:sat/uav)  (并行:sat/uav)  (并行:sat/uav)  (并行:各模型)
        │                │                │                │
        └────────────────┴────────────────┴────────────────┘
                                  │
                                  ▼
                         [P7 综合结果整理]
```

> **说明：**
> - P0、P1、P2 必须严格串行（前置依赖）
> - P3、P4、P5、P6 均依赖 P2 达标，四者之间**完全独立，可同时推进**
> - 每个阶段内，卫星（SAT）和无人机（UAV）分支**始终可并行**
> - 标注 `【本地】` 的任务在开发机编写代码；标注 `【服务器】` 的任务传输到实验室 GPU 服务器执行

---

## P0：环境与基础配置

> 前置条件：无。所有后续工作的基础。【本地编写，服务器验证】

- [ ] **P0.1** 编写服务器环境安装脚本 `scripts/install_env.sh`
  - 安装 ultralytics、rasterio/GDAL（方案A）、simplekml、pyshp、matplotlib、seaborn、pandas
  - 固定版本号，保证可复现
- [ ] **P0.2** 创建项目目录结构脚本 `scripts/init_dirs.sh`
  - 创建 `src/`、`runs/`、`results/` 等全部目录
- [ ] **P0.3** 编写全局配置文件
  - `configs/satellite_config.yaml`：卫星数据集超参数、路径
  - `configs/uav_config.yaml`：无人机数据集超参数、路径
  - `configs/predict_config.yaml`：大图预测参数（置信度阈值、NMS等）
- [ ] **P0.4** 编写 matplotlib 中文字体工具模块 `src/utils/plot_utils.py`
  - 统一图表风格，支持中文标题，DPI≥300
- [ ] **P0.5** 编写日志工具模块 `src/utils/logger.py`
  - 统一日志格式，同时输出到控制台和文件

**P0 完成标志：** 服务器上 `python -c "import ultralytics, rasterio, simplekml"` 无报错

---

## P1：数据准备

> 前置条件：P0 完成。【本地编写脚本 → 服务器执行】
> P1-SAT 与 P1-UAV **可完全并行执行**。

### P1-SAT：卫星数据处理

- [ ] **P1-SAT.1** 编写卫星数据预处理脚本 `src/data_preparation/preprocess_satellite.py`
  - 将 256×256 图像缩放至 640×640（双线性插值）
  - 标注坐标随缩放比例同步更新
  - 输出统计：各图像目标框数量分布图
- [ ] **P1-SAT.2** 编写卫星数据增强脚本 `src/data_preparation/augment.py`（支持 `--dataset satellite`）
  - 水平/垂直翻转、随机旋转90°×4、随机缩放（±20%）
  - HSV 色彩抖动、高斯噪声、随机模糊
  - Mosaic 拼接（4图合一，同步处理标注框）
  - 仅对 train 集执行，增强倍数可配置
- [ ] **P1-SAT.3** 编写数据集构建脚本 `src/data_preparation/build_dataset.py`（支持 `--dataset satellite`）
  - 按原始图像划分 train 70% / val 15% / test 15%（固定 seed=42）
  - 生成 `data/yolo_dataset_satellite/dataset.yaml`
  - 输出划分统计：各子集图像数、标注框数
- [ ] **P1-SAT.4** 【服务器执行】运行上述脚本，验证输出
  - 检查 `data/yolo_dataset_satellite/` 目录完整性
  - 抽查增强样本可视化（标注框是否正确变换）

### P1-UAV：无人机数据处理

- [ ] **P1-UAV.1** 编写无人机切片脚本 `src/data_preparation/preprocess_uav.py`
  - 滑动窗口切片：640×640，步长 320（50% 重叠）
  - 同步裁剪并转换 YOLO 归一化标注坐标（处理跨边界框的截断与过滤）
  - 过滤空白切片（保留约 10% 负样本）
  - 输出切片统计：原始图/切片数对应关系
- [ ] **P1-UAV.2** 复用增强脚本 `src/data_preparation/augment.py`（`--dataset uav`）
- [ ] **P1-UAV.3** 复用数据集构建脚本 `src/data_preparation/build_dataset.py`（`--dataset uav`）
  - 按**原始图像**（非切片）划分，同一原始图的切片归同一子集
  - 生成 `data/yolo_dataset_uav/dataset.yaml`
- [ ] **P1-UAV.4** 【服务器执行】运行上述脚本，验证输出
  - 抽查切片结果（标注框位置是否准确）
  - 检查 `data/yolo_dataset_uav/` 目录完整性

**P1 完成标志：**
- `data/yolo_dataset_satellite/` 和 `data/yolo_dataset_uav/` 均存在
- 各子集图像与标注文件一一对应，无遗漏

---

## P2：YOLOv8 模型训练（迭代阶段）

> 前置条件：P1 完成。【服务器执行，可能多轮迭代】
> P2-SAT 与 P2-UAV **GPU资源允许时可并行提交作业**。

- [ ] **P2.1** 编写 YOLOv8 训练脚本 `src/train/train_yolov8.py`
  - 支持 `--dataset [satellite|uav]`，读取对应配置文件
  - 支持 `--model [yolov8n|yolov8s|yolov8m]`
  - 训练日志保存到 `runs/{satellite|uav}/yolov8/{exp_name}/`
  - 记录所有超参数到 `train_config.yaml`

### P2-SAT：卫星模型训练

- [ ] **P2-SAT.1** 【服务器执行】第一轮：训练 `yolov8n`，baseline
  ```bash
  python src/train/train_yolov8.py --dataset satellite --model yolov8n --epochs 100
  ```
- [ ] **P2-SAT.2** 查看训练曲线，判断是否过拟合/欠拟合，决定下一轮策略
- [ ] **P2-SAT.3** 【视情况迭代】调整模型规模、超参数或增强策略，重新训练
- [ ] **P2-SAT.4** 达标确认：test 集 mAP@0.5 ≥ 0.70，无明显过拟合

### P2-UAV：无人机模型训练

- [ ] **P2-UAV.1** 【服务器执行】第一轮：训练 `yolov8n`，baseline
  ```bash
  python src/train/train_yolov8.py --dataset uav --model yolov8n --epochs 100
  ```
- [ ] **P2-UAV.2** 查看训练曲线，判断是否过拟合/欠拟合，决定下一轮策略
- [ ] **P2-UAV.3** 【视情况迭代】调整模型规模、超参数或增强策略，重新训练
- [ ] **P2-UAV.4** 达标确认：test 集 mAP@0.5 ≥ 0.70，无明显过拟合

> ⚠️ **关卡：** P2-SAT.4 和 P2-UAV.4 **均通过**后，P3/P4/P5/P6 方可开始。

---

## P3：评估与可视化

> 前置条件：P2 达标（关卡通过）。【本地编写脚本 → 服务器执行】
> P3-SAT 与 P3-UAV **完全并行**。

- [ ] **P3.1** 编写评估脚本 `src/evaluate/evaluate.py`（支持 `--dataset [satellite|uav]`）
  - 在 test 集上计算：Precision、Recall、mAP@0.5、mAP@0.5:0.95、F1
  - 输出结果到 `results/phase1_training/{satellite|uav}/metrics.csv`

- [ ] **P3.2** 编写训练过程可视化脚本 `src/visualize/plot_training.py`
  - Loss 曲线（box_loss / cls_loss / dfl_loss）
  - mAP@0.5 和 mAP@0.5:0.95 随 epoch 变化曲线
  - 输出：`results/phase1_training/{satellite|uav}/training_curves.png`

- [ ] **P3.3** 编写检测结果可视化脚本 `src/visualize/plot_predictions.py`
  - 混淆矩阵
  - PR 曲线（Precision-Recall Curve）
  - F1-Confidence 曲线
  - 测试集样本预测结果图（原图 + 预测框 + GT框 + 置信度，随机抽取 20 张）
  - 数据集标注框尺寸分布直方图
  - 输出：`results/phase1_training/{satellite|uav}/`

- [ ] **P3.4** 【服务器执行】对两个模型分别运行评估与可视化，检查输出图表质量

**P3 完成标志：** `results/phase1_training/satellite/` 和 `results/phase1_training/uav/` 均含完整图表

---

## P4：GeoTIFF 大图预测

> 前置条件：P2 达标（关卡通过）。【本地编写脚本 → 服务器执行】
> P4-SAT 与 P4-UAV **完全并行**；可与 P3/P5/P6 **同时推进**。

- [ ] **P4.1** 编写大图预测脚本 `src/predict/predict_geotiff.py`
  - **方案 A（优先）：** 读取 `.tfw` 地理变换参数 → 滑动窗口推理 → 像素坐标转地理坐标 → 输出 KML/KMZ + SHP
    - 使用 rasterio 或 GDAL 读取 TIF
    - 使用 simplekml 生成 KML，pyshp 生成 SHP
    - NMS 去重（跨切片重叠区域的重复框）
  - **方案 B（备用）：** 切片保存为 JPEG → 逐片推理 → 收集含检测结果的切片图像
  - 支持参数：`--model_path`、`--tif_path`、`--output_dir`、`--conf_thres`、`--iou_thres`

- [ ] **P4.2** 【服务器执行】测试方案 A 可行性（rasterio/GDAL 安装、坐标精度验证）
  - 若成功：对全部 6 个位置分别运行，输出 KML 和 SHP
  - 若失败：切换方案 B

- [ ] **P4-SAT.3** 【服务器执行】卫星模型推理全部 6 个 TIF
  - 输出到 `results/phase1_predict/satellite/{位置名}/`
- [ ] **P4-UAV.3** 【服务器执行】无人机模型推理全部 6 个 TIF
  - 输出到 `results/phase1_predict/uav/{位置名}/`

- [ ] **P4.4** 编写预测结果统计可视化脚本 `src/visualize/plot_predict_summary.py`
  - 各位置检测框数量柱状图
  - 检测置信度分布直方图
  - 若方案 A：在地图底图上叠加检测框可视化（可选，需网络支持）

**P4 完成标志：** 每个位置均有预测输出（KML/SHP 或检测切片），含 summary 统计

---

## P5：分辨率影响分析

> 前置条件：P2 达标（关卡通过）。【可与 P3/P4/P6 并行】

- [ ] **P5.1** 编写多分辨率数据集构建脚本 `src/data_preparation/build_resolution_datasets.py`
  - 对原始图像进行双线性插值下采样（100% / 75% / 50% / 25%）
  - 下采样后统一缩放回 640×640 送入模型
  - YOLO 归一化标注直接复用
  - 输出：`data/resolution_datasets_{satellite|uav}/scale_{100|75|50|25}/`

- [ ] **P5.2** 编写分辨率实验训练脚本（复用 `train_yolov8.py`，新增 `--scale` 参数）

- [ ] **P5-SAT.3** 【服务器执行】卫星数据4个分辨率级别依次训练（共 4 次训练，GPU串行）
- [ ] **P5-UAV.3** 【服务器执行】无人机数据4个分辨率级别依次训练（共 4 次训练）

- [ ] **P5.4** 编写分辨率影响可视化脚本 `src/visualize/plot_resolution_analysis.py`
  - 各指标（mAP/Precision/Recall/F1）随分辨率变化折线图（卫星与无人机双线并排）
  - 不同分辨率下同一区域检测结果对比图（4列横向拼图）
  - 分辨率 × 目标尺寸分布关联散点图

- [ ] **P5.5** 【服务器执行】运行可视化，输出到 `results/phase2_resolution/`

**P5 完成标志：** `results/phase2_resolution/` 含完整分析图表

---

## P6：对比实验（YOLOv5 / U-Net / FCN）

> 前置条件：P2 达标（关卡通过）。【可与 P3/P4/P5 并行，各模型间可并行】

### P6.1：标注转换（分割模型前置，仅需一次）

- [ ] **P6.1.1** 编写 YOLO 框→像素掩码转换脚本 `src/data_preparation/yolo_to_mask.py`
  - 将 YOLO 格式标注框转为二值掩码图（目标区域=255，背景=0）
  - 为 U-Net 和 FCN 生成分割标签
  - 输出：`data/yolo_dataset_{satellite|uav}/masks/train|val|test/`

### P6.2：YOLOv5 对比（可与 P6.3/P6.4 并行）

- [ ] **P6.2.1** 编写 YOLOv5 训练脚本 `src/train/train_yolov5.py`
  - 使用相同数据集，对齐超参数策略
  - 支持 `--dataset [satellite|uav]`
- [ ] **P6.2.2** 【服务器执行】训练卫星 YOLOv5 模型
- [ ] **P6.2.3** 【服务器执行】训练无人机 YOLOv5 模型

### P6.3：U-Net 对比（可与 P6.2/P6.4 并行）

- [ ] **P6.3.1** 编写 U-Net 模型定义与训练脚本 `src/train/train_unet.py`
  - 二值分割输出，BCE + Dice Loss
  - 后处理：连通域分析提取实例边界框（转为检测格式用于指标计算）
  - 支持 `--dataset [satellite|uav]`
- [ ] **P6.3.2** 【服务器执行】训练卫星 U-Net 模型
- [ ] **P6.3.3** 【服务器执行】训练无人机 U-Net 模型

### P6.4：FCN 对比（可与 P6.2/P6.3 并行）

- [ ] **P6.4.1** 编写 FCN 模型定义与训练脚本 `src/train/train_fcn.py`
  - 同 U-Net 标注转换与后处理策略
  - 支持 `--dataset [satellite|uav]`
- [ ] **P6.4.2** 【服务器执行】训练卫星 FCN 模型
- [ ] **P6.4.3** 【服务器执行】训练无人机 FCN 模型

### P6.5：统一对比评估与可视化（依赖 P6.2~P6.4 全部完成）

- [ ] **P6.5.1** 编写统一对比评估脚本 `src/evaluate/compare_models.py`
  - 在相同测试集上计算所有模型的：Precision / Recall / mAP@0.5 / F1 / FPS / Params / FLOPs
  - 输出汇总表格到 `results/phase3_comparison/metrics_summary.csv`
- [ ] **P6.5.2** 编写对比可视化脚本 `src/visualize/plot_comparison.py`
  - 各模型指标对比柱状图（卫星 / 无人机分两组）
  - 速度 vs. 精度散点图（气泡大小 = 参数量）
  - 典型测试图像4模型预测结果横向对比拼图
  - 综合性能雷达图（4个维度：精度/召回/速度/轻量化）
- [ ] **P6.5.3** 【服务器执行】运行对比评估与可视化，输出到 `results/phase3_comparison/`

**P6 完成标志：** `results/phase3_comparison/` 含所有模型的指标与可视化

---

## P7：综合结果整理

> 前置条件：P3/P4/P5/P6 全部完成。

- [ ] **P7.1** 汇总所有阶段结果，整理关键指标对比表
- [ ] **P7.2** 检查所有可视化图表（风格统一、中文正常显示、DPI≥300）
- [ ] **P7.3** 更新 `README.md`，补充项目运行说明
- [ ] **P7.4** 归档最优模型权重文件到 `runs/best_weights/`

---

## 快速参考：并行执行矩阵

| 任务 | 依赖 | 可与哪些任务并行 |
|------|------|-----------------|
| P0 环境配置 | 无 | — |
| P1-SAT 卫星数据处理 | P0 | P1-UAV |
| P1-UAV 无人机数据处理 | P0 | P1-SAT |
| P2-SAT 卫星训练 | P1-SAT | P2-UAV（GPU允许） |
| P2-UAV 无人机训练 | P1-UAV | P2-SAT（GPU允许） |
| **【关卡】** | P2-SAT + P2-UAV 均达标 | — |
| P3 评估可视化 | 关卡 | P4, P5, P6 |
| P4 大图预测 | 关卡 | P3, P5, P6 |
| P5 分辨率分析 | 关卡 | P3, P4, P6 |
| P6.2 YOLOv5 | 关卡 | P3, P4, P5, P6.3, P6.4 |
| P6.3 U-Net | 关卡 + P6.1.1 | P3, P4, P5, P6.2, P6.4 |
| P6.4 FCN | 关卡 + P6.1.1 | P3, P4, P5, P6.2, P6.3 |
| P6.5 对比汇总 | P6.2+P6.3+P6.4 完成 | — |
| P7 综合整理 | P3+P4+P5+P6 完成 | — |

---

## 当前进度

- [x] 需求文档 `vibe-docs/requirements.md` 已完成
- [x] 执行计划 `vibe-docs/TODO.md` 已完成
- [ ] **下一步：开始 P0 环境配置与 P1 数据准备脚本编写**
