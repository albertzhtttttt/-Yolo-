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
| `src/utils/plot_utils.py` | 中文字体自动检测、统一图表风格、PR曲线/混淆矩阵绘图函数 |
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
| `src/evaluate/evaluate.py` | 完整评估脚本：test 集指标（P/R/mAP/F1）、训练曲线、混淆矩阵、PR 曲线、F1-Confidence 曲线、测试集预测可视化（GT vs 预测框对比） |

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
