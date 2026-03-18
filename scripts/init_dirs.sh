#!/bin/bash
# =============================================================================
# 项目目录初始化脚本
# 执行方式：bash scripts/init_dirs.sh
# 说明：在服务器上首次部署时运行，创建全部必要目录
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "项目根目录：$PROJECT_ROOT"
cd "$PROJECT_ROOT"

echo "创建项目目录结构..."

# ---------- 源码目录 ----------
mkdir -p src/utils
mkdir -p src/data_preparation
mkdir -p src/train
mkdir -p src/evaluate
mkdir -p src/predict
mkdir -p src/visualize

# ---------- 配置文件目录 ----------
mkdir -p configs

# ---------- 数据目录（orig_data 已存在，只创建输出目录）----------
mkdir -p data/yolo_dataset_satellite/images/{train,val,test}
mkdir -p data/yolo_dataset_satellite/labels/{train,val,test}
mkdir -p data/yolo_dataset_uav/images/{train,val,test}
mkdir -p data/yolo_dataset_uav/labels/{train,val,test}

# 多分辨率数据集（阶段二）
for scale in 100 75 50 25; do
    mkdir -p data/resolution_datasets_satellite/scale_${scale}/images/{train,val,test}
    mkdir -p data/resolution_datasets_satellite/scale_${scale}/labels/{train,val,test}
    mkdir -p data/resolution_datasets_uav/scale_${scale}/images/{train,val,test}
    mkdir -p data/resolution_datasets_uav/scale_${scale}/labels/{train,val,test}
done

# 分割模型掩码标签
mkdir -p data/yolo_dataset_satellite/masks/{train,val,test}
mkdir -p data/yolo_dataset_uav/masks/{train,val,test}

# ---------- 训练输出目录 ----------
for dataset in satellite uav; do
    for model in yolov8 yolov5 unet fcn; do
        mkdir -p runs/${dataset}/${model}
    done
done
mkdir -p runs/best_weights

# ---------- 结果输出目录 ----------
mkdir -p results/phase1_training/{satellite,uav}
mkdir -p results/phase1_predict/{satellite,uav}
mkdir -p results/phase2_resolution/{satellite,uav}
mkdir -p results/phase3_comparison/{satellite,uav}

# ---------- 脚本目录 ----------
mkdir -p scripts
mkdir -p logs

echo ""
echo "目录结构创建完成！当前结构："
find . -type d | grep -v "__pycache__" | grep -v ".git" | sort | head -60
