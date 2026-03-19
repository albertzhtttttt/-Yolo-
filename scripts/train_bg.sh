#!/bin/bash
# =============================================================================
# train_bg.sh - 后台运行训练流程，日志同时写入文件
#
# 用法：
#   bash scripts/train_bg.sh [run_train_pipeline.py 的所有参数]
#
# 示例：
#   bash scripts/train_bg.sh --dataset all
#   bash scripts/train_bg.sh --dataset satellite --model yolov8s --epochs 200
#   bash scripts/train_bg.sh --dataset satellite --eval_only \
#       --weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt
#
# 查看日志：
#   tail -f logs/train_bg_<timestamp>.log
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$PROJECT_ROOT/logs"
ulimit -n 65536

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/logs/train_bg_${TIMESTAMP}.log"

echo "启动后台训练..."
echo "日志文件：$LOG_FILE"
echo "追踪命令：tail -f $LOG_FILE"
echo ""

nohup python "$PROJECT_ROOT/scripts/run_train_pipeline.py" "$@" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "后台进程 PID：$PID"
echo "停止命令：kill $PID"
