# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOv8-based forest gap detection system for high-resolution remote sensing imagery (satellite + UAV). The core workflow covers data preparation → training → evaluation → large GeoTIFF prediction → resolution analysis → multi-model comparison, and the repository also includes historical temporal prediction for archived satellite imagery.

**Server**: All training and heavy computation runs on `lfy@172.31.226.112`, project path `/home6/zht/-Yolo-`, conda env `forest_gap`. See memory files for SSH/SCP patterns.

## Key Commands

### Data Preparation (P1)
```bash
python scripts/run_data_pipeline.py --dataset all   # satellite + uav
python scripts/run_data_pipeline.py --dataset satellite
```

### Training (P2/P3)
```bash
bash scripts/train_bg.sh --dataset all              # background training
python scripts/run_train_pipeline.py --dataset satellite --device 0
python src/evaluate/evaluate.py --dataset satellite \
    --weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt
```

### Large-Image Prediction (P4)
```bash
python scripts/run_predict.py --dataset satellite \
    --weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt

python scripts/run_predict.py --dataset uav \
    --weights runs/uav/yolov8/uav_yolov8_cbam/weights/best.pt
```

### Resolution Analysis (P5)
```bash
python scripts/run_resolution_pipeline.py --dataset satellite --device 0
python scripts/run_resolution_pipeline.py --dataset uav --device 1
```

### Multi-Model Comparison (P6)
```bash
python src/data_preparation/yolo_to_mask.py --dataset all
python src/train/train_unet.py   --dataset satellite --device 0 --batch 2
python src/train/train_fcn.py    --dataset satellite --device 1
python src/evaluate/compare_models.py --dataset satellite \
    --yolov8_weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt \
    --yolov5_weights runs/satellite/yolov5/satellite_yolov5n/weights/best.pt \
    --unet_weights   runs/satellite/unet/satellite_unet/best.pt \
    --fcn_weights    runs/satellite/fcn/satellite_fcn/best.pt
```

### Historical Temporal Prediction
```bash
python scripts/run_history_predict.py \
    --weights runs/satellite/yolov8/satellite_yolov8/weights/best.pt \
    --device 0
```

## Architecture

### Data Flow
```
Original Data (satellite 256×256 / UAV variable)
  → Preprocess (scale to 640×640 or slice with 50% overlap)
  → Augment (8× for satellite, 5× for UAV, offline)
  → YOLO dataset (train/val/test split, seed=42)
  → Train → best.pt
  → Evaluate (metrics/curves/confusion matrix)
  → Predict on large GeoTIFF (windowed reading) → KML/SHP
  → Optional temporal batch prediction on archived GeoTIFFs
```

### Configuration-Driven Design
Core hyperparameters live in `configs/{satellite,uav,predict}_config.yaml`, while the UAV CBAM architecture is defined in `configs/yolov8_gap_cbam_p2.yaml`. Scripts load these configs and allow CLI overrides. Never hardcode reusable paths or hyperparameters — add them to the YAML configs.

### Key Design Patterns
- **Windowed GeoTIFF reading**: `src/predict/predict_geotiff.py` uses `rasterio` windowed reading for very large TIFs and should reuse a single opened dataset handle per image whenever possible.
- **Geo-reference precedence**: Prediction should prefer embedded GeoTIFF `transform` / `crs`; fall back to external `.tfw` / `.prj` only when embedded spatial reference is unavailable.
- **Custom CBAM registration**: `src/train/yolo_model_loader.py` registers `CBAM`, `ChannelAttention`, and `SpatialAttention` into Ultralytics runtime so custom YAML structures and CBAM weights can be loaded consistently in training, evaluation, and prediction.
- **Pipeline orchestration**: `scripts/run_*.py` call subprocesses in sequence with error handling. Each phase is independently re-runnable.
- **GPU selection**: Pass `--device 0/1/2` to training or prediction scripts. Heavy jobs are expected to run on the remote server.
- **Single class**: All models detect only one class — `gap` (林窗). Class index is always `0`.
- **Offline augmentation**: Augmentation is applied before training (generates new image files), not via Ultralytics' online augmentation.
- **Temporal batch prediction**: `scripts/run_history_predict.py` scans `data/predict/history/<date>/Level18/` directories and writes per-year outputs plus summary charts.

### Module Responsibilities
| Module | Purpose |
|--------|---------|
| `src/data_preparation/` | Preprocessing, slicing, augmentation, dataset building, mask conversion |
| `src/train/` | YOLOv8, YOLOv5, U-Net, FCN training scripts |
| `src/train/yolo_model_loader.py` | Runtime CBAM registration and unified loading for custom YOLO models |
| `src/evaluate/` | YOLOv8 evaluation + multi-model comparison metrics |
| `src/predict/predict_geotiff.py` | Windowed inference on large GeoTIFFs, prefers embedded spatial reference, outputs KML/SHP |
| `scripts/run_history_predict.py` | Historical temporal GeoTIFF batch prediction and yearly summaries |
| `src/visualize/` | Resolution analysis charts, model comparison plots |
| `src/utils/logger.py` | Unified logging (timestamp + console + file) |
| `src/utils/plot_utils.py` | Matplotlib styling, color-blind-friendly palette |

### Output Directories
- `runs/{satellite,uav}/{yolov8,yolov5,unet,fcn}/` — training checkpoints
- `results/phase1_training/` — evaluation metrics and plots
- `results/phase1_predict/` — KML/SHP prediction outputs per location
- `results/phase2_resolution/` — resolution analysis charts
- `results/phase3_comparison/` — multi-model comparison outputs
- `results/phase4_temporal/predictions/` — per-year temporal prediction outputs
- `results/phase4_temporal/summaries/` — temporal CSV and charts
- `logs/` — pipeline execution logs

### Best Model Weights
- Satellite YOLOv8: `runs/satellite/yolov8/satellite_yolov8/weights/best.pt` (mAP@0.5=0.418)
- UAV YOLOv8 + CBAM: `runs/uav/yolov8/uav_yolov8_cbam/weights/best.pt` (mAP@0.5=0.974)
