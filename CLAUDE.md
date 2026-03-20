# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOv8-based forest gap detection system for high-resolution remote sensing imagery (satellite + UAV). Six-phase pipeline: data preparation → training → evaluation → large GeoTIFF prediction → resolution analysis → multi-model comparison.

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
```

### Configuration-Driven Design
All hyperparameters live in `configs/{satellite,uav,predict}_config.yaml`. Scripts load these configs and allow CLI overrides. Never hardcode paths or hyperparameters — add them to the YAML configs.

### Key Design Patterns
- **Windowed GeoTIFF reading**: `src/predict/predict_geotiff.py` uses `rasterio` for images up to 28928×51200 px to avoid OOM. Always use windowed reading for large TIFs.
- **Pipeline orchestration**: `scripts/run_*.py` call subprocesses in sequence with error handling. Each phase is independently re-runnable.
- **GPU selection**: Pass `--device 0/1/2` to training scripts. The train scripts auto-detect free GPUs if not specified.
- **Single class**: All models detect only one class — `gap` (林窗). Class index is always 0.
- **Offline augmentation**: Augmentation is applied before training (generates new image files), not via Ultralytics' online augmentation.

### Module Responsibilities
| Module | Purpose |
|--------|---------|
| `src/data_preparation/` | Preprocessing, slicing, augmentation, dataset building, mask conversion |
| `src/train/` | YOLOv8, YOLOv5, U-Net, FCN training scripts |
| `src/evaluate/` | YOLOv8 evaluation + multi-model comparison metrics |
| `src/predict/predict_geotiff.py` | Windowed inference on large GeoTIFFs, outputs KML/SHP with geo-coordinates |
| `src/visualize/` | Resolution analysis charts, model comparison plots |
| `src/utils/logger.py` | Unified logging (timestamp + console + file) |
| `src/utils/plot_utils.py` | Matplotlib styling, color-blind-friendly palette |

### Output Directories
- `runs/{satellite,uav}/{yolov8,yolov5,unet,fcn}/` — training checkpoints
- `results/phase1_training/` — evaluation metrics and plots
- `results/phase1_predict/` — KML/SHP prediction outputs per location
- `results/phase2_resolution/` — resolution analysis charts
- `results/phase3_comparison/` — multi-model comparison outputs
- `logs/` — pipeline execution logs

### Best Model Weights
- Satellite YOLOv8: `runs/satellite/yolov8/satellite_yolov8/weights/best.pt` (mAP@0.5=0.418)
- UAV YOLOv8: `runs/uav/yolov8/uav_yolov8/weights/best.pt` (mAP@0.5=0.974)
