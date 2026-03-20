# 最终执行清单

**基于：** `vibe-docs/requirements.md` v1.2
**更新日期：** 2026-03-21
**说明：** 本文件已从历史执行计划收敛为最终状态记录，保留当前有效结果、产出目录和后续待办。

---

## 阶段状态总览

| 阶段 | 状态 | 主要产出 | 当前说明 |
|------|------|----------|----------|
| P0 | 已完成 | `scripts/install_env.sh`、`scripts/init_dirs.sh`、`configs/*.yaml`、`src/utils/plot_utils.py`、`src/utils/logger.py` | 环境与基础配置已就位 |
| P1 | 已完成 | `src/data_preparation/`、`scripts/run_data_pipeline.py` | 卫星与 UAV 数据准备流程已落地 |
| P2/P3 | 已完成 | `results/phase1_training/{satellite,uav}/` | UAV 达标；satellite 当前基线偏低 |
| P4 | 已完成 | `results/phase1_predict/{satellite,uav}/` | 6 个位置均已生成预测输出与统计结果 |
| P5 | 已完成 | `results/phase2_resolution/` | satellite 结果存在于 `satellite/scale_*/metrics.csv` |
| P6 | 已完成 | `results/phase3_comparison/` | 对比指标与图表已生成，文档已按修正后结果同步 |
| P7 | 部分完成 | `README.md`、`vibe-docs/deliverables.md`、`vibe-docs/summary.md` | 结果文档已更新，仍可继续补充原因分析与归档 |

---

## 最新结果快照

### P2/P3：YOLOv8 基准模型

| 数据集 | mAP@0.5 | Precision | Recall | F1 | 状态 |
|--------|---------|-----------|--------|----|------|
| satellite | 0.418 | 0.920 | 0.222 | 0.358 | 需改进 |
| uav | 0.974 | 0.974 | 0.950 | 0.962 | 达标 |

### P4：大图预测检测框统计

| 数据集 | Site1 | Site2_1 | Site2_2 | Site2_3 | Site3_1 | Site3_2 | 合计 |
|--------|-------|---------|---------|---------|---------|---------|------|
| satellite | 2704 | 1409 | 4046 | 3941 | 7785 | 3827 | 23,712 |
| uav | 13036 | 3407 | 7721 | 9594 | 7616 | 5822 | 47,196 |

### P5：分辨率影响分析（mAP@0.5）

| 数据集 | 100% | 75% | 50% | 25% |
|--------|------|-----|-----|-----|
| satellite | 0.667 | 0.446 | 0.645 | 0.556 |
| uav | 0.950 | 0.877 | 0.957 | 0.897 |

### P6：多模型对比（Satellite）

| 模型 | mAP@0.5 | Precision | Recall | F1 | FPS |
|------|---------|-----------|--------|----|-----|
| YOLOv8 | 0.418 | 0.920 | 0.222 | 0.358 | 106.2 |
| YOLOv5 | 0.630 | 0.987 | 0.444 | 0.613 | 107.0 |
| U-Net | 0.196 | 0.667 | 0.444 | 0.533 | 119.5 |
| FCN | 0.064 | 0.333 | 0.333 | 0.333 | 186.7 |

### P6：多模型对比（UAV）

| 模型 | mAP@0.5 | Precision | Recall | F1 | FPS |
|------|---------|-----------|--------|----|-----|
| YOLOv8 | 0.974 | 0.974 | 0.950 | 0.962 | 112.9 |
| YOLOv5 | 0.986 | 1.000 | 0.972 | 0.986 | 87.4 |
| U-Net | 0.210 | 0.274 | 1.000 | 0.430 | 155.1 |
| FCN | 0.267 | 0.333 | 0.975 | 0.497 | 337.6 |

---

## 关键输出目录

- `results/phase1_training/{satellite,uav}/`：P3 评估指标、曲线、混淆矩阵、预测样例
- `results/phase1_predict/{satellite,uav}/{位置名}/`：P4 大图预测输出，含 KML/KMZ/SHP 与统计图
- `results/phase2_resolution/`：P5 分辨率实验总目录
- `results/phase2_resolution/satellite/scale_{100,75,50,25}/metrics.csv`：satellite 分辨率实验明细
- `results/phase2_resolution/uav/scale_{100,75,50,25}/metrics.csv`：UAV 分辨率实验明细
- `results/phase3_comparison/metrics_summary_satellite.csv`：P6 卫星对比指标
- `results/phase3_comparison/metrics_summary_uav.csv`：P6 无人机对比指标

---

## 当前已知问题

- [x] `src/evaluate/compare_models.py` 的 AP 积分逻辑已修正，分割模型负 `mAP@0.5` 问题已消除
- [x] README 与交付文档中的旧实验结果、旧权重路径已同步更新
- [ ] satellite YOLOv8 基线结果 `mAP@0.5=0.418` 明显低于早期文档记录，需补充原因说明或重新复核实验来源
- [ ] `results/phase2_resolution/metrics_by_scale.csv` 当前仅汇总 UAV 行，如需统一对外展示，建议重新导出包含 satellite 的总表

---

## 后续待办

- [ ] 复核 satellite 基线实验来源、权重路径与评估配置，解释与早期结果不一致的原因
- [ ] 重新生成 `results/phase2_resolution/metrics_by_scale.csv`，补齐 satellite 行
- [ ] 视交付需要归档最终最优权重与代表性图表到统一目录
- [x] 更新 `README.md`、`vibe-docs/deliverables.md`、`vibe-docs/summary.md`、`CLAUDE.md`
