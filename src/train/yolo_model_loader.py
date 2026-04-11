"""
yolo_model_loader.py - YOLO 模型加载辅助模块
路径：src/train/yolo_model_loader.py

功能：
    1. 在运行时向 Ultralytics 注册 CBAM 相关模块，
       使自定义 YAML 结构和带 CBAM 的权重都能被正常解析；
    2. 统一解析模型来源，兼容内置模型名、本地 YAML、已有权重路径；
    3. 为训练、评估、预测脚本提供统一的 YOLO 加载入口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CBAM(nn.Module):
    """
    惰性初始化版 CBAM 模块。

    设计原因：
        Ultralytics 的 `parse_model()` 在解析未内置注册的模块时，
        不会自动把输入通道数 `c1` 注入到模块构造函数中。若直接复用
        官方实现的 `CBAM(c1, kernel_size)`，在宽度缩放场景下容易出现
        YAML 中声明通道数与实际输入通道数不一致的问题。

    这里改为：
        1. 构造阶段仅记录 `kernel_size`；
        2. 首次前向传播时，根据真实输入张量的通道数动态创建
           `ChannelAttention` 与 `SpatialAttention`；
        3. 后续前向直接复用已创建的注意力子模块。

    参数：
        channels_hint: YAML 中保留的占位参数，仅用于保持配置可读性，
                       实际计算时不会依赖它；
        kernel_size:   空间注意力卷积核大小。
    """

    def __init__(self, channels_hint: int, kernel_size: int = 7):
        super().__init__()
        self.channels_hint = channels_hint
        self.kernel_size = kernel_size
        self.channel_attention = None
        self.spatial_attention = None

    def _build_if_needed(self, x: torch.Tensor) -> None:
        """根据真实输入通道数延迟构建注意力模块。"""
        if self.channel_attention is not None and self.spatial_attention is not None:
            return

        from ultralytics.nn.modules import ChannelAttention, SpatialAttention

        channels = int(x.shape[1])
        self.channel_attention = ChannelAttention(channels).to(device=x.device, dtype=x.dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """按“通道注意力 → 空间注意力”的顺序增强特征。"""
        self._build_if_needed(x)
        return self.spatial_attention(self.channel_attention(x))


def register_cbam_modules(logger=None) -> None:
    """
    向 Ultralytics 的 tasks 命名空间注册 CBAM 相关模块。

    说明：
        Ultralytics 在解析 model.yaml 时，会从 tasks 模块的全局命名空间
        中查找模块名。虽然安装包内部已经实现了 CBAM，但默认并不会把
        `CBAM`、`ChannelAttention`、`SpatialAttention` 暴露到解析器使用的
        globals() 中，因此这里需要在运行时补注册一次。

    参数：
        logger: 可选日志对象，用于输出注册状态。
    """
    from ultralytics.nn import tasks as ultralytics_tasks
    from ultralytics.nn.modules import ChannelAttention, SpatialAttention

    registered_names = []
    module_mapping = {
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
    }

    for module_name, module_obj in module_mapping.items():
        if getattr(ultralytics_tasks, module_name, None) is not module_obj:
            setattr(ultralytics_tasks, module_name, module_obj)
            registered_names.append(module_name)

    if logger is not None and registered_names:
        logger.info(f"已向 Ultralytics 注册自定义模块：{', '.join(registered_names)}")



def resolve_model_source(model_name: str, use_pretrained: bool) -> str:
    """
    解析 YOLO 模型来源。

    规则：
        1. 若 `model_name` 已是仓库内存在的文件路径，则直接返回绝对路径；
        2. 若 `model_name` 已显式带有 `.yaml` / `.yml` / `.pt` 后缀，则原样返回；
        3. 若 `model_name` 仅是 Ultralytics 内置模型简称（如 `yolov8n`），
           则按是否使用预训练权重自动补全为 `.pt` 或 `.yaml`。

    参数：
        model_name:      模型标识或路径
        use_pretrained:  是否优先解析到预训练权重入口

    返回值：
        可直接传入 `YOLO(...)` 的来源字符串。
    """
    candidate_path = Path(model_name)
    if not candidate_path.is_absolute():
        candidate_path = PROJECT_ROOT / candidate_path

    if candidate_path.is_file():
        return str(candidate_path)

    if model_name.endswith((".yaml", ".yml", ".pt")):
        return model_name

    return model_name + (".pt" if use_pretrained else ".yaml")



def build_yolo_model(
    model_name: str,
    use_pretrained: bool = True,
    pretrained_weights: Optional[str] = None,
    logger=None,
):
    """
    构建 YOLO 模型对象，并按需加载预训练权重。

    说明：
        - 常规内置模型：`model_name='yolov8n'` 且 `use_pretrained=True`，
          会直接加载 `yolov8n.pt`；
        - 自定义结构 YAML：`model_name='configs/xxx.yaml'` 且提供
          `pretrained_weights='yolov8n.pt'`，会先按 YAML 构建网络，再把
          预训练权重迁移加载进去；
        - 评估/预测阶段：直接传入 `best.pt` 即可，函数也会先注册 CBAM，
          确保带 CBAM 的权重可以被正常反序列化。

    参数：
        model_name:           模型标识、结构 YAML 或权重路径
        use_pretrained:       是否优先走预训练初始化流程
        pretrained_weights:   自定义 YAML 场景下的预训练权重入口
        logger:               可选日志对象

    返回值：
        (model, model_source)
        - model: Ultralytics YOLO 对象
        - model_source: 实际使用的模型来源字符串
    """
    from ultralytics import YOLO

    register_cbam_modules(logger=logger)

    if pretrained_weights:
        model_source = resolve_model_source(model_name, use_pretrained=False)
        model = YOLO(model_source)
        model.load(pretrained_weights)
        if logger is not None:
            logger.info(f"已加载自定义结构：{model_source}")
            logger.info(f"已迁移加载预训练权重：{pretrained_weights}")
        return model, model_source

    model_source = resolve_model_source(model_name, use_pretrained=use_pretrained)
    model = YOLO(model_source)
    if logger is not None:
        logger.info(f"已加载模型来源：{model_source}")
    return model, model_source
