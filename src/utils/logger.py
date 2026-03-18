"""
logger.py - 日志工具模块
路径：src/utils/logger.py

功能：
    - 统一日志格式（时间戳 + 级别 + 消息）
    - 同时输出到控制台和日志文件
    - 提供全局获取 logger 的接口

使用方式：
    from src.utils.logger import get_logger
    logger = get_logger("train", log_dir="logs")
    logger.info("开始训练...")
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    创建或获取命名 logger，同时输出到控制台和文件。

    参数：
        name:    logger 名称（通常为脚本名或阶段名，如 "train_satellite"）
        log_dir: 日志文件保存目录，默认 "logs/"
        level:   日志级别，默认 INFO
        console: 是否同时输出到控制台，默认 True

    返回值：
        配置好的 logging.Logger 对象

    示例：
        logger = get_logger("preprocess_satellite", log_dir="logs")
        logger.info("处理第 1 张图像")
        logger.warning("标注框数量为 0，跳过该图像")
        logger.error("文件不存在：xxx.jpg")
    """
    # 如果同名 logger 已存在且有 handler，直接返回避免重复添加
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 统一日志格式：时间 | 级别 | 模块 | 消息
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 控制台 Handler ────────────────────────────────────────────────────────
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ── 文件 Handler ──────────────────────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    # 日志文件名：{name}_{日期}.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{name}_{timestamp}.log")

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 防止日志向上传播到根 logger（避免重复打印）
    logger.propagate = False

    logger.info(f"日志文件：{os.path.abspath(log_filename)}")
    return logger


def set_log_level(logger: logging.Logger, level: str) -> None:
    """
    动态修改 logger 的日志级别。

    参数：
        logger: 目标 logger 对象
        level:  级别字符串（"DEBUG" / "INFO" / "WARNING" / "ERROR"）
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)
