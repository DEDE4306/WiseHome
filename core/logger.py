import logging
import sys

def setup_logger(level=logging.INFO, name="wise_home", logfile=None):
    """
    创建独立的系统日志 Logger
    :param level: 初始日志级别
    :param name: 日志名（独立命名空间）
    :param logfile: 可选日志文件路径
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 不冒泡到根 Logger，防止影响其他模块

    # 如果已经有 handler，先移除，避免重复输出
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台 Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件 Handler（可选）
    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# 创建全局系统日志
logger = setup_logger(level=logging.INFO, name="wise_home")