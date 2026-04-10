import logging
import sys

def setup_logger(level=logging.INFO, name="wise_home", logfile=None):
    """
    创建独立的系统日志 Logger
    :param level: 初始日志级别（控制输出）
    :param name: 日志名（独立命名空间）
    :param logfile: 可选日志文件路径
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 不冒泡到根 Logger，防止影响其他模块

    # 如果已有 handler，先移除
    if logger.hasHandlers():
        logger.handlers.clear()

    # 格式化
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)  # 永远通过 handler，实际显示由 logger.level 控制
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件 Handler（可选）
    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(logging.DEBUG)  # 同样保持 DEBUG，输出由 logger 控制
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# 全局系统日志
logger = setup_logger(level=logging.INFO, name="wise_home")