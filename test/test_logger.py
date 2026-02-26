import logging

from core.logger import logger



if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.debug("你好")