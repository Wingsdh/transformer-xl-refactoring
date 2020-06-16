# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    log.py
   Description :
   Author :       Wings DH
   Time：         2020/1/3 2:00 下午
-------------------------------------------------
   Change Activity:
                   2020/1/3: Create
-------------------------------------------------
"""
import sys

from loguru import logger

logger = logger
logger.remove()

log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <6}</level> | <level>{message}</level>"
level = 'INFO'
logger.add(sys.stderr, format=log_format, level=level)
