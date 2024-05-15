import os
import sys
from pathlib import Path

from loguru import logger as lg

from src.utils.utils import BASE_DIR

if BASE_DIR is None:
    log_path = Path("logs")
else:
    log_path = Path(BASE_DIR, "logs")
try:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
except Exception:
    pass

lazy = True
stout = True
fout = False


class Logging:
    __instance = None
    lg.remove()
    if not lazy:
        if stout:
            lg.add(sys.stdout, colorize=True,
                   format="<yellow>{level}</yellow> | <red>{file}</red> | <red>{module}</red> | " \
                          "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{message}</level>",
                   level="INFO")
        if fout:
            lg.add(os.path.join(log_path, 'log_{time}.txt'), rotation="500MB", encoding="utf-8", enqueue=True,
                   retention="30 days",
                   format="{level} | {file} | {module} | {time:YYYY-MM-DD at HH:mm:ss} | {message}")

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Logging, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def info(self, msg, p=True):
        if lazy or not p:
            return
        return lg.info(msg)

    def debug(self, msg):
        return lg.debug(msg)

    def warning(self, msg):
        return lg.warning(msg)

    def error(self, msg):
        return lg.error(msg)


logger = Logging()
