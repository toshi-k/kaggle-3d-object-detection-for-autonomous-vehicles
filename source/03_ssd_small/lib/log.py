import logging
from pathlib import Path


def init_logger(path, name='root', level=logging.DEBUG):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(message)s")

    handler1 = logging.StreamHandler()
    handler1.setFormatter(formatter)

    dir_log = Path(path).parent

    dir_log.mkdir(exist_ok=True)

    handler2 = logging.FileHandler(filename=path)
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger
