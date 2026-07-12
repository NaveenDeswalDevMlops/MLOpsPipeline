import logging
import os

BASE_WORKSPACE = os.getenv("WORKSPACE", os.getcwd())
LOG_DIR = os.path.join(BASE_WORKSPACE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith(filename)
        for handler in logger.handlers
    ):
        file_path = os.path.join(LOG_DIR, filename)
        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        )
        logger.addHandler(ch)
    logger.propagate = False
    return logger
