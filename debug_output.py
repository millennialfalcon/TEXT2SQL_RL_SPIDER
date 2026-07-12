import json
import logging
import sys
from pathlib import Path


class JSONLWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def build_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("spider_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )

    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)
    return logger
