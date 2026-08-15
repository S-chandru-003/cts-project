import logging
from datetime import datetime
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Central log directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create one log file for each application run
LOG_FILE = LOG_DIR / (
    f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
)


LOG_FORMAT = (
    "[%(asctime)s] %(levelname)s "
    "%(name)s:%(lineno)d - %(message)s"
)


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured application logger.

    All project components write their logs to the
    central project-level logs directory.
    """

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        )

        formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        logger.propagate = False

    return logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Logging has started.")