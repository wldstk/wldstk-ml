"""This module provides a shared logging setup for the package and its scripts."""

import logging

PACKAGE_LOGGER_NAME = "wldstk_ml"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_package_logger() -> logging.Logger:
    """Return the package logger, attaching its console handler on first use."""

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger that writes to the console at DEBUG level."""

    package_logger = _configure_package_logger()
    if not name or name == PACKAGE_LOGGER_NAME:
        return package_logger
    if name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")
