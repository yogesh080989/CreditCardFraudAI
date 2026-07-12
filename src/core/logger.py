"""
Centralized logging utility for CreditCardFraudAI.

This module provides a singleton logger that is shared across the
entire application. It logs to both console and file.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

import logging
from pathlib import Path


class LoggerManager:
    """
    Creates and manages the project logger.

    This class ensures that only one logger instance exists
    throughout the application.
    """

    _logger = None

    @classmethod
    def get_logger(
        cls,
        name: str = "CreditCardFraudAI",
        log_file: str = "logs/pipeline.log",
        level: int = logging.INFO,
    ) -> logging.Logger:
        """
        Returns the singleton logger.

        Parameters
        ----------
        name : str
            Logger name.

        log_file : str
            Log file path.

        level : int
            Logging level.

        Returns
        -------
        logging.Logger
        """

        if cls._logger is not None:
            return cls._logger

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(level)

        if logger.handlers:
            cls._logger = logger
            return logger

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # File Handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.propagate = False

        cls._logger = logger

        return logger