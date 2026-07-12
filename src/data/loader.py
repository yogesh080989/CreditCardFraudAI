"""
Data loading utilities for CreditCardFraudAI.

This module provides the DataLoader class responsible for loading
datasets into pandas DataFrames while performing basic validation.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.config import ConfigManager
from src.core.exceptions import DataLoadingError
from src.core.logger import LoggerManager


class DataLoader:
    """
    Loads datasets used throughout the project.

    This class is the single entry point for loading datasets.
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
    ) -> None:

        self.config = config or ConfigManager()
        self.logger = LoggerManager.get_logger()

        # Resolve dataset path from project root
        self.dataset_path = (
            self.config.project_root
            / self.config.get("dataset_path")
        )

        self._dataframe: Optional[pd.DataFrame] = None

    def validate_file(self) -> None:
        """
        Validate dataset path.
        """

        self.logger.info("Validating dataset path.")

        if not self.dataset_path.exists():
            raise DataLoadingError(
                f"Dataset not found: {self.dataset_path}"
            )

        if not self.dataset_path.is_file():
            raise DataLoadingError(
                f"Invalid dataset path: {self.dataset_path}"
            )

    def load(self) -> pd.DataFrame:
        """
        Load dataset.

        Returns
        -------
        pandas.DataFrame
        """

        self.validate_file()

        self.logger.info(
            "Loading dataset from %s",
            self.dataset_path,
        )

        try:

            self._dataframe = pd.read_csv(
                self.dataset_path
            )

            self.logger.info(
                "Dataset loaded successfully."
            )

            self.logger.info(
                "Rows: %d | Columns: %d",
                self._dataframe.shape[0],
                self._dataframe.shape[1],
            )

            return self._dataframe

        except Exception as exc:

            raise DataLoadingError(
                f"Unable to load dataset: {exc}"
            ) from exc

    def validate_required_columns(
        self,
        required_columns: list[str],
    ) -> None:
        """
        Validate required columns.
        """

        if self._dataframe is None:
            raise DataLoadingError(
                "Dataset not loaded."
            )

        missing = [
            column
            for column in required_columns
            if column not in self._dataframe.columns
        ]

        if missing:

            raise DataLoadingError(
                f"Missing columns: {missing}"
            )

        self.logger.info(
            "Required columns validated."
        )

    def get_dataset_summary(self) -> dict:
        """
        Return basic dataset summary.
        """

        if self._dataframe is None:
            raise DataLoadingError(
                "Dataset not loaded."
            )

        return {
            "rows": self._dataframe.shape[0],
            "columns": self._dataframe.shape[1],
            "memory_mb": round(
                self._dataframe.memory_usage(
                    deep=True
                ).sum()
                / (1024 * 1024),
                2,
            ),
            "column_names": list(
                self._dataframe.columns
            ),
        }

    def shape(self) -> tuple[int, int]:
        """
        Return dataframe shape.
        """

        if self._dataframe is None:
            raise DataLoadingError(
                "Dataset not loaded."
            )

        return self._dataframe.shape

    def columns(self) -> list[str]:
        """
        Return dataframe columns.
        """

        if self._dataframe is None:
            raise DataLoadingError(
                "Dataset not loaded."
            )

        return list(self._dataframe.columns)