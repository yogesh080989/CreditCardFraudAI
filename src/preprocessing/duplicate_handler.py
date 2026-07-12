"""
Duplicate row handling utilities for CreditCardFraudAI.

This module provides functionality to analyze and remove
duplicate rows from datasets.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

import pandas as pd

from src.core.logger import LoggerManager


class DuplicateHandler:
    """
    Handles duplicate rows in datasets.
    """

    def __init__(self) -> None:

        self.logger = LoggerManager.get_logger()

    # ---------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------

    def analyze(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:

        duplicate_count = int(dataframe.duplicated().sum())

        return {
            "total_rows": len(dataframe),
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": round(
                duplicate_count / len(dataframe) * 100,
                2,
            ),
        }

    # ---------------------------------------------------------
    # Fit
    # ---------------------------------------------------------

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "DuplicateHandler":

        return self

    # ---------------------------------------------------------
    # Transform
    # ---------------------------------------------------------

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        self.logger.info(
            "Removing duplicate rows."
        )

        return dataframe.drop_duplicates().reset_index(drop=True)

    # ---------------------------------------------------------
    # Fit + Transform
    # ---------------------------------------------------------

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(dataframe)

        return self.transform(dataframe)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:

        before = self.analyze(dataframe)

        cleaned = self.fit_transform(dataframe)

        after = self.analyze(cleaned)

        return {
            "before": before,
            "after": after,
            "rows_removed": (
                before["duplicate_rows"]
            ),
        }