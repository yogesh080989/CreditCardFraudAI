"""
Feature engineering utilities for CreditCardFraudAI.

This module prepares the feature matrix and target vector
for machine learning models.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.logger import LoggerManager


class FeatureEngineering:
    """
    Prepare features and target for model training.
    """

    def __init__(
        self,
        target_column: str = "Class",
        exclude_columns: list[str] | None = None,
    ) -> None:
        """
        Initialize FeatureEngineering.

        Parameters
        ----------
        target_column : str
            Target column name.

        exclude_columns : list[str] | None
            Optional columns to exclude from features.
        """

        self.target_column = target_column
        self.exclude_columns = exclude_columns or []

        self.logger = LoggerManager.get_logger()

        self.feature_columns_: list[str] = []

    # ---------------------------------------------------------
    # Transform
    # ---------------------------------------------------------

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split dataframe into feature matrix (X)
        and target vector (y).
        """

        self.logger.info(
            "Preparing feature matrix and target vector."
        )

        columns_to_drop = [
            self.target_column,
            *self.exclude_columns,
        ]

        X = dataframe.drop(
            columns=columns_to_drop,
            errors="ignore",
        )

        y = dataframe[self.target_column]

        self.feature_columns_ = list(X.columns)

        self.logger.info(
            "Feature count : %d",
            len(self.feature_columns_),
        )

        return X, y

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Return feature engineering summary.
        """

        return {
            "target_column": self.target_column,
            "feature_count": len(self.feature_columns_),
            "feature_columns": self.feature_columns_,
        }

    # ---------------------------------------------------------
    # Feature Names
    # ---------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """
        Return feature names.
        """

        return self.feature_columns_

    # ---------------------------------------------------------
    # Feature Count
    # ---------------------------------------------------------

    def get_feature_count(self) -> int:
        """
        Return total feature count.
        """

        return len(self.feature_columns_)