"""
Missing value handling utilities for CreditCardFraudAI.

This module provides reusable strategies for handling missing values
in tabular datasets.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.logger import LoggerManager


class MissingValueHandler:
    """
    Handles missing values using configurable strategies.
    """

    VALID_STRATEGIES = {
        "mean",
        "median",
        "mode",
        "constant",
        "drop_rows",
        "drop_columns",
        "ignore",
    }

    def __init__(
        self,
        strategy: str = "ignore",
        fill_value: Any = None,
    ) -> None:

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy: {strategy}"
            )

        self.strategy = strategy
        self.fill_value = fill_value

        self.logger = LoggerManager.get_logger()

        self.statistics_: dict[str, Any] = {}

    # -----------------------------------------------------
    # Analysis
    # -----------------------------------------------------

    def analyze(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:

        missing = dataframe.isna().sum()

        return {
            "total_missing": int(missing.sum()),
            "missing_per_column": missing.to_dict(),
        }

    # -----------------------------------------------------
    # Fit
    # -----------------------------------------------------

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "MissingValueHandler":

        if self.strategy == "mean":

            for column in dataframe.select_dtypes(
                include="number"
            ):
                self.statistics_[column] = dataframe[
                    column
                ].mean()

        elif self.strategy == "median":

            for column in dataframe.select_dtypes(
                include="number"
            ):
                self.statistics_[column] = dataframe[
                    column
                ].median()

        elif self.strategy == "mode":

            for column in dataframe.columns:
                mode = dataframe[column].mode()

                if not mode.empty:
                    self.statistics_[column] = mode.iloc[0]

        elif self.strategy == "constant":

            for column in dataframe.columns:
                self.statistics_[column] = self.fill_value

        return self

    # -----------------------------------------------------
    # Transform
    # -----------------------------------------------------

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        self.logger.info(
            "Applying missing value strategy: %s",
            self.strategy,
        )

        if self.strategy == "ignore":
            return df

        if self.strategy == "drop_rows":
            return df.dropna()

        if self.strategy == "drop_columns":
            return df.dropna(axis=1)

        return df.fillna(self.statistics_)

    # -----------------------------------------------------
    # Fit + Transform
    # -----------------------------------------------------

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(dataframe)

        return self.transform(dataframe)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    def summary(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:

        before = self.analyze(dataframe)

        after = self.analyze(
            self.fit_transform(dataframe)
        )

        return {
            "strategy": self.strategy,
            "before": before,
            "after": after,
        }