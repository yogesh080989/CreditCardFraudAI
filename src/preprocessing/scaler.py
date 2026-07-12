"""
Feature scaling utilities for CreditCardFraudAI.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

import pandas as pd

from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from src.core.logger import LoggerManager


class FeatureScaler:
    """
    Scale numerical features using different scaling strategies.
    """

    VALID_STRATEGIES = {
        "standard",
        "minmax",
        "robust",
        "none",
    }

    def __init__(
        self,
        strategy: str = "standard",
        target_column: str = "Class",
    ) -> None:

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unsupported scaling strategy: {strategy}"
            )

        self.strategy = strategy
        self.target_column = target_column

        self.logger = LoggerManager.get_logger()

        self.scaler = None

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "FeatureScaler":

        if self.strategy == "none":
            return self

        features = dataframe.drop(
            columns=[self.target_column],
            errors="ignore",
        )

        if self.strategy == "standard":
            self.scaler = StandardScaler()

        elif self.strategy == "minmax":
            self.scaler = MinMaxScaler()

        elif self.strategy == "robust":
            self.scaler = RobustScaler()

        self.scaler.fit(features)

        return self

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.strategy == "none":
            return dataframe.copy()

        if self.scaler is None:
            raise RuntimeError(
                "Scaler has not been fitted."
            )

        df = dataframe.copy()

        feature_columns = [
            column
            for column in df.columns
            if column != self.target_column
        ]

        df[feature_columns] = self.scaler.transform(
            df[feature_columns]
        )

        return df

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(dataframe)

        return self.transform(dataframe)

    def summary(self) -> dict:

        return {
            "strategy": self.strategy,
            "scaler": (
                self.scaler.__class__.__name__
                if self.scaler
                else None
            ),
        }