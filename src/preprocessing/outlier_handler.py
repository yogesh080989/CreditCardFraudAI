"""
Outlier handling utilities for CreditCardFraudAI.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

import pandas as pd

from src.core.logger import LoggerManager


class OutlierHandler:
    """
    Detect and handle outliers using the IQR method.
    """

    VALID_STRATEGIES = {
        "ignore",
        "remove",
        "cap",
    }

    def __init__(
        self,
        strategy: str = "ignore",
    ) -> None:

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy: {strategy}"
            )

        self.strategy = strategy

        self.logger = LoggerManager.get_logger()

        self.bounds_: dict[str, tuple[float, float]] = {}

    # ---------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------

    def analyze(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:

        report = {}

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:

            q1 = dataframe[column].quantile(0.25)
            q3 = dataframe[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = (
                (
                    (dataframe[column] < lower)
                    | (dataframe[column] > upper)
                )
            ).sum()

            report[column] = {
                "count": int(count),
                "percentage": round(
                    count / len(dataframe) * 100,
                    2,
                ),
                "lower_bound": lower,
                "upper_bound": upper,
            }

        return report

    # ---------------------------------------------------------
    # Fit
    # ---------------------------------------------------------

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "OutlierHandler":

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:

            q1 = dataframe[column].quantile(0.25)
            q3 = dataframe[column].quantile(0.75)

            iqr = q3 - q1

            self.bounds_[column] = (
                q1 - 1.5 * iqr,
                q3 + 1.5 * iqr,
            )

        return self

    # ---------------------------------------------------------
    # Transform
    # ---------------------------------------------------------

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        if self.strategy == "ignore":
            return df

        if self.strategy == "remove":

            for column, (lower, upper) in self.bounds_.items():

                df = df[
                    (df[column] >= lower)
                    & (df[column] <= upper)
                ]

            return df.reset_index(drop=True)

        if self.strategy == "cap":

            for column, (lower, upper) in self.bounds_.items():

                df[column] = df[column].clip(
                    lower=lower,
                    upper=upper,
                )

            return df

        return df

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

        return {
            "strategy": self.strategy,
            "outliers": self.analyze(dataframe),
        }