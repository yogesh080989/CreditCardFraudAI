"""
Data profiling module for CreditCardFraudAI.

This module provides the DataProfiler class which generates
comprehensive statistics about a dataset.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.config import ConfigManager
from src.core.logger import LoggerManager
from src.core.exceptions import DataProfilingError


class DataProfiler:
    """
    Generates profiling statistics for a pandas DataFrame.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: ConfigManager | None = None,
    ) -> None:

        if dataframe is None or dataframe.empty:
            raise DataProfilingError(
                "Cannot profile an empty dataframe."
            )

        self.df = dataframe

        self.config = config or ConfigManager()

        self.logger = LoggerManager.get_logger()

        self.target_column = self.config.get(
            "target_column",
            "Class",
        )

    # ---------------------------------------------------------
    # Main Method
    # ---------------------------------------------------------

    def profile(self) -> dict[str, Any]:
        """
        Generate complete profiling report.
        """

        self.logger.info("Starting data profiling.")

        report = {
            "dataset": self.dataset_summary(),
            "memory": self.memory_summary(),
            "missing": self.missing_summary(),
            "duplicates": self.duplicate_summary(),
            "target_distribution": self.target_summary(),
            "numerical_statistics": self.numerical_summary(),
            "outliers": self.outlier_summary(),
            "correlations": self.correlation_summary(),
        }

        self.logger.info("Data profiling completed.")

        return report

    # ---------------------------------------------------------
    # Dataset Summary
    # ---------------------------------------------------------

    def dataset_summary(self) -> dict[str, Any]:

        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": list(self.df.columns),
            "data_types": {
                col: str(dtype)
                for col, dtype in self.df.dtypes.items()
            },
        }

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------

    def memory_summary(self) -> dict[str, Any]:

        memory = self.df.memory_usage(deep=True)

        return {
            "total_memory_mb": round(
                memory.sum() / (1024 * 1024),
                2,
            ),
            "memory_per_column": {
                col: round(value / 1024, 2)
                for col, value in memory.items()
            },
        }

    # ---------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------

    def missing_summary(self) -> dict[str, Any]:

        missing = self.df.isnull().sum()

        return {
            "total_missing": int(missing.sum()),
            "missing_per_column": missing.to_dict(),
            "missing_percentage": (
                missing / len(self.df) * 100
            ).round(2).to_dict(),
        }

    # ---------------------------------------------------------
    # Duplicates
    # ---------------------------------------------------------

    def duplicate_summary(self) -> dict[str, Any]:

        duplicate_count = int(self.df.duplicated().sum())

        return {
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": round(
                duplicate_count / len(self.df) * 100,
                2,
            ),
        }

    # ---------------------------------------------------------
    # Target Distribution
    # ---------------------------------------------------------

    def target_summary(self) -> dict[str, Any]:

        if self.target_column not in self.df.columns:
            return {}

        target = self.df[self.target_column]

        value_counts = target.value_counts()

        fraud = int(value_counts.get(1, 0))
        genuine = int(value_counts.get(0, 0))

        return {
            "fraud": fraud,
            "non_fraud": genuine,
            "fraud_percentage": round(
                fraud / len(self.df) * 100,
                4,
            ),
            "class_distribution": value_counts.to_dict(),
        }

    # ---------------------------------------------------------
    # Numerical Statistics
    # ---------------------------------------------------------

    def numerical_summary(self) -> dict[str, Any]:

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        report = {}

        for column in numerical_columns:

            series = self.df[column]

            report[column] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "variance": float(series.var()),
                "min": float(series.min()),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurt()),
            }

        return report

    # ---------------------------------------------------------
    # Outliers
    # ---------------------------------------------------------

    def outlier_summary(self) -> dict[str, Any]:

        report = {}

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for column in numerical_columns:

            q1 = self.df[column].quantile(0.25)
            q3 = self.df[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)

            outliers = self.df[
                (self.df[column] < lower)
                | (self.df[column] > upper)
            ]

            report[column] = {
                "count": len(outliers),
                "percentage": round(
                    len(outliers) / len(self.df) * 100,
                    2,
                ),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }

        return report

    # ---------------------------------------------------------
    # Correlation
    # ---------------------------------------------------------

    def correlation_summary(self) -> dict[str, Any]:

        corr = self.df.corr(numeric_only=True)

        correlations = []

        columns = corr.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):

                correlations.append(
                    {
                        "feature_1": columns[i],
                        "feature_2": columns[j],
                        "correlation": float(
                            corr.iloc[i, j]
                        ),
                    }
                )

        correlations = sorted(
            correlations,
            key=lambda x: abs(x["correlation"]),
            reverse=True,
        )

        return {
            "top_correlations": correlations[:20]
        }

    # ---------------------------------------------------------
    # Console Summary
    # ---------------------------------------------------------

    def print_summary(self) -> None:

        report = self.profile()

        print("\n========== DATA PROFILE ==========")

        print(report["dataset"])

        print(report["missing"])

        print(report["duplicates"])

        print(report["target_distribution"])

        print("==================================")