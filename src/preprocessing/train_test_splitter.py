"""
Train/Test splitting utilities for CreditCardFraudAI.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split

from src.core.logger import LoggerManager


class TrainTestSplitter:
    """
    Wrapper around sklearn train_test_split.
    """

    def __init__(
        self,
        target_column: str = "Class",
        test_size: float = 0.20,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

        self.logger = LoggerManager.get_logger()

    def split(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Split dataframe into train and test sets.
        """

        self.logger.info(
            "Splitting dataset into train and test."
        )

        X = dataframe.drop(
            columns=[self.target_column]
        )

        y = dataframe[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )

        self.logger.info(
            "Train Shape : %s",
            X_train.shape,
        )

        self.logger.info(
            "Test Shape : %s",
            X_test.shape,
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )

    def summary(self) -> dict:

        return {
            "target_column": self.target_column,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratify": self.stratify,
        }