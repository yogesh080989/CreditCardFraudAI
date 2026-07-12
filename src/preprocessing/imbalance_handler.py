"""
Class imbalance handling utilities for CreditCardFraudAI.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

import pandas as pd

from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.core.logger import LoggerManager


class ImbalanceHandler:
    """
    Handle class imbalance using different sampling techniques.
    """

    VALID_STRATEGIES = {
        "none",
        "smote",
        "adasyn",
        "random_under",
    }

    def __init__(
        self,
        strategy: str = "smote",
        random_state: int = 42,
    ) -> None:

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy: {strategy}"
            )

        self.strategy = strategy
        self.random_state = random_state

        self.logger = LoggerManager.get_logger()

        self.sampler = None

    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):

        self.logger.info(
            "Applying imbalance strategy: %s",
            self.strategy,
        )

        if self.strategy == "none":
            return X.copy(), y.copy()

        if self.strategy == "smote":
            self.sampler = SMOTE(
                random_state=self.random_state
            )

        elif self.strategy == "adasyn":
            self.sampler = ADASYN(
                random_state=self.random_state
            )

        elif self.strategy == "random_under":
            self.sampler = RandomUnderSampler(
                random_state=self.random_state
            )

        X_resampled, y_resampled = self.sampler.fit_resample(
            X,
            y,
        )

        self.logger.info(
            "Original Shape : %s",
            X.shape,
        )

        self.logger.info(
            "Resampled Shape : %s",
            X_resampled.shape,
        )

        return X_resampled, y_resampled

    def summary(self) -> dict:

        return {
            "strategy": self.strategy,
            "sampler": (
                self.sampler.__class__.__name__
                if self.sampler
                else None
            ),
        }