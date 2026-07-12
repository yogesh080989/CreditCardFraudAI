"""
Model training utilities for CreditCardFraudAI.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd

from src.core.logger import LoggerManager


class ModelTrainer:
    """
    Train machine learning models and maintain
    training metadata history.
    """

    def __init__(self) -> None:

        self.logger = LoggerManager.get_logger()

        self.training_metadata_: dict[str, Any] = {}

        self.training_history_: list[dict[str, Any]] = []

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    def train(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        """
        Train supplied model.
        """

        self.logger.info(
            "Training model: %s",
            model.__class__.__name__,
        )

        start = perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        elapsed = round(
            perf_counter() - start,
            3,
        )

        self.training_metadata_ = {
            "model_name": model.__class__.__name__,
            "training_time_seconds": elapsed,
            "training_timestamp": datetime.now().isoformat(),
            "training_samples": len(X_train),
            "feature_count": X_train.shape[1],
        }

        self.training_history_.append(
            self.training_metadata_.copy()
        )

        self.logger.info(
            "Training completed in %.3f seconds.",
            elapsed,
        )

        return model

    # ---------------------------------------------------------
    # Latest Summary
    # ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:

        return self.training_metadata_

    # ---------------------------------------------------------
    # History
    # ---------------------------------------------------------

    def history(self) -> list[dict[str, Any]]:

        return self.training_history_