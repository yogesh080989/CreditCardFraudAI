"""
Preprocessing pipeline for CreditCardFraudAI.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

import pandas as pd

from src.core.logger import LoggerManager
from src.preprocessing.missing_value_handler import (
    MissingValueHandler,
)
from src.preprocessing.duplicate_handler import (
    DuplicateHandler,
)
from src.preprocessing.scaler import (
    FeatureScaler,
)
from src.preprocessing.train_test_splitter import (
    TrainTestSplitter,
)
from src.preprocessing.imbalance_handler import (
    ImbalanceHandler,
)


class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline.
    """

    def __init__(
        self,
        missing_strategy: str = "ignore",
        scaling_strategy: str = "standard",
        imbalance_strategy: str = "smote",
    ) -> None:

        self.logger = LoggerManager.get_logger()

        self.missing_handler = MissingValueHandler(
            strategy=missing_strategy
        )

        self.duplicate_handler = DuplicateHandler()

        self.scaler = FeatureScaler(
            strategy=scaling_strategy
        )

        self.splitter = TrainTestSplitter()

        self.imbalance_handler = ImbalanceHandler(
            strategy=imbalance_strategy
        )

    def run(
        self,
        dataframe: pd.DataFrame,
    ):

        self.logger.info(
            "Starting preprocessing pipeline."
        )

        # ----------------------------------------

        dataframe = self.missing_handler.fit_transform(
            dataframe
        )

        # ----------------------------------------

        dataframe = self.duplicate_handler.fit_transform(
            dataframe
        )

        # ----------------------------------------

        dataframe = self.scaler.fit_transform(
            dataframe
        )

        # ----------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.splitter.split(dataframe)

        # ----------------------------------------

        (
            X_train,
            y_train,
        ) = self.imbalance_handler.fit_resample(
            X_train,
            y_train,
        )

        self.logger.info(
            "Preprocessing pipeline completed."
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )