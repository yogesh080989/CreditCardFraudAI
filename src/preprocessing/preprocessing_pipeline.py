"""
Preprocessing pipeline for CreditCardFraudAI.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.core.config import ConfigManager
from src.core.exceptions import PreprocessingError
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

    Pipeline Steps
    --------------
    1. Missing value handling
    2. Duplicate removal
    3. Feature scaling
    4. Train/Test split
    5. Class imbalance handling
    6. Save processed datasets
    """

    def __init__(
        self,
        missing_strategy: str = "ignore",
        scaling_strategy: str = "standard",
        imbalance_strategy: str = "smote",
    ) -> None:

        self.logger = LoggerManager.get_logger(
            self.__class__.__name__
        )

        self.config = ConfigManager()

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

    ###########################################################################
    # Run Pipeline
    ###########################################################################

    def run(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        """
        Execute complete preprocessing pipeline.
        """

        try:

            self.logger.info(
                "=" * 70
            )
            self.logger.info(
                "Starting preprocessing pipeline."
            )

            ###############################################################
            # Missing Values
            ###############################################################

            dataframe = (
                self.missing_handler.fit_transform(
                    dataframe
                )
            )

            ###############################################################
            # Duplicates
            ###############################################################

            dataframe = (
                self.duplicate_handler.fit_transform(
                    dataframe
                )
            )

            ###############################################################
            # Feature Scaling
            ###############################################################

            dataframe = (
                self.scaler.fit_transform(
                    dataframe
                )
            )

            ###############################################################
            # Train Test Split
            ###############################################################

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = self.splitter.split(
                dataframe
            )

            ###############################################################
            # Handle Imbalance
            ###############################################################

            (
                X_train,
                y_train,
            ) = self.imbalance_handler.fit_resample(
                X_train,
                y_train,
            )

            ###############################################################
            # Save Processed Data
            ###############################################################

            self._save_processed_data(
                X_train,
                X_test,
                y_train,
                y_test,
            )

            self.logger.info(
                "Preprocessing pipeline completed successfully."
            )

            self.logger.info(
                "=" * 70
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test,
            )

        except Exception as ex:

            self.logger.exception(ex)

            raise PreprocessingError(
                f"Preprocessing pipeline failed: {ex}"
            )

    ###########################################################################
    # Save Processed Dataset
    ###########################################################################

    def _save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """
        Persist processed datasets.
        """

        try:

            relative_path = Path(
                self.config.get(
                    "paths.processed_data",
                    "data/processed",
                )
            )

            processed_dir = (
                    self.config.project_root
                    / relative_path
            )

            processed_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            X_train.to_csv(
                processed_dir / "X_train.csv",
                index=False,
            )

            X_test.to_csv(
                processed_dir / "X_test.csv",
                index=False,
            )

            y_train.to_frame(
                name="Class"
            ).to_csv(
                processed_dir / "y_train.csv",
                index=False,
            )

            y_test.to_frame(
                name="Class"
            ).to_csv(
                processed_dir / "y_test.csv",
                index=False,
            )

            self.logger.info(
                "Processed datasets saved successfully."
            )

            self.logger.info(
                "Location : %s",
                processed_dir.resolve(),
            )

        except Exception as ex:

            raise PreprocessingError(
                f"Unable to save processed data: {ex}"
            )