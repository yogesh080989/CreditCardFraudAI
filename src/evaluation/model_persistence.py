"""
===============================================================================
CreditCardFraudAI
Module : Model Persistence

Enterprise-grade persistence utility for machine learning artifacts.

Features
--------
* Save trained models
* Load trained models
* Save metrics
* Load metrics
* Save predictions
* Load predictions
* Generic object persistence

Author : Yogesh Ahuja
Project : CreditCardFraudAI
===============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import pandas as pd

from src.core.base import BaseComponent
from src.core.exceptions import ModelSerializationError
from src.core.logger import LoggerManager


class ModelPersistence(BaseComponent):
    """
    Enterprise persistence utility for ML artifacts.
    """

    def __init__(self, logger=None) -> None:

        super().__init__()

        self.logger = logger or LoggerManager.get_logger(
            self.__class__.__name__
        )

    ###########################################################################
    # Generic Object
    ###########################################################################

    @staticmethod
    def save_object(
        obj: Any,
        file_path: Union[str, Path],
    ) -> Path:
        """
        Save any Python object using joblib.
        """

        try:

            file_path = Path(file_path)

            file_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            joblib.dump(
                obj,
                file_path,
            )

            return file_path

        except Exception as ex:

            raise ModelSerializationError(
                f"Unable to save object : {ex}"
            )

    @staticmethod
    def load_object(
        file_path: Union[str, Path],
    ) -> Any:
        """
        Load serialized object.
        """

        try:

            file_path = Path(file_path)

            if not file_path.exists():

                raise FileNotFoundError(
                    file_path
                )

            return joblib.load(
                file_path
            )

        except Exception as ex:

            raise ModelSerializationError(
                f"Unable to load object : {ex}"
            )

    ###########################################################################
    # Model
    ###########################################################################

    @classmethod
    def save_model(
        cls,
        model: Any,
        file_path: Union[str, Path],
    ) -> Path:
        """
        Save trained model.
        """

        return cls.save_object(
            model,
            file_path,
        )

    @classmethod
    def load_model(
        cls,
        file_path: Union[str, Path],
    ) -> Any:
        """
        Load trained model.
        """

        return cls.load_object(
            file_path,
        )

    ###########################################################################
    # Metrics
    ###########################################################################

    @staticmethod
    def save_metrics(
        metrics: Dict,
        file_path: Union[str, Path],
    ) -> Path:
        """
        Save evaluation metrics as JSON.
        """

        try:

            file_path = Path(file_path)

            file_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                file_path,
                "w",
                encoding="utf-8",
            ) as fp:

                json.dump(
                    metrics,
                    fp,
                    indent=4,
                )

            return file_path

        except Exception as ex:

            raise ModelSerializationError(
                f"Unable to save metrics : {ex}"
            )

    @staticmethod
    def load_metrics(
        file_path: Union[str, Path],
    ) -> Dict:
        """
        Load metrics JSON.
        """

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as fp:

                return json.load(fp)

        except Exception as ex:

            raise ModelSerializationError(
                f"Unable to load metrics : {ex}"
            )

    ###########################################################################
    # Predictions
    ###########################################################################

    @staticmethod
    def save_predictions(
        predictions: pd.DataFrame,
        file_path: Union[str, Path],
    ) -> Path:
        """
        Save predictions to CSV.
        """

        try:

            file_path = Path(file_path)

            file_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            predictions.to_csv(
                file_path,
                index=False,
            )

            return file_path

        except Exception as ex:

            raise ModelSerializationError(
                f"Unable to save predictions : {ex}"
            )

    @staticmethod
    def load_predictions(
        file_path: Union[str, Path],
    ) -> pd.DataFrame:
        """
        Load predictions CSV.
        """

        try:

            return pd.read_csv(
                file_path
            )

        except Exception as ex:

            raise ModelSerializationError(
                f"Unable to load predictions : {ex}"
            )

    ###########################################################################
    # Exists
    ###########################################################################

    @staticmethod
    def exists(
        file_path: Union[str, Path],
    ) -> bool:
        """
        Check whether an artifact exists.
        """

        return Path(file_path).exists()

    ###########################################################################
    # Delete
    ###########################################################################

    @staticmethod
    def delete(
        file_path: Union[str, Path],
    ) -> None:
        """
        Delete an artifact.
        """

        file_path = Path(file_path)

        if file_path.exists():

            file_path.unlink()

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            "()"
        )