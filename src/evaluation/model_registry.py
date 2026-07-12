"""
Enterprise Model Registry for CreditCardFraudAI.

Author: Yogesh Ahuja
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from src.core.config import ConfigManager
from src.core.logger import LoggerManager


class ModelRegistry:
    """
    Enterprise model registry.

    Responsible for:

    • Saving trained models
    • Loading models
    • Saving metadata
    • Listing registered models
    """

    def __init__(self):

        self.logger = LoggerManager.get_logger()

        self.config = ConfigManager()

        self.project_root = self.config.project_root

        self.model_directory = (
            self.project_root
            / self.config.get("model_directory")
        )

        self.metadata_directory = (
            self.project_root
            / self.config.get("metadata_directory")
        )

        self.model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------

    def save_model(
        self,
        model,
        metadata: dict[str, Any],
    ) -> Path:

        model_name = model.__class__.__name__

        model_path = (
            self.model_directory
            / f"{model_name}.pkl"
        )

        metadata_path = (
            self.metadata_directory
            / f"{model_name}.json"
        )

        self.logger.info(
            "Saving model: %s",
            model_name,
        )

        joblib.dump(
            model,
            model_path,
        )

        metadata = metadata.copy()

        metadata["saved_timestamp"] = (
            datetime.now().isoformat()
        )

        metadata["model_path"] = str(model_path)

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        self.logger.info(
            "Model saved successfully."
        )

        return model_path

    # -----------------------------------------------------

    def load_model(
        self,
        model_name: str,
    ):

        model_path = (
            self.model_directory
            / f"{model_name}.pkl"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                model_path
            )

        self.logger.info(
            "Loading model: %s",
            model_name,
        )

        return joblib.load(
            model_path
        )

    # -----------------------------------------------------

    def load_metadata(
        self,
        model_name: str,
    ) -> dict:

        metadata_path = (
            self.metadata_directory
            / f"{model_name}.json"
        )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # -----------------------------------------------------

    def list_models(
        self,
    ) -> list[str]:

        return sorted(
            file.stem
            for file in self.model_directory.glob(
                "*.pkl"
            )
        )