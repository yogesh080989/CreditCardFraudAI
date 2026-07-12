"""
Configuration manager for CreditCardFraudAI.

This module provides a centralized configuration manager that loads
application settings from a YAML configuration file.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ConfigurationError


class ConfigManager:
    """
    Centralized configuration manager.

    Automatically locates the project root and loads config/config.yaml.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        """
        Initialize the configuration manager.

        Parameters
        ----------
        config_path : str | Path | None
            Optional custom configuration file path.
            If None, the default config/config.yaml from the project root
            will be used.
        """

        # CreditCardFraudAI/
        # ├── config/
        # │   └── config.yaml
        # └── src/
        #     └── core/
        #         └── config.py

        self.project_root = Path(__file__).resolve().parents[2]

        if config_path is None:
            self.config_path = (
                self.project_root
                / "config"
                / "config.yaml"
            )
        else:
            self.config_path = Path(config_path).resolve()

        self._config: dict[str, Any] = {}

        self._load()

    def _load(self) -> None:
        """
        Load configuration from YAML.
        """

        if not self.config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}"
            )

        try:
            with self.config_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                self._config = yaml.safe_load(file) or {}

        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Invalid YAML configuration: {exc}"
            ) from exc

        except Exception as exc:
            raise ConfigurationError(
                f"Unable to load configuration: {exc}"
            ) from exc

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a configuration value.

        Supports nested keys using dot notation.

        Example
        -------
        config.get("model.learning_rate")
        """

        keys = key.split(".")

        value: Any = self._config

        for item in keys:

            if isinstance(value, dict) and item in value:
                value = value[item]
            else:
                return default

        return value

    def get_all(self) -> dict[str, Any]:
        """
        Return the complete configuration.
        """
        return self._config

    def contains(self, key: str) -> bool:
        """
        Check whether a configuration key exists.
        """
        return self.get(key) is not None

    def reload(self) -> None:
        """
        Reload configuration from disk.
        """
        self._load()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(config_path='{self.config_path}')"
        )