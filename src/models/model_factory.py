"""
Model factory for CreditCardFraudAI.

This module provides a centralized factory for creating
machine learning models.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


class ModelFactory:
    """
    Factory class for creating ML models.
    """

    SUPPORTED_MODELS = {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }

    def __init__(
        self,
        random_state: int = 42,
    ) -> None:

        self.random_state = random_state

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def get_model(
        self,
        model_name: str,
    ):

        model_name = model_name.lower()

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        if model_name == "logistic_regression":
            return self._logistic_regression()

        if model_name == "decision_tree":
            return self._decision_tree()

        if model_name == "random_forest":
            return self._random_forest()

        if model_name == "xgboost":
            return self._xgboost()

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    # ---------------------------------------------------------
    # Individual Model Builders
    # ---------------------------------------------------------

    def _logistic_regression(self):

        return LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight="balanced",
        )

    def _decision_tree(self):

        return DecisionTreeClassifier(
            random_state=self.random_state,
        )

    def _random_forest(self):

        return RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _xgboost(self):

        return XGBClassifier(
            random_state=self.random_state,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="logloss",
            use_label_encoder=False,
        )

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def available_models(self) -> list[str]:
        """
        Return supported model names.
        """

        return sorted(self.SUPPORTED_MODELS)