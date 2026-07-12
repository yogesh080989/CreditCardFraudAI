"""
Model evaluation utilities for CreditCardFraudAI.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.core.logger import LoggerManager


class ModelEvaluator:
    """
    Evaluate trained machine learning models.
    """

    def __init__(self) -> None:

        self.logger = LoggerManager.get_logger()

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """
        Evaluate a trained model.
        """

        self.logger.info(
            "Evaluating model: %s",
            model.__class__.__name__,
        )

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = None

        report = {
            "model_name": model.__class__.__name__,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix(
                y_test,
                y_pred,
            ),
            "classification_report": classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
        }

        return report