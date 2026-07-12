"""
Model comparison utilities for CreditCardFraudAI.

This module compares evaluation metrics from multiple
machine learning models.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class ModelComparison:
    """
    Compare evaluation metrics across multiple models.
    """

    DEFAULT_METRIC = "f1_score"

    def compare(
        self,
        evaluation_results: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Convert evaluation results into a comparison DataFrame.

        Parameters
        ----------
        evaluation_results : list
            Output from ModelEvaluator.

        Returns
        -------
        pandas.DataFrame
        """

        comparison = []

        for result in evaluation_results:

            comparison.append(
                {
                    "Model": result["model_name"],
                    "Accuracy": result["accuracy"],
                    "Precision": result["precision"],
                    "Recall": result["recall"],
                    "F1 Score": result["f1_score"],
                    "ROC AUC": result["roc_auc"],
                }
            )

        df = pd.DataFrame(comparison)

        df = df.sort_values(
            by="F1 Score",
            ascending=False,
        ).reset_index(drop=True)

        return df

    # ---------------------------------------------------------

    def best_model(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "F1 Score",
    ) -> dict[str, Any]:
        """
        Return the best-performing model.
        """

        row = comparison_df.sort_values(
            metric,
            ascending=False,
        ).iloc[0]

        return row.to_dict()

    # ---------------------------------------------------------

    def rank_models(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "F1 Score",
    ) -> pd.DataFrame:
        """
        Rank models based on a selected metric.
        """

        ranking = comparison_df.sort_values(
            metric,
            ascending=False,
        ).reset_index(drop=True)

        ranking.insert(
            0,
            "Rank",
            range(1, len(ranking) + 1),
        )

        return ranking

    # ---------------------------------------------------------

    def summary(
        self,
        comparison_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Return summary statistics.
        """

        best = self.best_model(comparison_df)

        return {
            "total_models": len(comparison_df),
            "best_model": best["Model"],
            "best_f1_score": best["F1 Score"],
            "best_roc_auc": comparison_df["ROC AUC"].max(),
        }