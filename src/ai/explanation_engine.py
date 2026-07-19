"""
Module: explanation_engine.py

Description:
------------
Enterprise-grade AI Explanation Engine for Credit Card Fraud Detection.

This module converts model predictions and SHAP explanations into a
structured explanation object that can later be consumed by:

- Project Owner Avatar
- LangChain
- Enterprise Reports
- REST APIs
- Streamlit UI
- Evaluation Framework

Author: CreditCardFraudAI
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger_manager import LoggerManager
from src.utils.config_manager import ConfigManager
from src.utils.custom_exception import CreditCardFraudException


# -------------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------------


@dataclass(slots=True)
class FeatureContribution:
    """
    Represents the contribution of a single feature.

    Attributes
    ----------
    rank : int
        Ranking based on SHAP importance.

    feature : str
        Feature name.

    feature_value : Any
        Actual transaction value.

    shap_value : float
        Raw SHAP value.

    absolute_importance : float
        Absolute SHAP magnitude.

    impact : str
        Positive / Negative / Neutral
    """

    rank: int
    feature: str
    feature_value: Any
    shap_value: float
    absolute_importance: float
    impact: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(slots=True)
class ExplanationResult:
    """
    Final explanation object returned by ExplanationEngine.
    """

    prediction: str
    confidence: float
    risk_level: str
    model_name: str

    business_summary: str
    technical_summary: str
    recommendation: str

    top_contributors: List[FeatureContribution] = field(default_factory=list)

    explanation_timestamp: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["top_contributors"] = [
            item.to_dict()
            for item in self.top_contributors
        ]
        return result


# -------------------------------------------------------------------------
# Explanation Engine
# -------------------------------------------------------------------------


class ExplanationEngine:
    """
    Enterprise AI Explanation Engine.

    Responsibilities
    ----------------

    1. Validate model outputs
    2. Process SHAP values
    3. Extract important features
    4. Determine risk level
    5. Build structured explanation
    """

    DEFAULT_TOP_FEATURES = 5

    DEFAULT_THRESHOLDS = {
        "LOW": 0.40,
        "MEDIUM": 0.70,
        "HIGH": 0.90,
    }

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        logger: Optional[Any] = None,
    ) -> None:
        """
        Initialize Explanation Engine.
        """

        self.config = config or ConfigManager()

        self.logger = logger or LoggerManager(
            name=self.__class__.__name__
        ).get_logger()

        self.logger.info(
            "Initializing ExplanationEngine..."
        )

        self.top_features = self.DEFAULT_TOP_FEATURES

        self.thresholds = self.DEFAULT_THRESHOLDS.copy()

        self.logger.info(
            "ExplanationEngine initialized successfully."
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        prediction: int,
        probability: float,
        shap_values: np.ndarray,
        feature_names: List[str],
        feature_values: List[Any],
    ) -> None:
        """
        Validate all inputs.
        """

        if prediction not in (0, 1):
            raise CreditCardFraudException(
                "Prediction must be either 0 or 1."
            )

        if not 0.0 <= probability <= 1.0:
            raise CreditCardFraudException(
                "Probability must lie between 0 and 1."
            )

        if len(feature_names) != len(feature_values):
            raise CreditCardFraudException(
                "Feature names and values length mismatch."
            )

        if len(shap_values) != len(feature_names):
            raise CreditCardFraudException(
                "SHAP values length mismatch."
            )

        self.logger.debug(
            "Input validation successful."
        )

    # ------------------------------------------------------------------
    # Risk Level
    # ------------------------------------------------------------------

    def determine_risk_level(
        self,
        probability: float,
    ) -> str:
        """
        Determine risk category.
        """

        if probability < self.thresholds["LOW"]:
            return "LOW"

        if probability < self.thresholds["MEDIUM"]:
            return "MEDIUM"

        if probability < self.thresholds["HIGH"]:
            return "HIGH"

        return "CRITICAL"

    # ------------------------------------------------------------------
    # Prediction Label
    # ------------------------------------------------------------------

    @staticmethod
    def prediction_label(prediction: int) -> str:
        """
        Convert numeric prediction into text.
        """

        return "Fraud" if prediction == 1 else "Legitimate"

    # ------------------------------------------------------------------
    # SHAP Processing
    # ------------------------------------------------------------------

    @staticmethod
    def _impact(shap_value: float) -> str:
        """
        Determine feature impact direction.
        """

        if shap_value > 0:
            return "Positive"

        if shap_value < 0:
            return "Negative"

        return "Neutral"

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------

    def extract_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        feature_values: List[Any],
        top_n: Optional[int] = None,
    ) -> List[FeatureContribution]:
        """
        Extract top SHAP contributors.
        """

        self.logger.info(
            "Extracting top SHAP contributors..."
        )

        if top_n is None:
            top_n = self.top_features

        absolute_values = np.abs(shap_values)

        sorted_indices = np.argsort(
            absolute_values
        )[::-1]

        contributors: List[
            FeatureContribution
        ] = []

        for rank, idx in enumerate(
            sorted_indices[:top_n],
            start=1,
        ):

            contribution = FeatureContribution(
                rank=rank,
                feature=feature_names[idx],
                feature_value=feature_values[idx],
                shap_value=float(shap_values[idx]),
                absolute_importance=float(
                    absolute_values[idx]
                ),
                impact=self._impact(
                    float(shap_values[idx])
                ),
            )

            contributors.append(contribution)

        self.logger.info(
            "Top %s contributors extracted.",
            len(contributors),
        )

        return contributors

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata(
        probability: float,
        contributors: List[FeatureContribution],
    ) -> Dict:
        """
        Build metadata dictionary.
        """

        return {
            "confidence_percentage": round(
                probability * 100,
                2,
            ),
            "top_feature_count": len(
                contributors
            ),
            "generated_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp() -> str:
        """
        Current UTC timestamp.
        """

        return datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )