"""
===============================================================================
CreditCardFraudAI

Module : explanation_engine.py

Enterprise AI Explanation Engine

Responsibilities
----------------
* Consume SHAP explainability report
* Generate business explanation
* Generate technical explanation
* Generate executive summary
* Generate recommendation
* Produce AIExplanation object

Author : Yogesh Ahuja
Project : CreditCardFraudAI
===============================================================================
"""

from __future__ import annotations

from typing import Dict
from typing import List

import pandas as pd

from src.ai.explanation_models import (
    AIExplanation,
    BusinessSummary,
    ComplianceSummary,
    ExecutiveSummary,
    ExplanationMetadata,
    FeatureContribution,
    TechnicalSummary,
)

from src.core.base import BaseComponent
from src.core.logger import LoggerManager
from src.core.exceptions import ExplainabilityError


class ExplanationEngine(BaseComponent):
    """
    Converts SHAP explainability into business-friendly AI explanations.

    This class does NOT compute SHAP values.

    It consumes:

        SHAPExplainer.generate_explainability_report()

    and produces

        AIExplanation
    """

    ###########################################################################
    # Constructor
    ###########################################################################

    def __init__(
        self,
        logger=None,
    ):

        self.logger = logger or LoggerManager.get_logger(
            self.__class__.__name__
        )

        self.logger.info(
            "Initializing ExplanationEngine..."
        )

    ###########################################################################
    # Validation
    ###########################################################################

    def _validate_report(
        self,
        report: Dict,
    ) -> None:

        required = [

            "prediction",

            "probability",

            "risk_level",

            "top_features",

        ]

        for key in required:

            if key not in report:

                raise ExplainabilityError(
                    f"Missing field '{key}' "
                    "inside explainability report."
                )

    ###########################################################################
    # Confidence
    ###########################################################################

    @staticmethod
    def _confidence(
        probability: float,
    ) -> float:

        return round(probability * 100, 2)

    ###########################################################################
    # Confidence Statement
    ###########################################################################

    def _confidence_statement(
        self,
        probability: float,
    ) -> str:

        if probability >= 0.95:

            return (
                "The model has VERY HIGH confidence "
                "in this prediction."
            )

        if probability >= 0.80:

            return (
                "The model has HIGH confidence "
                "in this prediction."
            )

        if probability >= 0.60:

            return (
                "The model has MODERATE confidence "
                "in this prediction."
            )

        return (
            "The model confidence is LOW. "
            "Manual review is recommended."
        )

    ###########################################################################
    # Recommendation
    ###########################################################################

    def _recommendation(
        self,
        risk_level: str,
    ) -> str:

        mapping = {

            "LOW": "Approve transaction.",

            "MEDIUM": (
                "Manual review recommended."
            ),

            "HIGH": (
                "Temporarily hold the transaction "
                "for investigation."
            ),

            "VERY HIGH": (
                "Block transaction immediately."
            ),

            "CRITICAL": (
                "Block transaction immediately "
                "and notify fraud operations."
            ),
        }

        return mapping.get(
            risk_level.upper(),
            "Manual review required.",
        )

    ###########################################################################
    # Feature Extraction
    ###########################################################################

    def _feature_contributions(
        self,
        top_features: pd.DataFrame,
    ) -> List[FeatureContribution]:

        contributors = []

        if top_features is None:

            return contributors

        for rank, (_, row) in enumerate(
            top_features.iterrows(),
            start=1,
        ):

            contribution = row.get(
                "SHAP",
                row.get(
                    "Contribution",
                    0.0,
                ),
            )

            contributors.append(

                FeatureContribution(

                    feature=row["Feature"],

                    value=row.get(
                        "Value",
                        None,
                    ),

                    contribution=float(
                        contribution
                    ),

                    absolute_contribution=abs(
                        float(contribution)
                    ),

                    rank=rank,

                    impact=(
                        "Positive"
                        if contribution >= 0
                        else "Negative"
                    ),
                )

            )

        return contributors

    ###########################################################################
    # Business Summary
    ###########################################################################

    def _business_summary(
        self,
        prediction: int,
        probability: float,
        risk_level: str,
        features: List[FeatureContribution],
    ) -> BusinessSummary:

        prediction_text = (
            "Fraud"
            if prediction == 1
            else "Legitimate"
        )

        feature_text = ", ".join(

            f.feature

            for f in features[:3]

        )

        summary = (

            f"The transaction has been classified as "

            f"{prediction_text} with "

            f"{self._confidence(probability):.2f}% "

            f"confidence. "

            f"The primary contributing features are "

            f"{feature_text}. "

            f"Overall risk level is "

            f"{risk_level}."

        )

        return BusinessSummary(

            title="Fraud Risk Assessment",

            summary=summary,

            recommendation=self._recommendation(
                risk_level
            ),

            action_required=(
                risk_level.upper()
                in [
                    "HIGH",
                    "VERY HIGH",
                    "CRITICAL",
                ]
            ),
        )

    ###########################################################################
    # Technical Summary
    ###########################################################################

    def _technical_summary(
        self,
        model_name: str,
        prediction: int,
        probability: float,
        risk_level: str,
        features: List[FeatureContribution],
    ) -> TechnicalSummary:

        prediction_text = (
            "Fraud"
            if prediction == 1
            else "Legitimate"
        )

        explanation = (

            f"The {model_name} model predicted "

            f"{prediction_text} with probability "

            f"{probability:.4f}. "

            f"SHAP analysis identified "

            f"{', '.join(f.feature for f in features[:5])} "

            f"as the dominant contributing features."

        )

        return TechnicalSummary(

            model_name=model_name,

            prediction=prediction,

            probability=probability,

            risk_level=risk_level,

            explanation=explanation,

            top_features=features,
        )
        ###########################################################################
    # Executive Summary
    ###########################################################################

    def _executive_summary(
        self,
        probability: float,
        risk_level: str,
    ) -> ExecutiveSummary:
        """
        Generate executive summary.
        """

        confidence = self._confidence(probability)

        headline = (
            "High Risk Fraud Detected"
            if risk_level.upper() in ["HIGH", "VERY HIGH", "CRITICAL"]
            else "Transaction Appears Safe"
        )

        one_line = (
            f"Fraud probability is {confidence:.2f}% "
            f"with {risk_level.upper()} risk."
        )

        return ExecutiveSummary(
            headline=headline,
            risk_level=risk_level,
            confidence=confidence,
            recommendation=self._recommendation(risk_level),
            one_line_summary=one_line,
        )

    ###########################################################################
    # Compliance Summary
    ###########################################################################

    def _compliance_summary(self) -> ComplianceSummary:
        """
        Build compliance summary.
        """

        return ComplianceSummary(
            framework="SHAP",
            explainability_enabled=True,
            audit_ready=True,
            reproducible=True,
            notes=(
                "Prediction supported using SHAP feature attribution."
            ),
        )

    ###########################################################################
    # Metadata
    ###########################################################################

    def _metadata(
        self,
        model_name: str,
    ) -> ExplanationMetadata:
        """
        Build explanation metadata.
        """

        return ExplanationMetadata(
            model_name=model_name,
            explainer_name="SHAPExplainer",
        )

    ###########################################################################
    # Public API
    ###########################################################################

    def generate(
        self,
        explainability_report: Dict,
        model_name: str,
    ) -> AIExplanation:
        """
        Convert SHAP report into enterprise AI explanation.

        Parameters
        ----------
        explainability_report : Dict
            Output from SHAPExplainer.generate_explainability_report()

        model_name : str
            Trained model name.

        Returns
        -------
        AIExplanation
        """

        self.logger.info(
            "Generating AI explanation..."
        )

        try:

            self._validate_report(
                explainability_report
            )

            prediction = explainability_report["prediction"]

            probability = float(
                explainability_report["probability"]
            )

            risk_level = explainability_report["risk_level"]

            top_features = explainability_report[
                "top_features"
            ]

            if top_features is None:
                top_features = pd.DataFrame()

            contributors = self._feature_contributions(
                top_features
            )

            business_summary = self._business_summary(
                prediction,
                probability,
                risk_level,
                contributors,
            )

            technical_summary = self._technical_summary(
                model_name,
                prediction,
                probability,
                risk_level,
                contributors,
            )

            executive_summary = self._executive_summary(
                probability,
                risk_level,
            )

            compliance_summary = (
                self._compliance_summary()
            )

            metadata = self._metadata(
                model_name
            )

            explanation = AIExplanation(

                prediction=prediction,

                probability=probability,

                confidence=self._confidence(
                    probability
                ),

                risk_level=risk_level,

                confidence_statement=(
                    self._confidence_statement(
                        probability
                    )
                ),

                feature_contributions=contributors,

                business_summary=business_summary,

                technical_summary=technical_summary,

                executive_summary=executive_summary,

                compliance_summary=compliance_summary,

                metadata=metadata,

                raw_report=explainability_report,
            )

            self.logger.info(
                "AI explanation generated successfully."
            )

            return explanation

        except Exception as ex:

            self.logger.exception(ex)

            raise ExplainabilityError(
                f"Unable to generate AI explanation : {ex}"
            )

    ###########################################################################
    # BaseComponent Implementation
    ###########################################################################

    def execute(
        self,
        explainability_report: Dict,
        model_name: str,
    ) -> AIExplanation:
        """
        Default execution entry point.
        """

        return self.generate(
            explainability_report=explainability_report,
            model_name=model_name,
        )

    ###########################################################################
    # Convenience Methods
    ###########################################################################

    def generate_dict(
        self,
        explainability_report: Dict,
        model_name: str,
    ) -> Dict:
        """
        Generate explanation as dictionary.
        """

        return self.generate(
            explainability_report,
            model_name,
        ).to_dict()

    ###########################################################################
    # String Representation
    ###########################################################################

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            "source='SHAPExplainer')"
        )