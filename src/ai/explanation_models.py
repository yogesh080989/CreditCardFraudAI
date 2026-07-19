"""
===============================================================================
CreditCardFraudAI

Module : explanation_models.py

Description
-----------
Domain models for the AI Explanation Layer.

These models provide strongly typed objects exchanged between:

    SHAPExplainer
            │
            ▼
    ExplanationEngine
            │
            ▼
    PromptManager
            │
            ▼
    ProjectOwnerAvatar
            │
            ▼
      Enterprise Reports

The models intentionally contain no business logic. They only represent
structured explanation data.

Author : Yogesh Ahuja
Project : CreditCardFraudAI
===============================================================================
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd


###############################################################################
# Feature Contribution
###############################################################################


@dataclass(slots=True)
class FeatureContribution:
    """
    Represents a single feature contribution.

    Attributes
    ----------
    feature
        Feature name.

    value
        Actual feature value.

    contribution
        SHAP contribution.

    absolute_contribution
        Absolute SHAP contribution.

    rank
        Importance rank.

    impact
        Positive / Negative / Neutral.
    """

    feature: str

    value: Any

    contribution: float

    absolute_contribution: float

    rank: int

    impact: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


###############################################################################
# Business Summary
###############################################################################


@dataclass(slots=True)
class BusinessSummary:

    title: str

    summary: str

    recommendation: str

    action_required: bool = True

    def to_dict(self):

        return asdict(self)


###############################################################################
# Technical Summary
###############################################################################


@dataclass(slots=True)
class TechnicalSummary:

    model_name: str

    prediction: int

    probability: float

    risk_level: str

    explanation: str

    top_features: List[FeatureContribution]

    def to_dict(self):

        result = asdict(self)

        result["top_features"] = [
            f.to_dict()
            for f in self.top_features
        ]

        return result


###############################################################################
# Executive Summary
###############################################################################


@dataclass(slots=True)
class ExecutiveSummary:

    headline: str

    risk_level: str

    confidence: float

    recommendation: str

    one_line_summary: str

    def to_dict(self):

        return asdict(self)


###############################################################################
# Compliance Summary
###############################################################################


@dataclass(slots=True)
class ComplianceSummary:

    framework: str = "SHAP"

    explainability_enabled: bool = True

    audit_ready: bool = True

    reproducible: bool = True

    notes: Optional[str] = None

    def to_dict(self):

        return asdict(self)


@dataclass(slots=True)
class PromptContext:
    audience: str          # business, auditor, executive, customer
    tone: str              # concise, detailed, technical
    max_tokens: int
    include_recommendation: bool = True
    include_compliance: bool = False

###############################################################################
# Metadata
###############################################################################


@dataclass(slots=True)
class ExplanationMetadata:

    project: str = "CreditCardFraudAI"

    engine_version: str = "1.0.0"

    report_version: str = "1.0.0"

    generated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    model_name: str = ""

    explainer_name: str = ""

    def to_dict(self):

        return asdict(self)


###############################################################################
# AI Explanation
###############################################################################


@dataclass(slots=True)
class AIExplanation:
    """
    Complete explanation object.

    This object becomes the standard payload exchanged between
    all AI modules.
    """

    prediction: int

    probability: float

    confidence: float

    risk_level: str

    confidence_statement: str

    feature_contributions: List[FeatureContribution]

    business_summary: BusinessSummary

    technical_summary: TechnicalSummary

    executive_summary: ExecutiveSummary

    compliance_summary: ComplianceSummary

    metadata: ExplanationMetadata

    raw_report: Optional[Dict[str, Any]] = None

    def top_feature_names(self) -> List[str]:

        return [
            feature.feature
            for feature in self.feature_contributions
        ]

    def top_features_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame(
            [
                feature.to_dict()
                for feature in self.feature_contributions
            ]
        )

    def to_dict(self) -> Dict[str, Any]:

        return {

            "prediction": self.prediction,

            "probability": self.probability,

            "confidence": self.confidence,

            "risk_level": self.risk_level,

            "confidence_statement":
                self.confidence_statement,

            "feature_contributions": [

                feature.to_dict()

                for feature in self.feature_contributions

            ],

            "business_summary":
                self.business_summary.to_dict(),

            "technical_summary":
                self.technical_summary.to_dict(),

            "executive_summary":
                self.executive_summary.to_dict(),

            "compliance_summary":
                self.compliance_summary.to_dict(),

            "metadata":
                self.metadata.to_dict(),

            "raw_report":
                self.raw_report,
        }

    @property
    def is_high_risk(self) -> bool:

        return self.risk_level.upper() in {

            "HIGH",

            "VERY HIGH",

            "CRITICAL",

        }

    @property
    def recommendation(self) -> str:

        return self.business_summary.recommendation

    def __repr__(self) -> str:

        return (

            "AIExplanation("

            f"risk={self.risk_level}, "

            f"probability={self.probability:.4f}, "

            f"features={len(self.feature_contributions)})"

        )