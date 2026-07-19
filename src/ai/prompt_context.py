"""
===============================================================================
CreditCardFraudAI

Module : prompt_context.py

Description
-----------
Defines strongly typed prompt configuration objects used by the AI Layer.

The PromptContext object provides a consistent configuration shared by:

    ExplanationEngine
    PromptManager
    ProjectOwnerAvatar
    AI Pipeline
    Agent Evaluator
    Enterprise Reports

Using enums avoids string comparisons and improves IDE support.

Author : Yogesh Ahuja
Project : CreditCardFraudAI
===============================================================================
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict
from typing import Any


###############################################################################
# Audience
###############################################################################


class AudienceType(str, Enum):
    """
    Target audience for generated explanations.
    """

    BUSINESS = "business"

    EXECUTIVE = "executive"

    CUSTOMER = "customer"

    AUDITOR = "auditor"

    RISK_OFFICER = "risk_officer"

    DATA_SCIENTIST = "data_scientist"

    DEVELOPER = "developer"


###############################################################################
# Tone
###############################################################################


class ToneType(str, Enum):
    """
    Tone of generated explanation.
    """

    PROFESSIONAL = "professional"

    TECHNICAL = "technical"

    EXECUTIVE = "executive"

    FRIENDLY = "friendly"

    ACADEMIC = "academic"

    CONCISE = "concise"


###############################################################################
# Verbosity
###############################################################################


class VerbosityType(str, Enum):
    """
    Response verbosity.
    """

    SHORT = "short"

    MEDIUM = "medium"

    LONG = "long"


###############################################################################
# Output Format
###############################################################################


class OutputFormat(str, Enum):
    """
    Desired response format.
    """

    TEXT = "text"

    MARKDOWN = "markdown"

    HTML = "html"

    JSON = "json"


###############################################################################
# Prompt Context
###############################################################################


@dataclass(slots=True)
class PromptContext:
    """
    Shared AI configuration object.

    Every AI module receives this object instead
    of multiple individual parameters.
    """

    audience: AudienceType = AudienceType.BUSINESS

    tone: ToneType = ToneType.PROFESSIONAL

    verbosity: VerbosityType = VerbosityType.MEDIUM

    output_format: OutputFormat = OutputFormat.MARKDOWN

    max_tokens: int = 1024

    temperature: float = 0.20

    include_confidence: bool = True

    include_top_features: bool = True

    include_recommendation: bool = True

    include_compliance: bool = False

    include_metadata: bool = False

    include_probability: bool = True

    include_risk_level: bool = True

    include_feature_values: bool = False

    include_shap_values: bool = False

    include_model_details: bool = False

    markdown_tables: bool = True

    def validate(self) -> None:
        """
        Validate prompt configuration.
        """

        if self.max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0 and 2."
            )

    @property
    def is_technical(self) -> bool:

        return self.audience in {

            AudienceType.DATA_SCIENTIST,

            AudienceType.DEVELOPER,

        }

    @property
    def is_business(self) -> bool:

        return self.audience in {

            AudienceType.BUSINESS,

            AudienceType.EXECUTIVE,

            AudienceType.RISK_OFFICER,

        }

    @property
    def is_customer(self) -> bool:

        return self.audience == AudienceType.CUSTOMER

    @property
    def is_auditor(self) -> bool:

        return self.audience == AudienceType.AUDITOR

    def to_dict(self) -> Dict[str, Any]:

        return asdict(self)

    def __repr__(self) -> str:

        return (

            "PromptContext("

            f"audience={self.audience.value}, "

            f"tone={self.tone.value}, "

            f"verbosity={self.verbosity.value})"

        )