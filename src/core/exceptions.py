"""
Custom exception hierarchy for the CreditCardFraudAI project.

This module defines all project-specific exceptions used across the
application. Having a dedicated exception hierarchy allows consistent
error handling, cleaner logging, and easier debugging.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations


class CreditCardFraudAIError(Exception):
    """
    Base exception for the CreditCardFraudAI project.

    All project-specific exceptions should inherit from this class.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

class ConfigurationError(CreditCardFraudAIError):
    """Raised when configuration is invalid or missing."""


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

class DataLoadingError(CreditCardFraudAIError):
    """Raised when dataset loading fails."""


class DataSavingError(CreditCardFraudAIError):
    """Raised when saving datasets or artifacts fails."""


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

class DatasetValidationError(CreditCardFraudAIError):
    """Base exception for dataset validation."""


class SchemaValidationError(DatasetValidationError):
    """Raised when dataset schema validation fails."""


class BusinessRuleValidationError(DatasetValidationError):
    """Raised when business rules are violated."""


class DataProfilingError(DatasetValidationError):
    """Raised when dataset profiling fails."""


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

class PreprocessingError(CreditCardFraudAIError):
    """Raised during preprocessing failures."""


# ---------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------

class FeatureEngineeringError(CreditCardFraudAIError):
    """Raised during feature engineering."""


# ---------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------

class ModelTrainingError(CreditCardFraudAIError):
    """Raised when model training fails."""


class ModelSerializationError(CreditCardFraudAIError):
    """Raised when model saving/loading fails."""


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

class ModelEvaluationError(CreditCardFraudAIError):
    """Raised during model evaluation."""


# ---------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------

class ExplainabilityError(CreditCardFraudAIError):
    """Raised when SHAP or explanation generation fails."""


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

class PipelineError(CreditCardFraudAIError):
    """Raised for end-to-end pipeline failures."""