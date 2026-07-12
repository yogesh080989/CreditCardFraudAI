import pytest

from src.core.exceptions import (
    ConfigurationError,
    DataLoadingError,
    SchemaValidationError,
    ModelTrainingError,
)


def test_configuration_error():
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Invalid config")


def test_data_loading_error():
    with pytest.raises(DataLoadingError):
        raise DataLoadingError("Dataset missing")


def test_schema_validation_error():
    with pytest.raises(SchemaValidationError):
        raise SchemaValidationError("Schema mismatch")


def test_model_training_error():
    with pytest.raises(ModelTrainingError):
        raise ModelTrainingError("Training failed")