# Schema validation utilities for dataset structure checks
"""
schema_validator.py

Purpose:
---------
Validate whether the incoming dataset satisfies the expected schema
required for Credit Card Fraud Detection.

Author:
Yogesh Ahuja Capstone Project
"""

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class ValidationResult:
    check_name: str
    status: str
    message: str


class SchemaValidator:

    def __init__(self):

        self.expected_columns = [
            "Time",
            "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
            "V11","V12","V13","V14","V15","V16","V17","V18","V19",
            "V20","V21","V22","V23","V24","V25","V26","V27","V28",
            "Amount",
            "Class"
        ]

    def validate_columns(self, df: pd.DataFrame) -> ValidationResult:

        missing = [
            column
            for column in self.expected_columns
            if column not in df.columns
        ]

        if missing:
            return ValidationResult(
                check_name="Column Validation",
                status="FAIL",
                message=f"Missing Columns : {missing}"
            )

        return ValidationResult(
            check_name="Column Validation",
            status="PASS",
            message="All expected columns are present."
        )

    def validate_data_types(self, df: pd.DataFrame):

        invalid_columns = []

        for column in self.expected_columns:

            if column == "Class":

                if not pd.api.types.is_integer_dtype(df[column]):
                    invalid_columns.append(column)

            else:

                if not pd.api.types.is_numeric_dtype(df[column]):
                    invalid_columns.append(column)

        if invalid_columns:

            return ValidationResult(
                "Datatype Validation",
                "FAIL",
                f"Invalid datatype found in {invalid_columns}"
            )

        return ValidationResult(
            "Datatype Validation",
            "PASS",
            "All datatypes are valid."
        )

    def validate_empty_dataset(self, df):

        if len(df) == 0:

            return ValidationResult(
                "Dataset Validation",
                "FAIL",
                "Dataset is empty."
            )

        return ValidationResult(
            "Dataset Validation",
            "PASS",
            f"Dataset contains {len(df):,} rows."
        )

    def run(self, df):

        return [

            self.validate_empty_dataset(df),

            self.validate_columns(df),

            self.validate_data_types(df)

        ]