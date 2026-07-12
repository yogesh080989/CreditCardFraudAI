"""
business_rules.py

Business Rule Validation Engine

This module validates business-specific rules for the
Credit Card Fraud Detection dataset.

Author:
Yogesh Ahuja
"""

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class ValidationResult:
    """
    Represents the result of a business rule validation.
    """
    rule_name: str
    status: str
    message: str


class BusinessRuleValidator:

    def __init__(
            self,
            target_column: str = "Class",
            amount_column: str = "Amount",
            time_column: str = "Time",
    ):

        self.target_column = target_column
        self.amount_column = amount_column
        self.time_column = time_column

    # --------------------------------------------------
    # Rule 1
    # --------------------------------------------------

    def validate_amount(self, df: pd.DataFrame):

        invalid = (df[self.amount_column] < 0).sum()

        if invalid > 0:
            return ValidationResult(
                "Amount Validation",
                "FAIL",
                f"{invalid} transaction(s) contain negative Amount."
            )

        return ValidationResult(
            "Amount Validation",
            "PASS",
            "All transaction amounts are valid."
        )

    # --------------------------------------------------
    # Rule 2
    # --------------------------------------------------

    def validate_time(self, df: pd.DataFrame):

        invalid = (df[self.time_column] < 0).sum()

        if invalid > 0:
            return ValidationResult(
                "Time Validation",
                "FAIL",
                f"{invalid} transaction(s) contain negative Time."
            )

        return ValidationResult(
            "Time Validation",
            "PASS",
            "All transaction timestamps are valid."
        )

    # --------------------------------------------------
    # Rule 3
    # --------------------------------------------------

    def validate_target(self, df: pd.DataFrame):

        invalid = df[
            ~df[self.target_column].isin([0, 1])
        ]

        if len(invalid):
            return ValidationResult(
                "Target Validation",
                "FAIL",
                "Target column contains values other than 0 and 1."
            )

        return ValidationResult(
            "Target Validation",
            "PASS",
            "Target column contains only valid labels."
        )

    # --------------------------------------------------
    # Rule 4
    # --------------------------------------------------

    def validate_fraud_presence(self, df: pd.DataFrame):

        fraud_count = (df[self.target_column] == 1).sum()

        if fraud_count == 0:
            return ValidationResult(
                "Fraud Presence",
                "FAIL",
                "No fraud records found."
            )

        return ValidationResult(
            "Fraud Presence",
            "PASS",
            f"{fraud_count:,} fraud transactions detected."
        )

    # --------------------------------------------------
    # Rule 5
    # --------------------------------------------------

    def validate_duplicate_ratio(self, df: pd.DataFrame):

        duplicates = df.duplicated().sum()

        duplicate_percentage = round(
            duplicates / len(df) * 100,
            2
        )

        if duplicate_percentage > 5:
            return ValidationResult(
                "Duplicate Validation",
                "WARN",
                f"{duplicate_percentage}% duplicate rows detected."
            )

        return ValidationResult(
            "Duplicate Validation",
            "PASS",
            f"{duplicate_percentage}% duplicate rows."
        )

    # --------------------------------------------------
    # Rule 6
    # --------------------------------------------------

    def validate_missing_values(self, df: pd.DataFrame):

        missing = int(df.isna().sum().sum())

        if missing > 0:
            return ValidationResult(
                "Missing Value Validation",
                "WARN",
                f"{missing} missing values detected."
            )

        return ValidationResult(
            "Missing Value Validation",
            "PASS",
            "No missing values."
        )

    # --------------------------------------------------
    # Rule 7
    # --------------------------------------------------

    def validate_class_distribution(self, df: pd.DataFrame):

        fraud = (df[self.target_column] == 1).sum()
        genuine = (df[self.target_column] == 0).sum()

        fraud_ratio = round(
            fraud / len(df) * 100,
            4
        )

        return ValidationResult(
            "Class Distribution",
            "PASS",
            f"Genuine={genuine:,}, Fraud={fraud:,}, Fraud Ratio={fraud_ratio}%"
        )

    # --------------------------------------------------
    # Execute all rules
    # --------------------------------------------------

    def run(self, df: pd.DataFrame) -> List[ValidationResult]:

        return [

            self.validate_amount(df),

            self.validate_time(df),

            self.validate_target(df),

            self.validate_fraud_presence(df),

            self.validate_duplicate_ratio(df),

            self.validate_missing_values(df),

            self.validate_class_distribution(df),

        ]
# Business rule checks for fraud detection datasets
