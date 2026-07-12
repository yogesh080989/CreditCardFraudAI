"""
Validation engine for CreditCardFraudAI.

This module orchestrates all validation components including
schema validation, business rule validation and data profiling.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd

from src.core.logger import LoggerManager
from src.validation.schema_validator import SchemaValidator
from src.validation.business_rules import BusinessRuleValidator
from src.validation.profiler import DataProfiler


class ValidationEngine:
    """
    Orchestrates dataset validation.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to validate.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:

        self.df = dataframe
        self.logger = LoggerManager.get_logger()

    def validate(self) -> dict[str, Any]:
        """
        Execute the complete validation workflow.

        Returns
        -------
        dict
            Complete validation report.
        """

        self.logger.info("Starting validation engine.")

        start_time = perf_counter()

        # ---------------------------------------------------------
        # Schema Validation
        # ---------------------------------------------------------

        self.logger.info("Running schema validation.")

        schema_validator = SchemaValidator()

        schema_report = schema_validator.run(self.df)

        # ---------------------------------------------------------
        # Business Rules
        # ---------------------------------------------------------

        self.logger.info("Running business rule validation.")

        business_validator = BusinessRuleValidator()

        business_report = business_validator.run(self.df)

        # ---------------------------------------------------------
        # Data Profiling
        # ---------------------------------------------------------

        self.logger.info("Running data profiler.")

        profiler = DataProfiler(self.df)

        profile_report = profiler.profile()

        elapsed = round(perf_counter() - start_time, 3)

        report = {
            "status": "SUCCESS",
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": elapsed,
            "schema_validation": schema_report,
            "business_rule_validation": business_report,
            "data_profile": profile_report,
        }

        self.logger.info(
            "Validation completed successfully in %.3f seconds.",
            elapsed,
        )

        return report

    def print_summary(self) -> None:
        """
        Print validation summary.
        """

        report = self.validate()

        print("\n========== VALIDATION REPORT ==========")

        print(f"Status : {report['status']}")

        print(f"Execution Time : {report['execution_time_seconds']} sec")

        print(f"Timestamp : {report['timestamp']}")

        print("\nSchema Validation")
        print("-----------------")
        print(report["schema_validation"])

        print("\nBusiness Rule Validation")
        print("------------------------")
        print(report["business_rule_validation"])

        print("\nData Profile")
        print("------------")
        print(report["data_profile"]["dataset"])

        print("=======================================\n")