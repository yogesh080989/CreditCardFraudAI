"""
Report generation utilities for CreditCardFraudAI.

This module provides functionality to export validation reports
into different formats such as JSON and plain text.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class ReportGenerator:
    """
    Generates validation reports in different formats.
    """

    def __init__(
        self,
        output_directory: str = "reports/validation",
    ) -> None:

        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # Public Methods
    # ---------------------------------------------------------

    def save_json(
        self,
        report: dict[str, Any],
        filename: str = "validation_report.json",
    ) -> Path:
        """
        Save validation report as JSON.
        """

        report = self._convert_dataclasses(report)

        output_file = self.output_directory / filename

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        return output_file

    def save_text(
        self,
        report: dict[str, Any],
        filename: str = "validation_report.txt",
    ) -> Path:
        """
        Save validation report as text.
        """

        output_file = self.output_directory / filename

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write("VALIDATION REPORT\n")
            file.write("=" * 60)
            file.write("\n\n")

            for key, value in report.items():

                file.write(f"{key.upper()}\n")
                file.write("-" * 60)
                file.write("\n")

                file.write(f"{value}\n\n")

        return output_file

    def print_console(
        self,
        report: dict[str, Any],
    ) -> None:
        """
        Print report to console.
        """

        print("\n")
        print("=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        for key, value in report.items():

            print(f"\n{key.upper()}")

            print("-" * 60)

            print(value)

        print("=" * 60)

    # ---------------------------------------------------------
    # Private Methods
    # ---------------------------------------------------------

    def _convert_dataclasses(
        self,
        obj: Any,
    ) -> Any:
        """
        Recursively convert dataclasses into dictionaries.
        """

        if is_dataclass(obj):
            return asdict(obj)

        if isinstance(obj, list):
            return [
                self._convert_dataclasses(item)
                for item in obj
            ]

        if isinstance(obj, dict):
            return {
                key: self._convert_dataclasses(value)
                for key, value in obj.items()
            }

        return obj