"""
Base classes for CreditCardFraudAI.

This module defines the abstract base component used throughout the
project. All major processing components should inherit from this class
to provide a consistent interface.

Author: Yogesh Ahuja
Project: CreditCardFraudAI
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class BaseComponent(ABC):
    """
    Abstract base class for all project components.

    Every processing component should inherit from this class and
    implement the execute() method.
    """

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.created_at = datetime.utcnow()

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the component.

        Returns
        -------
        Any
            Component-specific output.
        """
        raise NotImplementedError

    def get_component_name(self) -> str:
        """
        Returns the component name.
        """
        return self.component_name

    def metadata(self) -> dict:
        """
        Returns metadata about the component.
        """
        return {
            "component": self.component_name,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(component='{self.component_name}')"
        )