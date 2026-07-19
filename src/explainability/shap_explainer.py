"""
===============================================================================
CreditCardFraudAI
Module : SHAP Explainer

Enterprise-grade SHAP Explainability Engine

Supported Models
----------------
* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Generic sklearn estimators

Features
--------
* Automatic SHAP explainer selection
* Global explainability
* Local explainability
* Feature importance
* Visualization support
* Enterprise logging
* Artifact persistence

Author : Yogesh Ahuja
Project : CreditCardFraudAI
===============================================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.core.base import BaseComponent
from src.core.logger import LoggerManager
from src.core.exceptions import ExplainabilityError

warnings.filterwarnings("ignore")


class SHAPExplainer:
    """
    Enterprise SHAP Explainability Engine.

    Responsibilities
    ----------------
    * Automatically select the appropriate SHAP explainer.
    * Compute SHAP values.
    * Generate global explanations.
    * Generate local explanations.
    * Produce explainability plots.
    * Export explainability artifacts.
    """

    ###########################################################################
    # Constructor
    ###########################################################################

    def __init__(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        output_dir: Union[str, Path],
        background_data: Optional[pd.DataFrame] = None,
        logger=None,
    ) -> None:



        self.model = model
        self.feature_names = feature_names

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.background_data = background_data

        self.logger = logger or LoggerManager.get_logger(
            self.__class__.__name__
        )

        # SHAP objects
        self.explainer = None
        self.explanation = None
        self.shap_values = None
        self.expected_value = None
        self.explanation_data = None

        self.logger.info("Initializing SHAP Explainer...")

        self._validate_inputs()
        self._prepare_background()
        self._initialize_explainer()

    ###########################################################################
    # Validation
    ###########################################################################

    def _validate_inputs(self) -> None:
        """
        Validate constructor inputs.
        """

        if self.model is None:
            raise ExplainabilityError(
                "Model cannot be None."
            )

        if not isinstance(
            self.feature_names,
            (list, tuple),
        ):
            raise ExplainabilityError(
                "feature_names must be a list."
            )

        if len(self.feature_names) == 0:
            raise ExplainabilityError(
                "Feature names cannot be empty."
            )

    ###########################################################################
    # Background Dataset
    ###########################################################################

    def _prepare_background(self) -> None:
        """
        Prepare background dataset required for Linear/Kernel SHAP.
        """

        if self.background_data is None:

            self.logger.warning(
                "Background dataset not supplied."
            )

            return

        if not isinstance(
            self.background_data,
            pd.DataFrame,
        ):

            self.background_data = pd.DataFrame(
                self.background_data,
                columns=self.feature_names,
            )

        self.background_data = (
            self.background_data.copy()
        )

        self.logger.info(
            "Background dataset shape : %s",
            self.background_data.shape,
        )

    ###########################################################################
    # Explainer Selection
    ###########################################################################

    def _initialize_explainer(self) -> None:
        """
        Automatically choose the best SHAP explainer.
        """

        self.logger.info(
            "Selecting SHAP explainer..."
        )

        try:

            ###############################################################
            # Tree Models
            ###############################################################

            if isinstance(
                self.model,
                (
                    DecisionTreeClassifier,
                    RandomForestClassifier,
                ),
            ):

                self.logger.info(
                    "TreeExplainer selected."
                )

                self.explainer = shap.TreeExplainer(
                    self.model
                )

                return

            ###############################################################
            # Logistic Regression
            ###############################################################

            if isinstance(
                self.model,
                LogisticRegression,
            ):

                if self.background_data is None:

                    raise ExplainabilityError(
                        "LinearExplainer requires "
                        "background_data."
                    )

                self.logger.info(
                    "LinearExplainer selected."
                )

                self.explainer = shap.LinearExplainer(
                    self.model,
                    self.background_data,
                )

                return

            ###############################################################
            # Gradient Boosting
            ###############################################################

            module_name = (
                self.model.__class__.__module__.lower()
            )

            if any(
                library in module_name
                for library in (
                    "xgboost",
                    "lightgbm",
                    "catboost",
                )
            ):

                self.logger.info(
                    "Gradient boosting model detected."
                )

                self.explainer = shap.TreeExplainer(
                    self.model
                )

                return

            ###############################################################
            # Generic fallback
            ###############################################################

            if self.background_data is None:

                raise ExplainabilityError(
                    "KernelExplainer requires "
                    "background_data."
                )

            self.logger.warning(
                "Falling back to KernelExplainer."
            )

            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                self.background_data,
            )

        except Exception as ex:

            self.logger.exception(ex)

            raise ExplainabilityError(
                f"Unable to initialize SHAP explainer: {ex}"
            )

    ###########################################################################
    # Utility Methods
    ###########################################################################

    @staticmethod
    def save_object(
        obj: Any,
        file_path: Union[str, Path],
    ) -> None:
        """
        Save any Python object using joblib.
        """

        file_path = Path(file_path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            obj,
            file_path,
        )

    @staticmethod
    def load_object(
        file_path: Union[str, Path],
    ) -> Any:
        """
        Load serialized object.
        """

        return joblib.load(file_path)

    @staticmethod
    def save_figure(
        filename: Union[str, Path],
        dpi: int = 300,
    ) -> None:
        """
        Save current matplotlib figure.
        """

        plt.tight_layout()

        plt.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close()

    ###########################################################################
    # Internal Helpers
    ###########################################################################

    def _validate_explanation_ready(self) -> None:
        """
        Ensure SHAP values have already been computed.
        """

        if self.shap_values is None:
            raise ExplainabilityError(
                "Compute SHAP values first."
            )

    def _get_positive_class_values(self):
        """
        Return SHAP values for the positive class.

        Handles binary classifiers and SHAP API variations.
        """

        self._validate_explanation_ready()

        values = self.shap_values

        if isinstance(values, list):

            if len(values) == 2:
                return values[1]

            return values[0]

        if isinstance(values, np.ndarray):

            if values.ndim == 3:
                return values[:, :, 1]

        return values

    def _get_single_base_value(self):
        """
        Return a scalar expected value safely.
        """
        if self.expected_value is None:
            return 0.0

        value = self.expected_value

        # If it is already a simple scalar, return it
        if np.isscalar(value):
            return float(value)

        # If it is a numpy array
        if isinstance(value, np.ndarray):
            # If it's 0-dimensional (e.g., array(0.5)), use .item() to get the scalar
            if value.ndim == 0:
                return float(value.item())

            # If it's a 1D array (e.g., [base_legit, base_fraud])
            if value.ndim == 1:
                # If we have multiple classes, return the positive class (index 1)
                if len(value) > 1:
                    return float(value[1])
                return float(value[0])

        return 0.0

        ###########################################################################
    # SHAP Value Computation
    ###########################################################################

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        check_additivity: bool = False,
    ):
        """
        Compute SHAP values for the supplied dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset to explain.

        check_additivity : bool
            Used by TreeExplainer.

        Returns
        -------
        SHAP values
        """

        self.logger.info(
            "Computing SHAP values for %d records...",
            len(X),
        )

        try:

            if not isinstance(X, pd.DataFrame):

                X = pd.DataFrame(
                    X,
                    columns=self.feature_names,
                )

            self.explanation_data = X.copy()

            ###################################################################
            # Modern SHAP API
            ###################################################################

            try:

                explanation = self.explainer(X)

                self.explanation = explanation

                self.shap_values = explanation.values

                self.expected_value = explanation.base_values

            ###################################################################
            # Legacy SHAP API
            ###################################################################

            except Exception:

                if isinstance(
                    self.explainer,
                    shap.TreeExplainer,
                ):

                    self.shap_values = self.explainer.shap_values(
                        X,
                        check_additivity=check_additivity,
                    )

                else:

                    self.shap_values = self.explainer.shap_values(X)

                if hasattr(
                    self.explainer,
                    "expected_value",
                ):

                    self.expected_value = (
                        self.explainer.expected_value
                    )

                self.explanation = None

            self.logger.info(
                "SHAP values computed successfully."
            )

            return self.shap_values

        except Exception as ex:

            self.logger.exception(ex)

            raise ExplainabilityError(
                f"Unable to compute SHAP values : {ex}"
            )

    ###########################################################################
    # Global Feature Importance
    ###########################################################################

    def get_feature_importance(
        self,
    ) -> pd.DataFrame:
        """
        Calculate mean absolute SHAP importance.
        """

        values = self._get_positive_class_values()

        importance = np.abs(values).mean(axis=0)

        df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Importance": importance,
            }
        )

        df.sort_values(
            "Importance",
            ascending=False,
            inplace=True,
        )

        df.reset_index(
            drop=True,
            inplace=True,
        )

        return df

    ###########################################################################
    # Summary Plot
    ###########################################################################

    def summary_plot(
        self,
        max_display: int = 20,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> None:
        """
        Generate SHAP summary plot.
        """

        self._validate_explanation_ready()

        self.logger.info(
            "Generating summary plot..."
        )

        values = self._get_positive_class_values()

        plt.figure(figsize=(10, 7))

        shap.summary_plot(
            values,
            self.explanation_data,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )

        if save_path:

            self.save_figure(save_path)

            self.logger.info(
                "Summary plot saved to %s",
                save_path,
            )

        if show:
            plt.show()

    ###########################################################################
    # Beeswarm Plot
    ###########################################################################

    def beeswarm_plot(
            self,
            max_display: int = 20,
            save_path: Optional[Union[str, Path]] = None,
            show: bool = False,
    ) -> None:
        """
        Generate SHAP beeswarm plot with dimension handling.
        """

        self._validate_explanation_ready()

        self.logger.info("Generating beeswarm plot...")

        plt.figure(figsize=(10, 7))

        if self.explanation is not None:
            # Check if explanation is 3D (samples, features, classes)
            # If so, slice it to get only the positive class (index 1)
            if hasattr(self.explanation, 'values') and self.explanation.values.ndim == 3:
                explanation_to_plot = self.explanation[:, :, 1]
            else:
                explanation_to_plot = self.explanation

            shap.plots.beeswarm(
                explanation_to_plot,
                max_display=max_display,
                show=False,
            )

        else:
            # Fallback for legacy API
            shap.summary_plot(
                self._get_positive_class_values(),
                self.explanation_data,
                max_display=max_display,
                show=False,
            )

        if save_path:
            self.save_figure(save_path)

        if show:
            plt.show()

    ###########################################################################
    # Bar Plot
    ###########################################################################

    def bar_plot(
        self,
        max_display: int = 20,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> None:
        """
        Generate global feature importance plot.
        """

        self._validate_explanation_ready()

        plt.figure(figsize=(10, 7))

        shap.summary_plot(
            self._get_positive_class_values(),
            self.explanation_data,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )

        if save_path:
            self.save_figure(save_path)

        if show:
            plt.show()

    ###########################################################################
    # Dependence Plot
    ###########################################################################

    def dependence_plot(
        self,
        feature_name: str,
        interaction_index: str = "auto",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> None:
        """
        Generate SHAP dependence plot.
        """

        self._validate_explanation_ready()

        shap.dependence_plot(
            feature_name,
            self._get_positive_class_values(),
            self.explanation_data,
            interaction_index=interaction_index,
            show=False,
        )

        if save_path:
            self.save_figure(save_path)

        if show:
            plt.show()

    ###########################################################################
    # Waterfall Plot
    ###########################################################################

    def waterfall_plot(
        self,
        row_index: int = 0,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> None:
        """
        Generate SHAP waterfall plot for a single observation.
        """

        self._validate_explanation_ready()

        self.logger.info("Generating waterfall plot...")

        # 1. Get the explanation for the specific row
        if self.explanation is not None:
            # If the explanation is multi-dimensional (e.g., has class dimension),
            # slice to the positive class (index 1)
            explanation = self.explanation[row_index]
            if explanation.values.ndim > 1:
                explanation = explanation[:, 1]
        else:
            # Legacy path for when we manually build the Explanation object
            explanation = shap.Explanation(
                values=self._get_positive_class_values()[row_index],
                base_values=self._get_single_base_value(),
                data=self.explanation_data.iloc[row_index],
                feature_names=self.feature_names,
            )

        plt.figure(figsize=(8, 6))

        # 2. Plot the 1D explanation
        shap.plots.waterfall(
            explanation,
            show=False,
        )

        if save_path:
            self.save_figure(save_path)

        if show:
            plt.show()

    ###########################################################################
    # Force Plot
    ###########################################################################

    def force_plot(
        self,
        row_index: int = 0,
        save_path: Optional[Union[str, Path]] = None,
    ):
        """
        Generate an interactive SHAP force plot.

        Parameters
        ----------
        row_index : int
            Index of the observation to explain.

        save_path : str | Path, optional
            HTML output path.

        Returns
        -------
        SHAP force plot object
        """

        self._validate_explanation_ready()

        self.logger.info(
            "Generating SHAP force plot..."
        )

        try:

            values = self._get_positive_class_values()

            expected = self._get_single_base_value()

            record = self.explanation_data.iloc[row_index]

            force = shap.force_plot(
                expected,
                values[row_index],
                record,
                matplotlib=False,
            )

            if save_path:

                save_path = Path(save_path)

                save_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shap.save_html(
                    str(save_path),
                    force,
                )

                self.logger.info(
                    "Force plot saved to %s",
                    save_path,
                )

            return force

        except Exception as ex:

            self.logger.exception(ex)

            raise ExplainabilityError(str(ex))

    ###########################################################################
    # Explain Single Prediction
    ###########################################################################

    def explain_prediction(
        self,
        row_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Return a detailed explanation for one prediction.
        """

        self._validate_explanation_ready()

        values = self._get_positive_class_values()

        row = self.explanation_data.iloc[row_index]

        contribution = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Value": row.values,
                "SHAP": values[row_index],
            }
        )

        contribution["ABS_SHAP"] = np.abs(
            contribution["SHAP"]
        )

        contribution.sort_values(
            "ABS_SHAP",
            ascending=False,
            inplace=True,
        )

        prediction = None
        probability = None

        ###############################################################
        # Prediction
        ###############################################################

        try:

            prediction = int(
                self.model.predict(
                    row.to_frame().T
                )[0]
            )

        except Exception:

            pass

        ###############################################################
        # Probability
        ###############################################################

        try:

            if hasattr(
                self.model,
                "predict_proba",
            ):

                probability = float(
                    self.model.predict_proba(
                        row.to_frame().T
                    )[0][1]
                )

        except Exception:

            pass

        return {

            "prediction": prediction,

            "probability": probability,

            "base_value": self._get_single_base_value(),

            "feature_contributions": contribution,

        }

    ###########################################################################
    # Top Positive Contributors
    ###########################################################################

    def top_positive_features(
        self,
        row_index: int,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return strongest positive contributors.
        """

        values = self._get_positive_class_values()

        row = self.explanation_data.iloc[row_index]

        df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Value": row.values,
                "Contribution": values[row_index],
            }
        )

        df.sort_values(
            "Contribution",
            ascending=False,
            inplace=True,
        )

        return df.head(top_n).reset_index(drop=True)

    ###########################################################################
    # Top Negative Contributors
    ###########################################################################

    def top_negative_features(
        self,
        row_index: int,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return strongest negative contributors.
        """

        values = self._get_positive_class_values()

        row = self.explanation_data.iloc[row_index]

        df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Value": row.values,
                "Contribution": values[row_index],
            }
        )

        df.sort_values(
            "Contribution",
            ascending=True,
            inplace=True,
        )

        return df.head(top_n).reset_index(drop=True)

    ###########################################################################
    # Export Feature Importance
    ###########################################################################

    def export_feature_importance(
        self,
        output_file: Union[str, Path],
    ) -> Path:
        """
        Export feature importance to CSV.
        """

        df = self.get_feature_importance()

        output_file = Path(output_file)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            output_file,
            index=False,
        )

        self.logger.info(
            "Feature importance exported to %s",
            output_file,
        )

        return output_file

    ###########################################################################
    # Save SHAP Explanation
    ###########################################################################

    def save_explanation(
        self,
        output_file: Union[str, Path],
    ) -> None:
        """
        Persist computed SHAP explanation.
        """

        self._validate_explanation_ready()

        self.save_object(
            self.explanation
            if self.explanation is not None
            else self.shap_values,
            output_file,
        )

        self.logger.info(
            "Explanation saved to %s",
            output_file,
        )

    ###########################################################################
    # Load SHAP Explanation
    ###########################################################################

    def load_explanation(
        self,
        input_file: Union[str, Path],
    ):
        """
        Load previously saved explanation.
        """

        explanation = self.load_object(
            input_file
        )

        self.logger.info(
            "Explanation loaded from %s",
            input_file,
        )

        return explanation

    ###########################################################################
    # Explainability Report
    ###########################################################################

    def generate_explainability_report(
        self,
        row_index: int = 0,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate a business-friendly explainability report.
        """

        explanation = self.explain_prediction(
            row_index=row_index
        )

        feature_importance = (
            explanation[
                "feature_contributions"
            ]
            .head(top_n)
            .copy()
        )

        probability = explanation["probability"]

        if probability is None:

            risk_level = "UNKNOWN"

        elif probability >= 0.90:

            risk_level = "VERY HIGH"

        elif probability >= 0.70:

            risk_level = "HIGH"

        elif probability >= 0.40:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        return {

            "prediction": explanation["prediction"],

            "probability": probability,

            "risk_level": risk_level,

            "base_value": explanation["base_value"],

            "top_features": feature_importance,

        }
        ###########################################################################
    # Top Features
    ###########################################################################

    def top_features(
        self,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Return the top N globally important features.
        """

        return self.get_feature_importance().head(top_n)

    ###########################################################################
    # Summary Statistics
    ###########################################################################

    def get_summary_statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Return summary statistics for the computed SHAP values.
        """

        self._validate_explanation_ready()

        values = self._get_positive_class_values()

        return {
            "num_samples": int(values.shape[0]),
            "num_features": int(values.shape[1]),
            "mean_abs_shap": float(np.mean(np.abs(values))),
            "max_abs_shap": float(np.max(np.abs(values))),
            "min_abs_shap": float(np.min(np.abs(values))),
        }

    ###########################################################################
    # Save Complete Explainability Report
    ###########################################################################

    def save_report(
        self,
        output_file: Union[str, Path],
        row_index: int = 0,
        top_n: int = 10,
    ) -> Path:
        """
        Save a complete explainability report as JSON.
        """

        import json

        report = self.generate_explainability_report(
            row_index=row_index,
            top_n=top_n,
        )

        serializable = {
            "prediction": report["prediction"],
            "probability": report["probability"],
            "risk_level": report["risk_level"],
            "base_value": float(report["base_value"]),
            "top_features": report[
                "top_features"
            ].to_dict(
                orient="records"
            ),
        }

        output_file = Path(output_file)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                serializable,
                fp,
                indent=4,
            )

        self.logger.info(
            "Explainability report saved to %s",
            output_file,
        )

        return output_file

    ###########################################################################
    # Reset State
    ###########################################################################

    def reset(
        self,
    ) -> None:
        """
        Reset computed SHAP artifacts while keeping the explainer.
        """

        self.shap_values = None
        self.expected_value = None
        self.explanation = None
        self.explanation_data = None

        self.logger.info(
            "SHAP state has been reset."
        )

    ###########################################################################
    # Validate Ready State
    ###########################################################################

    @property
    def is_fitted(
        self,
    ) -> bool:
        """
        Indicates whether SHAP values have already been computed.
        """

        return self.shap_values is not None

    ###########################################################################
    # Model Information
    ###########################################################################

    @property
    def model_name(
        self,
    ) -> str:
        """
        Return the underlying estimator name.
        """

        return self.model.__class__.__name__

    ###########################################################################
    # Explainer Information
    ###########################################################################

    @property
    def explainer_name(
        self,
    ) -> str:
        """
        Return SHAP explainer name.
        """

        if self.explainer is None:
            return "Not Initialized"

        return self.explainer.__class__.__name__

    ###########################################################################
    # String Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            f"SHAPExplainer("
            f"model={self.model_name}, "
            f"explainer={self.explainer_name}, "
            f"features={len(self.feature_names)}, "
            f"computed={self.is_fitted})"
        )

    ###########################################################################
    # Destructor
    ###########################################################################

    def __del__(
        self,
    ):
        """
        Release memory occupied by SHAP artifacts.
        """

        try:

            self.reset()

        except Exception:

            pass
###############################################################################
# Execute
###############################################################################

def execute(self):
    """
    Default execute implementation required by BaseComponent.

    SHAPExplainer exposes specialized methods instead of a single execute()
    workflow, so this method simply satisfies the abstract interface.
    """
    pass