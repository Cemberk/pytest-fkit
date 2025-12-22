"""
PySR Runner: Automatic symbolic regression on trace data.

Discovers mathematical formulas from collected metrics and adds
predictions back to trace rows for database upload.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from .collector import TraceRow

logger = logging.getLogger(__name__)


@dataclass
class FormulaResult:
    """Result of PySR symbolic regression for a target metric."""

    target: str
    equation: str
    equation_latex: str = ""
    complexity: int = 0
    r2_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    feature_names: List[str] = field(default_factory=list)
    n_samples: int = 0
    # Store the model for predictions (optional, can be None if serializing)
    _model: Any = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "target": self.target,
            "equation": self.equation,
            "equation_latex": self.equation_latex,
            "complexity": self.complexity,
            "r2_score": self.r2_score,
            "mse": self.mse,
            "mae": self.mae,
            "feature_names": self.feature_names,
            "n_samples": self.n_samples,
        }


# Default target metrics for formula discovery
DEFAULT_TARGETS = [
    "duration_ms",
    "metric_duration_ms",
    "metric_memory_delta_mb",
    "metric_call_ms",
]


class PySRRunner:
    """
    Runs PySR symbolic regression on trace data.

    Discovers mathematical formulas that explain relationships between
    collected metrics. Automatically detects available targets and features.

    Usage:
        runner = PySRRunner()
        if runner.is_available():
            formulas = runner.discover_formulas(trace_rows)
            rows_with_predictions = runner.add_predictions_to_rows(trace_rows, formulas)
    """

    def __init__(
        self,
        min_samples: int = 20,
        niterations: int = 40,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        populations: int = 8,
        population_size: int = 33,
        maxsize: int = 20,
        timeout_seconds: int = 300,
    ):
        """
        Initialize PySR runner.

        Args:
            min_samples: Minimum number of samples required for regression
            niterations: Number of PySR iterations
            binary_operators: Binary operators to use (default: +, -, *, /)
            unary_operators: Unary operators to use (default: log, exp, sqrt)
            populations: Number of populations for genetic algorithm
            population_size: Size of each population
            maxsize: Maximum equation size
            timeout_seconds: Timeout for PySR fitting
        """
        self.min_samples = min_samples
        self.niterations = niterations
        self.binary_operators = binary_operators or ["+", "-", "*", "/"]
        self.unary_operators = unary_operators or ["log", "exp", "sqrt"]
        self.populations = populations
        self.population_size = population_size
        self.maxsize = maxsize
        self.timeout_seconds = timeout_seconds

        # Cache availability check
        self._pysr_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if PySR and numpy are installed."""
        if self._pysr_available is not None:
            return self._pysr_available

        try:
            import numpy  # noqa: F401
            from pysr import PySRRegressor  # noqa: F401

            self._pysr_available = True
        except ImportError:
            self._pysr_available = False

        return self._pysr_available

    def _extract_numeric_data(
        self, rows: List[Union[TraceRow, Dict[str, Any]]]
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """
        Extract numeric columns from trace rows.

        Returns:
            (list of numeric row dicts, list of column names)
        """
        numeric_rows: List[Dict[str, float]] = []

        for row in rows:
            if isinstance(row, TraceRow):
                row_dict = row.to_dict()
            else:
                row_dict = row

            # Filter to numeric values only
            numeric_row: Dict[str, float] = {}
            for key, value in row_dict.items():
                if isinstance(value, bool):
                    # Convert booleans to 0.0/1.0
                    numeric_row[key] = 1.0 if value else 0.0
                elif isinstance(value, (int, float)):
                    try:
                        numeric_row[key] = float(value)
                    except (ValueError, TypeError):
                        continue
                elif isinstance(value, str):
                    # Try to parse string as number
                    try:
                        numeric_row[key] = float(value)
                    except (ValueError, TypeError):
                        continue

            numeric_rows.append(numeric_row)

        # Get all column names
        all_columns: set = set()
        for row in numeric_rows:
            all_columns.update(row.keys())

        return numeric_rows, sorted(all_columns)

    def _find_available_targets(
        self, columns: List[str], requested_targets: Optional[List[str]] = None
    ) -> List[str]:
        """Find which target columns are available in the data."""
        if requested_targets:
            return [t for t in requested_targets if t in columns]

        # Use defaults, filtering to available columns
        available = []
        for target in DEFAULT_TARGETS:
            if target in columns:
                available.append(target)

        return available

    def _get_features_for_target(
        self, target: str, columns: List[str], exclude_prefixes: Optional[List[str]] = None
    ) -> List[str]:
        """Get feature columns for a target, excluding the target itself."""
        exclude_prefixes = exclude_prefixes or ["tag_", "calc_"]

        features = []
        for col in columns:
            # Skip the target
            if col == target:
                continue

            # Skip excluded prefixes (tags, pysr results)
            if any(col.startswith(prefix) for prefix in exclude_prefixes):
                continue

            # Skip non-varying columns (like timestamps)
            if col in ["timestamp", "trace_id", "start_time", "end_time"]:
                continue

            features.append(col)

        return features

    def discover_formulas(
        self,
        rows: List[Union[TraceRow, Dict[str, Any]]],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, FormulaResult]:
        """
        Run PySR symbolic regression on trace data.

        Args:
            rows: Trace rows to analyze
            targets: Target metrics to predict (auto-detect if None)

        Returns:
            Dictionary mapping target names to FormulaResult objects
        """
        if not self.is_available():
            logger.warning("PySR not available, skipping formula discovery")
            return {}

        if len(rows) < self.min_samples:
            logger.warning(
                f"Not enough samples for PySR ({len(rows)} < {self.min_samples})"
            )
            return {}

        import numpy as np
        from pysr import PySRRegressor

        # Extract numeric data
        numeric_rows, columns = self._extract_numeric_data(rows)

        if not numeric_rows:
            logger.warning("No numeric data found in trace rows")
            return {}

        # Find available targets
        available_targets = self._find_available_targets(columns, targets)

        if not available_targets:
            logger.warning("No target columns found in data")
            return {}

        formulas: Dict[str, FormulaResult] = {}

        for target in available_targets:
            logger.info(f"Running PySR for target: {target}")

            # Get features
            feature_names = self._get_features_for_target(target, columns)

            if len(feature_names) < 1:
                logger.warning(f"No features available for target: {target}")
                continue

            # Build arrays
            X = []
            y = []
            valid_indices = []

            for i, row in enumerate(numeric_rows):
                target_val = row.get(target)
                if target_val is None or np.isnan(target_val):
                    continue

                feature_vals = [row.get(f, 0.0) for f in feature_names]

                # Skip if any feature is NaN
                if any(np.isnan(v) if isinstance(v, float) else False for v in feature_vals):
                    continue

                X.append(feature_vals)
                y.append(target_val)
                valid_indices.append(i)

            if len(X) < self.min_samples:
                logger.warning(
                    f"Not enough valid samples for {target} ({len(X)} < {self.min_samples})"
                )
                continue

            X_arr = np.array(X, dtype=np.float64)
            y_arr = np.array(y, dtype=np.float64)

            # Handle constant columns (zero variance)
            valid_features = []
            valid_feature_indices = []
            for i, fname in enumerate(feature_names):
                col = X_arr[:, i]
                if np.std(col) > 1e-10:  # Non-constant
                    valid_features.append(fname)
                    valid_feature_indices.append(i)

            if len(valid_features) < 1:
                logger.warning(f"All features are constant for target: {target}")
                continue

            X_arr = X_arr[:, valid_feature_indices]
            feature_names = valid_features

            try:
                # Create and fit PySR model
                model = PySRRegressor(
                    niterations=self.niterations,
                    binary_operators=self.binary_operators,
                    unary_operators=self.unary_operators,
                    populations=self.populations,
                    population_size=self.population_size,
                    maxsize=self.maxsize,
                    timeout_in_seconds=self.timeout_seconds,
                    progress=False,  # Disable progress bar for automation
                    verbosity=0,  # Minimal output
                )

                model.fit(X_arr, y_arr, variable_names=feature_names)

                # Get best equation
                best_eq = model.get_best()
                if best_eq is None:
                    logger.warning(f"PySR found no equations for target: {target}")
                    continue

                # Calculate metrics
                y_pred = model.predict(X_arr)
                mse = float(np.mean((y_arr - y_pred) ** 2))
                mae = float(np.mean(np.abs(y_arr - y_pred)))

                # R-squared
                ss_res = np.sum((y_arr - y_pred) ** 2)
                ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                # Get equation string
                equation_str = str(best_eq["equation"]) if "equation" in best_eq else str(best_eq)

                # Try to get LaTeX
                try:
                    equation_latex = model.latex()
                except Exception:
                    equation_latex = equation_str

                # Get complexity
                complexity = int(best_eq.get("complexity", 0)) if isinstance(best_eq, dict) else 0

                formulas[target] = FormulaResult(
                    target=target,
                    equation=equation_str,
                    equation_latex=equation_latex,
                    complexity=complexity,
                    r2_score=float(r2),
                    mse=mse,
                    mae=mae,
                    feature_names=feature_names,
                    n_samples=len(X_arr),
                    _model=model,
                )

                logger.info(f"Found formula for {target}: {equation_str} (R²={r2:.3f})")

            except Exception as e:
                logger.error(f"PySR failed for target {target}: {e}")
                continue

        return formulas

    def add_predictions_to_rows(
        self,
        rows: List[Union[TraceRow, Dict[str, Any]]],
        formulas: Dict[str, FormulaResult],
    ) -> List[Dict[str, Any]]:
        """
        Add predictions and formula metadata to trace rows.

        Args:
            rows: Original trace rows
            formulas: Discovered formulas from discover_formulas()

        Returns:
            List of dictionaries with added calc_* columns
        """
        if not formulas:
            # No formulas, just convert rows to dicts
            result = []
            for row in rows:
                if isinstance(row, TraceRow):
                    result.append(row.to_dict())
                else:
                    result.append(dict(row))
            return result

        if not self.is_available():
            # No numpy, can't compute predictions
            result = []
            for row in rows:
                if isinstance(row, TraceRow):
                    row_dict = row.to_dict()
                else:
                    row_dict = dict(row)

                # Add formula metadata only
                for target, formula in formulas.items():
                    row_dict[f"calc_formula_{target}"] = formula.equation
                    row_dict[f"calc_r2_{target}"] = formula.r2_score
                    row_dict[f"calc_complexity_{target}"] = formula.complexity

                result.append(row_dict)
            return result

        import numpy as np

        # Extract numeric data for predictions
        numeric_rows, _ = self._extract_numeric_data(rows)

        result = []
        for i, row in enumerate(rows):
            if isinstance(row, TraceRow):
                row_dict = row.to_dict()
            else:
                row_dict = dict(row)

            numeric_row = numeric_rows[i] if i < len(numeric_rows) else {}

            for target, formula in formulas.items():
                # Add formula metadata
                row_dict[f"calc_formula_{target}"] = formula.equation
                row_dict[f"calc_r2_{target}"] = formula.r2_score
                row_dict[f"calc_complexity_{target}"] = formula.complexity

                # Compute prediction if model available
                if formula._model is not None and formula.feature_names:
                    try:
                        # Build feature vector
                        feature_vals = [
                            numeric_row.get(f, 0.0) for f in formula.feature_names
                        ]
                        X = np.array([feature_vals], dtype=np.float64)
                        prediction = formula._model.predict(X)[0]

                        row_dict[f"calc_predicted_{target}"] = float(prediction)

                        # Compute residual if actual value exists
                        actual = numeric_row.get(target)
                        if actual is not None:
                            row_dict[f"calc_residual_{target}"] = float(actual - prediction)

                    except Exception as e:
                        logger.debug(f"Could not compute prediction for {target}: {e}")

            result.append(row_dict)

        return result


def export_formulas_summary(
    formulas: Dict[str, FormulaResult],
    output_path: str,
) -> str:
    """
    Export discovered formulas to a JSON summary file.

    Args:
        formulas: Dictionary of FormulaResult objects
        output_path: Output file path (typically .json)

    Returns:
        Path to the exported file
    """
    summary = {
        "export_time": datetime.now().isoformat(),
        "formula_count": len(formulas),
        "formulas": {
            target: formula.to_dict() for target, formula in formulas.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return output_path
