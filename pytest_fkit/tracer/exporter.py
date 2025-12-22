"""
Trace Exporter: Export trace data to CSV, JSON, and PySR-compatible formats.

Zero-dependency export using Python stdlib (csv, json).
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TextIO, Union

from .collector import TraceRow


class CSVExporter:
    """
    Export trace data to CSV format.

    Uses Python's built-in csv module (no pandas required).
    """

    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path
        self.rows: List[Dict[str, Any]] = []

    def add_row(self, row: Union[TraceRow, Dict[str, Any]]) -> None:
        """Add a trace row to the export."""
        if isinstance(row, TraceRow):
            self.rows.append(row.to_dict())
        else:
            self.rows.append(row)

    def add_rows(self, rows: List[Union[TraceRow, Dict[str, Any]]]) -> None:
        """Add multiple trace rows."""
        for row in rows:
            self.add_row(row)

    def export(self, output_path: Optional[str] = None) -> str:
        """
        Export all rows to CSV.

        Args:
            output_path: Output file path (overrides constructor path)

        Returns:
            Path to the exported file
        """
        path = output_path or self.output_path
        if not path:
            path = f"traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        if not self.rows:
            # Export empty file with minimal headers
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "name", "duration_ms"])
            return path

        # Get all fieldnames from all rows (handle varying schemas)
        all_fields: set = set()
        for row in self.rows:
            all_fields.update(row.keys())
        fieldnames = sorted(all_fields)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

        return path

    def export_to_stream(self, stream: TextIO) -> None:
        """Export to a stream/file object."""
        if not self.rows:
            return

        all_fields: set = set()
        for row in self.rows:
            all_fields.update(row.keys())
        fieldnames = sorted(all_fields)

        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.rows)


class JSONExporter:
    """
    Export trace data to JSON format.

    Supports both single-file export and JSONL (newline-delimited).
    """

    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path
        self.rows: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def add_row(self, row: Union[TraceRow, Dict[str, Any]]) -> None:
        """Add a trace row to the export."""
        if isinstance(row, TraceRow):
            self.rows.append(row.to_dict())
        else:
            self.rows.append(row)

    def add_rows(self, rows: List[Union[TraceRow, Dict[str, Any]]]) -> None:
        """Add multiple trace rows."""
        for row in rows:
            self.add_row(row)

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set export metadata (included in JSON output)."""
        self.metadata = metadata

    def export(self, output_path: Optional[str] = None, pretty: bool = True) -> str:
        """
        Export all rows to JSON.

        Args:
            output_path: Output file path
            pretty: Whether to format with indentation

        Returns:
            Path to the exported file
        """
        path = output_path or self.output_path
        if not path:
            path = f"traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "trace_count": len(self.rows),
                **self.metadata,
            },
            "traces": self.rows,
        }

        with open(path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(output, f, indent=2, default=str)
            else:
                json.dump(output, f, default=str)

        return path

    def export_jsonl(self, output_path: Optional[str] = None) -> str:
        """
        Export to JSONL (newline-delimited JSON) format.

        Better for streaming and large datasets.
        """
        path = output_path or self.output_path
        if not path:
            path = f"traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, default=str) + "\n")

        return path


def export_to_csv(
    rows: List[Union[TraceRow, Dict[str, Any]]],
    output_path: str,
) -> str:
    """
    Convenience function to export trace rows to CSV.
    """
    exporter = CSVExporter(output_path)
    exporter.add_rows(rows)
    return exporter.export()


def export_to_json(
    rows: List[Union[TraceRow, Dict[str, Any]]],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    pretty: bool = True,
) -> str:
    """
    Convenience function to export trace rows to JSON.
    """
    exporter = JSONExporter(output_path)
    exporter.add_rows(rows)
    if metadata:
        exporter.set_metadata(metadata)
    return exporter.export(pretty=pretty)


def export_for_pysr(
    rows: List[Union[TraceRow, Dict[str, Any]]],
    output_path: str,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> str:
    """
    Export numeric data in PySR-compatible format.

    PySR expects a CSV with numeric columns only.
    This function filters to only include numeric metrics.

    Args:
        rows: Trace rows to export
        output_path: Output CSV path
        target_column: Column to predict (y variable)
        feature_columns: Columns to use as features (X variables)

    Returns:
        Path to the exported file
    """
    # Extract numeric data
    numeric_rows = []
    for row in rows:
        if isinstance(row, TraceRow):
            numeric_rows.append(row.to_pysr_row())
        else:
            # Filter to numeric values
            numeric_row = {
                k: v
                for k, v in row.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            numeric_rows.append(numeric_row)

    if not numeric_rows:
        # Create empty file
        with open(output_path, "w") as f:
            f.write("duration_ms\n")
        return output_path

    # Determine columns
    all_cols: set = set()
    for row in numeric_rows:
        all_cols.update(row.keys())

    if feature_columns:
        cols = [c for c in feature_columns if c in all_cols]
    else:
        cols = sorted(all_cols)

    # Put target column last (PySR convention)
    if target_column and target_column in cols:
        cols.remove(target_column)
        cols.append(target_column)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in numeric_rows:
            # Fill missing values with 0
            filled_row = {c: row.get(c, 0.0) for c in cols}
            writer.writerow(filled_row)

    return output_path


class PySRDataPreparer:
    """
    Prepare trace data for PySR symbolic regression.

    Handles:
    - Collecting numeric metrics from traces
    - Separating features (X) from target (y)
    - Converting to numpy arrays if available
    - Generating PySR-compatible data structures

    Usage:
        preparer = PySRDataPreparer(target="duration_ms")

        for row in trace_rows:
            preparer.add(row)

        X, y = preparer.get_arrays()

        # Use with PySR
        from pysr import PySRRegressor
        model = PySRRegressor()
        model.fit(X, y)
    """

    def __init__(
        self,
        target: str = "duration_ms",
        features: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.target = target
        self.features = features
        self.exclude = exclude or []
        self._data: List[Dict[str, float]] = []

    def add(self, row: Union[TraceRow, Dict[str, Any]]) -> None:
        """Add a trace row."""
        if isinstance(row, TraceRow):
            numeric = row.to_pysr_row()
        else:
            numeric = {
                k: float(v)
                for k, v in row.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        self._data.append(numeric)

    def add_many(self, rows: List[Union[TraceRow, Dict[str, Any]]]) -> None:
        """Add multiple rows."""
        for row in rows:
            self.add(row)

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        if not self._data:
            return []

        all_cols: set = set()
        for row in self._data:
            all_cols.update(row.keys())

        # Remove target and excluded columns
        all_cols.discard(self.target)
        for ex in self.exclude:
            all_cols.discard(ex)

        if self.features:
            return [f for f in self.features if f in all_cols]
        return sorted(all_cols)

    def get_dataframe(self) -> "Any":
        """
        Get data as pandas DataFrame if available.

        Returns:
            pandas.DataFrame or raises ImportError
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for get_dataframe()")

        return pd.DataFrame(self._data)

    def get_arrays(self) -> "tuple":
        """
        Get X (features) and y (target) as numpy arrays.

        Returns:
            Tuple of (X, y) numpy arrays
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for get_arrays()")

        if not self._data:
            return np.array([]).reshape(0, 0), np.array([])

        feature_names = self.get_feature_names()

        X = []
        y = []
        for row in self._data:
            X.append([row.get(f, 0.0) for f in feature_names])
            y.append(row.get(self.target, 0.0))

        return np.array(X), np.array(y)

    def export_csv(self, output_path: str) -> str:
        """Export to CSV file."""
        return export_for_pysr(
            self._data,
            output_path,
            target_column=self.target,
            feature_columns=self.features,
        )

    def __len__(self) -> int:
        return len(self._data)


def export_to_csv_with_formulas(
    rows: List[Union[TraceRow, Dict[str, Any]]],
    output_path: str,
    pysr_targets: Optional[List[str]] = None,
    pysr_min_samples: int = 20,
    formulas_output_path: Optional[str] = None,
) -> tuple:
    """
    Export traces to CSV with automatic PySR formula discovery.

    If PySR is available and there are enough samples, runs symbolic
    regression and adds predictions as new columns.

    Args:
        rows: Trace rows to export
        output_path: Output CSV path
        pysr_targets: Target metrics for PySR (auto-detect if None)
        pysr_min_samples: Minimum samples required for PySR
        formulas_output_path: Optional path for formulas JSON (default: {output_path}_formulas.json)

    Returns:
        Tuple of (csv_path, formulas_dict or None)
    """
    from .pysr_runner import PySRRunner, export_formulas_summary

    formulas = None
    rows_to_export = rows

    try:
        runner = PySRRunner(min_samples=pysr_min_samples)

        if runner.is_available() and len(rows) >= pysr_min_samples:
            # Discover formulas
            formulas = runner.discover_formulas(rows, targets=pysr_targets)

            if formulas:
                # Add predictions to rows
                rows_to_export = runner.add_predictions_to_rows(rows, formulas)

                # Export formulas summary
                if formulas_output_path is None:
                    # Default to same path with _formulas.json suffix
                    base = output_path.rsplit(".", 1)[0]
                    formulas_output_path = f"{base}_formulas.json"

                export_formulas_summary(formulas, formulas_output_path)

    except ImportError:
        # PySR not installed, continue with normal export
        pass
    except Exception as e:
        # Log error but don't fail the export
        import logging
        logging.getLogger(__name__).warning(f"PySR analysis failed: {e}")

    # Export CSV
    csv_path = export_to_csv(rows_to_export, output_path)

    return csv_path, formulas
