"""
Trace Collector: Core data collection for execution tracing.

Captures structured trace data including timing, metrics, and metadata
for analysis and export to tools like PySR.
"""

import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

T = TypeVar("T")


def content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class TraceRow:
    """A single trace record with timing and metrics."""

    # Identifiers
    timestamp: str
    trace_id: str
    name: str = ""
    source: str = ""

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0

    # Metrics (numeric data for PySR)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    parent_id: Optional[str] = None

    # Status
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        base = {
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "name": self.name,
            "source": self.source,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error or "",
            "parent_id": self.parent_id or "",
        }
        # Flatten metrics with prefix
        for k, v in self.metrics.items():
            base[f"metric_{k}"] = v
        # Flatten tags with prefix
        for k, v in self.tags.items():
            base[f"tag_{k}"] = v
        return base

    def to_pysr_row(self) -> Dict[str, float]:
        """
        Convert to PySR-compatible row (numeric only).

        Returns only numeric metrics suitable for symbolic regression.
        """
        row = {"duration_ms": self.duration_ms}
        row.update(self.metrics)
        return row


class TraceContext:
    """
    Context for a single trace span.

    Automatically captures timing and allows adding metrics/tags.
    """

    def __init__(
        self,
        name: str,
        collector: "TraceCollector",
        parent_id: Optional[str] = None,
        source: str = "",
    ):
        self.name = name
        self.collector = collector
        self.parent_id = parent_id
        self.source = source

        self.trace_id = content_hash(f"{name}:{time.time()}")[:12]
        self.start_time = 0.0
        self.end_time = 0.0
        self.metrics: Dict[str, float] = {}
        self.tags: Dict[str, str] = {}
        self.success = True
        self.error: Optional[str] = None
        self._row: Optional[TraceRow] = None

    def add_metric(self, name: str, value: float) -> "TraceContext":
        """Add a numeric metric."""
        self.metrics[name] = value
        return self

    def add_metrics(self, **kwargs: float) -> "TraceContext":
        """Add multiple metrics at once."""
        self.metrics.update(kwargs)
        return self

    def add_tag(self, name: str, value: str) -> "TraceContext":
        """Add a string tag."""
        self.tags[name] = value
        return self

    def add_tags(self, **kwargs: str) -> "TraceContext":
        """Add multiple tags at once."""
        self.tags.update(kwargs)
        return self

    def set_error(self, error: str) -> "TraceContext":
        """Mark trace as failed with error message."""
        self.success = False
        self.error = error
        return self

    def __enter__(self) -> "TraceContext":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000

        if exc_val is not None:
            self.success = False
            self.error = str(exc_val)

        self._row = TraceRow(
            timestamp=datetime.now().isoformat(),
            trace_id=self.trace_id,
            name=self.name,
            source=self.source,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_ms=duration_ms,
            metrics=self.metrics.copy(),
            tags=self.tags.copy(),
            parent_id=self.parent_id,
            success=self.success,
            error=self.error,
        )

        self.collector._add_row(self._row)

    @property
    def row(self) -> Optional[TraceRow]:
        """Get the finalized trace row (available after context exit)."""
        return self._row


class TraceCollector:
    """
    Collects trace data across multiple operations.

    Usage:
        collector = TraceCollector()

        with collector.trace("operation1") as ctx:
            result = do_work()
            ctx.add_metric("items_processed", len(result))

        with collector.trace("operation2") as ctx:
            # nested tracing
            ...

        # Export for analysis
        collector.export_csv("traces.csv")
        collector.export_for_pysr("metrics.csv")  # numeric only
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.rows: List[TraceRow] = []
        self._current_trace_id: Optional[str] = None
        self._start_time = time.time()

    def trace(
        self,
        name: str,
        source: str = "",
    ) -> TraceContext:
        """
        Create a trace context for an operation.

        Args:
            name: Name of the operation being traced
            source: Source identifier (file, function, etc.)

        Returns:
            TraceContext to use as context manager
        """
        return TraceContext(
            name=name,
            collector=self,
            parent_id=self._current_trace_id,
            source=source,
        )

    def _add_row(self, row: TraceRow) -> None:
        """Internal: add a completed trace row."""
        self.rows.append(row)

    def add_metric_row(
        self,
        name: str,
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        source: str = "",
    ) -> TraceRow:
        """
        Add a metric row directly without timing context.

        Useful for capturing external metrics or batch data.
        """
        row = TraceRow(
            timestamp=datetime.now().isoformat(),
            trace_id=content_hash(f"{name}:{time.time()}")[:12],
            name=name,
            source=source,
            metrics=metrics,
            tags=tags or {},
        )
        self.rows.append(row)
        return row

    def get_metrics_dataframe(self) -> List[Dict[str, float]]:
        """
        Get all numeric metrics as list of dicts (PySR-compatible).

        Returns data suitable for symbolic regression.
        """
        return [row.to_pysr_row() for row in self.rows]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected traces."""
        if not self.rows:
            return {"total_traces": 0}

        durations = [r.duration_ms for r in self.rows]
        successful = sum(1 for r in self.rows if r.success)

        # Collect all unique metric names
        all_metrics = set()
        for row in self.rows:
            all_metrics.update(row.metrics.keys())

        return {
            "total_traces": len(self.rows),
            "successful": successful,
            "failed": len(self.rows) - successful,
            "total_duration_ms": sum(durations),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "metric_names": list(all_metrics),
            "unique_names": list(set(r.name for r in self.rows)),
        }

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Convert all rows to dictionaries."""
        return [row.to_dict() for row in self.rows]

    def clear(self) -> None:
        """Clear all collected traces."""
        self.rows = []
