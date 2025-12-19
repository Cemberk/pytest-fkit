"""
pytest-fkit tracer: Standalone tracing module for AI tool execution analysis.

Provides:
- Trace collection with structured data capture
- Incremental computation for on-the-fly formula discovery
- Export to CSV/JSON/JSONL for PySR integration
- Pytest plugin hooks for test execution tracing

Usage (standalone):
    from pytest_fkit.tracer import Tracer, TraceCollector

    tracer = Tracer()
    with tracer.trace("my_operation"):
        result = expensive_computation()

    tracer.export_csv("traces.csv")

Usage (pytest):
    pytest --trace-metrics  # Enable tracing during test runs
"""

from .collector import TraceCollector, TraceRow, TraceContext
from .incremental import (
    DependencyTracker,
    IncrementalCache,
    IncrementalDataAccumulator,
    compute_hash,
    memoize,
)
from .exporter import CSVExporter, JSONExporter, export_for_pysr
from .tracer import Tracer, trace

__all__ = [
    # Core
    "Tracer",
    "trace",
    "TraceCollector",
    "TraceRow",
    "TraceContext",
    # Incremental
    "DependencyTracker",
    "IncrementalCache",
    "IncrementalDataAccumulator",
    "compute_hash",
    "memoize",
    # Export
    "CSVExporter",
    "JSONExporter",
    "export_for_pysr",
]

__version__ = "0.1.0"
