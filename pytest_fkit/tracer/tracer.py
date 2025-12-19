"""
Tracer: High-level API for execution tracing and analysis.

Provides a simple interface for tracing code execution with
automatic data collection, incremental computation, and export.
"""

import functools
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .collector import TraceCollector, TraceContext, TraceRow
from .exporter import (
    CSVExporter,
    JSONExporter,
    PySRDataPreparer,
    export_for_pysr,
    export_to_csv,
    export_to_json,
)
from .incremental import (
    IncrementalCache,
    IncrementalDataAccumulator,
    DependencyTracker,
    compute_hash,
    memoize,
)

T = TypeVar("T")
Func = TypeVar("Func", bound=Callable)

# Global tracer instance
_global_tracer: Optional["Tracer"] = None


def get_tracer() -> "Tracer":
    """Get or create the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def reset_tracer() -> None:
    """Reset the global tracer instance."""
    global _global_tracer
    _global_tracer = None


class Tracer:
    """
    Main tracer interface for execution tracing and analysis.

    Combines trace collection, incremental computation, and export
    into a simple, unified API.

    Usage:
        tracer = Tracer()

        # Context manager tracing
        with tracer.trace("my_operation") as ctx:
            result = expensive_computation()
            ctx.add_metric("items", len(result))

        # Decorator tracing
        @tracer.traced
        def my_function(x, y):
            return x + y

        # Export results
        tracer.export_csv("traces.csv")
        tracer.export_for_pysr("metrics.csv")

    For pytest integration, use the pytest plugin hooks instead.
    """

    def __init__(
        self,
        name: str = "default",
        cache_size: int = 1000,
        cache_ttl: Optional[float] = 3600.0,
    ):
        self.name = name
        self.collector = TraceCollector(name)
        self.cache = IncrementalCache(max_size=cache_size, default_ttl=cache_ttl)
        self.dependency_tracker = DependencyTracker()
        self._pysr_preparer: Optional[PySRDataPreparer] = None

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

        Example:
            with tracer.trace("process_batch") as ctx:
                result = process(data)
                ctx.add_metric("batch_size", len(data))
        """
        return self.collector.trace(name, source)

    def traced(
        self,
        name: Optional[str] = None,
        track_args: bool = False,
    ) -> Callable[[Func], Func]:
        """
        Decorator to trace a function's execution.

        Args:
            name: Custom name (defaults to function name)
            track_args: Whether to add args as metrics

        Example:
            @tracer.traced()
            def my_function(x, y):
                return x + y
        """

        def decorator(fn: Func) -> Func:
            trace_name = name or fn.__name__

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                source = f"{fn.__module__}.{fn.__name__}"
                with self.trace(trace_name, source=source) as ctx:
                    if track_args:
                        # Add numeric args as metrics
                        for i, arg in enumerate(args):
                            if isinstance(arg, (int, float)):
                                ctx.add_metric(f"arg_{i}", float(arg))
                        for k, v in kwargs.items():
                            if isinstance(v, (int, float)):
                                ctx.add_metric(f"kwarg_{k}", float(v))

                    result = fn(*args, **kwargs)

                    # Add result size if iterable
                    if hasattr(result, "__len__"):
                        try:
                            ctx.add_metric("result_size", float(len(result)))
                        except TypeError:
                            pass

                    return result

            return wrapper  # type: ignore

        return decorator

    def memoized(
        self,
        ttl: Optional[float] = None,
    ) -> Callable[[Func], Func]:
        """
        Decorator for memoized functions with tracing.

        Combines tracing with caching for expensive computations.

        Example:
            @tracer.memoized(ttl=3600)
            def expensive_analysis(data):
                ...
        """
        return memoize(cache=self.cache, ttl=ttl)

    def add_metric(
        self,
        name: str,
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        source: str = "",
    ) -> TraceRow:
        """
        Add a metric row directly (without timing context).

        Useful for capturing external metrics or batch data.
        """
        return self.collector.add_metric_row(name, metrics, tags, source)

    # Export methods
    def export_csv(self, output_path: str) -> str:
        """Export all traces to CSV."""
        return export_to_csv(self.collector.rows, output_path)

    def export_json(
        self,
        output_path: str,
        pretty: bool = True,
    ) -> str:
        """Export all traces to JSON."""
        return export_to_json(
            self.collector.rows,
            output_path,
            metadata={"tracer_name": self.name},
            pretty=pretty,
        )

    def export_for_pysr(
        self,
        output_path: str,
        target: str = "duration_ms",
        features: Optional[List[str]] = None,
    ) -> str:
        """
        Export numeric metrics for PySR symbolic regression.

        Args:
            output_path: Output CSV path
            target: Target variable (y)
            features: Feature variables (X), or None for all

        Returns:
            Path to exported file
        """
        return export_for_pysr(
            self.collector.rows,
            output_path,
            target_column=target,
            feature_columns=features,
        )

    def get_pysr_preparer(
        self,
        target: str = "duration_ms",
        features: Optional[List[str]] = None,
    ) -> PySRDataPreparer:
        """
        Get a PySRDataPreparer populated with current trace data.

        Can be used directly with PySR:
            preparer = tracer.get_pysr_preparer()
            X, y = preparer.get_arrays()
            model = PySRRegressor()
            model.fit(X, y)
        """
        preparer = PySRDataPreparer(target=target, features=features)
        preparer.add_many(self.collector.rows)
        return preparer

    def create_accumulator(
        self,
        batch_size: int = 100,
        on_batch: Optional[Callable[[List[Dict[str, float]]], Any]] = None,
    ) -> IncrementalDataAccumulator:
        """
        Create an incremental data accumulator for streaming analysis.

        Args:
            batch_size: Number of points before triggering batch processing
            on_batch: Callback for batch processing (e.g., PySR fit)

        Returns:
            IncrementalDataAccumulator instance
        """
        return IncrementalDataAccumulator(
            batch_size=batch_size,
            on_batch=on_batch,
            cache=self.cache,
        )

    # Summary and inspection
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected traces."""
        summary = self.collector.get_summary()
        summary["cache_stats"] = self.cache.stats()
        return summary

    def get_rows(self) -> List[TraceRow]:
        """Get all collected trace rows."""
        return self.collector.rows

    def get_metrics(self) -> List[Dict[str, float]]:
        """Get all numeric metrics (PySR-compatible format)."""
        return self.collector.get_metrics_dataframe()

    def clear(self) -> None:
        """Clear all collected traces and cache."""
        self.collector.clear()
        self.cache.clear()
        self.dependency_tracker.clear()

    def __len__(self) -> int:
        """Number of collected traces."""
        return len(self.collector.rows)


# Convenience functions using global tracer


@contextmanager
def trace(name: str, source: str = ""):
    """
    Context manager for tracing using the global tracer.

    Example:
        from pytest_fkit.tracer import trace

        with trace("my_operation") as ctx:
            result = do_work()
            ctx.add_metric("count", len(result))
    """
    tracer = get_tracer()
    with tracer.trace(name, source) as ctx:
        yield ctx


def traced(
    name: Optional[str] = None,
    track_args: bool = False,
) -> Callable[[Func], Func]:
    """
    Decorator for tracing using the global tracer.

    Example:
        from pytest_fkit.tracer import traced

        @traced()
        def my_function(x, y):
            return x + y
    """
    return get_tracer().traced(name, track_args)
