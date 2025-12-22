"""
Pytest plugin for automatic test execution tracing.

Captures timing, metrics, and test outcomes for analysis and
formula discovery with PySR.

Enable with: pytest --trace-metrics

Usage:
    # Run tests with tracing
    pytest --trace-metrics --trace-output=traces.csv

    # Export for PySR analysis
    pytest --trace-metrics --trace-pysr=metrics.csv
"""

import os
import time
from typing import Any, Dict, List, Optional

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.reports import TestReport

from .collector import TraceCollector, TraceRow
from .exporter import export_for_pysr, export_to_csv, export_to_csv_with_formulas, export_to_json


class TracerPlugin:
    """
    Pytest plugin that traces test execution.

    Collects:
    - Test timing (setup, call, teardown)
    - Memory usage (if psutil available)
    - Test outcomes
    - Custom metrics via fixtures
    """

    def __init__(self, config: Config):
        self.config = config
        self.collector = TraceCollector(name="pytest")

        # Options
        self.output_path = config.getoption("--trace-output")
        self.pysr_path = config.getoption("--trace-pysr")
        self.json_path = config.getoption("--trace-json")

        # Current test state
        self._current_test: Optional[str] = None
        self._test_start: float = 0.0
        self._phase_times: Dict[str, float] = {}
        self._test_metrics: Dict[str, float] = {}

        # Memory tracking
        self._track_memory = self._can_track_memory()
        self._initial_memory: float = 0.0

    def _can_track_memory(self) -> bool:
        """Check if memory tracking is available."""
        try:
            import psutil

            return True
        except ImportError:
            return False

    def _get_memory_mb(self) -> float:
        """Get current process memory in MB."""
        if not self._track_memory:
            return 0.0
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    # Pytest hooks
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item: Item) -> None:
        """Called before test setup."""
        self._current_test = item.nodeid
        self._test_start = time.perf_counter()
        self._phase_times = {}
        self._test_metrics = {}
        self._initial_memory = self._get_memory_mb()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: Item, call) -> None:
        """Called after each test phase (setup, call, teardown)."""
        outcome = yield
        report: TestReport = outcome.get_result()

        # Record phase timing
        self._phase_times[report.when] = report.duration

        # On final phase, create trace row
        if report.when == "teardown":
            self._finalize_test(item, report)

    def _finalize_test(self, item: Item, final_report: TestReport) -> None:
        """Create trace row for completed test."""
        total_duration = sum(self._phase_times.values()) * 1000  # to ms

        # Collect test info
        metrics = {
            "duration_ms": total_duration,
            "setup_ms": self._phase_times.get("setup", 0) * 1000,
            "call_ms": self._phase_times.get("call", 0) * 1000,
            "teardown_ms": self._phase_times.get("teardown", 0) * 1000,
        }

        # Memory delta
        if self._track_memory:
            final_memory = self._get_memory_mb()
            metrics["memory_delta_mb"] = final_memory - self._initial_memory
            metrics["memory_peak_mb"] = final_memory

        # Add any custom metrics from fixture
        metrics.update(self._test_metrics)

        # Outcome
        passed = final_report.outcome == "passed"
        metrics["passed"] = 1.0 if passed else 0.0
        metrics["failed"] = 0.0 if passed else 1.0

        # Tags
        tags = {
            "test_name": item.name,
            "module": item.module.__name__ if item.module else "",
            "outcome": final_report.outcome,
        }

        # Add markers as tags
        for marker in item.iter_markers():
            tags[f"marker_{marker.name}"] = "true"

        # Create trace row
        self.collector.add_metric_row(
            name=item.nodeid,
            metrics=metrics,
            tags=tags,
            source=str(item.fspath) if item.fspath else "",
        )

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        """Called after all tests complete."""
        # Export results with automatic PySR formula discovery
        if self.output_path:
            csv_path, formulas = export_to_csv_with_formulas(
                self.collector.rows,
                self.output_path,
            )
            print(f"\nTrace data exported to: {csv_path}")

            if formulas:
                print(f"PySR discovered {len(formulas)} formula(s):")
                for target, result in formulas.items():
                    print(f"  {target}: {result.equation} (R²={result.r2_score:.3f})")

        if self.pysr_path:
            export_for_pysr(
                self.collector.rows,
                self.pysr_path,
                target_column="duration_ms",
            )
            print(f"PySR-compatible data exported to: {self.pysr_path}")

        if self.json_path:
            export_to_json(self.collector.rows, self.json_path)
            print(f"JSON trace data exported to: {self.json_path}")

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config) -> None:
        """Add tracing summary to terminal output."""
        if not self.collector.rows:
            return

        summary = self.collector.get_summary()

        terminalreporter.section("Test Tracing Summary")
        terminalreporter.write_line(f"Total tests traced: {summary['total_traces']}")
        terminalreporter.write_line(
            f"Total duration: {summary['total_duration_ms']:.2f}ms"
        )
        terminalreporter.write_line(
            f"Avg duration: {summary['avg_duration_ms']:.2f}ms"
        )
        terminalreporter.write_line(
            f"Min/Max: {summary['min_duration_ms']:.2f}ms / {summary['max_duration_ms']:.2f}ms"
        )

        if summary.get("metric_names"):
            terminalreporter.write_line(
                f"Metrics collected: {', '.join(summary['metric_names'][:5])}"
            )


def pytest_addoption(parser: Parser) -> None:
    """Add tracing options to pytest."""
    group = parser.getgroup("tracing")

    group.addoption(
        "--trace-metrics",
        action="store_true",
        default=False,
        help="Enable test execution tracing",
    )

    group.addoption(
        "--trace-output",
        action="store",
        default=None,
        help="Output path for trace CSV (default: traces.csv)",
    )

    group.addoption(
        "--trace-pysr",
        action="store",
        default=None,
        help="Output path for PySR-compatible CSV",
    )

    group.addoption(
        "--trace-json",
        action="store",
        default=None,
        help="Output path for JSON trace data",
    )


def pytest_configure(config: Config) -> None:
    """Register the tracer plugin if enabled."""
    if config.getoption("--trace-metrics"):
        # Set default output if not specified
        if not config.getoption("--trace-output"):
            config.option.trace_output = "traces.csv"

        plugin = TracerPlugin(config)
        config.pluginmanager.register(plugin, "tracer_plugin")


# Fixture for adding custom metrics
@pytest.fixture
def trace_metrics(request) -> Dict[str, float]:
    """
    Fixture for adding custom metrics to the current test trace.

    Usage:
        def test_example(trace_metrics):
            # ... do work ...
            trace_metrics["custom_value"] = 42.0
            trace_metrics["another_metric"] = calculate_something()
    """
    metrics: Dict[str, float] = {}

    yield metrics

    # Store metrics for the plugin to pick up
    plugin = request.config.pluginmanager.get_plugin("tracer_plugin")
    if plugin:
        plugin._test_metrics.update(metrics)
