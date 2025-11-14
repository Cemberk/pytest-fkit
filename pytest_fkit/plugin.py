"""
pytest-fkit plugin: Isolate test crashes and convert them to ERROR results

Inspired by fkitpy - when tests crash (SIGABRT, SIGSEGV, etc.),
catch them and report as normal pytest errors instead of killing the entire run.

This plugin runs each test in a subprocess to isolate crashes.
"""
import sys
import os
import subprocess
import pytest
import signal
import pickle
import tempfile
import time
from pathlib import Path
from _pytest.runner import pytest_runtest_makereport


def pytest_addoption(parser):
    """Add command-line options for pytest-fkit."""
    group = parser.getgroup("fkit")
    group.addoption(
        "--fkit",
        action="store_true",
        default=False,
        help="Enable crash isolation (convert crashes to ERROR results)",
    )
    group.addoption(
        "--fkit-timeout",
        action="store",
        type=int,
        default=600,
        help="Timeout per test in seconds (default: 600 = 10 min)",
    )


def pytest_configure(config):
    """Register the plugin markers."""
    config.addinivalue_line(
        "markers",
        "fkit_skip: Skip crash isolation for this test (run normally)"
    )

    # Only register if enabled
    if config.getoption("--fkit"):
        config.pluginmanager.register(CrashIsolationPlugin(config), "fkit_plugin")


class CrashIsolationPlugin:
    """Plugin that runs each test in a subprocess to catch crashes."""

    def __init__(self, config):
        self.config = config
        self.timeout = config.getoption("--fkit-timeout")

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        """Hook that runs before each test - we'll run it in a subprocess."""

        # Check if test wants to skip isolation
        if item.get_closest_marker("fkit_skip"):
            # Run normally - return None to let other hooks handle it
            return None

        # We'll handle this test ourselves
        # Setup phase
        item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)

        # Call phase - run in subprocess for isolation
        call_report = self._run_test_in_subprocess(item)

        # Send report
        item.ihook.pytest_runtest_logreport(report=call_report)

        # Teardown phase
        item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)

        # Return True to tell pytest we handled this test completely
        return True

    def _run_test_in_subprocess(self, item):
        """Run a single test in an isolated subprocess."""
        import pickle
        import tempfile

        # Create a script that will run just this test
        # IMPORTANT: Disable fkit plugin in subprocess using PYTEST_DISABLE_PLUGIN_AUTOLOAD
        test_script = f"""
import sys
import os

# Disable fkit plugin in subprocess to prevent recursion
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'

import pytest

# Run the test without fkit plugin
exit_code = pytest.main([
    '{item.nodeid}',
    '-v',
    '--tb=short',
    '--continue-on-collection-errors',
    '-p', 'no:cacheprovider',
    '-p', 'no:fkit',  # Explicitly disable fkit
])

sys.exit(exit_code)
"""

        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name

        try:
            # Run subprocess
            start_time = time.time()

            try:
                # Prepare environment - disable fkit in subprocess
                env = os.environ.copy()
                env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'

                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(Path.cwd()),
                    env=env,
                )

                stop_time = time.time()
                duration = stop_time - start_time

                # Determine outcome based on return code
                if result.returncode == 0:
                    # Test passed
                    return self._make_report(item, "call", "passed", duration=duration)

                elif result.returncode < 0:
                    # Process was killed by signal (CRASH!)
                    signal_num = -result.returncode

                    # Map common signals
                    signal_names = {
                        signal.SIGABRT: "SIGABRT (Aborted)",
                        signal.SIGSEGV: "SIGSEGV (Segmentation Fault)",
                        signal.SIGTERM: "SIGTERM (Terminated)",
                        signal.SIGKILL: "SIGKILL (Killed)",
                    }

                    signal_name = signal_names.get(signal_num, f"Signal {signal_num}")

                    crash_info = (
                        f"\n{'='*70}\n"
                        f"💥 TEST CRASHED: {signal_name}\n"
                        f"{'='*70}\n"
                        f"\nThis test caused Python to crash with {signal_name}.\n"
                        f"pytest-fkit caught it and converted it to an ERROR.\n"
                        f"\n--- STDOUT ---\n{result.stdout}\n"
                        f"\n--- STDERR ---\n{result.stderr}\n"
                        f"{'='*70}\n"
                    )

                    return self._make_report(
                        item, "call", "failed",
                        longrepr=crash_info,
                        duration=duration,
                        crash=True
                    )

                else:
                    # Test failed normally
                    fail_info = f"\n--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}"
                    return self._make_report(
                        item, "call", "failed",
                        longrepr=fail_info,
                        duration=duration
                    )

            except subprocess.TimeoutExpired as e:
                # Test timed out
                stop_time = time.time()
                duration = stop_time - start_time

                timeout_info = (
                    f"\n{'='*70}\n"
                    f"⏱️  TEST TIMEOUT\n"
                    f"{'='*70}\n"
                    f"\nTest exceeded timeout of {self.timeout} seconds.\n"
                    f"pytest-fkit terminated it and converted it to an ERROR.\n"
                    f"\n--- PARTIAL STDOUT ---\n{e.stdout if e.stdout else '(none)'}\n"
                    f"\n--- PARTIAL STDERR ---\n{e.stderr if e.stderr else '(none)'}\n"
                    f"{'='*70}\n"
                )

                return self._make_report(
                    item, "call", "failed",
                    longrepr=timeout_info,
                    duration=duration,
                    timeout=True
                )

        finally:
            # Clean up temp script
            try:
                os.unlink(script_path)
            except:
                pass

    def _make_report(self, item, when, outcome, longrepr=None, duration=0, crash=False, timeout=False):
        """Create a test report."""
        from _pytest.reports import TestReport

        # Create report
        report = TestReport(
            nodeid=item.nodeid,
            location=item.location,
            keywords=item.keywords,
            outcome=outcome,
            longrepr=longrepr,
            when=when,
            duration=duration,
        )

        # Add custom attributes
        if crash:
            report.crash = True
        if timeout:
            report.timeout = True

        return report


def pytest_report_teststatus(report, config):
    """Customize test status reporting for crashes."""
    if hasattr(report, 'crash') and report.crash:
        return 'failed', '💥', ('CRASH', {'red': True})
    if hasattr(report, 'timeout') and report.timeout:
        return 'failed', '⏱️', ('TIMEOUT', {'yellow': True})


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add summary section for crashes and timeouts."""
    if not config.getoption("--fkit"):
        return

    # Count crashes and timeouts
    crashes = []
    timeouts = []

    for report in terminalreporter.stats.get('failed', []):
        if hasattr(report, 'crash') and report.crash:
            crashes.append(report.nodeid)
        elif hasattr(report, 'timeout') and report.timeout:
            timeouts.append(report.nodeid)

    if crashes or timeouts:
        terminalreporter.section("pytest-fkit summary")

        if crashes:
            terminalreporter.write_line(
                f"\n💥 {len(crashes)} test(s) CRASHED (converted to ERROR by pytest-fkit):",
                bold=True,
                red=True
            )
            for nodeid in crashes:
                terminalreporter.write_line(f"  - {nodeid}")

        if timeouts:
            terminalreporter.write_line(
                f"\n⏱️  {len(timeouts)} test(s) TIMED OUT (converted to ERROR by pytest-fkit):",
                bold=True,
                yellow=True
            )
            for nodeid in timeouts:
                terminalreporter.write_line(f"  - {nodeid}")

        terminalreporter.write_line(
            f"\n✅ pytest-fkit prevented {len(crashes) + len(timeouts)} crashes from killing your test suite!",
            bold=True,
            green=True
        )
