"""
Standalone CLI for tracing command execution.

Usage:
    # Trace a single command
    python -m pytest_fkit.tracer run "python train.py"

    # Trace multiple commands
    python -m pytest_fkit.tracer run "python preprocess.py" "python train.py"

    # Trace with output
    python -m pytest_fkit.tracer run --output traces.csv "python train.py"

    # Export for PySR
    python -m pytest_fkit.tracer run --pysr metrics.csv "python train.py"

    # Analyze existing traces
    python -m pytest_fkit.tracer analyze traces.csv --formula

    # Convert trace formats
    python -m pytest_fkit.tracer convert traces.csv --to json
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from .collector import TraceCollector
from .exporter import export_for_pysr, export_to_csv, export_to_csv_with_formulas, export_to_json


def run_command(
    command: str,
    collector: TraceCollector,
    timeout: int = 600,
    capture: bool = True,
) -> int:
    """
    Run a command with tracing.

    Returns exit code.
    """
    with collector.trace(command[:50], source=f"cmd:{command}") as ctx:
        start = time.perf_counter()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=capture,
                text=True,
                timeout=timeout,
            )

            duration_ms = (time.perf_counter() - start) * 1000
            ctx.add_metric("duration_ms", duration_ms)
            ctx.add_metric("exit_code", float(result.returncode))

            if result.returncode != 0:
                ctx.set_error(f"Exit code: {result.returncode}")

            # Capture output metrics
            if capture:
                ctx.add_metric("stdout_lines", float(result.stdout.count("\n")))
                ctx.add_metric("stderr_lines", float(result.stderr.count("\n")))

                # Print output
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr)

            return result.returncode

        except subprocess.TimeoutExpired:
            ctx.set_error(f"Timeout after {timeout}s")
            ctx.add_metric("timeout", 1.0)
            return -1
        except Exception as e:
            ctx.set_error(str(e))
            return -1


def cmd_run(args: argparse.Namespace) -> int:
    """Run commands with tracing."""
    collector = TraceCollector(name="cli")

    exit_code = 0
    for command in args.commands:
        print(f"\n{'='*60}")
        print(f"Running: {command}")
        print("=" * 60)

        code = run_command(
            command,
            collector,
            timeout=args.timeout,
            capture=not args.no_capture,
        )

        if code != 0:
            exit_code = code
            if args.stop_on_error:
                break

    # Export results with automatic PySR formula discovery
    if args.output:
        csv_path, formulas = export_to_csv_with_formulas(
            collector.rows,
            args.output,
        )
        print(f"\nTraces exported to: {csv_path}")

        if formulas:
            print(f"PySR discovered {len(formulas)} formula(s):")
            for target, result in formulas.items():
                print(f"  {target}: {result.equation} (R²={result.r2_score:.3f})")

    if args.pysr:
        export_for_pysr(collector.rows, args.pysr)
        print(f"PySR data exported to: {args.pysr}")

    if args.json:
        export_to_json(collector.rows, args.json)
        print(f"JSON exported to: {args.json}")

    # Print summary
    summary = collector.get_summary()
    print(f"\n{'='*60}")
    print("Tracing Summary")
    print("=" * 60)
    print(f"Commands traced: {summary['total_traces']}")
    print(f"Total duration: {summary['total_duration_ms']:.2f}ms")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")

    return exit_code


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze existing trace data."""
    import csv
    import json

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    # Load data based on extension
    if input_path.suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
            rows = data.get("traces", [])
    elif input_path.suffix == ".jsonl":
        rows = []
        with open(input_path) as f:
            for line in f:
                rows.append(json.loads(line))
    else:  # CSV
        with open(input_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    if not rows:
        print("No data found")
        return 0

    # Basic statistics
    print(f"\nAnalyzing: {input_path}")
    print(f"Total rows: {len(rows)}")

    # Find numeric columns
    numeric_cols = []
    for key in rows[0].keys():
        try:
            float(rows[0][key])
            numeric_cols.append(key)
        except (ValueError, TypeError):
            pass

    print(f"Numeric columns: {', '.join(numeric_cols[:10])}")

    # Compute stats for key columns
    if "duration_ms" in numeric_cols:
        durations = [float(r.get("duration_ms", 0)) for r in rows]
        print(f"\nDuration statistics:")
        print(f"  Total: {sum(durations):.2f}ms")
        print(f"  Mean: {sum(durations)/len(durations):.2f}ms")
        print(f"  Min: {min(durations):.2f}ms")
        print(f"  Max: {max(durations):.2f}ms")

    # Formula discovery with PySR
    if args.formula:
        from .pysr_runner import PySRRunner, export_formulas_summary

        runner = PySRRunner(
            min_samples=10,
            niterations=args.iterations or 40,
        )

        if not runner.is_available():
            print("\nPySR not installed. Install with: pip install pytest-fkit[pysr]")
            return 1

        print("\nRunning symbolic regression with PySR...")

        # Determine target
        targets = [args.target] if args.target else None

        try:
            formulas = runner.discover_formulas(rows, targets=targets)

            if not formulas:
                print("No formulas discovered (not enough data or no valid targets)")
                return 1

            print("\nDiscovered formulas:")
            for target, result in formulas.items():
                print(f"\n  Target: {target}")
                print(f"    Equation: {result.equation}")
                print(f"    R² Score: {result.r2_score:.4f}")
                print(f"    MSE: {result.mse:.4f}")
                print(f"    Complexity: {result.complexity}")
                print(f"    Features: {', '.join(result.feature_names)}")
                print(f"    Samples: {result.n_samples}")

            if args.formula_output:
                export_formulas_summary(formulas, args.formula_output)
                print(f"\nFormulas saved to: {args.formula_output}")

        except Exception as e:
            print(f"Error running PySR: {e}")
            return 1

    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert between trace formats."""
    import csv
    import json

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    # Load data
    if input_path.suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
            rows = data.get("traces", [])
    elif input_path.suffix == ".jsonl":
        rows = []
        with open(input_path) as f:
            for line in f:
                rows.append(json.loads(line))
    else:  # CSV
        with open(input_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    if not rows:
        print("No data to convert")
        return 0

    # Determine output format and path
    output_format = args.to
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_suffix(f".{output_format}"))

    # Convert
    if output_format == "csv":
        export_to_csv(rows, output_path)
    elif output_format == "json":
        export_to_json(rows, output_path)
    elif output_format == "jsonl":
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
    elif output_format == "pysr":
        export_for_pysr(rows, output_path)
    else:
        print(f"Unknown format: {output_format}")
        return 1

    print(f"Converted {len(rows)} rows to: {output_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tracer",
        description="Standalone tracer for command execution",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run commands with tracing")
    run_parser.add_argument("commands", nargs="+", help="Commands to run")
    run_parser.add_argument(
        "-o", "--output", help="Output CSV path (default: traces.csv)"
    )
    run_parser.add_argument("--pysr", help="Output PySR-compatible CSV")
    run_parser.add_argument("--json", help="Output JSON path")
    run_parser.add_argument("--timeout", type=int, default=600, help="Command timeout")
    run_parser.add_argument(
        "--stop-on-error", action="store_true", help="Stop on first error"
    )
    run_parser.add_argument(
        "--no-capture", action="store_true", help="Don't capture output"
    )
    run_parser.set_defaults(func=cmd_run)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze trace data")
    analyze_parser.add_argument("input", help="Input trace file")
    analyze_parser.add_argument(
        "--formula", action="store_true", help="Run symbolic regression with PySR"
    )
    analyze_parser.add_argument("--target", help="Target variable for regression")
    analyze_parser.add_argument("--iterations", type=int, help="PySR iterations")
    analyze_parser.add_argument("--formula-output", help="Output path for formulas")
    analyze_parser.set_defaults(func=cmd_analyze)

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert trace formats")
    convert_parser.add_argument("input", help="Input trace file")
    convert_parser.add_argument(
        "--to",
        choices=["csv", "json", "jsonl", "pysr"],
        required=True,
        help="Output format",
    )
    convert_parser.add_argument("-o", "--output", help="Output path")
    convert_parser.set_defaults(func=cmd_convert)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
