"""
pytest-fkit tracer: Comprehensive multi-dimensional tracing for AI/ML execution analysis.

Provides:
- Trace collection with structured data capture
- Incremental computation for on-the-fly formula discovery
- Hardware profiling (GPU detection, specs, monitoring)
- Performance analysis (GEMM, roofline, bottlenecks)
- Graph compilation tracing (torch.compile, JAX jit)
- Stack frame capture (call stacks, GIL, GC state)
- Dataflow tracking (computation DAG, value provenance)
- Multi-dimensional correlation (cross-layer bottleneck analysis)
- Export to CSV/JSON/JSONL for PySR integration
- Pytest plugin hooks for test execution tracing

Usage (standalone):
    from pytest_fkit.tracer import Tracer, HardwareProfiler, PerformanceProfiler

    tracer = Tracer()
    with tracer.trace("my_operation") as ctx:
        result = expensive_computation()
        ctx.add_metric("items", len(result))

    tracer.export_csv("traces.csv")
    tracer.export_for_pysr("metrics.csv")

Usage (unified multi-dimensional tracing):
    from pytest_fkit.tracer import UnifiedTracer

    tracer = UnifiedTracer()
    tracer.start()

    # Run your code - all dimensions are tracked
    result = my_computation()

    profile = tracer.stop()
    bottlenecks = tracer.analyze_bottlenecks()
    for b in bottlenecks:
        print(f"{b.source}: {b.suggested_fix}")

Usage (stack tracing):
    from pytest_fkit.tracer import StackTracer

    tracer = StackTracer()
    snapshot = tracer.capture()
    for thread in snapshot.threads:
        print(f"{thread.name}: {thread.call_stack}")

Usage (dataflow tracking):
    from pytest_fkit.tracer import DataflowTracker

    tracker = DataflowTracker()

    @tracker.track
    def my_function(x, y):
        return x + y

    result = my_function(1, 2)
    lineage = tracker.get_value_lineage(result)

Usage (hardware profiling):
    from pytest_fkit.tracer import detect_gpus, roofline_analysis

    gpus = detect_gpus()
    for gpu in gpus:
        print(f"{gpu.name}: {gpu.peak_tflops} TFLOPS")

Usage (performance analysis):
    from pytest_fkit.tracer import GEMMAnalysis, detect_bottlenecks

    gemm = GEMMAnalysis(M=1024, N=2048, K=512, precision="fp16")
    print(f"FLOPS: {gemm.flops}, AI: {gemm.arithmetic_intensity}")

Usage (graph compilation):
    from pytest_fkit.tracer import CompilationAnalyzer

    analyzer = CompilationAnalyzer()
    analysis = analyzer.analyze_code(source_code)

Usage (pytest):
    pytest --trace-metrics  # Enable tracing during test runs
"""

# Core tracing
from .collector import TraceCollector, TraceRow, TraceContext
from .tracer import Tracer, trace, get_tracer, reset_tracer, traced

# Incremental computation
from .incremental import (
    DependencyTracker,
    IncrementalCache,
    IncrementalDataAccumulator,
    CacheScope,
    CacheResult,
    compute_hash,
    memoize,
)

# Export
from .exporter import (
    CSVExporter,
    JSONExporter,
    PySRDataPreparer,
    export_for_pysr,
    export_to_csv,
    export_to_csv_with_formulas,
    export_to_json,
)

# PySR symbolic regression
from .pysr_runner import (
    FormulaResult,
    PySRRunner,
    export_formulas_summary,
)

# Hardware profiling
from .hardware import (
    GPUInfo,
    HardwareProfiler,
    GPU_SPECS,
    detect_gpus,
    get_gpu_specs,
    get_peak_performance,
    get_memory_bandwidth,
    precision_to_bytes,
    sample_gpu_metrics,
)

# Performance analysis
from .profiling import (
    GEMMAnalysis,
    RooflineAnalysis,
    PerformanceBottleneck,
    BottleneckType,
    PerformanceProfiler,
    roofline_analysis,
    analyze_gemm_from_code,
    analyze_attention,
    detect_bottlenecks,
    calculate_throughput,
    calculate_flops_throughput,
    calculate_gemm_flops,
    calculate_gemm_bytes,
)

# Graph compilation
from .compilation import (
    GraphBackend,
    GraphCompilation,
    GraphOptimization,
    CompilationAnalyzer,
    detect_graph_backend,
    find_compilations,
    find_graph_breaks,
    detect_optimizations,
    analyze_compile_mode,
    analyze_graph_code,
)

# Stack frame capture
from .stack import (
    FrameLocation,
    StackFrame,
    ThreadState,
    StackSnapshot,
    StackTracer,
    capture_thread_stack,
    capture_all_stacks,
    capture_with_pystack,
)

# Dataflow tracking
from .dataflow import (
    ValueNode,
    CallNode,
    ArgumentNode,
    DataflowEdge,
    NodeType,
    ExecutionContext,
    DataflowTracker,
    TrackedExecution,
    compute_value_hash,
    compute_call_hash,
    visualize_dataflow,
)

# Multi-dimensional correlation
from .correlation import (
    BottleneckSource,
    CorrelationPoint,
    BottleneckAnalysis,
    ExecutionProfile,
    CorrelationEngine,
    UnifiedTracer,
)

# Expert perspectives
from .perspectives import (
    ExpertPerspective,
    PerspectiveInsight,
    PerspectiveRegistry,
    BlameTarget,
    SeverityLevel,
    FrameworkExpert,
    FrameworkInfo,
    FrameworkBlame,
    FRAMEWORK_PATTERNS,
    RefactorExpert,
    CodeLocation,
    CommitBlame,
    RefactorSuggestion,
)

__all__ = [
    # Core tracing
    "Tracer",
    "trace",
    "traced",
    "get_tracer",
    "reset_tracer",
    "TraceCollector",
    "TraceRow",
    "TraceContext",
    # Incremental computation
    "DependencyTracker",
    "IncrementalCache",
    "IncrementalDataAccumulator",
    "CacheScope",
    "CacheResult",
    "compute_hash",
    "memoize",
    # Export
    "CSVExporter",
    "JSONExporter",
    "PySRDataPreparer",
    "export_for_pysr",
    "export_to_csv",
    "export_to_csv_with_formulas",
    "export_to_json",
    # PySR symbolic regression
    "FormulaResult",
    "PySRRunner",
    "export_formulas_summary",
    # Hardware profiling
    "GPUInfo",
    "HardwareProfiler",
    "GPU_SPECS",
    "detect_gpus",
    "get_gpu_specs",
    "get_peak_performance",
    "get_memory_bandwidth",
    "precision_to_bytes",
    "sample_gpu_metrics",
    # Performance analysis
    "GEMMAnalysis",
    "RooflineAnalysis",
    "PerformanceBottleneck",
    "BottleneckType",
    "PerformanceProfiler",
    "roofline_analysis",
    "analyze_gemm_from_code",
    "analyze_attention",
    "detect_bottlenecks",
    "calculate_throughput",
    "calculate_flops_throughput",
    "calculate_gemm_flops",
    "calculate_gemm_bytes",
    # Graph compilation
    "GraphBackend",
    "GraphCompilation",
    "GraphOptimization",
    "CompilationAnalyzer",
    "detect_graph_backend",
    "find_compilations",
    "find_graph_breaks",
    "detect_optimizations",
    "analyze_compile_mode",
    "analyze_graph_code",
    # Stack frame capture
    "FrameLocation",
    "StackFrame",
    "ThreadState",
    "StackSnapshot",
    "StackTracer",
    "capture_thread_stack",
    "capture_all_stacks",
    "capture_with_pystack",
    # Dataflow tracking
    "ValueNode",
    "CallNode",
    "ArgumentNode",
    "DataflowEdge",
    "NodeType",
    "ExecutionContext",
    "DataflowTracker",
    "TrackedExecution",
    "compute_value_hash",
    "compute_call_hash",
    "visualize_dataflow",
    # Multi-dimensional correlation
    "BottleneckSource",
    "CorrelationPoint",
    "BottleneckAnalysis",
    "ExecutionProfile",
    "CorrelationEngine",
    "UnifiedTracer",
    # Expert perspectives
    "ExpertPerspective",
    "PerspectiveInsight",
    "PerspectiveRegistry",
    "BlameTarget",
    "SeverityLevel",
    "FrameworkExpert",
    "FrameworkInfo",
    "FrameworkBlame",
    "FRAMEWORK_PATTERNS",
    "RefactorExpert",
    "CodeLocation",
    "CommitBlame",
    "RefactorSuggestion",
]

__version__ = "0.2.0"
