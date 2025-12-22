"""
Correlation Layer: Unified multi-dimensional execution analysis.

Brings together all trace dimensions:
- Stack frames (call stack state)
- Dataflow (computation DAG)
- Hardware metrics (GPU, memory)
- Performance metrics (timing, bottlenecks)
- Graph compilation (torch.compile, JAX)

Enables:
- Cross-dimensional correlation to identify bottleneck sources
- Holistic view of execution from multiple perspectives
- Root cause analysis by tracing through dimensions
"""

import hashlib
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Iterator

from .stack import (
    StackSnapshot,
    StackTracer,
    ThreadState,
    StackFrame,
    capture_all_stacks,
)
from .dataflow import (
    DataflowTracker,
    CallNode,
    ValueNode,
    DataflowEdge,
    compute_value_hash,
)

# Lazy import for perspectives to avoid circular imports
def _get_perspective_registry():
    from .perspectives import PerspectiveRegistry, FrameworkExpert, RefactorExpert
    return PerspectiveRegistry, FrameworkExpert, RefactorExpert


class BottleneckSource(Enum):
    """Source layer of a bottleneck."""
    PYTHON_GIL = "python_gil"
    PYTHON_GC = "python_gc"
    PYTHON_STACK = "python_stack"
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    GPU_THERMAL = "gpu_thermal"
    GPU_POWER = "gpu_power"
    GPU_SYNC = "gpu_sync"
    GRAPH_BREAK = "graph_break"
    GRAPH_RECOMPILE = "graph_recompile"
    IO_WAIT = "io_wait"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class CorrelationPoint:
    """
    A point correlating multiple trace dimensions at a moment in time.

    Represents a snapshot of execution state across all dimensions.
    """
    timestamp: float
    correlation_id: str

    # Stack dimension
    stack_snapshot: Optional[StackSnapshot] = None
    active_function: Optional[str] = None
    call_depth: int = 0
    holds_gil: bool = False
    is_gc_collecting: bool = False

    # Dataflow dimension
    call_hash: Optional[str] = None
    func_name: Optional[str] = None
    dataflow_depth: int = 0
    arg_count: int = 0

    # Hardware dimension
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_temperature: float = 0.0
    gpu_power_watts: float = 0.0

    # Performance dimension
    duration_ms: float = 0.0
    arithmetic_intensity: float = 0.0
    is_compute_bound: bool = False
    is_memory_bound: bool = False

    # Graph dimension
    is_compiled: bool = False
    has_graph_break: bool = False
    compile_mode: Optional[str] = None

    # Analysis results
    bottleneck_source: Optional[BottleneckSource] = None
    bottleneck_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            # Stack
            "active_function": self.active_function,
            "call_depth": self.call_depth,
            "holds_gil": self.holds_gil,
            "is_gc_collecting": self.is_gc_collecting,
            # Dataflow
            "call_hash": self.call_hash,
            "func_name": self.func_name,
            "dataflow_depth": self.dataflow_depth,
            "arg_count": self.arg_count,
            # Hardware
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "gpu_temperature": self.gpu_temperature,
            "gpu_power_watts": self.gpu_power_watts,
            # Performance
            "duration_ms": self.duration_ms,
            "arithmetic_intensity": self.arithmetic_intensity,
            "is_compute_bound": self.is_compute_bound,
            "is_memory_bound": self.is_memory_bound,
            # Graph
            "is_compiled": self.is_compiled,
            "has_graph_break": self.has_graph_break,
            "compile_mode": self.compile_mode,
            # Analysis
            "bottleneck_source": self.bottleneck_source.value if self.bottleneck_source else None,
            "bottleneck_confidence": self.bottleneck_confidence,
        }

    def to_metrics(self) -> Dict[str, float]:
        """Convert to numeric metrics for PySR."""
        return {
            "corr_call_depth": float(self.call_depth),
            "corr_dataflow_depth": float(self.dataflow_depth),
            "corr_arg_count": float(self.arg_count),
            "corr_gpu_util": self.gpu_utilization,
            "corr_gpu_mem_gb": self.gpu_memory_used_gb,
            "corr_gpu_temp": self.gpu_temperature,
            "corr_gpu_power": self.gpu_power_watts,
            "corr_duration_ms": self.duration_ms,
            "corr_arith_intensity": self.arithmetic_intensity,
            "corr_holds_gil": 1.0 if self.holds_gil else 0.0,
            "corr_is_gc": 1.0 if self.is_gc_collecting else 0.0,
            "corr_compute_bound": 1.0 if self.is_compute_bound else 0.0,
            "corr_memory_bound": 1.0 if self.is_memory_bound else 0.0,
            "corr_compiled": 1.0 if self.is_compiled else 0.0,
            "corr_graph_break": 1.0 if self.has_graph_break else 0.0,
        }


@dataclass
class BottleneckAnalysis:
    """Analysis of a bottleneck across dimensions."""
    source: BottleneckSource
    confidence: float
    evidence: List[str]
    affected_functions: List[str]
    correlation_points: List[str]  # correlation_ids
    suggested_fix: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "affected_functions": self.affected_functions,
            "correlation_points": self.correlation_points,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class ExecutionProfile:
    """Complete execution profile across all dimensions."""
    start_time: float
    end_time: float
    correlation_points: List[CorrelationPoint]
    bottlenecks: List[BottleneckAnalysis]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": (self.end_time - self.start_time) * 1000,
            "correlation_count": len(self.correlation_points),
            "bottleneck_count": len(self.bottlenecks),
            "summary": self.summary,
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
        }


class CorrelationEngine:
    """
    Engine for correlating trace data across multiple dimensions.

    Usage:
        engine = CorrelationEngine()

        # Add tracers
        engine.add_stack_tracer(stack_tracer)
        engine.add_dataflow_tracker(dataflow_tracker)

        # Start correlation
        engine.start()

        # ... execute code ...

        # Get analysis
        profile = engine.stop()
        bottlenecks = profile.bottlenecks

        # Get perspective insights
        insights = engine.get_perspective_insights()
    """

    def __init__(self, enable_perspectives: bool = True):
        self.stack_tracer: Optional[StackTracer] = None
        self.dataflow_tracker: Optional[DataflowTracker] = None

        self.correlation_points: List[CorrelationPoint] = []
        self._running = False
        self._sample_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Hardware metrics cache
        self._gpu_metrics: Dict[str, float] = {}

        # Graph compilation cache
        self._compilation_info: Dict[str, Any] = {}

        # Expert perspectives
        self._perspective_registry = None
        self._enable_perspectives = enable_perspectives
        if enable_perspectives:
            self._init_perspectives()

    def _init_perspectives(self) -> None:
        """Initialize default expert perspectives."""
        try:
            PerspectiveRegistry, FrameworkExpert, RefactorExpert = _get_perspective_registry()
            self._perspective_registry = PerspectiveRegistry()
            self._perspective_registry.register(FrameworkExpert())
            self._perspective_registry.register(RefactorExpert())
        except ImportError:
            self._perspective_registry = None

    def add_perspective(self, perspective) -> None:
        """Add a custom expert perspective."""
        if self._perspective_registry is None:
            PerspectiveRegistry, _, _ = _get_perspective_registry()
            self._perspective_registry = PerspectiveRegistry()
        self._perspective_registry.register(perspective)

    def remove_perspective(self, name: str) -> None:
        """Remove a perspective by name."""
        if self._perspective_registry:
            self._perspective_registry.unregister(name)

    def list_perspectives(self) -> List[str]:
        """List registered perspective names."""
        if self._perspective_registry:
            return self._perspective_registry.list_perspectives()
        return []

    def add_stack_tracer(self, tracer: StackTracer) -> None:
        """Add stack tracer for stack dimension."""
        self.stack_tracer = tracer

    def add_dataflow_tracker(self, tracker: DataflowTracker) -> None:
        """Add dataflow tracker for computation dimension."""
        self.dataflow_tracker = tracker

    def set_gpu_metrics(self, metrics: Dict[str, float]) -> None:
        """Update GPU metrics for hardware dimension."""
        with self._lock:
            self._gpu_metrics = metrics

    def set_compilation_info(self, info: Dict[str, Any]) -> None:
        """Update graph compilation info."""
        with self._lock:
            self._compilation_info = info

    def correlate_now(self) -> CorrelationPoint:
        """Create a correlation point from current state."""
        with self._lock:
            timestamp = time.time()
            corr_id = hashlib.sha256(
                f"{timestamp}:{id(self)}".encode()
            ).hexdigest()[:12]

            point = CorrelationPoint(
                timestamp=timestamp,
                correlation_id=corr_id,
            )

            # Stack dimension
            if self.stack_tracer:
                try:
                    snapshot = self.stack_tracer.capture()
                    point.stack_snapshot = snapshot

                    current = snapshot.get_current_thread()
                    if current and current.frames:
                        point.active_function = current.frames[0].location.function
                        point.call_depth = len(current.frames)
                        point.holds_gil = current.holds_gil
                        point.is_gc_collecting = current.is_gc_collecting
                except Exception:
                    pass

            # Dataflow dimension
            if self.dataflow_tracker and self.dataflow_tracker._call_stack:
                try:
                    current_call = self.dataflow_tracker._call_stack[-1]
                    if current_call in self.dataflow_tracker.calls:
                        call = self.dataflow_tracker.calls[current_call]
                        point.call_hash = current_call
                        point.func_name = call.func_name
                        point.dataflow_depth = self.dataflow_tracker.get_call_depth(current_call)
                        point.arg_count = len(call.arguments)
                except Exception:
                    pass

            # Hardware dimension
            if self._gpu_metrics:
                point.gpu_utilization = self._gpu_metrics.get("gpu_utilization_pct", 0)
                point.gpu_memory_used_gb = self._gpu_metrics.get("gpu_memory_used_gb", 0)
                point.gpu_temperature = self._gpu_metrics.get("gpu_temperature_c", 0)
                point.gpu_power_watts = self._gpu_metrics.get("gpu_power_watts", 0)

            # Graph dimension
            if self._compilation_info:
                point.is_compiled = self._compilation_info.get("is_compiled", False)
                point.has_graph_break = self._compilation_info.get("has_graph_break", False)
                point.compile_mode = self._compilation_info.get("mode")

            # Analyze bottleneck
            self._analyze_bottleneck(point)

            self.correlation_points.append(point)
            return point

    def _analyze_bottleneck(self, point: CorrelationPoint) -> None:
        """Analyze potential bottleneck source for a correlation point."""
        evidence = []
        source = BottleneckSource.UNKNOWN
        confidence = 0.0

        # Python GIL contention
        if point.holds_gil and point.call_depth > 10:
            evidence.append("GIL held with deep call stack")
            source = BottleneckSource.PYTHON_GIL
            confidence = 0.7

        # Python GC
        if point.is_gc_collecting:
            evidence.append("GC collection in progress")
            source = BottleneckSource.PYTHON_GC
            confidence = 0.9

        # GPU thermal throttling
        if point.gpu_temperature > 80:
            evidence.append(f"GPU temperature {point.gpu_temperature}C > 80C")
            source = BottleneckSource.GPU_THERMAL
            confidence = 0.8

        # GPU power throttling
        if point.gpu_power_watts > 0 and point.gpu_utilization < 50:
            if point.gpu_power_watts > 250:  # Typical power limit
                evidence.append("High power draw with low utilization")
                source = BottleneckSource.GPU_POWER
                confidence = 0.6

        # Graph break
        if point.has_graph_break:
            evidence.append("Graph break detected in compiled code")
            source = BottleneckSource.GRAPH_BREAK
            confidence = 0.85

        # Compute vs memory bound
        if point.arithmetic_intensity > 0:
            if point.is_compute_bound:
                source = BottleneckSource.COMPUTE_BOUND
                confidence = 0.75
            elif point.is_memory_bound:
                source = BottleneckSource.MEMORY_BOUND
                confidence = 0.75

        if evidence:
            point.bottleneck_source = source
            point.bottleneck_confidence = confidence

    def start(self, sample_interval_ms: int = 10) -> None:
        """Start correlation sampling."""
        if self._running:
            return

        self._running = True
        self.start_time = time.time()
        self.correlation_points.clear()

        def sample_loop():
            while self._running:
                try:
                    self.correlate_now()
                except Exception:
                    pass
                time.sleep(sample_interval_ms / 1000.0)

        self._sample_thread = threading.Thread(target=sample_loop, daemon=True)
        self._sample_thread.start()

    def stop(self) -> ExecutionProfile:
        """Stop correlation and return execution profile."""
        self._running = False
        self.end_time = time.time()

        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)
            self._sample_thread = None

        # Analyze bottlenecks across all points
        bottlenecks = self._analyze_all_bottlenecks()

        # Build summary
        summary = self._build_summary()

        return ExecutionProfile(
            start_time=self.start_time or time.time(),
            end_time=self.end_time,
            correlation_points=self.correlation_points,
            bottlenecks=bottlenecks,
            summary=summary,
        )

    def _analyze_all_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze bottlenecks across all correlation points."""
        bottleneck_groups: Dict[BottleneckSource, List[CorrelationPoint]] = defaultdict(list)

        for point in self.correlation_points:
            if point.bottleneck_source:
                bottleneck_groups[point.bottleneck_source].append(point)

        analyses = []
        for source, points in bottleneck_groups.items():
            if len(points) < 2:  # Require multiple observations
                continue

            avg_confidence = sum(p.bottleneck_confidence for p in points) / len(points)

            affected_funcs = list(set(
                p.active_function for p in points
                if p.active_function
            ))

            evidence = self._gather_evidence(source, points)
            suggested_fix = self._suggest_fix(source)

            analyses.append(BottleneckAnalysis(
                source=source,
                confidence=avg_confidence,
                evidence=evidence,
                affected_functions=affected_funcs[:10],
                correlation_points=[p.correlation_id for p in points[:100]],
                suggested_fix=suggested_fix,
            ))

        return sorted(analyses, key=lambda a: a.confidence, reverse=True)

    def _gather_evidence(
        self,
        source: BottleneckSource,
        points: List[CorrelationPoint],
    ) -> List[str]:
        """Gather evidence for a bottleneck type."""
        evidence = []

        if source == BottleneckSource.PYTHON_GIL:
            avg_depth = sum(p.call_depth for p in points) / len(points)
            evidence.append(f"Average call depth: {avg_depth:.1f}")
            evidence.append(f"Observed in {len(points)} samples")

        elif source == BottleneckSource.PYTHON_GC:
            evidence.append(f"GC active in {len(points)} samples")

        elif source == BottleneckSource.GPU_THERMAL:
            avg_temp = sum(p.gpu_temperature for p in points) / len(points)
            max_temp = max(p.gpu_temperature for p in points)
            evidence.append(f"Average GPU temp: {avg_temp:.1f}C")
            evidence.append(f"Max GPU temp: {max_temp:.1f}C")

        elif source == BottleneckSource.GRAPH_BREAK:
            evidence.append(f"Graph breaks in {len(points)} samples")
            evidence.append("Prevents kernel fusion optimization")

        elif source == BottleneckSource.COMPUTE_BOUND:
            avg_util = sum(p.gpu_utilization for p in points) / len(points)
            evidence.append(f"Average GPU utilization: {avg_util:.1f}%")
            evidence.append("High compute vs memory ratio")

        elif source == BottleneckSource.MEMORY_BOUND:
            avg_mem = sum(p.gpu_memory_used_gb for p in points) / len(points)
            evidence.append(f"Average GPU memory used: {avg_mem:.2f} GB")
            evidence.append("Low compute vs memory ratio")

        return evidence

    def _suggest_fix(self, source: BottleneckSource) -> str:
        """Suggest fix for bottleneck type."""
        suggestions = {
            BottleneckSource.PYTHON_GIL: "Consider using multiprocessing or releasing GIL in C extensions",
            BottleneckSource.PYTHON_GC: "Reduce object allocations or disable GC during critical sections",
            BottleneckSource.COMPUTE_BOUND: "Consider using lower precision (fp16/bf16) or Tensor Cores",
            BottleneckSource.MEMORY_BOUND: "Optimize memory access patterns, use memory-efficient algorithms",
            BottleneckSource.MEMORY_BANDWIDTH: "Batch operations, reduce memory transfers",
            BottleneckSource.GPU_THERMAL: "Reduce workload, improve cooling, or throttle",
            BottleneckSource.GPU_POWER: "Reduce power limit or optimize for efficiency",
            BottleneckSource.GPU_SYNC: "Use async operations, overlap compute and transfers",
            BottleneckSource.GRAPH_BREAK: "Use torch._dynamo.explain() to identify breaks, fix Python constructs",
            BottleneckSource.GRAPH_RECOMPILE: "Stabilize input shapes, use dynamic=True in torch.compile",
            BottleneckSource.IO_WAIT: "Use async I/O, prefetching, or caching",
            BottleneckSource.NETWORK: "Batch network calls, use compression, optimize serialization",
        }
        return suggestions.get(source, "Profile further to identify root cause")

    def _build_summary(self) -> Dict[str, Any]:
        """Build execution summary."""
        if not self.correlation_points:
            return {}

        # Aggregate metrics
        avg_depth = sum(p.call_depth for p in self.correlation_points) / len(self.correlation_points)
        avg_gpu_util = sum(p.gpu_utilization for p in self.correlation_points) / len(self.correlation_points)
        max_gpu_temp = max((p.gpu_temperature for p in self.correlation_points), default=0)

        # Count bottleneck types
        bottleneck_counts: Dict[str, int] = defaultdict(int)
        for p in self.correlation_points:
            if p.bottleneck_source:
                bottleneck_counts[p.bottleneck_source.value] += 1

        # Top functions
        func_counts: Dict[str, int] = defaultdict(int)
        for p in self.correlation_points:
            if p.active_function:
                func_counts[p.active_function] += 1

        top_functions = sorted(
            func_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "sample_count": len(self.correlation_points),
            "duration_ms": (
                (self.end_time - self.start_time) * 1000
                if self.end_time and self.start_time else 0
            ),
            "avg_call_depth": avg_depth,
            "avg_gpu_utilization": avg_gpu_util,
            "max_gpu_temperature": max_gpu_temp,
            "bottleneck_distribution": dict(bottleneck_counts),
            "top_functions": dict(top_functions),
            "gil_held_pct": (
                sum(1 for p in self.correlation_points if p.holds_gil)
                / len(self.correlation_points) * 100
            ),
            "gc_active_pct": (
                sum(1 for p in self.correlation_points if p.is_gc_collecting)
                / len(self.correlation_points) * 100
            ),
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics for PySR."""
        if not self.correlation_points:
            return {}

        metrics = {}

        # Aggregate from all points
        for key in self.correlation_points[0].to_metrics().keys():
            values = [p.to_metrics().get(key, 0) for p in self.correlation_points]
            metrics[f"{key}_mean"] = sum(values) / len(values)
            metrics[f"{key}_max"] = max(values)

        metrics["corr_sample_count"] = float(len(self.correlation_points))

        return metrics

    def clear(self) -> None:
        """Clear all correlation data."""
        with self._lock:
            self.correlation_points.clear()
            self._gpu_metrics.clear()
            self._compilation_info.clear()
            self.start_time = None
            self.end_time = None

    # ============== Perspective Analysis Methods ==============

    def get_perspective_context(self) -> Dict[str, Any]:
        """Get context dict for perspective analysis."""
        return {
            "stack_tracer": self.stack_tracer,
            "dataflow_tracker": self.dataflow_tracker,
            "gpu_metrics": self._gpu_metrics,
            "compilation_info": self._compilation_info,
        }

    def get_perspective_insights(self) -> Dict[str, List[Any]]:
        """
        Get insights from all registered perspectives.

        Returns:
            Dict mapping perspective name to list of PerspectiveInsight
        """
        if not self._perspective_registry:
            return {}

        context = self.get_perspective_context()
        return self._perspective_registry.analyze_all(self.correlation_points, context)

    def get_all_blame_targets(self) -> List[Any]:
        """
        Get blame targets from all perspectives.

        Returns:
            List of BlameTarget from all perspectives
        """
        if not self._perspective_registry:
            return []

        context = self.get_perspective_context()
        return self._perspective_registry.get_all_blame_targets(self.correlation_points, context)

    def get_framework_report(self) -> Dict[str, Any]:
        """
        Get framework-specific analysis report.

        Returns detailed breakdown of which frameworks are consuming time.
        """
        if not self._perspective_registry:
            return {}

        framework_expert = self._perspective_registry.get("framework_expert")
        if framework_expert and hasattr(framework_expert, "get_framework_report"):
            context = self.get_perspective_context()
            return framework_expert.get_framework_report(self.correlation_points, context)
        return {}

    def get_hotspot_report(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Get code hotspot analysis report.

        Returns specific file:line locations consuming the most time,
        with git blame information when available.
        """
        if not self._perspective_registry:
            return {}

        refactor_expert = self._perspective_registry.get("refactor_expert")
        if refactor_expert and hasattr(refactor_expert, "get_hotspot_report"):
            context = self.get_perspective_context()
            return refactor_expert.get_hotspot_report(self.correlation_points, context, top_n)
        return {}

    def get_perspective_summary(self) -> Dict[str, Any]:
        """Get combined summary from all perspectives."""
        if not self._perspective_registry:
            return {"perspectives_enabled": False}

        context = self.get_perspective_context()
        summary = self._perspective_registry.get_summary(self.correlation_points, context)
        summary["perspectives_enabled"] = True
        return summary


class UnifiedTracer:
    """
    Unified tracer that coordinates all trace dimensions.

    Provides a single interface for comprehensive execution tracing.

    Usage:
        tracer = UnifiedTracer()

        with tracer.trace("my_operation") as ctx:
            # All dimensions are tracked
            result = my_function()
            ctx.add_metric("items", len(result))

        # Get multi-dimensional analysis
        profile = tracer.get_profile()
        bottlenecks = tracer.analyze_bottlenecks()
    """

    def __init__(
        self,
        enable_stack: bool = True,
        enable_dataflow: bool = True,
        enable_hardware: bool = True,
        sample_interval_ms: int = 10,
    ):
        self.stack_tracer = StackTracer() if enable_stack else None
        self.dataflow_tracker = DataflowTracker() if enable_dataflow else None
        self.correlation_engine = CorrelationEngine()

        if self.stack_tracer:
            self.correlation_engine.add_stack_tracer(self.stack_tracer)
        if self.dataflow_tracker:
            self.correlation_engine.add_dataflow_tracker(self.dataflow_tracker)

        self.sample_interval_ms = sample_interval_ms
        self.enable_hardware = enable_hardware
        self._hw_sample_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start all tracers."""
        self._running = True
        self.correlation_engine.start(self.sample_interval_ms)

        if self.enable_hardware:
            self._start_hardware_sampling()

    def stop(self) -> ExecutionProfile:
        """Stop all tracers and return profile."""
        self._running = False

        if self._hw_sample_thread:
            self._hw_sample_thread.join(timeout=1.0)

        return self.correlation_engine.stop()

    def _start_hardware_sampling(self) -> None:
        """Start hardware metrics sampling."""
        def sample_hardware():
            try:
                from .hardware import sample_gpu_metrics
            except ImportError:
                return

            while self._running:
                try:
                    metrics_list = sample_gpu_metrics()
                    if metrics_list:
                        # Use first GPU's metrics
                        self.correlation_engine.set_gpu_metrics(metrics_list[0])
                except Exception:
                    pass
                time.sleep(self.sample_interval_ms / 1000.0)

        self._hw_sample_thread = threading.Thread(target=sample_hardware, daemon=True)
        self._hw_sample_thread.start()

    def track(self, func):
        """Decorator to track function with all dimensions."""
        if self.dataflow_tracker:
            func = self.dataflow_tracker.track(func)
        return func

    def get_profile(self) -> Optional[ExecutionProfile]:
        """Get the current or last execution profile."""
        if self.correlation_engine.end_time:
            return ExecutionProfile(
                start_time=self.correlation_engine.start_time or time.time(),
                end_time=self.correlation_engine.end_time,
                correlation_points=self.correlation_engine.correlation_points,
                bottlenecks=self.correlation_engine._analyze_all_bottlenecks(),
                summary=self.correlation_engine._build_summary(),
            )
        return None

    def analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze bottlenecks across all dimensions."""
        return self.correlation_engine._analyze_all_bottlenecks()

    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics from all tracers for PySR."""
        metrics = {}

        if self.stack_tracer:
            metrics.update(self.stack_tracer.get_metrics())

        if self.dataflow_tracker:
            metrics.update(self.dataflow_tracker.get_metrics())

        metrics.update(self.correlation_engine.get_metrics())

        return metrics

    def clear(self) -> None:
        """Clear all tracers."""
        if self.stack_tracer:
            self.stack_tracer.clear()
        if self.dataflow_tracker:
            self.dataflow_tracker.clear()
        self.correlation_engine.clear()

    # ============== Perspective Analysis Methods ==============

    def get_perspective_insights(self) -> Dict[str, List[Any]]:
        """Get insights from all registered perspectives."""
        return self.correlation_engine.get_perspective_insights()

    def get_all_blame_targets(self) -> List[Any]:
        """Get blame targets from all perspectives."""
        return self.correlation_engine.get_all_blame_targets()

    def get_framework_report(self) -> Dict[str, Any]:
        """Get framework-specific analysis report."""
        return self.correlation_engine.get_framework_report()

    def get_hotspot_report(self, top_n: int = 20) -> Dict[str, Any]:
        """Get code hotspot analysis report with git blame."""
        return self.correlation_engine.get_hotspot_report(top_n)

    def get_perspective_summary(self) -> Dict[str, Any]:
        """Get combined summary from all perspectives."""
        return self.correlation_engine.get_perspective_summary()

    def add_perspective(self, perspective) -> None:
        """Add a custom expert perspective."""
        self.correlation_engine.add_perspective(perspective)

    def list_perspectives(self) -> List[str]:
        """List registered perspective names."""
        return self.correlation_engine.list_perspectives()
