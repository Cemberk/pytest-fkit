"""
Framework Expert: Identifies which frameworks are causing bottlenecks.

Analyzes execution traces to determine which ML/DL frameworks are responsible
for performance issues, providing blame attribution and optimization suggestions.

Supports:
- PyTorch (torch, torchvision, torchaudio)
- TensorFlow (tensorflow, tf, keras)
- JAX (jax, flax, optax)
- ONNX Runtime
- Transformers (huggingface)
- Other common ML libraries
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .base import (
    ExpertPerspective,
    PerspectiveInsight,
    BlameTarget,
    SeverityLevel,
)

if TYPE_CHECKING:
    from ..correlation import CorrelationPoint
    from ..dataflow import DataflowTracker, CallNode


class FrameworkCategory(Enum):
    """Categories of ML frameworks."""
    DEEP_LEARNING = "deep_learning"
    DATA_PROCESSING = "data_processing"
    SCIENTIFIC = "scientific"
    OPTIMIZATION = "optimization"
    SERVING = "serving"
    UTILITY = "utility"
    UNKNOWN = "unknown"


# Framework detection patterns with metadata
FRAMEWORK_PATTERNS = {
    # Deep Learning Frameworks
    "pytorch": {
        "patterns": [r"^torch\b", r"^torchvision\b", r"^torchaudio\b", r"^pytorch_"],
        "category": FrameworkCategory.DEEP_LEARNING,
        "known_bottlenecks": {
            "torch.cuda.synchronize": "GPU synchronization point",
            "torch.autograd": "Backward pass computation",
            "torch.nn.functional": "Operator execution",
            "torch.distributed": "Distributed communication",
            "torch._dynamo": "Graph compilation overhead",
            "torch._inductor": "Kernel compilation",
        },
        "optimization_hints": [
            "Use torch.compile() for graph optimization",
            "Enable cudnn.benchmark for consistent input sizes",
            "Use mixed precision (torch.cuda.amp)",
            "Batch operations to reduce kernel launch overhead",
        ],
    },
    "tensorflow": {
        "patterns": [r"^tensorflow\b", r"^tf\b", r"^keras\b"],
        "category": FrameworkCategory.DEEP_LEARNING,
        "known_bottlenecks": {
            "tf.function": "Graph tracing overhead",
            "tf.data": "Data pipeline processing",
            "tf.distribute": "Distribution strategy overhead",
        },
        "optimization_hints": [
            "Use tf.function for graph compilation",
            "Prefetch data with tf.data.Dataset.prefetch",
            "Enable XLA compilation",
        ],
    },
    "jax": {
        "patterns": [r"^jax\b", r"^flax\b", r"^optax\b", r"^equinox\b"],
        "category": FrameworkCategory.DEEP_LEARNING,
        "known_bottlenecks": {
            "jax.jit": "JIT compilation overhead",
            "jax.pmap": "Multi-device parallelism setup",
            "jax.grad": "Gradient computation",
        },
        "optimization_hints": [
            "Use jax.jit for compilation",
            "Use jax.lax.scan instead of Python loops",
            "Avoid Python control flow in jitted functions",
        ],
    },
    "transformers": {
        "patterns": [r"^transformers\b", r"^huggingface_hub\b", r"^tokenizers\b"],
        "category": FrameworkCategory.DEEP_LEARNING,
        "known_bottlenecks": {
            "transformers.modeling": "Model forward pass",
            "transformers.tokenization": "Tokenization overhead",
            "transformers.generation": "Autoregressive generation",
        },
        "optimization_hints": [
            "Use Flash Attention 2 for long sequences",
            "Enable BetterTransformer for inference",
            "Use torch.compile with the model",
            "Consider quantization (bitsandbytes, GPTQ)",
        ],
    },
    "numpy": {
        "patterns": [r"^numpy\b", r"^np\b"],
        "category": FrameworkCategory.SCIENTIFIC,
        "known_bottlenecks": {
            "numpy.linalg": "Linear algebra operations",
            "numpy.fft": "FFT computations",
        },
        "optimization_hints": [
            "Use vectorized operations instead of loops",
            "Consider using GPU arrays (CuPy)",
        ],
    },
    "pandas": {
        "patterns": [r"^pandas\b", r"^pd\b"],
        "category": FrameworkCategory.DATA_PROCESSING,
        "known_bottlenecks": {
            "pandas.core.frame": "DataFrame operations",
            "pandas.io": "Data I/O",
        },
        "optimization_hints": [
            "Use vectorized operations",
            "Consider Polars for better performance",
        ],
    },
    "scipy": {
        "patterns": [r"^scipy\b"],
        "category": FrameworkCategory.SCIENTIFIC,
        "known_bottlenecks": {
            "scipy.optimize": "Optimization routines",
            "scipy.sparse": "Sparse matrix operations",
        },
        "optimization_hints": [],
    },
    "sklearn": {
        "patterns": [r"^sklearn\b", r"^scikit_learn\b"],
        "category": FrameworkCategory.DEEP_LEARNING,
        "known_bottlenecks": {
            "sklearn.ensemble": "Ensemble methods",
            "sklearn.cluster": "Clustering algorithms",
        },
        "optimization_hints": [
            "Use n_jobs=-1 for parallelization",
        ],
    },
    "onnxruntime": {
        "patterns": [r"^onnxruntime\b", r"^ort\b"],
        "category": FrameworkCategory.SERVING,
        "known_bottlenecks": {
            "onnxruntime.InferenceSession": "Session initialization",
        },
        "optimization_hints": [
            "Use CUDA execution provider",
            "Enable graph optimizations",
        ],
    },
    "triton": {
        "patterns": [r"^triton\b"],
        "category": FrameworkCategory.OPTIMIZATION,
        "known_bottlenecks": {
            "triton.jit": "Kernel compilation",
        },
        "optimization_hints": [
            "Cache compiled kernels",
            "Tune block sizes for your GPU",
        ],
    },
    "deepspeed": {
        "patterns": [r"^deepspeed\b"],
        "category": FrameworkCategory.OPTIMIZATION,
        "known_bottlenecks": {
            "deepspeed.runtime": "ZeRO optimizer overhead",
            "deepspeed.comm": "Communication overhead",
        },
        "optimization_hints": [
            "Tune ZeRO stage based on model size",
            "Use activation checkpointing",
        ],
    },
    "accelerate": {
        "patterns": [r"^accelerate\b"],
        "category": FrameworkCategory.UTILITY,
        "known_bottlenecks": {
            "accelerate.Accelerator": "Device synchronization",
        },
        "optimization_hints": [],
    },
    "vllm": {
        "patterns": [r"^vllm\b"],
        "category": FrameworkCategory.SERVING,
        "known_bottlenecks": {
            "vllm.engine": "Engine initialization",
            "vllm.model_executor": "Model execution",
        },
        "optimization_hints": [
            "Tune tensor_parallel_size",
            "Use PagedAttention effectively",
        ],
    },
    "sglang": {
        "patterns": [r"^sglang\b"],
        "category": FrameworkCategory.SERVING,
        "known_bottlenecks": {
            "sglang.srt": "Runtime initialization",
        },
        "optimization_hints": [],
    },
}


@dataclass
class FrameworkInfo:
    """Information about a detected framework."""
    name: str
    category: FrameworkCategory
    modules_seen: Set[str] = field(default_factory=set)
    functions_seen: Set[str] = field(default_factory=set)
    call_count: int = 0
    total_duration_ms: float = 0.0
    bottleneck_functions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "modules_seen": list(self.modules_seen),
            "functions_seen": list(self.functions_seen)[:20],  # Limit size
            "call_count": self.call_count,
            "total_duration_ms": self.total_duration_ms,
            "bottleneck_functions": dict(
                sorted(self.bottleneck_functions.items(), key=lambda x: -x[1])[:10]
            ),
        }


@dataclass
class FrameworkBlame:
    """Blame attribution to a framework."""
    framework: str
    category: FrameworkCategory
    confidence: float
    impact_pct: float  # Percentage of total time
    duration_ms: float
    bottleneck_function: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_blame_target(self) -> BlameTarget:
        return BlameTarget(
            target_type="framework",
            name=self.framework,
            confidence=self.confidence,
            module_path=self.bottleneck_function,
            evidence=self.evidence,
            impact_pct=self.impact_pct,
        )


class FrameworkExpert(ExpertPerspective):
    """
    Expert perspective for framework-level bottleneck analysis.

    Identifies which ML/DL frameworks are causing performance issues
    and provides optimization suggestions.
    """

    @property
    def name(self) -> str:
        return "framework_expert"

    @property
    def description(self) -> str:
        return "Identifies and blames frameworks for performance bottlenecks"

    def __init__(self, patterns: Optional[Dict] = None):
        """
        Initialize the framework expert.

        Args:
            patterns: Custom framework patterns (merged with defaults)
        """
        self.patterns = dict(FRAMEWORK_PATTERNS)
        if patterns:
            self.patterns.update(patterns)

        # Compile regex patterns
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for name, info in self.patterns.items():
            self._compiled_patterns[name] = [
                re.compile(p) for p in info["patterns"]
            ]

    def detect_framework(self, module_path: Optional[str]) -> Optional[str]:
        """Detect which framework a module belongs to."""
        if not module_path:
            return None

        for name, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.match(module_path):
                    return name
        return None

    def analyze(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[PerspectiveInsight]:
        """Analyze correlation points to identify framework bottlenecks."""
        context = context or {}

        # Collect framework usage from correlation points
        framework_stats = self._collect_framework_stats(correlation_points, context)

        if not framework_stats:
            return []

        # Calculate total time for impact percentages
        total_duration = sum(f.total_duration_ms for f in framework_stats.values())
        if total_duration == 0:
            total_duration = 1  # Avoid division by zero

        # Generate insights
        insights = []

        # Sort frameworks by duration
        sorted_frameworks = sorted(
            framework_stats.values(),
            key=lambda f: f.total_duration_ms,
            reverse=True,
        )

        for fw_info in sorted_frameworks:
            impact_pct = (fw_info.total_duration_ms / total_duration) * 100

            # Only report frameworks with significant impact
            if impact_pct < 1.0:
                continue

            severity = self._calculate_severity(impact_pct, fw_info)
            blame_targets = self._create_blame_targets(fw_info, impact_pct, total_duration)
            suggestions = self._get_suggestions(fw_info)

            insight = PerspectiveInsight(
                perspective_name=self.name,
                category="framework_bottleneck",
                severity=severity,
                summary=f"{fw_info.name} accounts for {impact_pct:.1f}% of execution time",
                description=self._generate_description(fw_info, impact_pct),
                blame_targets=blame_targets,
                suggestions=suggestions,
                metrics={
                    "framework_duration_ms": fw_info.total_duration_ms,
                    "framework_impact_pct": impact_pct,
                    "framework_call_count": float(fw_info.call_count),
                },
                tags={
                    "framework": fw_info.name,
                    "category": fw_info.category.value,
                },
            )
            insights.append(insight)

        # Add comparative insight if multiple frameworks
        if len(sorted_frameworks) > 1:
            insights.append(self._create_comparison_insight(sorted_frameworks, total_duration))

        return insights

    def _collect_framework_stats(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Dict[str, Any],
    ) -> Dict[str, FrameworkInfo]:
        """Collect statistics about framework usage."""
        stats: Dict[str, FrameworkInfo] = {}

        # Analyze from dataflow tracker if available
        dataflow_tracker = context.get("dataflow_tracker")
        if dataflow_tracker:
            self._analyze_dataflow(dataflow_tracker, stats)

        # Analyze from correlation points
        for point in correlation_points:
            # Check function name from stack
            if point.active_function:
                self._analyze_function(point.active_function, point, stats)

            # Check func_name from dataflow
            if point.func_name:
                self._analyze_function(point.func_name, point, stats)

            # Check stack snapshot if available
            if point.stack_snapshot:
                for thread in point.stack_snapshot.threads:
                    for frame in thread.frames[:10]:  # Top 10 frames
                        if frame.location.module:
                            fw = self.detect_framework(frame.location.module)
                            if fw:
                                self._update_stats(stats, fw, frame.location.module, None, 0)

        return stats

    def _analyze_dataflow(
        self,
        tracker: "DataflowTracker",
        stats: Dict[str, FrameworkInfo],
    ) -> None:
        """Analyze dataflow tracker for framework usage."""
        for call_hash, call in tracker.calls.items():
            if call.func_module:
                fw = self.detect_framework(call.func_module)
                if fw:
                    self._update_stats(
                        stats, fw, call.func_module, call.func_name, call.duration_ms
                    )

    def _analyze_function(
        self,
        func_name: str,
        point: "CorrelationPoint",
        stats: Dict[str, FrameworkInfo],
    ) -> None:
        """Analyze a function name for framework attribution."""
        # Try to extract module from function name
        if "." in func_name:
            parts = func_name.rsplit(".", 1)
            module = parts[0]
            fw = self.detect_framework(module)
            if fw:
                self._update_stats(stats, fw, module, func_name, point.duration_ms)

    def _update_stats(
        self,
        stats: Dict[str, FrameworkInfo],
        framework: str,
        module: str,
        func_name: Optional[str],
        duration_ms: float,
    ) -> None:
        """Update framework statistics."""
        if framework not in stats:
            stats[framework] = FrameworkInfo(
                name=framework,
                category=self.patterns[framework]["category"],
            )

        info = stats[framework]
        info.modules_seen.add(module)
        if func_name:
            info.functions_seen.add(func_name)
        info.call_count += 1
        info.total_duration_ms += duration_ms

        # Track bottleneck functions
        if func_name and duration_ms > 0:
            known = self.patterns[framework].get("known_bottlenecks", {})
            for pattern, desc in known.items():
                if pattern in (module or "") or pattern in (func_name or ""):
                    info.bottleneck_functions[func_name] = (
                        info.bottleneck_functions.get(func_name, 0) + duration_ms
                    )

    def _calculate_severity(
        self,
        impact_pct: float,
        fw_info: FrameworkInfo,
    ) -> SeverityLevel:
        """Calculate severity based on impact and known issues."""
        if impact_pct > 50:
            return SeverityLevel.CRITICAL
        elif impact_pct > 30:
            return SeverityLevel.HIGH
        elif impact_pct > 15:
            return SeverityLevel.MEDIUM
        elif impact_pct > 5:
            return SeverityLevel.LOW
        return SeverityLevel.INFO

    def _create_blame_targets(
        self,
        fw_info: FrameworkInfo,
        impact_pct: float,
        total_duration: float,
    ) -> List[BlameTarget]:
        """Create blame targets for a framework."""
        targets = []

        # Framework-level blame
        targets.append(BlameTarget(
            target_type="framework",
            name=fw_info.name,
            confidence=min(0.9, impact_pct / 100 + 0.3),
            impact_pct=impact_pct,
            occurrence_count=fw_info.call_count,
            evidence=[
                f"Called {fw_info.call_count} times",
                f"Total duration: {fw_info.total_duration_ms:.1f}ms",
                f"Modules: {', '.join(list(fw_info.modules_seen)[:5])}",
            ],
        ))

        # Add specific bottleneck functions
        for func, duration in sorted(
            fw_info.bottleneck_functions.items(),
            key=lambda x: -x[1],
        )[:5]:
            func_impact = (duration / total_duration) * 100
            targets.append(BlameTarget(
                target_type="function",
                name=func,
                confidence=0.8,
                function_name=func,
                impact_pct=func_impact,
                evidence=[
                    f"Known bottleneck in {fw_info.name}",
                    f"Duration: {duration:.1f}ms ({func_impact:.1f}%)",
                ],
            ))

        return targets

    def _get_suggestions(self, fw_info: FrameworkInfo) -> List[str]:
        """Get optimization suggestions for a framework."""
        suggestions = list(self.patterns[fw_info.name].get("optimization_hints", []))

        # Add bottleneck-specific suggestions
        known = self.patterns[fw_info.name].get("known_bottlenecks", {})
        for func in fw_info.bottleneck_functions:
            for pattern, desc in known.items():
                if pattern in func:
                    suggestions.append(f"Optimize {pattern}: {desc}")

        return suggestions[:5]  # Limit suggestions

    def _generate_description(
        self,
        fw_info: FrameworkInfo,
        impact_pct: float,
    ) -> str:
        """Generate detailed description for framework insight."""
        desc = [
            f"Framework: {fw_info.name} ({fw_info.category.value})",
            f"Impact: {impact_pct:.1f}% of total execution time",
            f"Calls: {fw_info.call_count}",
            f"Duration: {fw_info.total_duration_ms:.1f}ms",
        ]

        if fw_info.bottleneck_functions:
            top_bottleneck = max(fw_info.bottleneck_functions.items(), key=lambda x: x[1])
            desc.append(f"Top bottleneck: {top_bottleneck[0]} ({top_bottleneck[1]:.1f}ms)")

        return "\n".join(desc)

    def _create_comparison_insight(
        self,
        frameworks: List[FrameworkInfo],
        total_duration: float,
    ) -> PerspectiveInsight:
        """Create comparison insight across frameworks."""
        comparison_data = []
        blame_targets = []

        for fw in frameworks[:5]:  # Top 5
            impact = (fw.total_duration_ms / total_duration) * 100
            comparison_data.append(f"  {fw.name}: {impact:.1f}% ({fw.total_duration_ms:.1f}ms)")
            blame_targets.append(BlameTarget(
                target_type="framework",
                name=fw.name,
                confidence=0.7,
                impact_pct=impact,
            ))

        return PerspectiveInsight(
            perspective_name=self.name,
            category="framework_comparison",
            severity=SeverityLevel.INFO,
            summary=f"Framework time distribution across {len(frameworks)} frameworks",
            description="Time spent per framework:\n" + "\n".join(comparison_data),
            blame_targets=blame_targets,
            suggestions=[
                "Focus optimization on the top framework by time",
                "Consider framework alternatives for major bottlenecks",
            ],
            metrics={
                "framework_count": float(len(frameworks)),
                "top_framework_pct": (frameworks[0].total_duration_ms / total_duration) * 100,
            },
        )

    def get_framework_report(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive framework usage report."""
        stats = self._collect_framework_stats(correlation_points, context or {})
        total_duration = sum(f.total_duration_ms for f in stats.values())

        return {
            "frameworks_detected": len(stats),
            "total_tracked_duration_ms": total_duration,
            "frameworks": {
                name: {
                    **info.to_dict(),
                    "impact_pct": (info.total_duration_ms / total_duration * 100)
                    if total_duration > 0 else 0,
                }
                for name, info in sorted(
                    stats.items(),
                    key=lambda x: -x[1].total_duration_ms,
                )
            },
            "category_breakdown": self._get_category_breakdown(stats),
        }

    def _get_category_breakdown(
        self,
        stats: Dict[str, FrameworkInfo],
    ) -> Dict[str, Dict[str, float]]:
        """Get breakdown by framework category."""
        breakdown: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"duration_ms": 0, "call_count": 0}
        )

        for fw in stats.values():
            cat = fw.category.value
            breakdown[cat]["duration_ms"] += fw.total_duration_ms
            breakdown[cat]["call_count"] += fw.call_count

        return dict(breakdown)
