"""
Graph Compilation Tracing: Analyze torch.compile, JAX JIT, and other graph compilers.

Provides:
- Compilation backend detection (torch.compile, JAX jit, TF function)
- Graph break detection for torch.compile/dynamo
- Optimization pass detection
- Compile mode analysis with recommendations
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class GraphBackend(Enum):
    """Computational graph backend types."""
    TORCH_COMPILE = "torch.compile"
    TORCH_TRACE = "torch.jit.trace"
    TORCH_SCRIPT = "torch.jit.script"
    TF_FUNCTION = "tf.function"
    TF_XLA = "tf.xla"
    JAX_JIT = "jax.jit"
    JAX_PMAP = "jax.pmap"
    TRITON = "triton"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    UNKNOWN = "unknown"


@dataclass
class GraphCompilation:
    """Information about a graph compilation."""
    backend: GraphBackend
    line_number: int
    function_name: Optional[str] = None
    mode: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.value,
            "line": self.line_number,
            "function": self.function_name,
            "mode": self.mode,
            "options": self.options,
        }


@dataclass
class GraphOptimization:
    """Information about a graph optimization pass."""
    name: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description}


# Compile patterns
COMPILE_PATTERNS = {
    "torch_compile": re.compile(
        r"torch\.compile\s*\(\s*(\w+)(?:\s*,\s*mode\s*=\s*['\"]([^'\"]+)['\"])?"
    ),
    "torch_compile_decorator": re.compile(
        r"@torch\.compile(?:\s*\(\s*mode\s*=\s*['\"]([^'\"]+)['\"])?",
    ),
    "torch_trace": re.compile(r"torch\.jit\.trace\s*\(\s*(\w+)"),
    "torch_script": re.compile(r"torch\.jit\.script\s*\(\s*(\w+)"),
    "torch_script_decorator": re.compile(r"@torch\.jit\.script"),
    "tf_function": re.compile(r"@tf\.function"),
    "tf_xla": re.compile(r"tf\.xla\.|xla_compile"),
    "jax_jit": re.compile(r"@jax\.jit|jax\.jit\s*\("),
    "jax_pmap": re.compile(r"@jax\.pmap|jax\.pmap\s*\("),
    "triton_jit": re.compile(r"@triton\.jit"),
}

# Graph break patterns
DYNAMO_PATTERNS = {
    "graph_break": re.compile(r"graph.?break|GRAPH_BREAK|TorchDynamoGraphBreak"),
    "recompile": re.compile(r"recompile|RECOMPILATION|triggered.?recompile"),
    "fallback": re.compile(r"fallback|eager.?mode|graph.?break.?reason"),
}

# Optimization patterns
OPTIMIZATION_PATTERNS = {
    "fusion": re.compile(r"fused|fusion|Fuse", re.IGNORECASE),
    "constant_folding": re.compile(r"constant.?fold", re.IGNORECASE),
    "dead_code": re.compile(r"dead.?code|remove.?unused", re.IGNORECASE),
    "inlining": re.compile(r"inline|inlining", re.IGNORECASE),
    "layout_opt": re.compile(r"layout|channels.?last|memory.?format", re.IGNORECASE),
    "quantization": re.compile(r"quantiz|int8|fp16.?convert", re.IGNORECASE),
}

# torch.compile mode info
TORCH_COMPILE_MODES = {
    "default": {
        "description": "Balanced compilation",
        "compile_time": "medium",
        "runtime": "good",
    },
    "reduce-overhead": {
        "description": "Minimal Python overhead, uses CUDA graphs",
        "compile_time": "fast",
        "runtime": "optimized for latency",
    },
    "max-autotune": {
        "description": "Maximum autotuning for best throughput",
        "compile_time": "slow (extensive search)",
        "runtime": "best throughput",
    },
    "max-autotune-no-cudagraphs": {
        "description": "Max autotune without CUDA graphs",
        "compile_time": "slow",
        "runtime": "good for dynamic shapes",
    },
}


def detect_graph_backend(content: str) -> GraphBackend:
    """Detect the graph compilation backend used in code."""
    if COMPILE_PATTERNS["triton_jit"].search(content):
        return GraphBackend.TRITON
    if COMPILE_PATTERNS["torch_compile"].search(content) or \
       COMPILE_PATTERNS["torch_compile_decorator"].search(content):
        return GraphBackend.TORCH_COMPILE
    if COMPILE_PATTERNS["torch_script"].search(content) or \
       COMPILE_PATTERNS["torch_script_decorator"].search(content):
        return GraphBackend.TORCH_SCRIPT
    if COMPILE_PATTERNS["torch_trace"].search(content):
        return GraphBackend.TORCH_TRACE
    if COMPILE_PATTERNS["tf_xla"].search(content):
        return GraphBackend.TF_XLA
    if COMPILE_PATTERNS["tf_function"].search(content):
        return GraphBackend.TF_FUNCTION
    if COMPILE_PATTERNS["jax_jit"].search(content):
        return GraphBackend.JAX_JIT
    if COMPILE_PATTERNS["jax_pmap"].search(content):
        return GraphBackend.JAX_PMAP
    return GraphBackend.UNKNOWN


def find_compilations(content: str) -> List[GraphCompilation]:
    """Find all graph compilation points in code."""
    compilations = []
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        # torch.compile
        match = COMPILE_PATTERNS["torch_compile"].search(line)
        if match:
            compilations.append(GraphCompilation(
                backend=GraphBackend.TORCH_COMPILE,
                line_number=i,
                function_name=match.group(1),
                mode=match.group(2) if match.lastindex and match.lastindex >= 2 else "default",
            ))
            continue

        # @torch.compile decorator
        match = COMPILE_PATTERNS["torch_compile_decorator"].search(line)
        if match:
            compilations.append(GraphCompilation(
                backend=GraphBackend.TORCH_COMPILE,
                line_number=i,
                mode=match.group(1) if match.lastindex and match.lastindex >= 1 else "default",
            ))
            continue

        # torch.jit.trace
        match = COMPILE_PATTERNS["torch_trace"].search(line)
        if match:
            compilations.append(GraphCompilation(
                backend=GraphBackend.TORCH_TRACE,
                line_number=i,
                function_name=match.group(1),
            ))
            continue

        # torch.jit.script
        match = COMPILE_PATTERNS["torch_script"].search(line)
        if match:
            compilations.append(GraphCompilation(
                backend=GraphBackend.TORCH_SCRIPT,
                line_number=i,
                function_name=match.group(1),
            ))
            continue

        if COMPILE_PATTERNS["torch_script_decorator"].search(line):
            compilations.append(GraphCompilation(
                backend=GraphBackend.TORCH_SCRIPT,
                line_number=i,
            ))
            continue

        if COMPILE_PATTERNS["jax_jit"].search(line):
            compilations.append(GraphCompilation(
                backend=GraphBackend.JAX_JIT,
                line_number=i,
            ))
            continue

        if COMPILE_PATTERNS["triton_jit"].search(line):
            compilations.append(GraphCompilation(
                backend=GraphBackend.TRITON,
                line_number=i,
            ))

    return compilations


def find_graph_breaks(content: str) -> List[Dict[str, Any]]:
    """Find potential graph breaks in torch.compile code."""
    breaks = []
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        if DYNAMO_PATTERNS["graph_break"].search(line):
            breaks.append({"line": i, "type": "graph_break", "content": line.strip()[:100]})
        elif DYNAMO_PATTERNS["recompile"].search(line):
            breaks.append({"line": i, "type": "recompilation", "content": line.strip()[:100]})
        elif DYNAMO_PATTERNS["fallback"].search(line):
            breaks.append({"line": i, "type": "fallback", "content": line.strip()[:100]})

    return breaks


def detect_optimizations(content: str) -> List[GraphOptimization]:
    """Detect graph optimizations in compiler output."""
    optimizations = []
    seen = set()

    descriptions = {
        "fusion": "Operator fusion - combines operations into single kernel",
        "constant_folding": "Constant folding - precomputes constant expressions",
        "dead_code": "Dead code elimination - removes unused computations",
        "inlining": "Function inlining - reduces function call overhead",
        "layout_opt": "Memory layout optimization",
        "quantization": "Quantization - reduces precision for speed",
    }

    for opt_name, pattern in OPTIMIZATION_PATTERNS.items():
        if pattern.search(content) and opt_name not in seen:
            seen.add(opt_name)
            optimizations.append(GraphOptimization(
                name=opt_name,
                description=descriptions.get(opt_name, "Graph optimization"),
            ))

    return optimizations


def analyze_compile_mode(mode: str) -> Dict[str, Any]:
    """Analyze a torch.compile mode."""
    mode_info = TORCH_COMPILE_MODES.get(mode, {
        "description": f"Custom mode: {mode}",
        "compile_time": "unknown",
        "runtime": "unknown",
    })

    suggestions = []
    if mode == "default":
        suggestions.append("Try 'reduce-overhead' for latency-sensitive inference")
        suggestions.append("Try 'max-autotune' for maximum throughput")
    elif mode == "reduce-overhead":
        suggestions.append("Best for small batch sizes")
        suggestions.append("Uses CUDA graphs when possible")
    elif mode == "max-autotune":
        suggestions.append("First run slow due to autotuning")
        suggestions.append("Cache with TORCHINDUCTOR_CACHE_DIR")

    return {**mode_info, "suggestions": suggestions}


def analyze_graph_code(content: str) -> Dict[str, Any]:
    """Perform comprehensive analysis of graph compilation code."""
    backend = detect_graph_backend(content)
    compilations = find_compilations(content)
    graph_breaks = find_graph_breaks(content)
    optimizations = detect_optimizations(content)

    analysis = {
        "backend": backend.value,
        "compilations": [c.to_dict() for c in compilations],
        "graph_breaks": graph_breaks,
        "optimizations": [o.to_dict() for o in optimizations],
        "suggestions": [],
        "warnings": [],
    }

    if backend == GraphBackend.TORCH_COMPILE:
        if graph_breaks:
            analysis["warnings"].append(
                f"Found {len(graph_breaks)} potential graph breaks"
            )
            analysis["suggestions"].append(
                "Use TORCH_LOGS='graph_breaks' to debug"
            )

        modes = [c.mode for c in compilations if c.mode]
        if "default" in modes:
            analysis["suggestions"].append(
                "Consider 'reduce-overhead' or 'max-autotune'"
            )

    elif backend == GraphBackend.TORCH_TRACE:
        analysis["warnings"].append(
            "torch.jit.trace only captures single path - consider torch.compile"
        )

    elif backend == GraphBackend.JAX_JIT:
        analysis["suggestions"].append(
            "Use jax.lax.cond for conditionals to avoid recompilation"
        )

    return analysis


# Debug environment variables
DEBUG_ENV_VARS = {
    "torch.compile": [
        ("TORCH_LOGS", "Enable logging: 'graph_breaks,recompiles'"),
        ("TORCH_COMPILE_DEBUG", "Enable detailed debug"),
        ("TORCHINDUCTOR_CACHE_DIR", "Cache compiled kernels"),
    ],
    "JAX": [
        ("JAX_LOG_COMPILES", "Log JAX compilations"),
        ("XLA_FLAGS", "XLA compiler flags"),
    ],
}


def get_debug_env_vars(backend: GraphBackend) -> List[Tuple[str, str]]:
    """Get relevant debug environment variables."""
    if backend in (GraphBackend.TORCH_COMPILE, GraphBackend.TORCH_TRACE, GraphBackend.TORCH_SCRIPT):
        return DEBUG_ENV_VARS["torch.compile"]
    elif backend in (GraphBackend.JAX_JIT, GraphBackend.JAX_PMAP):
        return DEBUG_ENV_VARS["JAX"]
    return []


class CompilationAnalyzer:
    """
    Analyzer for graph compilation in ML code.

    Usage:
        analyzer = CompilationAnalyzer()

        # Analyze source code
        analysis = analyzer.analyze_code(source_code)

        # Analyze compiler output/logs
        optimizations = analyzer.analyze_logs(compiler_output)
    """

    def __init__(self):
        self.compilations: List[GraphCompilation] = []
        self.graph_breaks: List[Dict[str, Any]] = []
        self.optimizations: List[GraphOptimization] = []

    def analyze_code(self, content: str) -> Dict[str, Any]:
        """Analyze source code for compilation patterns."""
        result = analyze_graph_code(content)
        self.compilations.extend(find_compilations(content))
        self.graph_breaks.extend(find_graph_breaks(content))
        self.optimizations.extend(detect_optimizations(content))
        return result

    def analyze_logs(self, logs: str) -> Dict[str, Any]:
        """Analyze compiler logs for optimizations."""
        opts = detect_optimizations(logs)
        breaks = find_graph_breaks(logs)
        self.optimizations.extend(opts)
        self.graph_breaks.extend(breaks)
        return {
            "optimizations": [o.to_dict() for o in opts],
            "graph_breaks": breaks,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        return {
            "total_compilations": len(self.compilations),
            "graph_break_count": len(self.graph_breaks),
            "optimization_count": len(self.optimizations),
            "backends": list(set(c.backend.value for c in self.compilations)),
            "optimization_types": list(set(o.name for o in self.optimizations)),
        }

    def get_recommendations(self) -> List[str]:
        """Get recommendations based on analysis."""
        recs = []
        if self.graph_breaks:
            recs.append(f"Fix {len(self.graph_breaks)} graph breaks for better performance")
        if not self.optimizations:
            recs.append("No optimizations detected - ensure compiler is running")
        if any(c.mode == "default" for c in self.compilations):
            recs.append("Consider 'max-autotune' mode for production")
        return recs

    def clear(self):
        """Clear analysis state."""
        self.compilations = []
        self.graph_breaks = []
        self.optimizations = []
