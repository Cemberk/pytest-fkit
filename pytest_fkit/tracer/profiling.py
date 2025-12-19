"""
Performance Profiling: GEMM analysis, roofline modeling, bottleneck detection.

Provides:
- GEMM FLOPS and memory analysis
- Roofline model for compute vs memory bound detection
- GPU bottleneck detection (power, thermal, occupancy)
- Attention operation analysis (Flash Attention)
- Performance recommendations
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .hardware import precision_to_bytes, get_peak_performance, get_memory_bandwidth


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    COMPUTE_BOUND = "compute"
    MEMORY_BOUND = "memory"
    LATENCY_BOUND = "latency"
    IO_BOUND = "io"
    POWER_THROTTLE = "power"
    THERMAL_THROTTLE = "thermal"
    PCIE_BOUND = "pcie"
    SYNC_BOUND = "sync"
    KERNEL_LAUNCH = "launch"


@dataclass
class GEMMAnalysis:
    """Complete analysis of a GEMM operation."""
    M: int
    N: int
    K: int
    batch: int = 1
    precision: str = "fp32"

    # Computed fields (set in __post_init__)
    flops: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    total_bytes: int = 0
    arithmetic_intensity: float = 0.0
    tensor_core_aligned: bool = False
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        bpe = precision_to_bytes(self.precision)
        self.flops = 2 * self.batch * self.M * self.N * self.K
        self.bytes_read = self.batch * (self.M * self.K + self.K * self.N) * bpe
        self.bytes_written = self.batch * self.M * self.N * bpe
        self.total_bytes = self.bytes_read + self.bytes_written
        if self.total_bytes > 0:
            self.arithmetic_intensity = self.flops / self.total_bytes
        self.tensor_core_aligned = self._check_alignment()
        self._generate_recommendations()

    def _check_alignment(self) -> bool:
        alignment = 16 if self.precision in ["fp8", "int8"] else 8
        return self.M % alignment == 0 and self.N % alignment == 0 and self.K % alignment == 0

    def _generate_recommendations(self):
        recs = []
        if not self.tensor_core_aligned:
            alignment = 16 if self.precision in ["fp8", "int8"] else 8
            recs.append(f"Pad dimensions to multiples of {alignment} for tensor cores")
        if self.M < 32 or self.N < 32:
            recs.append("Small M/N - consider batching for better utilization")
        if self.K < 64:
            recs.append("Small K - GEMM is memory-bound; consider fusion")
        if self.arithmetic_intensity < 50:
            recs.append(f"Low arithmetic intensity ({self.arithmetic_intensity:.1f}) - memory-bound")
        elif self.arithmetic_intensity > 200:
            recs.append(f"High arithmetic intensity ({self.arithmetic_intensity:.1f}) - compute-bound")
        self.recommendations = recs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "M": self.M, "N": self.N, "K": self.K, "batch": self.batch,
            "precision": self.precision, "flops": self.flops,
            "bytes_read": self.bytes_read, "bytes_written": self.bytes_written,
            "total_bytes": self.total_bytes,
            "arithmetic_intensity": self.arithmetic_intensity,
            "tensor_core_aligned": self.tensor_core_aligned,
            "recommendations": self.recommendations,
        }

    def to_metrics(self) -> Dict[str, float]:
        """Convert to numeric metrics for PySR."""
        return {
            "gemm_m": float(self.M),
            "gemm_n": float(self.N),
            "gemm_k": float(self.K),
            "gemm_batch": float(self.batch),
            "gemm_flops": float(self.flops),
            "gemm_bytes": float(self.total_bytes),
            "arithmetic_intensity": self.arithmetic_intensity,
        }


@dataclass
class RooflineAnalysis:
    """Roofline model analysis results."""
    arithmetic_intensity: float
    peak_compute_tflops: float
    memory_bandwidth_tbps: float
    achievable_tflops: float
    ridge_point: float
    bottleneck: str  # "compute" or "memory"
    efficiency: float
    headroom: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arithmetic_intensity": self.arithmetic_intensity,
            "peak_compute_tflops": self.peak_compute_tflops,
            "memory_bandwidth_tbps": self.memory_bandwidth_tbps,
            "achievable_tflops": self.achievable_tflops,
            "ridge_point": self.ridge_point,
            "bottleneck": self.bottleneck,
            "efficiency": self.efficiency,
            "headroom": self.headroom,
        }

    def to_metrics(self) -> Dict[str, float]:
        return {
            "roofline_ai": self.arithmetic_intensity,
            "roofline_achievable_tflops": self.achievable_tflops,
            "roofline_efficiency": self.efficiency,
            "roofline_headroom": self.headroom,
            "is_memory_bound": 1.0 if self.bottleneck == "memory" else 0.0,
            "is_compute_bound": 1.0 if self.bottleneck == "compute" else 0.0,
        }


@dataclass
class PerformanceBottleneck:
    """A detected performance bottleneck."""
    bottleneck_type: BottleneckType
    severity: float  # 0-1
    details: str
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.bottleneck_type.value,
            "severity": self.severity,
            "details": self.details,
            "suggestions": self.suggestions,
        }


def calculate_gemm_flops(M: int, N: int, K: int, batch: int = 1, bias: bool = False) -> int:
    """Calculate FLOPS for GEMM C = A @ B (+ bias)."""
    flops = 2 * batch * M * N * K
    if bias:
        flops += batch * M * N
    return flops


def calculate_gemm_bytes(
    M: int, N: int, K: int, batch: int = 1,
    precision: str = "fp32", bias: bool = False
) -> Dict[str, int]:
    """Calculate memory bytes for GEMM."""
    bpe = precision_to_bytes(precision)
    bytes_a = batch * M * K * bpe
    bytes_b = batch * K * N * bpe
    bytes_c = batch * M * N * bpe
    bytes_bias = N * bpe if bias else 0
    return {
        "bytes_a": bytes_a, "bytes_b": bytes_b, "bytes_c": bytes_c,
        "bytes_read": bytes_a + bytes_b + bytes_bias,
        "bytes_written": bytes_c,
        "total_bytes": bytes_a + bytes_b + bytes_c + bytes_bias,
    }


def roofline_analysis(
    flops: int,
    total_bytes: int,
    peak_compute_tflops: float,
    memory_bandwidth_tbps: float,
    actual_time_ms: Optional[float] = None
) -> RooflineAnalysis:
    """
    Perform roofline model analysis.

    Args:
        flops: Total FLOPS
        total_bytes: Total bytes transferred
        peak_compute_tflops: Peak GPU compute (TFLOPS)
        memory_bandwidth_tbps: Memory bandwidth (TB/s)
        actual_time_ms: Actual execution time for efficiency calc
    """
    ai = flops / total_bytes if total_bytes > 0 else 0
    ridge_point = peak_compute_tflops / memory_bandwidth_tbps if memory_bandwidth_tbps > 0 else 0

    if ai < ridge_point:
        achievable_tflops = ai * memory_bandwidth_tbps
        bottleneck = "memory"
    else:
        achievable_tflops = peak_compute_tflops
        bottleneck = "compute"

    efficiency = min(achievable_tflops / peak_compute_tflops, 1.0) if peak_compute_tflops > 0 else 0

    if actual_time_ms and actual_time_ms > 0:
        actual_tflops = (flops / 1e12) / (actual_time_ms / 1000)
        efficiency = actual_tflops / achievable_tflops if achievable_tflops > 0 else 0

    return RooflineAnalysis(
        arithmetic_intensity=ai,
        peak_compute_tflops=peak_compute_tflops,
        memory_bandwidth_tbps=memory_bandwidth_tbps,
        achievable_tflops=achievable_tflops,
        ridge_point=ridge_point,
        bottleneck=bottleneck,
        efficiency=min(efficiency, 1.0),
        headroom=max(0, 1.0 - efficiency),
    )


# Dimension extraction patterns
DIMENSION_PATTERNS = [
    r"[#\/]\s*M\s*[=:]\s*(\d+).*N\s*[=:]\s*(\d+).*K\s*[=:]\s*(\d+)",
    r"(\d+)\s*[xX×]\s*(\d+)\s*[xX×]\s*(\d+)",
    r"nn\.Linear\s*\(\s*(\d+)\s*,\s*(\d+)",
]


def extract_dimensions(content: str) -> List[Tuple[int, int, int]]:
    """Extract GEMM dimensions (M, N, K) from code."""
    dimensions = []

    # Explicit M, N, K comments
    for match in re.finditer(DIMENSION_PATTERNS[0], content, re.IGNORECASE):
        m, n, k = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if m > 0 and n > 0 and k > 0:
            dimensions.append((m, n, k))

    # Multiplication notation
    for match in re.finditer(DIMENSION_PATTERNS[1], content):
        m, k, n = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if m > 0 and n > 0 and k > 0:
            dimensions.append((m, n, k))

    # nn.Linear
    for match in re.finditer(DIMENSION_PATTERNS[2], content):
        in_f, out_f = int(match.group(1)), int(match.group(2))
        dimensions.append((1, out_f, in_f))

    return dimensions


def analyze_gemm_from_code(content: str) -> List[GEMMAnalysis]:
    """Extract and analyze all GEMM operations from code."""
    precision = "fp32"
    if re.search(r"\bbf16\b|\bbfloat16\b", content, re.IGNORECASE):
        precision = "bf16"
    elif re.search(r"\bfp16\b|\bfloat16\b|\bhalf\b", content, re.IGNORECASE):
        precision = "fp16"
    elif re.search(r"\bfp8\b|\bfloat8\b", content, re.IGNORECASE):
        precision = "fp8"

    dims = extract_dimensions(content)
    return [GEMMAnalysis(M=m, N=n, K=k, precision=precision) for m, n, k in dims]


def analyze_attention(
    batch_size: int, seq_len: int, num_heads: int, head_dim: int,
    precision: str = "fp16"
) -> Dict[str, Any]:
    """Analyze attention operation complexity."""
    bpe = precision_to_bytes(precision)

    # Q @ K^T -> softmax -> @ V
    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    av_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    total_flops = qk_flops + softmax_flops + av_flops

    # Memory
    qkv_bytes = 3 * batch_size * num_heads * seq_len * head_dim * bpe
    attn_scores_bytes = batch_size * num_heads * seq_len * seq_len * bpe
    output_bytes = batch_size * num_heads * seq_len * head_dim * bpe

    standard_bytes = qkv_bytes + attn_scores_bytes + output_bytes
    flash_bytes = qkv_bytes + output_bytes  # No full attention matrix

    memory_savings = 1 - (flash_bytes / standard_bytes) if standard_bytes > 0 else 0

    recommendations = []
    if seq_len >= 512:
        recommendations.append("Flash Attention recommended for memory efficiency")
    if seq_len >= 2048:
        recommendations.append(f"Long sequence ({seq_len}) - Flash saves {memory_savings:.1%} memory")

    return {
        "total_flops": total_flops,
        "standard_bytes": standard_bytes,
        "flash_bytes": flash_bytes,
        "memory_savings_pct": memory_savings * 100,
        "complexity": f"O(S²) where S={seq_len}",
        "recommendations": recommendations,
    }


# Performance thresholds
PERF_THRESHOLDS = {
    "power_throttle": 0.95,
    "thermal_throttle": 83,
    "low_occupancy": 0.5,
    "memory_bound": 0.8,
    "compute_bound": 0.8,
    "pcie_bound": 0.7,
}


def detect_bottlenecks(
    compute_throughput: float = 0,
    memory_throughput: float = 0,
    occupancy: float = 1,
    power_draw: Optional[float] = None,
    power_limit: Optional[float] = None,
    temperature: Optional[float] = None,
    data_load_time: float = 0,
    compute_time: float = 1,
) -> List[PerformanceBottleneck]:
    """
    Detect performance bottlenecks from metrics.

    Args:
        compute_throughput: Compute utilization (0-1)
        memory_throughput: Memory bandwidth utilization (0-1)
        occupancy: GPU occupancy (0-1)
        power_draw: Current power draw (watts)
        power_limit: Power limit (watts)
        temperature: GPU temperature (Celsius)
        data_load_time: Time spent loading data
        compute_time: Time spent in compute
    """
    bottlenecks = []

    # Memory bound
    if memory_throughput > PERF_THRESHOLDS["memory_bound"] and compute_throughput < 0.5:
        bottlenecks.append(PerformanceBottleneck(
            bottleneck_type=BottleneckType.MEMORY_BOUND,
            severity=memory_throughput,
            details=f"Memory throughput {memory_throughput*100:.1f}% vs compute {compute_throughput*100:.1f}%",
            suggestions=[
                "Increase arithmetic intensity",
                "Use mixed precision (FP16/BF16)",
                "Improve data locality with tiling",
            ],
        ))

    # Compute bound
    elif compute_throughput > PERF_THRESHOLDS["compute_bound"] and memory_throughput < 0.5:
        bottlenecks.append(PerformanceBottleneck(
            bottleneck_type=BottleneckType.COMPUTE_BOUND,
            severity=compute_throughput,
            details=f"Compute throughput {compute_throughput*100:.1f}%",
            suggestions=["Good utilization! Consider reduced precision for more throughput."],
        ))

    # Low occupancy
    if occupancy < PERF_THRESHOLDS["low_occupancy"]:
        bottlenecks.append(PerformanceBottleneck(
            bottleneck_type=BottleneckType.LATENCY_BOUND,
            severity=1 - occupancy,
            details=f"Occupancy only {occupancy*100:.1f}%",
            suggestions=[
                "Reduce registers per thread",
                "Reduce shared memory usage",
                "Increase threads per block",
            ],
        ))

    # Power throttling
    if power_draw and power_limit:
        ratio = power_draw / power_limit
        if ratio > PERF_THRESHOLDS["power_throttle"]:
            bottlenecks.append(PerformanceBottleneck(
                bottleneck_type=BottleneckType.POWER_THROTTLE,
                severity=ratio,
                details=f"Power at {ratio*100:.1f}% of limit",
                suggestions=["Increase power limit", "Improve cooling"],
            ))

    # Thermal throttling
    if temperature and temperature >= PERF_THRESHOLDS["thermal_throttle"]:
        bottlenecks.append(PerformanceBottleneck(
            bottleneck_type=BottleneckType.THERMAL_THROTTLE,
            severity=min(temperature / 100, 1.0),
            details=f"Temperature {temperature}°C",
            suggestions=["Improve cooling", "Reduce power limit"],
        ))

    # I/O bound
    total_time = data_load_time + compute_time
    if total_time > 0 and data_load_time / total_time > 0.3:
        io_ratio = data_load_time / total_time
        bottlenecks.append(PerformanceBottleneck(
            bottleneck_type=BottleneckType.IO_BOUND,
            severity=io_ratio,
            details=f"Data loading takes {io_ratio*100:.1f}% of time",
            suggestions=[
                "Use more DataLoader workers",
                "Enable pin_memory=True",
                "Use prefetching",
            ],
        ))

    return bottlenecks


def calculate_throughput(items: int, time_ms: float, unit: str = "items") -> Dict[str, float]:
    """Calculate throughput metrics."""
    if time_ms <= 0:
        return {f"{unit}_per_second": 0, f"{unit}_per_minute": 0, "time_per_item_ms": 0}
    time_s = time_ms / 1000
    return {
        f"{unit}_per_second": items / time_s,
        f"{unit}_per_minute": items / time_s * 60,
        "time_per_item_ms": time_ms / items,
    }


def calculate_flops_throughput(flops: int, time_ms: float) -> Dict[str, float]:
    """Calculate FLOPS throughput."""
    if time_ms <= 0:
        return {"tflops": 0, "gflops": 0, "mflops": 0}
    flops_per_s = flops / (time_ms / 1000)
    return {
        "tflops": flops_per_s / 1e12,
        "gflops": flops_per_s / 1e9,
        "mflops": flops_per_s / 1e6,
    }


class PerformanceProfiler:
    """
    Performance profiler for ML workloads.

    Usage:
        profiler = PerformanceProfiler(gpu_specs={"fp16_tflops": 312, "memory_bw_tbps": 2.0})

        with profiler.profile_gemm(M=1024, N=2048, K=512) as gemm:
            result = torch.matmul(a, b)

        analysis = profiler.get_analysis()
    """

    def __init__(self, gpu_specs: Optional[Dict[str, Any]] = None):
        self.gpu_specs = gpu_specs or {}
        self.gemm_analyses: List[GEMMAnalysis] = []
        self.roofline_analyses: List[RooflineAnalysis] = []
        self.bottlenecks: List[PerformanceBottleneck] = []
        self.timings: Dict[str, float] = {}

    def analyze_gemm(
        self, M: int, N: int, K: int, batch: int = 1, precision: str = "fp16",
        actual_time_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """Analyze a GEMM operation."""
        gemm = GEMMAnalysis(M=M, N=N, K=K, batch=batch, precision=precision)
        self.gemm_analyses.append(gemm)

        result = {"gemm": gemm.to_dict()}

        # Roofline if GPU specs available
        if self.gpu_specs:
            peak_tflops = get_peak_performance(self.gpu_specs, precision)
            mem_bw = get_memory_bandwidth(self.gpu_specs)
            if peak_tflops > 0 and mem_bw > 0:
                roofline = roofline_analysis(
                    gemm.flops, gemm.total_bytes,
                    peak_tflops, mem_bw, actual_time_ms
                )
                self.roofline_analyses.append(roofline)
                result["roofline"] = roofline.to_dict()

        return result

    def analyze_attention(
        self, batch: int, seq_len: int, heads: int, head_dim: int,
        precision: str = "fp16"
    ) -> Dict[str, Any]:
        """Analyze attention operation."""
        return analyze_attention(batch, seq_len, heads, head_dim, precision)

    def detect_bottlenecks(self, **metrics) -> List[Dict[str, Any]]:
        """Detect bottlenecks from metrics."""
        bns = detect_bottlenecks(**metrics)
        self.bottlenecks.extend(bns)
        return [b.to_dict() for b in bns]

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        return {
            "gemm_count": len(self.gemm_analyses),
            "total_gemm_flops": sum(g.flops for g in self.gemm_analyses),
            "avg_arithmetic_intensity": (
                sum(g.arithmetic_intensity for g in self.gemm_analyses) / len(self.gemm_analyses)
                if self.gemm_analyses else 0
            ),
            "roofline_analyses": len(self.roofline_analyses),
            "bottleneck_count": len(self.bottlenecks),
            "bottleneck_types": list(set(b.bottleneck_type.value for b in self.bottlenecks)),
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics for PySR."""
        metrics = {}
        for i, gemm in enumerate(self.gemm_analyses):
            for k, v in gemm.to_metrics().items():
                metrics[f"{k}_{i}"] = v
        for i, roofline in enumerate(self.roofline_analyses):
            for k, v in roofline.to_metrics().items():
                metrics[f"{k}_{i}"] = v
        return metrics
