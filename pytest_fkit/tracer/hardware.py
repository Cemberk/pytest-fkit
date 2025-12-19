"""
Hardware Profiler: GPU detection, specifications, and monitoring.

Provides:
- Auto-detection of NVIDIA, AMD, Intel GPUs
- Peak performance and bandwidth specs
- Real-time monitoring metrics
- Hardware-aware performance predictions
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Precision to bytes mapping
PRECISION_BYTES = {
    "fp64": 8, "double": 8, "float64": 8,
    "fp32": 4, "float": 4, "float32": 4,
    "tf32": 4,
    "fp16": 2, "float16": 2, "half": 2,
    "bf16": 2, "bfloat16": 2,
    "fp8": 1, "float8": 1, "e4m3": 1, "e5m2": 1,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
}

# GPU specifications database
GPU_SPECS = {
    "nvidia": {
        "H100 SXM": {
            "fp32_tflops": 67.0, "fp16_tflops": 989.0, "bf16_tflops": 989.0,
            "fp8_tflops": 1979.0, "memory_gb": 80, "memory_bw_tbps": 3.35,
            "tensor_cores": True, "architecture": "Hopper", "tdp_watts": 700,
        },
        "H100 PCIe": {
            "fp32_tflops": 51.0, "fp16_tflops": 756.0, "bf16_tflops": 756.0,
            "memory_gb": 80, "memory_bw_tbps": 2.0, "architecture": "Hopper",
        },
        "H200": {
            "fp32_tflops": 67.0, "fp16_tflops": 989.0, "bf16_tflops": 989.0,
            "memory_gb": 141, "memory_bw_tbps": 4.8, "architecture": "Hopper",
        },
        "A100 SXM": {
            "fp32_tflops": 19.5, "fp16_tflops": 312.0, "bf16_tflops": 312.0,
            "memory_gb": 80, "memory_bw_tbps": 2.039, "architecture": "Ampere",
        },
        "A100 PCIe": {
            "fp32_tflops": 19.5, "fp16_tflops": 312.0, "bf16_tflops": 312.0,
            "memory_gb": 80, "memory_bw_tbps": 1.935, "architecture": "Ampere",
        },
        "L40S": {
            "fp32_tflops": 91.6, "fp16_tflops": 183.0, "bf16_tflops": 183.0,
            "memory_gb": 48, "memory_bw_tbps": 0.864, "architecture": "Ada Lovelace",
        },
        "RTX 4090": {
            "fp32_tflops": 82.6, "fp16_tflops": 165.0, "bf16_tflops": 165.0,
            "memory_gb": 24, "memory_bw_tbps": 1.008, "architecture": "Ada Lovelace",
        },
        "V100": {
            "fp32_tflops": 15.7, "fp16_tflops": 125.0,
            "memory_gb": 32, "memory_bw_tbps": 0.9, "architecture": "Volta",
        },
    },
    "amd": {
        "MI300X": {
            "fp32_tflops": 163.4, "fp16_tflops": 1307.0, "bf16_tflops": 1307.0,
            "fp8_tflops": 2615.0, "memory_gb": 192, "memory_bw_tbps": 5.3,
            "architecture": "CDNA3",
        },
        "MI325X": {
            "fp32_tflops": 163.4, "fp16_tflops": 1307.0, "bf16_tflops": 1307.0,
            "memory_gb": 256, "memory_bw_tbps": 6.0, "architecture": "CDNA3",
        },
        "MI250X": {
            "fp32_tflops": 95.7, "fp16_tflops": 383.0, "bf16_tflops": 383.0,
            "memory_gb": 128, "memory_bw_tbps": 3.2, "architecture": "CDNA2",
        },
        "MI210": {
            "fp32_tflops": 45.3, "fp16_tflops": 181.0, "bf16_tflops": 181.0,
            "memory_gb": 64, "memory_bw_tbps": 1.6, "architecture": "CDNA2",
        },
    },
    "intel": {
        "Max 1550": {
            "fp32_tflops": 52.0, "fp16_tflops": 839.0, "bf16_tflops": 839.0,
            "memory_gb": 128, "memory_bw_tbps": 3.2, "architecture": "Ponte Vecchio",
        },
    },
}


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    index: int
    name: str
    vendor: str  # nvidia, amd, intel
    memory_total_gb: float
    memory_used_gb: float = 0.0
    memory_free_gb: float = 0.0
    utilization_pct: float = 0.0
    temperature_c: float = 0.0
    power_watts: float = 0.0
    power_limit_watts: float = 0.0
    clock_mhz: int = 0
    driver_version: str = ""
    specs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "vendor": self.vendor,
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
            "memory_free_gb": self.memory_free_gb,
            "utilization_pct": self.utilization_pct,
            "temperature_c": self.temperature_c,
            "power_watts": self.power_watts,
            "power_limit_watts": self.power_limit_watts,
            "clock_mhz": self.clock_mhz,
            "driver_version": self.driver_version,
        }

    def to_metrics(self) -> Dict[str, float]:
        """Convert to numeric metrics for PySR."""
        return {
            "gpu_memory_total_gb": self.memory_total_gb,
            "gpu_memory_used_gb": self.memory_used_gb,
            "gpu_memory_free_gb": self.memory_free_gb,
            "gpu_utilization_pct": self.utilization_pct,
            "gpu_temperature_c": self.temperature_c,
            "gpu_power_watts": self.power_watts,
            "gpu_clock_mhz": float(self.clock_mhz),
        }

    @property
    def peak_tflops(self) -> float:
        """Get peak TFLOPS (FP16)."""
        return self.specs.get("fp16_tflops", self.specs.get("fp32_tflops", 0))

    @property
    def memory_bandwidth_tbps(self) -> float:
        """Get memory bandwidth in TB/s."""
        return self.specs.get("memory_bw_tbps", 0)


def precision_to_bytes(precision: str) -> int:
    """Convert precision string to bytes per element."""
    return PRECISION_BYTES.get(precision.lower(), 4)


def detect_gpus() -> List[GPUInfo]:
    """Detect all available GPUs on the system."""
    gpus = []
    gpus.extend(_detect_nvidia_gpus())
    gpus.extend(_detect_amd_gpus())
    gpus.extend(_detect_intel_gpus())
    return gpus


def _detect_nvidia_gpus() -> List[GPUInfo]:
    """Detect NVIDIA GPUs using nvidia-smi."""
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                "utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm,driver_version",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 11:
                    name = parts[1]
                    gpu = GPUInfo(
                        index=int(parts[0]),
                        name=name,
                        vendor="nvidia",
                        memory_total_gb=float(parts[2]) / 1024,
                        memory_used_gb=_safe_float(parts[3]) / 1024,
                        memory_free_gb=_safe_float(parts[4]) / 1024,
                        utilization_pct=_safe_float(parts[5]),
                        temperature_c=_safe_float(parts[6]),
                        power_watts=_safe_float(parts[7]),
                        power_limit_watts=_safe_float(parts[8]),
                        clock_mhz=int(_safe_float(parts[9])),
                        driver_version=parts[10] if parts[10] != '[N/A]' else "",
                        specs=_match_gpu_specs(name, "nvidia"),
                    )
                    gpus.append(gpu)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return gpus


def _detect_amd_gpus() -> List[GPUInfo]:
    """Detect AMD GPUs using rocm-smi."""
    gpus = []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            name_match = re.search(r'(MI\d+\w*)', result.stdout, re.IGNORECASE)
            name = name_match.group(1) if name_match else "AMD GPU"
            mem_match = re.search(r'(\d+)\s*(GB|GiB)', result.stdout, re.IGNORECASE)
            memory_gb = float(mem_match.group(1)) if mem_match else 0
            gpu = GPUInfo(
                index=0, name=name, vendor="amd",
                memory_total_gb=memory_gb,
                specs=_match_gpu_specs(name, "amd"),
            )
            gpus.append(gpu)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return gpus


def _detect_intel_gpus() -> List[GPUInfo]:
    """Detect Intel GPUs using sycl-ls."""
    gpus = []
    try:
        result = subprocess.run(["sycl-ls"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'intel' in line.lower() and 'gpu' in line.lower():
                    name_match = re.search(r'(Max\s*\d+)', line, re.IGNORECASE)
                    name = name_match.group(1) if name_match else "Intel GPU"
                    gpu = GPUInfo(
                        index=len(gpus), name=name, vendor="intel",
                        memory_total_gb=0,
                        specs=_match_gpu_specs(name, "intel"),
                    )
                    gpus.append(gpu)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return gpus


def _safe_float(s: str) -> float:
    """Safely parse float from string."""
    try:
        if s == '[N/A]' or not s:
            return 0.0
        return float(s)
    except ValueError:
        return 0.0


def _match_gpu_specs(gpu_name: str, vendor: str) -> Dict[str, Any]:
    """Match GPU name to specs database."""
    vendor_specs = GPU_SPECS.get(vendor, {})
    gpu_lower = gpu_name.lower()
    for spec_name, specs in vendor_specs.items():
        if spec_name.lower() in gpu_lower:
            return specs.copy()
    for spec_name, specs in vendor_specs.items():
        parts = spec_name.lower().split()
        if any(p in gpu_lower for p in parts if len(p) > 2):
            return specs.copy()
    return {}


def get_gpu_specs(gpu_name: str, vendor: Optional[str] = None) -> Dict[str, Any]:
    """Get specifications for a GPU by name."""
    if vendor:
        specs = _match_gpu_specs(gpu_name, vendor)
        if specs:
            return specs
    for v in ["nvidia", "amd", "intel"]:
        specs = _match_gpu_specs(gpu_name, v)
        if specs:
            return specs
    return {}


def get_peak_performance(specs: Dict[str, Any], precision: str = "fp16") -> float:
    """Get peak TFLOPS for given precision."""
    key = f"{precision}_tflops"
    return specs.get(key, specs.get("fp32_tflops", 0))


def get_memory_bandwidth(specs: Dict[str, Any]) -> float:
    """Get memory bandwidth in TB/s."""
    return specs.get("memory_bw_tbps", 0)


def sample_gpu_metrics() -> List[Dict[str, float]]:
    """Sample current GPU metrics for all GPUs."""
    gpus = detect_gpus()
    return [gpu.to_metrics() for gpu in gpus]


class HardwareProfiler:
    """
    Hardware profiler for continuous monitoring.

    Usage:
        profiler = HardwareProfiler()
        profiler.start_sampling(interval_ms=100)

        # ... do work ...

        metrics = profiler.stop_sampling()
    """

    def __init__(self):
        self.gpus: List[GPUInfo] = []
        self.samples: List[Dict[str, Any]] = []
        self._sampling = False
        self._detect_hardware()

    def _detect_hardware(self):
        """Detect available hardware."""
        self.gpus = detect_gpus()

    def get_system_info(self) -> Dict[str, Any]:
        """Get system hardware information."""
        return {
            "gpu_count": len(self.gpus),
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "total_gpu_memory_gb": sum(g.memory_total_gb for g in self.gpus),
            "peak_tflops": sum(g.peak_tflops for g in self.gpus),
            "total_memory_bw_tbps": sum(g.memory_bandwidth_tbps for g in self.gpus),
        }

    def sample(self) -> Dict[str, Any]:
        """Take a single sample of hardware metrics."""
        import time
        gpus = detect_gpus()
        sample = {
            "timestamp": time.time(),
            "gpus": [gpu.to_metrics() for gpu in gpus],
        }
        self.samples.append(sample)
        return sample

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across all samples."""
        if not self.samples:
            return {}

        # Aggregate GPU metrics
        all_gpu_metrics = []
        for sample in self.samples:
            for gpu_metrics in sample.get("gpus", []):
                all_gpu_metrics.append(gpu_metrics)

        if not all_gpu_metrics:
            return {}

        # Compute stats
        keys = all_gpu_metrics[0].keys()
        result = {}
        for key in keys:
            values = [m[key] for m in all_gpu_metrics if key in m]
            if values:
                result[f"{key}_mean"] = sum(values) / len(values)
                result[f"{key}_max"] = max(values)
                result[f"{key}_min"] = min(values)

        return result

    def clear_samples(self):
        """Clear collected samples."""
        self.samples = []
