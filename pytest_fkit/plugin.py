"""
pytest-fkit plugin: Isolate test crashes and convert them to ERROR results

Inspired by fkitpy - when tests crash (SIGABRT, SIGSEGV, etc.),
catch them and report as normal pytest errors instead of killing the entire run.

This plugin supports two execution modes:

1. BATCH MODE (default, recommended for speed):
   - Tests are sliced upfront and distributed to workers
   - Each worker runs its entire slice in a SINGLE subprocess
   - Much faster due to reduced subprocess overhead
   - Use --fkit-batch (default) or --fkit-mode=batch

2. ISOLATION MODE (for maximum crash isolation):
   - Each test runs in its own subprocess
   - Slower but provides per-test crash isolation
   - Use --fkit-mode=isolate

Supports parallel workers with GPU affinity for multi-GPU systems.
"""
import sys
import os
import subprocess
import pytest
import signal
import tempfile
import time
import threading
import queue
import re
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Tuple
from enum import Enum

# Base port for per-worker NCCL MASTER_PORT allocation.
# Each fkit worker gets NCCL_BASE_PORT + worker_id to avoid collisions
# when multiple workers run DataParallel tests simultaneously.
NCCL_BASE_PORT = 29500


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
    group.addoption(
        "--fkit-workers",
        action="store",
        type=str,
        default="1",
        help="Number of parallel workers (default: 1, use 'auto' for GPU-based auto-detection)",
    )
    group.addoption(
        "--fkit-gpus-per-worker",
        action="store",
        type=int,
        default=2,
        help="GPUs assigned to each worker (default: 2 for multi-GPU test support)",
    )
    group.addoption(
        "--fkit-mode",
        action="store",
        type=str,
        default="batch",
        choices=["batch", "isolate"],
        help="Execution mode: 'batch' (fast, slice tests per worker) or 'isolate' (slow, one subprocess per test)",
    )
    group.addoption(
        "--fkit-threads-per-worker",
        action="store",
        type=str,
        default="auto",
        help="CPU threads per worker for OMP/MKL (default: auto = total_cores/num_workers)",
    )
    group.addoption(
        "--fkit-batch",
        action="store_true",
        default=False,
        help="[DEPRECATED] Use batch mode (now default). Use --fkit-mode=isolate for per-test isolation.",
    )
    group.addoption(
        "--fkit-max-retries",
        action="store",
        type=int,
        default=3,
        help="Max retries for transient errors (GPU unavailable, DNS, network). Default: 3",
    )


def pytest_configure(config):
    """Register the plugin markers."""
    config.addinivalue_line(
        "markers",
        "fkit_skip: Skip crash isolation for this test (run normally)"
    )
    config.addinivalue_line(
        "markers",
        "fkit_multi_gpu: Mark test as requiring multiple GPUs"
    )
    config.addinivalue_line(
        "markers",
        "fkit_single_gpu: Mark test as requiring only single GPU"
    )

    # Only register if enabled
    if config.getoption("--fkit"):
        config.pluginmanager.register(CrashIsolationPlugin(config), "fkit_plugin")


@dataclass
class GPUInfo:
    """Information about available GPUs."""
    count: int
    vendor: str  # 'amd', 'nvidia', or 'none'
    ids: List[str]


@dataclass
class CPUInfo:
    """Information about available CPU cores."""
    total_cores: int
    physical_cores: int


def detect_cpus() -> CPUInfo:
    """Detect available CPU cores."""
    import multiprocessing
    
    total_cores = multiprocessing.cpu_count()
    
    # Try to get physical cores (excluding hyperthreading)
    physical_cores = total_cores
    try:
        import os
        # Linux: count physical cores
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo') as f:
                content = f.read()
                # Count unique physical id + core id combinations
                physical_ids = set()
                current_physical = None
                current_core = None
                for line in content.split('\n'):
                    if line.startswith('physical id'):
                        current_physical = line.split(':')[1].strip()
                    elif line.startswith('core id'):
                        current_core = line.split(':')[1].strip()
                        if current_physical is not None and current_core is not None:
                            physical_ids.add((current_physical, current_core))
                            current_physical = None
                            current_core = None
                if physical_ids:
                    physical_cores = len(physical_ids)
    except Exception:
        pass
    
    return CPUInfo(total_cores=total_cores, physical_cores=physical_cores)


def detect_gpus() -> GPUInfo:
    """Detect available GPUs (AMD or NVIDIA).

    IMPORTANT: Respects HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES /
    ROCR_VISIBLE_DEVICES environment variables.  These env vars restrict
    which GPUs the parent process intended to make available (e.g. set by
    priority_bucketing.py or Jenkins).  rocm-smi and nvidia-smi report ALL
    physical GPUs regardless of these env vars, so we must check the env
    vars first to avoid allocating workers to inaccessible GPUs.
    """
    # ---- Step 1: Respect visibility env vars ----
    # Check AMD-specific vars first, then the generic CUDA var.
    for var in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        val = os.environ.get(var, '').strip()
        if val:
            gpu_ids = [x.strip() for x in val.split(',') if x.strip()]
            if gpu_ids:
                # Determine vendor from available CLI tools
                vendor = 'amd' if shutil.which('rocm-smi') else (
                    'nvidia' if shutil.which('nvidia-smi') else 'unknown')
                print(f"[fkit] GPU detection: using {var}={val} "
                      f"({len(gpu_ids)} GPUs, vendor={vendor})")
                return GPUInfo(count=len(gpu_ids), vendor=vendor, ids=gpu_ids)

    # ---- Step 2: Fall back to hardware detection ----
    # Try AMD first (ROCm)
    try:
        result = subprocess.run(
            ['rocm-smi', '--showid'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse GPU IDs from rocm-smi output - deduplicate
            gpu_ids_set = set()
            for line in result.stdout.split('\n'):
                # Look for lines like "GPU[0]" or device indices
                if 'GPU[' in line:
                    match = re.search(r'GPU\[(\d+)\]', line)
                    if match:
                        gpu_ids_set.add(match.group(1))
            
            # Sort numerically
            gpu_ids = sorted(list(gpu_ids_set), key=int)
            
            if not gpu_ids:
                # Alternative: count from rocminfo
                result2 = subprocess.run(
                    ['rocminfo'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result2.returncode == 0:
                    # Filter to GPU agents only
                    gpu_count = result2.stdout.lower().count('type: gpu')
                    if gpu_count > 0:
                        gpu_ids = [str(i) for i in range(gpu_count)]
            
            if gpu_ids:
                print(f"[fkit] GPU detection: rocm-smi found {len(gpu_ids)} GPUs")
                return GPUInfo(count=len(gpu_ids), vendor='amd', ids=gpu_ids)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try NVIDIA
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpu_ids = []
            for line in result.stdout.split('\n'):
                if 'GPU ' in line:
                    match = re.search(r'GPU (\d+):', line)
                    if match:
                        gpu_ids.append(match.group(1))
            if gpu_ids:
                print(f"[fkit] GPU detection: nvidia-smi found {len(gpu_ids)} GPUs")
                return GPUInfo(count=len(gpu_ids), vendor='nvidia', ids=gpu_ids)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return GPUInfo(count=0, vendor='none', ids=[])


def allocate_gpus_to_workers(gpu_info: GPUInfo, num_workers: int, gpus_per_worker: int) -> List[str]:
    """
    Allocate GPUs to workers.
    
    Returns a list of GPU ID strings for CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES.
    Example: ["0,1", "2,3", "4,5", "6,7"] for 4 workers with 2 GPUs each.
    """
    if gpu_info.count == 0:
        # No GPUs - all workers share empty allocation
        return [""] * num_workers
    
    allocations = []
    gpu_ids = gpu_info.ids
    
    for worker_idx in range(num_workers):
        start_idx = worker_idx * gpus_per_worker
        end_idx = start_idx + gpus_per_worker
        
        if end_idx <= len(gpu_ids):
            # Assign specific GPUs to this worker
            worker_gpus = gpu_ids[start_idx:end_idx]
            allocations.append(",".join(worker_gpus))
        else:
            # Not enough GPUs - wrap around for load balancing
            worker_gpus = []
            for i in range(gpus_per_worker):
                idx = (start_idx + i) % len(gpu_ids)
                worker_gpus.append(gpu_ids[idx])
            allocations.append(",".join(worker_gpus))
    
    return allocations


class WorkerState(Enum):
    """State of a worker."""
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TestResult:
    """Result from running a test in subprocess."""
    nodeid: str
    outcome: str  # 'passed', 'failed', 'skipped'
    duration: float
    longrepr: Optional[str] = None
    crash: bool = False
    timeout: bool = False
    skip_reason: Optional[str] = None
    worker_id: int = 0
    gpu_error: bool = False  # New: indicates GPU-related failure


@dataclass
class WorkItem:
    """A test to be executed."""
    nodeid: str
    item: object  # pytest item
    retry_count: int = 0
    max_retries: int = 3  # Allow retries on transient errors (GPU unavailable, DNS, network)


# ---------------------------------------------------------------------------
# Transient error detection for retry logic
# ---------------------------------------------------------------------------

# Patterns that indicate a transient/retryable error (not a real test failure)
TRANSIENT_ERROR_PATTERNS = [
    # GPU availability (transient when running across workers with shared GPUs)
    'No HIP GPUs are available',
    'No CUDA GPUs are available',
    'RuntimeError: No HIP GPUs',
    'RuntimeError: No CUDA GPUs',
    'hipErrorNoDevice',
    'cudaErrorNoDevice',
    # Network / DNS (transient infrastructure issues)
    'Temporary failure in name resolution',
    'Name or service not known',
    'ConnectError',
    'ConnectionError',
    'Connection refused',
    'Connection reset by peer',
    'Connection timed out',
    'OSError: [Errno -3]',  # DNS resolution failure
    'OSError: [Errno -2]',  # DNS resolution failure
    'OSError: [Errno 101]',  # Network unreachable
    'OSError: [Errno 110]',  # Connection timed out
    'OSError: [Errno 111]',  # Connection refused
    # HuggingFace Hub transient errors
    'requests.exceptions.ConnectionError',
    'httpx.ConnectError',
    'httpx.ReadTimeout',
    'huggingface_hub.errors.HfHubHTTPError',
    'HTTP Error 5',  # 500, 502, 503, 504
    'Server Error',
    '502 Bad Gateway',
    '503 Service Unavailable',
    '504 Gateway Timeout',
    # NCCL transient errors (often recoverable on retry with different worker)
    'NCCL Error 2: unhandled system error',
    'NCCL error',
    # GPU memory (may succeed on a different worker or after GC)
    'CUDA out of memory',
    'hipErrorOutOfMemory',
    'HIP out of memory',
]


def _is_transient_error(result_or_text, stderr: str = "") -> bool:
    """Detect if a failure was due to a transient/retryable issue.
    
    This catches GPU availability, DNS resolution, network connectivity,
    HuggingFace Hub errors, and NCCL errors that may succeed on retry
    (especially when retried on a different worker with different GPUs).
    
    Args:
        result_or_text: Either a TestResult object or a string to check
        stderr: Additional stderr text to check
    
    Returns:
        True if the error appears transient and the test should be retried
    """
    if isinstance(result_or_text, TestResult):
        check_text = (result_or_text.longrepr or "") + stderr
    else:
        check_text = str(result_or_text) + stderr
    
    check_lower = check_text.lower()
    return any(pattern.lower() in check_lower for pattern in TRANSIENT_ERROR_PATTERNS)


def _reset_gpu(gpu_env: Dict[str, str], gpu_vendor: str) -> bool:
    """Attempt to reset the GPU after a crash.

    After a process crash (SIGABRT/SIGSEGV), the GPU driver may hold stale
    state from the dead process.  An explicit reset via ``rocm-smi --gpureset``
    or ``nvidia-smi --gpu-reset`` tells the driver to clean up immediately
    rather than waiting for a timeout.

    This is best-effort: not all environments allow resets (e.g. VMs,
    containers without --privileged).  A failed reset doesn't mean the GPU
    is dead -- the health probe that follows will determine that.

    Returns True if reset command succeeded (or was not needed), False on error.
    """
    gpu_ids_str = gpu_env.get('FKIT_GPU_IDS', '')
    if not gpu_ids_str:
        return True  # No GPU allocated, nothing to reset

    # Use the *physical* GPU IDs for the reset command
    # (ROCR_VISIBLE_DEVICES for AMD, CUDA_VISIBLE_DEVICES for NVIDIA)
    if gpu_vendor == 'amd':
        physical_ids = gpu_env.get('ROCR_VISIBLE_DEVICES', gpu_ids_str)
    else:
        physical_ids = gpu_env.get('CUDA_VISIBLE_DEVICES', gpu_ids_str)

    ok = True
    for gpu_id in physical_ids.split(','):
        gpu_id = gpu_id.strip()
        if not gpu_id:
            continue
        try:
            if gpu_vendor == 'amd':
                cmd = ['rocm-smi', '--gpureset', '-d', gpu_id]
            elif gpu_vendor == 'nvidia':
                cmd = ['nvidia-smi', '--gpu-reset', '-i', gpu_id]
            else:
                continue
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                print(f"      GPU reset succeeded for device {gpu_id}")
            else:
                # Non-fatal: reset may not be available in this environment
                print(f"      GPU reset returned {r.returncode} for device {gpu_id} "
                      f"(may require --privileged or root)")
                ok = False
        except FileNotFoundError:
            pass  # Tool not installed -- skip silently
        except subprocess.TimeoutExpired:
            print(f"      GPU reset timed out for device {gpu_id}")
            ok = False
        except Exception as e:
            print(f"      GPU reset error for device {gpu_id}: {e}")
            ok = False
    return ok


def _extract_failure_from_output(stdout: str, stderr: str) -> str:
    """Extract the meaningful failure message from pytest subprocess output.
    
    Instead of dumping the entire raw stdout/stderr (which includes the pytest
    session header, progress bars, etc.), this extracts just the failure section.
    Falls back to truncated raw output if extraction fails.
    """
    # Try to find the FAILURES section in pytest output
    fail_sections = []
    
    # Pattern 1: pytest's FAILURES section
    if '= FAILURES =' in stdout or '_ FAILURES _' in stdout:
        # Extract from FAILURES marker to the next section marker or end
        lines = stdout.split('\n')
        in_failures = False
        for line in lines:
            if '= FAILURES =' in line or '_ FAILURES _' in line:
                in_failures = True
                continue
            if in_failures:
                # Stop at the next section header (=== ... ===)
                if re.match(r'^={3,}\s+.+\s+={3,}$', line.strip()):
                    break
                fail_sections.append(line)
    
    # Pattern 2: pytest's short test summary
    if not fail_sections and '= short test summary info =' in stdout:
        lines = stdout.split('\n')
        in_summary = False
        for line in lines:
            if '= short test summary info =' in line:
                in_summary = True
                continue
            if in_summary:
                if re.match(r'^={3,}\s+.+\s+={3,}$', line.strip()):
                    break
                fail_sections.append(line)
    
    # Pattern 3: Look for FAILED lines with assertion info
    if not fail_sections:
        lines = stdout.split('\n')
        for i, line in enumerate(lines):
            if 'FAILED' in line or 'AssertionError' in line or 'Error' in line:
                # Grab context: a few lines before and after
                start = max(0, i - 5)
                end = min(len(lines), i + 10)
                fail_sections = lines[start:end]
                break
    
    if fail_sections:
        failure_text = '\n'.join(fail_sections).strip()
        # Add stderr if it has useful content (filter out common noise)
        stderr_useful = _filter_stderr(stderr)
        if stderr_useful:
            failure_text += f"\n\n--- STDERR ---\n{stderr_useful}"
        return failure_text
    
    # Fallback: truncated raw output
    stdout_tail = stdout[-4000:] if len(stdout) > 4000 else stdout
    stderr_tail = _filter_stderr(stderr)
    parts = []
    if stdout_tail.strip():
        parts.append(f"--- STDOUT ---\n{stdout_tail}")
    if stderr_tail:
        parts.append(f"--- STDERR ---\n{stderr_tail}")
    return '\n\n'.join(parts) if parts else "(no output captured)"


def _filter_stderr(stderr: str) -> str:
    """Filter stderr to remove common noisy lines, keep useful content."""
    if not stderr or not stderr.strip():
        return ""
    
    # Lines to filter out (common pytest/torch noise)
    noise_patterns = [
        'UserWarning:',
        'warnings.warn(',
        'FutureWarning:',
        'DeprecationWarning:',
        'from .compat import',
        'PytestUnraisableExceptionWarning',
    ]
    
    lines = stderr.split('\n')
    filtered = []
    for line in lines:
        if any(pattern in line for pattern in noise_patterns):
            continue
        if line.strip():
            filtered.append(line)
    
    result = '\n'.join(filtered).strip()
    # Truncate if still too long
    if len(result) > 2000:
        result = result[-2000:]
    return result


def _gpu_health_probe(gpu_env: Dict[str, str], timeout: int = 15) -> bool:
    """Probe GPU health by spawning a subprocess that actually allocates a tensor.

    After a crash (SIGABRT/SIGSEGV), the ROCm/CUDA driver may need time to
    reclaim the crashed process's GPU context.  ``torch.cuda.is_available()``
    can return True before the device is actually usable, so we go further and
    attempt a small tensor allocation + synchronization.

    Returns True if the GPU is healthy, False otherwise.
    """
    probe_script = (
        "import sys\n"
        "try:\n"
        "    import torch\n"
        "    if not torch.cuda.is_available():\n"
        "        print('probe:unavailable'); sys.exit(1)\n"
        "    n = torch.cuda.device_count()\n"
        "    if n == 0:\n"
        "        print('probe:no_devices'); sys.exit(1)\n"
        "    # Actually allocate a tensor and sync – this catches driver-level\n"
        "    # failures that is_available() misses.\n"
        "    t = torch.zeros(64, device='cuda:0')\n"
        "    torch.cuda.synchronize()\n"
        "    del t\n"
        "    print(f'probe:ok devices={n}')\n"
        "    sys.exit(0)\n"
        "except Exception as e:\n"
        "    print(f'probe:error {e}')\n"
        "    sys.exit(1)\n"
    )
    env = os.environ.copy()
    env.update(gpu_env)
    try:
        r = subprocess.run(
            [sys.executable, '-c', probe_script],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        output = (r.stdout.strip().split('\n') or [''])[-1]
        if r.returncode == 0 and 'probe:ok' in output:
            return True
        return False
    except Exception:
        return False


def slice_tests_to_workers(items: List, num_workers: int) -> List[List]:
    """
    Distribute tests across workers using round-robin for balance.
    
    This ensures:
    1. Deterministic distribution (same tests always go to same worker)
    2. Even distribution regardless of test count
    3. Sorted by nodeid for reproducibility
    
    Args:
        items: List of pytest items to distribute
        num_workers: Number of workers
        
    Returns:
        List of lists, where slices[i] contains items for worker i
    """
    if num_workers <= 1:
        return [items]
    
    # Sort by nodeid for deterministic distribution
    sorted_items = sorted(items, key=lambda x: x.nodeid)
    
    # Round-robin distribution
    slices = [[] for _ in range(num_workers)]
    for i, item in enumerate(sorted_items):
        worker_idx = i % num_workers
        slices[worker_idx].append(item)
    
    return slices


# =========================================================================
# WorkScheduler — Online-Thr-Restarts scheduling model
# =========================================================================

class WorkScheduler:
    """Capacity-aware work scheduler — no drops, no duplicates.

    Correctness invariant
    ---------------------
    Every submitted test is in **exactly one** of three states at all times:

        QUEUED      – sitting in the work queue
        IN_FLIGHT   – checked out by a worker (between get_work / report_result)
        RESOLVED    – terminal result emitted to pytest (exactly once)

    State transitions::

        submit()        →  QUEUED
        get_work()      →  QUEUED  ➜  IN_FLIGHT
        report_result() →  IN_FLIGHT  ➜  RESOLVED   (pass / fail / skip / max-crash)
                        →  IN_FLIGHT  ➜  QUEUED     (crash, restarts remaining)
        release_work()  →  IN_FLIGHT  ➜  QUEUED     (worker died before reporting)
        drain_unresolved() → QUEUED/IN_FLIGHT  ➜  RESOLVED  (all workers dead)

    No test can ever leave the system without reaching RESOLVED.
    No test can be emitted to pytest more than once (_resolved set guard).

    Crash-restart model
    -------------------
    When a test crashes, it is moved back to QUEUED (up to ``max_crashes_per_test``
    times).  The scheduler tracks which workers crashed which tests so workers
    can be avoided, but it will still assign to the same worker as a fallback
    if no other worker is alive.

    When all workers are dead and tests remain, ``drain_unresolved()`` reports
    them as errors so pytest sees every test exactly once.
    """

    def __init__(self, num_workers: int, result_callback: Callable,
                 max_crashes_per_test: int = 2):
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._result_callback = result_callback

        # --- Capacity tracking ---
        self._worker_health: Dict[int, str] = {}
        for i in range(num_workers):
            self._worker_health[i] = 'healthy'

        # --- Restart bookkeeping ---
        self._max_crashes = max_crashes_per_test
        self._crash_counts: Dict[str, int] = {}
        self._crash_workers: Dict[str, set] = {}

        # --- State tracking (the invariant) ---
        self._in_flight: Dict[int, object] = {}   # worker_id → item
        self._resolved: set = set()                # nodeids already emitted
        self._total_submitted = 0
        self._total_resolved = 0
        self._all_submitted = threading.Event()
        self._all_done = threading.Event()
        self._shutdown = threading.Event()

        # --- Stats ---
        self._stats = {
            'restarts': 0,
            'permanent_crashes': 0,
        }

    # ----- submit -----

    def submit(self, items: List):
        """QUEUED ← new tests."""
        with self._lock:
            self._total_submitted += len(items)
        for item in items:
            self._queue.put(item)
        self._all_submitted.set()

    # ----- get / return work -----

    def get_work(self, worker_id: int, timeout: float = 0.5):
        """QUEUED → IN_FLIGHT.  Returns item or None."""
        with self._lock:
            if self._worker_health.get(worker_id) != 'healthy':
                return None
            if self._shutdown.is_set():
                return None

        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

        with self._lock:
            self._in_flight[worker_id] = item
        return item

    def release_work(self, worker_id: int):
        """IN_FLIGHT → QUEUED (worker died without reporting).

        Called automatically by ``set_worker_health(…, 'dead')`` and by
        the worker loop's finally-block.  Safe to call when there is
        nothing in-flight for this worker.
        """
        with self._lock:
            item = self._in_flight.pop(worker_id, None)
        if item is not None:
            nodeid = item.nodeid if hasattr(item, 'nodeid') else str(item)
            print(f"   🔄 Scheduler: releasing in-flight test {nodeid} "
                  f"from worker {worker_id} back to queue")
            self._queue.put(item)

    # ----- report results -----

    def report_result(self, item, result: 'TestResult', worker_id: int) -> bool:
        """IN_FLIGHT → RESOLVED  or  IN_FLIGHT → QUEUED (crash restart).

        Returns True if the test was re-queued (not yet terminal).
        """
        # Remove from in-flight first (atomically)
        with self._lock:
            self._in_flight.pop(worker_id, None)

        if result.crash:
            return self._handle_crash(item, result, worker_id)

        # Terminal non-crash result
        self._emit(item, result)
        return False

    # ----- capacity management -----

    def set_worker_health(self, worker_id: int, health: str):
        """Update worker capacity.  If 'dead', release any in-flight work."""
        with self._lock:
            old = self._worker_health.get(worker_id)
            self._worker_health[worker_id] = health

        if health == 'dead' and old != 'dead':
            # Release in-flight test back to queue BEFORE logging
            self.release_work(worker_id)
            print(f"   📉 Scheduler: worker {worker_id} capacity → 0 (dead)")
        elif health == 'healthy' and old != 'healthy':
            print(f"   📈 Scheduler: worker {worker_id} capacity restored")

    def get_worker_health(self, worker_id: int) -> str:
        with self._lock:
            return self._worker_health.get(worker_id, 'dead')

    def healthy_worker_count(self) -> int:
        with self._lock:
            return sum(1 for h in self._worker_health.values()
                       if h == 'healthy')

    # ----- lifecycle -----

    def is_done(self) -> bool:
        return self._all_done.is_set()

    def wait(self, timeout=None):
        self._all_done.wait(timeout=timeout)

    def shutdown(self):
        self._shutdown.set()
        self._all_done.set()

    @property
    def stats(self) -> Dict:
        with self._lock:
            return dict(self._stats)

    # ----- drain (emergency: all workers dead) -----

    def drain_unresolved(self):
        """Report all remaining QUEUED + IN_FLIGHT tests as errors.

        Called by ``wait_for_completion`` when every worker has died.
        Guarantees no test is dropped — each gets exactly one callback.
        """
        # 1. Drain in-flight → RESOLVED
        with self._lock:
            stranded = dict(self._in_flight)
            self._in_flight.clear()

        for wid, item in stranded.items():
            nodeid = item.nodeid if hasattr(item, 'nodeid') else str(item)
            self._emit(item, TestResult(
                nodeid=nodeid, outcome='failed', duration=0,
                longrepr=(f"Worker {wid} died with this test in-flight; "
                          f"no healthy workers remain"),
            ))

        # 2. Drain queue → RESOLVED
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            nodeid = item.nodeid if hasattr(item, 'nodeid') else str(item)
            self._emit(item, TestResult(
                nodeid=nodeid, outcome='failed', duration=0,
                longrepr="All workers died; test was never executed",
            ))

    # ----- internals -----

    def _handle_crash(self, item, result: 'TestResult',
                      worker_id: int) -> bool:
        """IN_FLIGHT → QUEUED (restart) or IN_FLIGHT → RESOLVED (give up)."""
        nodeid = item.nodeid if hasattr(item, 'nodeid') else str(item)

        with self._lock:
            self._crash_counts[nodeid] = self._crash_counts.get(nodeid, 0) + 1
            self._crash_workers.setdefault(nodeid, set()).add(worker_id)
            count = self._crash_counts[nodeid]
            healthy = sum(1 for h in self._worker_health.values()
                          if h == 'healthy')

        if count > self._max_crashes or healthy == 0:
            with self._lock:
                self._stats['permanent_crashes'] += 1
            self._emit(item, result)
            return False

        with self._lock:
            self._stats['restarts'] += 1
        print(f"   🔁 Scheduler: re-queuing {nodeid} after crash "
              f"(attempt {count}/{self._max_crashes}, "
              f"worker {worker_id} crashed it)")
        self._queue.put(item)
        return True

    def _emit(self, item, result: 'TestResult'):
        """RESOLVED ← item.  Emits callback exactly once per nodeid."""
        nodeid = item.nodeid if hasattr(item, 'nodeid') else str(item)
        with self._lock:
            if nodeid in self._resolved:
                return  # Already emitted — guard against duplicates
            self._resolved.add(nodeid)
            self._total_resolved += 1

        if self._result_callback:
            self._result_callback(item, result)
        self._check_done()

    def _check_done(self):
        with self._lock:
            if (self._all_submitted.is_set()
                    and self._total_resolved >= self._total_submitted
                    and self._queue.empty()
                    and not self._in_flight):
                self._all_done.set()


class _ScheduledWorkerPool:
    """Base worker pool using WorkScheduler for capacity-aware scheduling.

    Implements the **Online-Thr-Restarts** model:

    * All tests go into a shared ``WorkScheduler`` queue — no static
      pre-assignment (which the scheduling paper proves has arbitrarily
      bad competitive ratio under adversarial crashes).
    * Workers pull tests on-demand; crashed tests are **re-queued** to
      healthy workers (Restarts model → 1/2-competitive throughput).
    * Each worker independently manages GPU recovery.  The scheduler
      tracks which workers are alive (capacity profile *c(t)*) and
      stops routing work to dead workers.

    Subclasses ``DynamicWorkerPool`` and ``SlicedWorkerPool`` provide
    backward-compatible constructors for the two ``--fkit-mode`` values.
    They share this implementation entirely.
    """

    def __init__(self, num_workers: int, gpu_allocations: List[str],
                 gpu_vendor: str, timeout: int, result_callback: Callable,
                 threads_per_worker: int = 4, max_retries: int = 3):
        self.num_workers = num_workers
        self.gpu_allocations = gpu_allocations
        self.gpu_vendor = gpu_vendor
        self.timeout = timeout
        self.threads_per_worker = threads_per_worker
        self._max_retries = max_retries

        # --- Central scheduler (paper §: Online-Thr-Restarts) ---
        self.scheduler = WorkScheduler(
            num_workers=num_workers,
            result_callback=result_callback,
            max_crashes_per_test=2,
        )

        # Worker state tracking
        self._worker_states = {i: WorkerState.IDLE for i in range(num_workers)}

        # Execution-level statistics (counts every attempt, not just terminals)
        self._lock = threading.Lock()
        self._stats = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'crashes': 0,
            'timeouts': 0,
            'gpu_errors': 0,
            'retries': 0,
            'workers_failed': 0,
        }

        # Control
        self._shutdown = threading.Event()
        self._workers: List[threading.Thread] = []

    # ------------------------------------------------------------------
    # GPU environment (identical for both modes)
    # ------------------------------------------------------------------

    def _get_gpu_env_vars(self, worker_id: int) -> Dict[str, str]:
        """Get GPU, CPU, and NCCL environment variables for a worker.

        AMD/ROCm GPU visibility layering:
          ROCR_VISIBLE_DEVICES  – physical GPU indices (ROCm runtime level).
          HIP_VISIBLE_DEVICES   – indices *relative to* the ROCR-visible set.
          CUDA_VISIBLE_DEVICES  – PyTorch ROCm maps this to HIP_VISIBLE_DEVICES.

        If a worker owns physical GPU 3, we set:
          ROCR_VISIBLE_DEVICES=3   (expose only that physical GPU)
          HIP_VISIBLE_DEVICES=0    (index 0 within that single-GPU ROCR set)
          CUDA_VISIBLE_DEVICES=0   (same, for PyTorch compat)

        NCCL/RCCL per-worker isolation:
          Each worker gets a unique MASTER_PORT so that DataParallel /
          DistributedDataParallel tests in different workers don't collide.
        """
        gpu_ids = self.gpu_allocations[worker_id] if worker_id < len(self.gpu_allocations) else ""

        env_vars = {
            'OMP_NUM_THREADS': str(self.threads_per_worker),
            'MKL_NUM_THREADS': str(self.threads_per_worker),
            'NUMEXPR_NUM_THREADS': str(self.threads_per_worker),
            'OPENBLAS_NUM_THREADS': str(self.threads_per_worker),
            'VECLIB_MAXIMUM_THREADS': str(self.threads_per_worker),
            'TORCH_NUM_THREADS': str(self.threads_per_worker),
            'FKIT_WORKER_ID': str(worker_id),
            'FKIT_THREADS': str(self.threads_per_worker),
            'MASTER_PORT': str(NCCL_BASE_PORT + worker_id),
            'NCCL_ASYNC_ERROR_HANDLING': '1',
            'NCCL_SOCKET_IFNAME': 'lo',
        }

        if gpu_ids:
            if self.gpu_vendor == 'amd':
                env_vars['ROCR_VISIBLE_DEVICES'] = gpu_ids
                num_gpus = len(gpu_ids.split(','))
                hip_ids = ','.join(str(i) for i in range(num_gpus))
                env_vars['HIP_VISIBLE_DEVICES'] = hip_ids
                env_vars['CUDA_VISIBLE_DEVICES'] = hip_ids
            elif self.gpu_vendor == 'nvidia':
                env_vars['CUDA_VISIBLE_DEVICES'] = gpu_ids
            env_vars['FKIT_WORKER_ID'] = str(worker_id)
            env_vars['FKIT_GPU_IDS'] = gpu_ids

        return env_vars

    # ------------------------------------------------------------------
    # Worker loop (unified — implements the paper's greedy algorithm)
    # ------------------------------------------------------------------

    def _worker_loop(self, worker_id: int):
        """Worker loop with guaranteed no-drop / no-duplicate semantics.

        Every test that is checked out from the scheduler via ``get_work``
        is **guaranteed** to be returned — either through ``report_result``
        (normal path) or ``release_work`` (exception / finally path).

        Flow::

            get_work   → item moves QUEUED → IN_FLIGHT
            try:
                run test
                report_result → item moves IN_FLIGHT → RESOLVED or QUEUED
            finally:
                release_work  → if still IN_FLIGHT, moves back to QUEUED
        """
        gpu_env = self._get_gpu_env_vars(worker_id)
        gpu_str = gpu_env.get('FKIT_GPU_IDS', 'N/A')
        max_consecutive_crashes = 3
        consecutive_crashes = 0

        with self._lock:
            self._worker_states[worker_id] = WorkerState.RUNNING

        while not self._shutdown.is_set():
            # --- Pull work from scheduler (QUEUED → IN_FLIGHT) ---
            item = self.scheduler.get_work(worker_id)
            if item is None:
                if self.scheduler.is_done() or self._shutdown.is_set():
                    break
                time.sleep(0.1)
                continue

            # --- Execute with finally-guard ---
            # If anything between get_work and report_result throws,
            # the finally block releases the test back to the queue.
            reported = False
            try:
                result = self._run_test(item.nodeid, worker_id, gpu_env)

                # Retry transient (non-crash) errors in-place
                retry_count = 0
                while (result.outcome == 'failed'
                       and not result.crash
                       and _is_transient_error(result)
                       and retry_count < self._max_retries):
                    retry_count += 1
                    with self._lock:
                        self._stats['retries'] += 1
                    print(f"   🔄 Worker {worker_id}: retrying {item.nodeid} "
                          f"(attempt {retry_count + 1}/{self._max_retries + 1}, "
                          f"transient error)")
                    time.sleep(min(2 ** (retry_count - 1), 4))
                    result = self._run_test(item.nodeid, worker_id, gpu_env)

                # Execution-level stats
                with self._lock:
                    self._stats['tests_run'] += 1
                    if result.outcome == 'passed':
                        self._stats['tests_passed'] += 1
                    elif result.outcome == 'skipped':
                        self._stats['tests_skipped'] += 1
                    else:
                        self._stats['tests_failed'] += 1
                    if result.crash:
                        self._stats['crashes'] += 1
                    if result.timeout:
                        self._stats['timeouts'] += 1
                    if (result.outcome == 'failed'
                            and not result.crash
                            and _is_transient_error(result)):
                        self._stats['gpu_errors'] += 1

                # Report (IN_FLIGHT → RESOLVED or IN_FLIGHT → QUEUED)
                self.scheduler.report_result(item, result, worker_id)
                reported = True

                # --- Crash recovery ---
                if result.crash:
                    consecutive_crashes += 1

                    if consecutive_crashes > max_consecutive_crashes:
                        print(f"   ❌ Worker {worker_id} (GPUs: {gpu_str}): "
                              f"{consecutive_crashes} consecutive crashes, "
                              f"disabling (capacity → 0)")
                        self.scheduler.set_worker_health(worker_id, 'dead')
                        with self._lock:
                            self._worker_states[worker_id] = WorkerState.FAILED
                            self._stats['workers_failed'] += 1
                        break

                    self.scheduler.set_worker_health(worker_id, 'recovering')
                    print(f"   🔧 Worker {worker_id} (GPUs: {gpu_str}): "
                          f"crash recovery {consecutive_crashes}/"
                          f"{max_consecutive_crashes}")

                    _reset_gpu(gpu_env, self.gpu_vendor)
                    time.sleep(5)

                    if _gpu_health_probe(gpu_env):
                        self.scheduler.set_worker_health(worker_id, 'healthy')
                        print(f"   ✓  Worker {worker_id}: GPU healthy, "
                              f"resuming")
                        continue

                    print(f"   ⚠️  Worker {worker_id}: GPU unhealthy, "
                          f"extended wait (10s)...")
                    time.sleep(10)

                    if _gpu_health_probe(gpu_env):
                        self.scheduler.set_worker_health(worker_id, 'healthy')
                        print(f"   ✓  Worker {worker_id}: GPU recovered")
                        continue

                    print(f"   ❌ Worker {worker_id} (GPUs: {gpu_str}): "
                          f"GPU unrecoverable, disabling")
                    self.scheduler.set_worker_health(worker_id, 'dead')
                    with self._lock:
                        self._worker_states[worker_id] = WorkerState.FAILED
                        self._stats['workers_failed'] += 1
                    break
                else:
                    consecutive_crashes = 0

            except Exception as e:
                print(f"\n❌ Worker {worker_id} exception: {e}")
                self.scheduler.set_worker_health(worker_id, 'dead')
                with self._lock:
                    self._worker_states[worker_id] = WorkerState.FAILED
                    self._stats['workers_failed'] += 1
                break

            finally:
                # Safety net: if report_result was never called, the test
                # is still IN_FLIGHT.  Release it back to QUEUED so another
                # worker (or drain_unresolved) can handle it.
                if not reported:
                    self.scheduler.release_work(worker_id)

        # Clean exit
        with self._lock:
            if self._worker_states[worker_id] != WorkerState.FAILED:
                self._worker_states[worker_id] = WorkerState.STOPPED

    # ------------------------------------------------------------------
    # Test execution (subprocess isolation)
    # ------------------------------------------------------------------

    def _run_test(self, nodeid: str, worker_id: int,
                  gpu_env: Dict[str, str]) -> TestResult:
        """Run a single test in an isolated subprocess."""
        junit_fd, junit_path = tempfile.mkstemp(
            suffix='.xml', prefix=f'fkit_w{worker_id}_')
        os.close(junit_fd)

        try:
            start_time = time.time()

            env = os.environ.copy()
            env.update(gpu_env)
            env.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
            env.setdefault('MASTER_ADDR', '127.0.0.1')
            env.setdefault('MASTER_PORT', str(NCCL_BASE_PORT + worker_id))

            for var in ('HF_TOKEN', 'RUN_SLOW', 'NCCL_DEBUG',
                        'PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH',
                        'TRANSFORMERS_VERBOSITY', 'TRANSFORMERS_CACHE'):
                if var in os.environ:
                    env[var] = os.environ[var]

            pytest_cmd = [
                sys.executable, '-m', 'pytest',
                nodeid, '-v', '--tb=short',
                '--continue-on-collection-errors',
                '-p', 'no:cacheprovider',
                '-p', 'no:fkit',
                f'--junitxml={junit_path}',
            ]

            try:
                proc = subprocess.run(
                    pytest_cmd, capture_output=True, text=True,
                    timeout=self.timeout, cwd=str(Path.cwd()), env=env,
                )

                duration = time.time() - start_time
                outcome, skip_reason, fail_message = \
                    self._parse_junit_result(junit_path)

                if proc.returncode == 0:
                    if outcome == 'skipped':
                        return TestResult(
                            nodeid=nodeid, outcome='skipped',
                            duration=duration, skip_reason=skip_reason,
                            worker_id=worker_id)
                    return TestResult(
                        nodeid=nodeid, outcome='passed',
                        duration=duration, worker_id=worker_id)

                if proc.returncode < 0:
                    sig = -proc.returncode
                    sig_names = {
                        signal.SIGABRT: "SIGABRT (Aborted)",
                        signal.SIGSEGV: "SIGSEGV (Segmentation Fault)",
                        signal.SIGTERM: "SIGTERM (Terminated)",
                        signal.SIGKILL: "SIGKILL (Killed)",
                    }
                    sig_name = sig_names.get(sig, f"Signal {sig}")
                    crash_info = (
                        f"\n{'='*70}\n"
                        f"💥 TEST CRASHED: {sig_name} "
                        f"(Worker {worker_id}, GPUs: "
                        f"{gpu_env.get('FKIT_GPU_IDS', 'N/A')})\n"
                        f"{'='*70}\n"
                        f"\nThis test caused Python to crash with "
                        f"{sig_name}.\npytest-fkit caught it and "
                        f"converted it to an ERROR.\n"
                        f"\n--- STDOUT ---\n"
                        f"{proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout}\n"
                        f"\n--- STDERR ---\n"
                        f"{proc.stderr[-4000:] if len(proc.stderr) > 4000 else proc.stderr}\n"
                        f"{'='*70}\n"
                    )
                    return TestResult(
                        nodeid=nodeid, outcome='failed',
                        duration=duration, longrepr=crash_info,
                        crash=True, worker_id=worker_id)

                # Normal failure
                fail_info = (fail_message if fail_message
                             else _extract_failure_from_output(
                                 proc.stdout, proc.stderr))
                return TestResult(
                    nodeid=nodeid, outcome='failed',
                    duration=duration, longrepr=fail_info,
                    worker_id=worker_id)

            except subprocess.TimeoutExpired as e:
                duration = time.time() - start_time
                timeout_info = (
                    f"\n{'='*70}\n"
                    f"⏱️  TEST TIMEOUT (Worker {worker_id})\n"
                    f"{'='*70}\n"
                    f"\nTest exceeded timeout of {self.timeout}s.\n"
                    f"pytest-fkit terminated it and converted it to an "
                    f"ERROR.\n"
                    f"\n--- PARTIAL STDOUT ---\n"
                    f"{e.stdout if e.stdout else '(none)'}\n"
                    f"\n--- PARTIAL STDERR ---\n"
                    f"{e.stderr if e.stderr else '(none)'}\n"
                    f"{'='*70}\n"
                )
                return TestResult(
                    nodeid=nodeid, outcome='failed',
                    duration=duration, longrepr=timeout_info,
                    timeout=True, worker_id=worker_id)
        finally:
            try:
                os.unlink(junit_path)
            except Exception:
                pass

    def _parse_junit_result(self, junit_path: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Parse JUnit XML for outcome, skip reason, and failure message."""
        try:
            if not os.path.exists(junit_path):
                return 'unknown', None, None
            tree = ET.parse(junit_path)
            root = tree.getroot()
            for tc in root.findall('.//testcase'):
                sk = tc.find('skipped')
                if sk is not None:
                    return ('skipped',
                            sk.get('message') or sk.text or "Skipped",
                            None)
                fa = tc.find('failure')
                if fa is not None:
                    return ('failed', None,
                            f"{fa.get('message', '')}\n{fa.text or ''}".strip())
                er = tc.find('error')
                if er is not None:
                    return ('failed', None,
                            f"{er.get('message', '')}\n{er.text or ''}".strip())
                return 'passed', None, None
            return 'unknown', None, None
        except Exception:
            return 'unknown', None, None

    # ------------------------------------------------------------------
    # Public API (same interface for both modes)
    # ------------------------------------------------------------------

    def submit_tests(self, items: List):
        """Submit all tests into the scheduler's shared work queue."""
        print(f"\n📊 Submitting {len(items)} tests to shared scheduler "
              f"({self.num_workers} workers)")
        self.scheduler.submit(items)

    def start(self):
        """Start all worker threads."""
        for wid in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop, args=(wid,),
                name=f"fkit-worker-{wid}", daemon=True)
            self._workers.append(t)
            t.start()

    def wait_for_completion(self):
        """Wait for all tests to reach a terminal state.

        If all workers die before all tests are resolved, calls
        ``drain_unresolved()`` to report remaining tests as errors.
        This guarantees **no test is dropped** — every submitted test
        gets exactly one result callback.
        """
        while not self.scheduler.is_done() and not self._shutdown.is_set():
            self.scheduler.wait(timeout=1.0)

            # Check if all workers have exited
            with self._lock:
                alive = any(
                    st not in (WorkerState.FAILED, WorkerState.STOPPED)
                    for st in self._worker_states.values()
                )

            if not alive and not self.scheduler.is_done():
                # All workers dead with tests remaining — drain them
                print("\n⚠️  All workers dead — draining remaining tests "
                      "as errors")
                self.scheduler.drain_unresolved()
                break

        self._shutdown.set()
        for t in self._workers:
            t.join(timeout=5.0)

    def shutdown(self):
        """Force shutdown."""
        self._shutdown.set()
        self.scheduler.shutdown()
        for t in self._workers:
            t.join(timeout=1.0)

    @property
    def stats(self):
        with self._lock:
            s = dict(self._stats)
        # Merge scheduler-level stats
        sched = self.scheduler.stats
        s['restarts'] = sched.get('restarts', 0)
        s['permanent_crashes'] = sched.get('permanent_crashes', 0)
        return s

    @property
    def active_workers(self) -> int:
        with self._lock:
            return sum(1 for st in self._worker_states.values()
                       if st not in (WorkerState.FAILED, WorkerState.STOPPED))


class DynamicWorkerPool(_ScheduledWorkerPool):
    """Dynamic scheduling mode (``--fkit-mode=isolate``).

    All tests go into a shared queue; workers pull on-demand.
    Backward-compatible constructor — delegates entirely to the base.
    """
    pass


class SlicedWorkerPool(_ScheduledWorkerPool):
    """Scheduled mode (``--fkit-mode=batch``).

    Formerly used static pre-slicing (which has arbitrarily bad
    competitive ratio under crashes — see scheduling paper §: NoRestarts).
    Now uses the same shared-queue scheduler as DynamicWorkerPool
    (Online-Thr-Restarts → 1/2-competitive).

    The ``slice_tests_to_workers`` function is retained for *display*
    purposes only (showing the initial round-robin distribution).
    """

    def submit_tests(self, items: List):
        """Submit tests and show the initial round-robin distribution."""
        slices = slice_tests_to_workers(items, self.num_workers)
        print(f"\n📊 Initial round-robin distribution "
              f"({self.num_workers} workers, "
              f"shared-queue scheduling):")
        for i, s in enumerate(slices):
            print(f"   Worker {i}: {len(s)} tests (initial)")
        # All tests go into the shared scheduler queue
        self.scheduler.submit(items)


class CrashIsolationPlugin:
    """Plugin that runs tests in subprocess workers to catch crashes."""
    
    def __init__(self, config):
        self.config = config
        self.timeout = config.getoption("--fkit-timeout")
        self.gpus_per_worker = config.getoption("--fkit-gpus-per-worker")
        self.execution_mode = config.getoption("--fkit-mode")
        self.max_retries = config.getoption("--fkit-max-retries")
        threads_per_worker_opt = config.getoption("--fkit-threads-per-worker")
        
        # Parse worker count
        workers_opt = config.getoption("--fkit-workers")
        
        # Detect GPUs and CPUs
        self.gpu_info = detect_gpus()
        self.cpu_info = detect_cpus()
        
        if workers_opt == 'auto':
            # Auto-detect based on GPUs
            if self.gpu_info.count > 0:
                self.num_workers = max(1, self.gpu_info.count // self.gpus_per_worker)
            else:
                # No GPUs - use CPU count
                self.num_workers = max(1, self.cpu_info.total_cores // 2)
        else:
            self.num_workers = max(1, int(workers_opt))
        
        # Calculate threads per worker
        if threads_per_worker_opt == 'auto':
            # Distribute CPU cores evenly across workers
            # Use physical cores if available to avoid hyperthreading contention
            available_cores = self.cpu_info.physical_cores or self.cpu_info.total_cores
            self.threads_per_worker = max(1, available_cores // self.num_workers)
        else:
            self.threads_per_worker = max(1, int(threads_per_worker_opt))
        
        # Allocate GPUs to workers
        self.gpu_allocations = allocate_gpus_to_workers(
            self.gpu_info, 
            self.num_workers, 
            self.gpus_per_worker
        )
        
        # For parallel execution
        self._collected_items = []
        self._item_map = {}
        self._results = {}
        self._results_lock = threading.Lock()
        self._parallel_mode = self.num_workers > 1
        
        # Worker pool (created later with callback)
        self.worker_pool = None
        
        # Determine scheduling mode description
        if self.execution_mode == 'batch':
            scheduling_desc = "sliced scheduling (tests pre-distributed to workers)"
        else:
            scheduling_desc = "dynamic scheduling (tests assigned on-demand)"
        
        # Print configuration
        if self.gpu_info.count > 0:
            print(f"\n🚀 pytest-fkit: {self.num_workers} workers, "
                  f"{self.gpu_info.count} {self.gpu_info.vendor.upper()} GPUs, "
                  f"{self.gpus_per_worker} GPU(s)/worker, "
                  f"{self.threads_per_worker} CPU threads/worker")
            print(f"   GPU allocations (physical): {self.gpu_allocations}")
            if self.gpu_info.vendor == 'amd':
                for w, phys in enumerate(self.gpu_allocations):
                    n = len(phys.split(','))
                    hip = ','.join(str(i) for i in range(n))
                    print(f"   Worker {w}: ROCR_VISIBLE_DEVICES={phys} "
                          f"HIP_VISIBLE_DEVICES={hip} "
                          f"(/dev/kfd + /dev/dri)")
            print(f"   CPU cores: {self.cpu_info.total_cores} total, {self.cpu_info.physical_cores} physical")
            print(f"   Mode: {self.execution_mode} - {scheduling_desc}")
            print(f"   Transient error retries: {self.max_retries} (GPU unavailable, DNS, network, NCCL)")
            print(f"   NCCL ports: {NCCL_BASE_PORT}-{NCCL_BASE_PORT + self.num_workers - 1} "
                  f"(per-worker isolation)")
            print(f"   Crash recovery: GPU health probe + 5-15s cooldown + auto-redistribute")
            # GPU preflight: verify each worker can see its GPU from a subprocess
            self._gpu_preflight()
        else:
            print(f"\n🚀 pytest-fkit: {self.num_workers} workers, "
                  f"{self.threads_per_worker} CPU threads/worker (no GPU detected)")
            print(f"   CPU cores: {self.cpu_info.total_cores} total, {self.cpu_info.physical_cores} physical")
            print(f"   Mode: {self.execution_mode} - {scheduling_desc}")
            print(f"   Transient error retries: {self.max_retries} (GPU unavailable, DNS, network, NCCL)")
    
    def _gpu_preflight(self):
        """Verify each worker can see its GPU from a subprocess.

        Spawns a tiny Python process per worker with the exact same env that
        _run_test would use.  Checks /dev/kfd, /dev/dri/renderD*, and
        torch.cuda.is_available().  Failures here surface the root cause
        immediately instead of manifesting as hundreds of mysterious skips.
        """
        probe_script = (
            "import os, sys, glob\n"
            "kfd = os.path.exists('/dev/kfd')\n"
            "drm = sorted(glob.glob('/dev/dri/renderD*'))\n"
            "hip = os.environ.get('HIP_VISIBLE_DEVICES', '')\n"
            "rocr = os.environ.get('ROCR_VISIBLE_DEVICES', '')\n"
            "cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')\n"
            "try:\n"
            "    import torch\n"
            "    avail = torch.cuda.is_available()\n"
            "    cnt = torch.cuda.device_count() if avail else 0\n"
            "except Exception as e:\n"
            "    avail = False; cnt = 0\n"
            "print(f'kfd={kfd} drm={len(drm)} avail={avail} cnt={cnt} '"
            "      f'ROCR={rocr} HIP={hip} CUDA={cuda}')\n"
            "sys.exit(0 if avail else 1)\n"
        )
        all_ok = True
        for w in range(min(self.num_workers, len(self.gpu_allocations))):
            env = os.environ.copy()
            gpu_env = self.gpu_allocations[w]
            if not gpu_env:
                continue
            # Build env exactly like _run_test would
            if self.gpu_info.vendor == 'amd':
                num = len(gpu_env.split(','))
                hip_ids = ','.join(str(i) for i in range(num))
                env['ROCR_VISIBLE_DEVICES'] = gpu_env
                env['HIP_VISIBLE_DEVICES'] = hip_ids
                env['CUDA_VISIBLE_DEVICES'] = hip_ids
            elif self.gpu_info.vendor == 'nvidia':
                env['CUDA_VISIBLE_DEVICES'] = gpu_env
            try:
                r = subprocess.run(
                    [sys.executable, '-c', probe_script],
                    capture_output=True, text=True, timeout=30, env=env
                )
                line = (r.stdout.strip().split('\n') or [''])[-1]
                if r.returncode != 0:
                    all_ok = False
                    print(f"   ⚠️  Worker {w} GPU PREFLIGHT FAILED: {line}")
                    if r.stderr.strip():
                        for s in r.stderr.strip().split('\n')[-3:]:
                            print(f"       {s}")
                else:
                    print(f"   ✓  Worker {w} GPU OK: {line}")
            except Exception as e:
                all_ok = False
                print(f"   ⚠️  Worker {w} GPU preflight error: {e}")
        if not all_ok:
            print("   ⚠️  Some workers cannot see their GPU – tests may skip "
                  "with 'test requires accelerator'")

    def _result_callback(self, item, result: TestResult):
        """Callback for when a test completes."""
        with self._results_lock:
            self._results[result.nodeid] = result
        self._report_result(item, result)
    
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items):
        """Collect all items for parallel distribution."""
        if self._parallel_mode:
            self._collected_items = list(items)
            self._item_map = {item.nodeid: item for item in items}
    
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session):
        """Override test loop for parallel execution with sliced or dynamic scheduling."""
        if not self._parallel_mode:
            return None
        
        if not self._collected_items:
            return None
        
        # Choose worker pool based on execution mode
        if self.execution_mode == 'batch':
            # Sliced mode: tests are pre-distributed to workers
            print(f"\n🔄 Running {len(self._collected_items)} tests across {self.num_workers} workers "
                  f"(sliced scheduling - each worker gets 1/{self.num_workers} of tests)...\n")
            
            self.worker_pool = SlicedWorkerPool(
                num_workers=self.num_workers,
                gpu_allocations=self.gpu_allocations,
                gpu_vendor=self.gpu_info.vendor,
                timeout=self.timeout,
                result_callback=self._result_callback,
                threads_per_worker=self.threads_per_worker,
                max_retries=self.max_retries,
            )
        else:
            # Dynamic mode: tests are assigned to workers on-demand
            print(f"\n🔄 Running {len(self._collected_items)} tests across {self.num_workers} workers "
                  f"(dynamic scheduling)...\n")
            
            self.worker_pool = DynamicWorkerPool(
                num_workers=self.num_workers,
                gpu_allocations=self.gpu_allocations,
                gpu_vendor=self.gpu_info.vendor,
                timeout=self.timeout,
                result_callback=self._result_callback,
                threads_per_worker=self.threads_per_worker,
                max_retries=self.max_retries,
            )
        
        # Submit all tests (sliced or queued depending on pool type)
        self.worker_pool.submit_tests(self._collected_items)
        
        # Start workers
        self.worker_pool.start()
        
        # Wait for completion
        try:
            self.worker_pool.wait_for_completion()
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted - shutting down workers...")
            self.worker_pool.shutdown()
            raise
        
        # Print summary
        stats = self.worker_pool.stats
        print(f"\n{'='*70}")
        print(f"✅ Completed {stats['tests_run']} test executions")
        print(f"   Passed: {stats['tests_passed']}, Failed: {stats['tests_failed']}, "
              f"Skipped: {stats['tests_skipped']}")
        if stats['crashes'] > 0:
            restarts = stats.get('restarts', 0)
            perm = stats.get('permanent_crashes', 0)
            restart_info = ""
            if restarts:
                restart_info = f" ({restarts} restarted on healthy workers"
                if perm:
                    restart_info += f", {perm} permanently failed"
                restart_info += ")"
            print(f"   💥 Crashes: {stats['crashes']}{restart_info}")
        if stats['timeouts'] > 0:
            print(f"   ⏱️  Timeouts: {stats['timeouts']}")
        if stats['gpu_errors'] > 0:
            retries = stats.get('retries', 0)
            extra = f" (retries: {retries})" if retries else ""
            print(f"   🎮 GPU errors: {stats['gpu_errors']}{extra}")
        if stats.get('workers_failed', 0) > 0:
            print(f"   ⚠️  Workers disabled: {stats['workers_failed']}")
        print(f"{'='*70}")
        
        return True
    
    def _report_result(self, item, result: TestResult):
        """Report a test result to pytest."""
        # Log start
        item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
        
        # Setup report
        setup_report = self._make_report(item, "setup", "passed", duration=0)
        item.ihook.pytest_runtest_logreport(report=setup_report)
        
        # Call report
        if result.outcome == 'skipped':
            skip_location = (str(item.fspath), item.location[1], result.skip_reason or "Skipped")
            call_report = self._make_report(
                item, "call", "skipped",
                longrepr=skip_location,
                duration=result.duration
            )
        elif result.outcome == 'passed':
            call_report = self._make_report(
                item, "call", "passed",
                duration=result.duration
            )
        else:
            call_report = self._make_report(
                item, "call", "failed",
                longrepr=result.longrepr,
                duration=result.duration,
                crash=result.crash,
                timeout=result.timeout
            )
        
        item.ihook.pytest_runtest_logreport(report=call_report)
        
        # Teardown report
        teardown_report = self._make_report(item, "teardown", "passed", duration=0)
        item.ihook.pytest_runtest_logreport(report=teardown_report)
        
        # Log finish
        item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        """Hook for single-worker mode."""
        if self._parallel_mode:
            if item.nodeid in self._results:
                return True
            return None
        
        # Single worker mode
        if item.get_closest_marker("fkit_skip"):
            return None
        
        # Create worker pool for single test if not exists
        if self.worker_pool is None:
            self.worker_pool = DynamicWorkerPool(
                num_workers=1,
                gpu_allocations=self.gpu_allocations,
                gpu_vendor=self.gpu_info.vendor,
                timeout=self.timeout,
                result_callback=lambda i, r: None,  # No-op callback
                max_retries=self.max_retries,
            )
        
        # Run test with retry logic for transient errors
        gpu_env = self.worker_pool._get_gpu_env_vars(0)
        result = self.worker_pool._run_test(item.nodeid, worker_id=0, gpu_env=gpu_env)
        
        retry_count = 0
        while (result.outcome == 'failed'
               and _is_transient_error(result)
               and retry_count < self.max_retries):
            retry_count += 1
            print(f"   🔄 Retrying {item.nodeid} "
                  f"(attempt {retry_count + 1}/{self.max_retries + 1}, transient error)")
            time.sleep(min(2 ** (retry_count - 1), 4))
            result = self.worker_pool._run_test(item.nodeid, worker_id=0, gpu_env=gpu_env)
        
        self._report_result(item, result)
        
        return True
    
    def _make_report(self, item, when, outcome, longrepr=None, duration=0, crash=False, timeout=False):
        """Create a test report compatible with pytest's reporting system."""
        from _pytest.reports import TestReport
        
        report = TestReport(
            nodeid=item.nodeid,
            location=item.location,
            keywords=item.keywords,
            outcome=outcome,
            longrepr=longrepr,
            when=when,
            duration=duration,
            sections=[],
            user_properties=[],
        )
        
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
