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


class DynamicWorkerPool:
    """
    Pool of workers with dynamic work queue scheduling.
    
    Tests are not pre-assigned to workers. Instead:
    1. All tests go into a shared queue
    2. Workers pull tests from the queue as they become available
    3. If a worker encounters a transient error (GPU, DNS, network), the test is retried
    4. Failed workers are marked and work continues with remaining workers
    """
    
    def __init__(self, num_workers: int, gpu_allocations: List[str], 
                 gpu_vendor: str, timeout: int, result_callback: Callable,
                 threads_per_worker: int = 4, max_retries: int = 3):
        self.num_workers = num_workers
        self.gpu_allocations = gpu_allocations
        self.gpu_vendor = gpu_vendor
        self.timeout = timeout
        self.result_callback = result_callback
        self.threads_per_worker = threads_per_worker
        self._max_retries = max_retries
        
        # Work queue - tests waiting to be executed
        self.work_queue = queue.Queue()
        
        # Results queue - completed test results
        self.results_queue = queue.Queue()
        
        # Worker state tracking
        self._worker_states = {i: WorkerState.IDLE for i in range(num_workers)}
        self._worker_error_counts = {i: 0 for i in range(num_workers)}
        self._max_worker_errors = 3  # Disable worker after this many consecutive errors
        
        # Statistics
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
        
        # Control flags
        self._shutdown = threading.Event()
        self._all_submitted = threading.Event()
        
        # Worker threads
        self._workers = []
    
    def _get_gpu_env_vars(self, worker_id: int) -> Dict[str, str]:
        """Get GPU and CPU environment variables for a worker.

        AMD/ROCm GPU visibility layering:
          ROCR_VISIBLE_DEVICES  – physical GPU indices (ROCm runtime level).
          HIP_VISIBLE_DEVICES   – indices *relative to* the ROCR-visible set.
          CUDA_VISIBLE_DEVICES  – PyTorch ROCm maps this to HIP_VISIBLE_DEVICES.

        If a worker owns physical GPU 3, we set:
          ROCR_VISIBLE_DEVICES=3   (expose only that physical GPU)
          HIP_VISIBLE_DEVICES=0    (index 0 within that single-GPU ROCR set)
          CUDA_VISIBLE_DEVICES=0   (same, for PyTorch compat)

        Setting all three to "3" would fail because HIP would look for
        ROCR-index 3 inside a set that only has ROCR-index 0.
        """
        gpu_ids = self.gpu_allocations[worker_id] if worker_id < len(self.gpu_allocations) else ""
        
        env_vars = {
            # CPU thread settings - prevent workers from fighting for cores
            'OMP_NUM_THREADS': str(self.threads_per_worker),
            'MKL_NUM_THREADS': str(self.threads_per_worker),
            'NUMEXPR_NUM_THREADS': str(self.threads_per_worker),
            'OPENBLAS_NUM_THREADS': str(self.threads_per_worker),
            'VECLIB_MAXIMUM_THREADS': str(self.threads_per_worker),
            'TORCH_NUM_THREADS': str(self.threads_per_worker),
            'FKIT_WORKER_ID': str(worker_id),
            'FKIT_THREADS': str(self.threads_per_worker),
        }
        
        if gpu_ids:
            if self.gpu_vendor == 'amd':
                # ROCR uses the physical device indices for /dev/kfd + /dev/dri isolation
                env_vars['ROCR_VISIBLE_DEVICES'] = gpu_ids
                # HIP and CUDA indices are 0-based *within* the ROCR-visible set
                num_gpus = len(gpu_ids.split(','))
                hip_ids = ','.join(str(i) for i in range(num_gpus))
                env_vars['HIP_VISIBLE_DEVICES'] = hip_ids
                env_vars['CUDA_VISIBLE_DEVICES'] = hip_ids
            elif self.gpu_vendor == 'nvidia':
                env_vars['CUDA_VISIBLE_DEVICES'] = gpu_ids
            env_vars['FKIT_WORKER_ID'] = str(worker_id)
            env_vars['FKIT_GPU_IDS'] = gpu_ids
        
        return env_vars
    
    def _is_gpu_error(self, result: TestResult, stderr: str = "") -> bool:
        """Detect if a failure was due to GPU issues."""
        return _is_transient_error(result, stderr)
    
    def _worker_loop(self, worker_id: int):
        """Main loop for a worker thread."""
        gpu_env = self._get_gpu_env_vars(worker_id)
        gpu_str = gpu_env.get('FKIT_GPU_IDS', 'N/A')
        
        while not self._shutdown.is_set():
            try:
                # Try to get work with timeout (allows checking shutdown flag)
                try:
                    work_item = self.work_queue.get(timeout=0.5)
                except queue.Empty:
                    # Check if all work is done
                    if self._all_submitted.is_set() and self.work_queue.empty():
                        break
                    continue
                
                # Check if worker is still healthy
                with self._lock:
                    if self._worker_states[worker_id] == WorkerState.FAILED:
                        # Put work back for another worker
                        self.work_queue.put(work_item)
                        break
                    self._worker_states[worker_id] = WorkerState.RUNNING
                
                # Run the test
                result = self._run_test(work_item.nodeid, worker_id, gpu_env)
                
                # Check for transient errors (GPU, DNS, network, NCCL)
                is_transient = result.outcome == 'failed' and _is_transient_error(result)
                if is_transient:
                    result.gpu_error = True
                    
                    with self._lock:
                        self._stats['gpu_errors'] += 1
                        self._worker_error_counts[worker_id] += 1
                        
                        # Check if worker should be disabled
                        if self._worker_error_counts[worker_id] >= self._max_worker_errors:
                            self._worker_states[worker_id] = WorkerState.FAILED
                            self._stats['workers_failed'] += 1
                            print(f"\n⚠️  Worker {worker_id} (GPUs: {gpu_str}) disabled after "
                                  f"{self._max_worker_errors} consecutive GPU errors")
                    
                    # Retry on another worker if allowed
                    if work_item.retry_count < work_item.max_retries:
                        work_item.retry_count += 1
                        with self._lock:
                            self._stats['retries'] += 1
                        print(f"   🔄 Retrying {work_item.nodeid} on another worker "
                              f"(attempt {work_item.retry_count + 1}/{work_item.max_retries})")
                        self.work_queue.put(work_item)
                        self.work_queue.task_done()
                        continue
                else:
                    # Reset error count on success
                    with self._lock:
                        self._worker_error_counts[worker_id] = 0
                
                # GPU cooldown after crash signals (SIGABRT, SIGSEGV, etc.)
                if result.crash:
                    print(f"   ⏳ Worker {worker_id}: GPU cooldown after crash "
                          f"(3s for driver recovery)")
                    time.sleep(3)
                
                # Update stats
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
                    
                    self._worker_states[worker_id] = WorkerState.IDLE
                
                # Report result via callback
                self.result_callback(work_item.item, result)
                
                # Mark work as done
                self.work_queue.task_done()
                
            except Exception as e:
                # Worker encountered an error - mark as failed
                with self._lock:
                    self._worker_states[worker_id] = WorkerState.FAILED
                    self._stats['workers_failed'] += 1
                print(f"\n❌ Worker {worker_id} encountered error: {e}")
                break
        
        with self._lock:
            if self._worker_states[worker_id] != WorkerState.FAILED:
                self._worker_states[worker_id] = WorkerState.STOPPED
    
    def _run_test(self, nodeid: str, worker_id: int, gpu_env: Dict[str, str]) -> TestResult:
        """Run a single test in an isolated subprocess."""
        import xml.etree.ElementTree as ET
        
        junit_fd, junit_path = tempfile.mkstemp(suffix='.xml', prefix=f'fkit_w{worker_id}_')
        os.close(junit_fd)
        
        try:
            start_time = time.time()
            
            # Prepare environment
            env = os.environ.copy()
            env.update(gpu_env)
            
            # NCCL error recovery: enable async error handling so NCCL
            # can recover from transient communication failures instead
            # of hanging or crashing the entire process group.
            env.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
            
            # Preserve critical variables
            critical_vars = ['HF_TOKEN', 'RUN_SLOW', 'NCCL_DEBUG',
                           'PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH',
                           'TRANSFORMERS_VERBOSITY', 'TRANSFORMERS_CACHE']
            for var in critical_vars:
                if var in os.environ:
                    env[var] = os.environ[var]
            
            # Build pytest command
            pytest_cmd = [
                sys.executable, '-m', 'pytest',
                nodeid,
                '-v',
                '--tb=short',
                '--continue-on-collection-errors',
                '-p', 'no:cacheprovider',
                '-p', 'no:fkit',
                f'--junitxml={junit_path}',
            ]
            
            try:
                result = subprocess.run(
                    pytest_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(Path.cwd()),
                    env=env,
                )
                
                duration = time.time() - start_time
                outcome, skip_reason, fail_message = self._parse_junit_result(junit_path)
                
                if result.returncode == 0:
                    if outcome == 'skipped':
                        return TestResult(
                            nodeid=nodeid,
                            outcome='skipped',
                            duration=duration,
                            skip_reason=skip_reason,
                            worker_id=worker_id
                        )
                    else:
                        return TestResult(
                            nodeid=nodeid,
                            outcome='passed',
                            duration=duration,
                            worker_id=worker_id
                        )
                
                elif result.returncode < 0:
                    # Process killed by signal - CRASH!
                    signal_num = -result.returncode
                    signal_names = {
                        signal.SIGABRT: "SIGABRT (Aborted)",
                        signal.SIGSEGV: "SIGSEGV (Segmentation Fault)",
                        signal.SIGTERM: "SIGTERM (Terminated)",
                        signal.SIGKILL: "SIGKILL (Killed)",
                    }
                    signal_name = signal_names.get(signal_num, f"Signal {signal_num}")
                    
                    crash_info = (
                        f"\n{'='*70}\n"
                        f"💥 TEST CRASHED: {signal_name} (Worker {worker_id}, GPUs: {gpu_env.get('FKIT_GPU_IDS', 'N/A')})\n"
                        f"{'='*70}\n"
                        f"\nThis test caused Python to crash with {signal_name}.\n"
                        f"pytest-fkit caught it and converted it to an ERROR.\n"
                        f"\n--- STDOUT ---\n{result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout}\n"
                        f"\n--- STDERR ---\n{result.stderr[-4000:] if len(result.stderr) > 4000 else result.stderr}\n"
                        f"{'='*70}\n"
                    )
                    
                    return TestResult(
                        nodeid=nodeid,
                        outcome='failed',
                        duration=duration,
                        longrepr=crash_info,
                        crash=True,
                        worker_id=worker_id
                    )
                
                else:
                    # Normal failure - prefer structured JUnit failure message
                    if fail_message:
                        fail_info = fail_message
                    else:
                        # Fallback: extract failure info from subprocess output
                        fail_info = _extract_failure_from_output(result.stdout, result.stderr)
                    return TestResult(
                        nodeid=nodeid,
                        outcome='failed',
                        duration=duration,
                        longrepr=fail_info,
                        worker_id=worker_id
                    )
            
            except subprocess.TimeoutExpired as e:
                duration = time.time() - start_time
                
                timeout_info = (
                    f"\n{'='*70}\n"
                    f"⏱️  TEST TIMEOUT (Worker {worker_id})\n"
                    f"{'='*70}\n"
                    f"\nTest exceeded timeout of {self.timeout} seconds.\n"
                    f"pytest-fkit terminated it and converted it to an ERROR.\n"
                    f"\n--- PARTIAL STDOUT ---\n{e.stdout if e.stdout else '(none)'}\n"
                    f"\n--- PARTIAL STDERR ---\n{e.stderr if e.stderr else '(none)'}\n"
                    f"{'='*70}\n"
                )
                
                return TestResult(
                    nodeid=nodeid,
                    outcome='failed',
                    duration=duration,
                    longrepr=timeout_info,
                    timeout=True,
                    worker_id=worker_id
                )
        
        finally:
            try:
                os.unlink(junit_path)
            except:
                pass
    
    def _parse_junit_result(self, junit_path: str) -> tuple:
        """Parse JUnit XML for outcome, skip reason, and failure message.
        
        Returns:
            (outcome, skip_reason, failure_message) tuple
        """
        import xml.etree.ElementTree as ET
        
        try:
            if not os.path.exists(junit_path):
                return 'unknown', None, None
            
            tree = ET.parse(junit_path)
            root = tree.getroot()
            
            for testcase in root.findall('.//testcase'):
                skipped = testcase.find('skipped')
                if skipped is not None:
                    reason = skipped.get('message') or skipped.text or "Skipped"
                    return 'skipped', reason, None
                
                failure = testcase.find('failure')
                if failure is not None:
                    fail_msg = failure.get('message', '')
                    fail_text = failure.text or ''
                    return 'failed', None, f"{fail_msg}\n{fail_text}".strip()
                
                error = testcase.find('error')
                if error is not None:
                    err_msg = error.get('message', '')
                    err_text = error.text or ''
                    return 'failed', None, f"{err_msg}\n{err_text}".strip()
                
                return 'passed', None, None
            
            return 'unknown', None, None
        except Exception:
            return 'unknown', None, None
    
    def submit_tests(self, items: List):
        """Submit tests to the work queue."""
        for item in items:
            work_item = WorkItem(nodeid=item.nodeid, item=item,
                                 max_retries=self._max_retries)
            self.work_queue.put(work_item)
        self._all_submitted.set()
    
    def start(self):
        """Start all worker threads."""
        for worker_id in range(self.num_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                name=f"fkit-worker-{worker_id}",
                daemon=True
            )
            self._workers.append(thread)
            thread.start()
    
    def wait_for_completion(self):
        """Wait for all tests to complete."""
        # Wait for queue to be empty
        self.work_queue.join()
        
        # Signal shutdown
        self._shutdown.set()
        
        # Wait for all workers to finish
        for thread in self._workers:
            thread.join(timeout=5.0)
    
    def shutdown(self):
        """Force shutdown all workers."""
        self._shutdown.set()
        for thread in self._workers:
            thread.join(timeout=1.0)
    
    @property
    def stats(self):
        with self._lock:
            return dict(self._stats)
    
    @property
    def active_workers(self) -> int:
        """Count of workers that haven't failed."""
        with self._lock:
            return sum(1 for state in self._worker_states.values() 
                      if state not in (WorkerState.FAILED, WorkerState.STOPPED))


class SlicedWorkerPool:
    """
    Pool of workers with pre-sliced test distribution and dynamic failover.
    
    Tests are distributed upfront using round-robin slicing:
    1. Tests are sorted and sliced across workers deterministically
    2. Each worker runs its slice of tests sequentially
    3. Each test still runs in its own subprocess for crash isolation
    4. Workers run in parallel for speed
    5. If a worker encounters GPU errors, remaining tests go to overflow queue
    6. Healthy workers pick up overflow tests when they finish their slice
    
    This provides:
    - Deterministic distribution (reproducible test assignments)
    - Crash isolation (subprocess per test)
    - GPU affinity (each worker has dedicated GPUs)
    - CPU thread affinity (each worker gets fair share of cores)
    - Parallel execution across workers
    - Dynamic failover for GPU failures
    """
    
    def __init__(self, num_workers: int, gpu_allocations: List[str], 
                 gpu_vendor: str, timeout: int, result_callback: Callable,
                 threads_per_worker: int = 4, max_retries: int = 3):
        self.num_workers = num_workers
        self.gpu_allocations = gpu_allocations
        self.gpu_vendor = gpu_vendor
        self.timeout = timeout
        self.result_callback = result_callback
        self.threads_per_worker = threads_per_worker
        self._max_retries = max_retries
        
        # Pre-sliced test lists for each worker
        self._worker_slices: List[List] = []
        
        # Overflow queue for tests from failed workers
        self._overflow_queue = queue.Queue()
        
        # Worker state tracking
        self._worker_states = {i: WorkerState.IDLE for i in range(num_workers)}
        self._worker_gpu_error_counts = {i: 0 for i in range(num_workers)}
        self._max_gpu_errors = 3  # After this many GPU errors, worker moves tests to overflow
        
        # Statistics
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
            'redistributed': 0,
            'workers_failed': 0,
        }
        
        # Control flags
        self._shutdown = threading.Event()
        self._all_slices_done = threading.Event()
        
        # Worker threads
        self._workers = []
        
        # Item map for result reporting
        self._item_map = {}
    
    def _get_gpu_env_vars(self, worker_id: int) -> Dict[str, str]:
        """Get GPU and CPU environment variables for a worker.

        AMD/ROCm GPU visibility layering:
          ROCR_VISIBLE_DEVICES  – physical GPU indices (ROCm runtime level).
          HIP_VISIBLE_DEVICES   – indices *relative to* the ROCR-visible set.
          CUDA_VISIBLE_DEVICES  – PyTorch ROCm maps this to HIP_VISIBLE_DEVICES.

        If a worker owns physical GPU 3, we set:
          ROCR_VISIBLE_DEVICES=3   (expose only that physical GPU)
          HIP_VISIBLE_DEVICES=0    (index 0 within that single-GPU ROCR set)
          CUDA_VISIBLE_DEVICES=0   (same, for PyTorch compat)

        Setting all three to "3" would fail because HIP would look for
        ROCR-index 3 inside a set that only has ROCR-index 0.
        """
        gpu_ids = self.gpu_allocations[worker_id] if worker_id < len(self.gpu_allocations) else ""
        
        env_vars = {
            # CPU thread settings - prevent workers from fighting for cores
            'OMP_NUM_THREADS': str(self.threads_per_worker),
            'MKL_NUM_THREADS': str(self.threads_per_worker),
            'NUMEXPR_NUM_THREADS': str(self.threads_per_worker),
            'OPENBLAS_NUM_THREADS': str(self.threads_per_worker),
            'VECLIB_MAXIMUM_THREADS': str(self.threads_per_worker),
            # PyTorch specific
            'TORCH_NUM_THREADS': str(self.threads_per_worker),
            # Worker identification
            'FKIT_WORKER_ID': str(worker_id),
            'FKIT_THREADS': str(self.threads_per_worker),
        }
        
        if gpu_ids:
            if self.gpu_vendor == 'amd':
                # ROCR uses the physical device indices for /dev/kfd + /dev/dri isolation
                env_vars['ROCR_VISIBLE_DEVICES'] = gpu_ids
                # HIP and CUDA indices are 0-based *within* the ROCR-visible set
                num_gpus = len(gpu_ids.split(','))
                hip_ids = ','.join(str(i) for i in range(num_gpus))
                env_vars['HIP_VISIBLE_DEVICES'] = hip_ids
                env_vars['CUDA_VISIBLE_DEVICES'] = hip_ids
            elif self.gpu_vendor == 'nvidia':
                env_vars['CUDA_VISIBLE_DEVICES'] = gpu_ids
            env_vars['FKIT_WORKER_ID'] = str(worker_id)
            env_vars['FKIT_GPU_IDS'] = gpu_ids
        
        return env_vars
    
    def _is_gpu_error(self, result: TestResult, stderr: str = "") -> bool:
        """Detect if a failure was due to GPU or transient issues."""
        return _is_transient_error(result, stderr)
    
    def _worker_loop(self, worker_id: int, test_slice: List):
        """Main loop for a worker - runs its pre-assigned slice with GPU failover.
        
        Enhanced with per-test retry for transient errors (GPU unavailable,
        DNS resolution, network failures, NCCL errors). A test is retried
        up to max_retries times on the SAME worker before being counted as
        a real failure. After max_gpu_errors consecutive transient failures,
        remaining tests are redistributed to healthy workers.
        """
        gpu_env = self._get_gpu_env_vars(worker_id)
        gpu_str = gpu_env.get('FKIT_GPU_IDS', 'N/A')
        
        slice_size = len(test_slice)
        print(f"   Worker {worker_id} (GPUs: {gpu_str}): {slice_size} tests")
        
        with self._lock:
            self._worker_states[worker_id] = WorkerState.RUNNING
        
        consecutive_gpu_errors = 0
        last_test_crashed = False  # Track crash state for accelerated redistribution
        
        for idx, item in enumerate(test_slice):
            if self._shutdown.is_set():
                break
            
            # After a crash, lower the redistribution threshold: if the NEXT
            # test also hits a transient GPU error, the GPU is likely stuck.
            effective_max_errors = 1 if last_test_crashed else self._max_gpu_errors
            
            # Check if this worker should stop due to GPU errors
            if consecutive_gpu_errors >= effective_max_errors:
                # Move remaining tests to overflow queue for healthy workers
                remaining = test_slice[idx:]
                with self._lock:
                    self._stats['redistributed'] += len(remaining)
                    self._stats['workers_failed'] += 1
                    self._worker_states[worker_id] = WorkerState.FAILED
                reason = "post-crash GPU failure" if last_test_crashed else "transient errors"
                print(f"   ⚠️  Worker {worker_id} (GPUs: {gpu_str}): {reason} detected, "
                      f"redistributing {len(remaining)} remaining tests")
                for remaining_item in remaining:
                    self._overflow_queue.put(remaining_item)
                return
            
            # Run the test with retry logic for transient errors
            max_retries = getattr(self, '_max_retries', 3)
            result = self._run_test(item.nodeid, worker_id, gpu_env)
            retry_count = 0
            
            while (result.outcome == 'failed' 
                   and _is_transient_error(result)
                   and retry_count < max_retries):
                retry_count += 1
                with self._lock:
                    self._stats['retries'] = self._stats.get('retries', 0) + 1
                print(f"   🔄 Worker {worker_id}: Retrying {item.nodeid} "
                      f"(attempt {retry_count + 1}/{max_retries + 1}, transient error)")
                # Brief pause before retry (exponential backoff: 1s, 2s, 4s)
                time.sleep(min(2 ** (retry_count - 1), 4))
                result = self._run_test(item.nodeid, worker_id, gpu_env)
            
            # Check if the final result is still a transient error
            if result.outcome == 'failed' and _is_transient_error(result):
                consecutive_gpu_errors += 1
                with self._lock:
                    self._stats['gpu_errors'] += 1
            else:
                consecutive_gpu_errors = 0  # Reset on success or non-transient failure
            
            # GPU cooldown after crash signals (SIGABRT, SIGSEGV, etc.)
            # The ROCm/CUDA driver may need time to reclaim resources from
            # the crashed process before the next subprocess can init the GPU.
            if result.crash:
                last_test_crashed = True
                print(f"   ⏳ Worker {worker_id}: GPU cooldown after crash "
                      f"(3s for driver recovery)")
                time.sleep(3)
            else:
                last_test_crashed = False
            
            # Update stats
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
            
            # Report result via callback
            self.result_callback(item, result)
        
        with self._lock:
            self._worker_states[worker_id] = WorkerState.STOPPED
        
        # After finishing slice, help with overflow queue
        self._process_overflow(worker_id, gpu_env)
    
    def _process_overflow(self, worker_id: int, gpu_env: Dict[str, str]):
        """Process tests from overflow queue (from failed workers).
        
        Redistributed tests also get retry logic for transient errors.
        """
        gpu_str = gpu_env.get('FKIT_GPU_IDS', 'N/A')
        
        while not self._shutdown.is_set():
            try:
                item = self._overflow_queue.get_nowait()
            except queue.Empty:
                break
            
            print(f"   Worker {worker_id} (GPUs: {gpu_str}): picking up redistributed test")
            result = self._run_test(item.nodeid, worker_id, gpu_env)
            
            # Retry transient errors on this (healthy) worker
            retry_count = 0
            while (result.outcome == 'failed'
                   and _is_transient_error(result)
                   and retry_count < self._max_retries):
                retry_count += 1
                with self._lock:
                    self._stats['retries'] += 1
                print(f"   🔄 Worker {worker_id}: Retrying redistributed {item.nodeid} "
                      f"(attempt {retry_count + 1}/{self._max_retries + 1})")
                time.sleep(min(2 ** (retry_count - 1), 4))
                result = self._run_test(item.nodeid, worker_id, gpu_env)
            
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
            
            self.result_callback(item, result)
            self._overflow_queue.task_done()
    
    def _run_test(self, nodeid: str, worker_id: int, gpu_env: Dict[str, str]) -> TestResult:
        """Run a single test in an isolated subprocess."""
        junit_fd, junit_path = tempfile.mkstemp(suffix='.xml', prefix=f'fkit_w{worker_id}_')
        os.close(junit_fd)
        
        try:
            start_time = time.time()
            
            # Prepare environment
            env = os.environ.copy()
            env.update(gpu_env)
            
            # NCCL error recovery: enable async error handling so NCCL
            # can recover from transient communication failures instead
            # of hanging or crashing the entire process group.
            env.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
            
            # Preserve critical variables
            critical_vars = ['HF_TOKEN', 'RUN_SLOW', 'NCCL_DEBUG',
                           'PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH',
                           'TRANSFORMERS_VERBOSITY', 'TRANSFORMERS_CACHE']
            for var in critical_vars:
                if var in os.environ:
                    env[var] = os.environ[var]
            
            # Build pytest command
            pytest_cmd = [
                sys.executable, '-m', 'pytest',
                nodeid,
                '-v',
                '--tb=short',
                '--continue-on-collection-errors',
                '-p', 'no:cacheprovider',
                '-p', 'no:fkit',  # Disable fkit in subprocess to prevent recursion
                f'--junitxml={junit_path}',
            ]
            
            try:
                result = subprocess.run(
                    pytest_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(Path.cwd()),
                    env=env,
                )
                
                duration = time.time() - start_time
                outcome, skip_reason, fail_message = self._parse_junit_result(junit_path)
                
                if result.returncode == 0:
                    if outcome == 'skipped':
                        return TestResult(
                            nodeid=nodeid,
                            outcome='skipped',
                            duration=duration,
                            skip_reason=skip_reason,
                            worker_id=worker_id
                        )
                    else:
                        return TestResult(
                            nodeid=nodeid,
                            outcome='passed',
                            duration=duration,
                            worker_id=worker_id
                        )
                
                elif result.returncode < 0:
                    # Process killed by signal - CRASH!
                    signal_num = -result.returncode
                    signal_names = {
                        signal.SIGABRT: "SIGABRT (Aborted)",
                        signal.SIGSEGV: "SIGSEGV (Segmentation Fault)",
                        signal.SIGTERM: "SIGTERM (Terminated)",
                        signal.SIGKILL: "SIGKILL (Killed)",
                    }
                    signal_name = signal_names.get(signal_num, f"Signal {signal_num}")
                    
                    crash_info = (
                        f"\n{'='*70}\n"
                        f"💥 TEST CRASHED: {signal_name} (Worker {worker_id}, GPUs: {gpu_env.get('FKIT_GPU_IDS', 'N/A')})\n"
                        f"{'='*70}\n"
                        f"\nThis test caused Python to crash with {signal_name}.\n"
                        f"pytest-fkit caught it and converted it to an ERROR.\n"
                        f"\n--- STDOUT ---\n{result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout}\n"
                        f"\n--- STDERR ---\n{result.stderr[-4000:] if len(result.stderr) > 4000 else result.stderr}\n"
                        f"{'='*70}\n"
                    )
                    
                    return TestResult(
                        nodeid=nodeid,
                        outcome='failed',
                        duration=duration,
                        longrepr=crash_info,
                        crash=True,
                        worker_id=worker_id
                    )
                
                else:
                    # Normal failure - prefer structured JUnit failure message
                    if fail_message:
                        fail_info = fail_message
                    else:
                        # Fallback: extract failure info from subprocess output
                        fail_info = _extract_failure_from_output(result.stdout, result.stderr)
                    return TestResult(
                        nodeid=nodeid,
                        outcome='failed',
                        duration=duration,
                        longrepr=fail_info,
                        worker_id=worker_id
                    )
            
            except subprocess.TimeoutExpired as e:
                duration = time.time() - start_time
                
                timeout_info = (
                    f"\n{'='*70}\n"
                    f"⏱️  TEST TIMEOUT (Worker {worker_id})\n"
                    f"{'='*70}\n"
                    f"\nTest exceeded timeout of {self.timeout} seconds.\n"
                    f"pytest-fkit terminated it and converted it to an ERROR.\n"
                    f"\n--- PARTIAL STDOUT ---\n{e.stdout if e.stdout else '(none)'}\n"
                    f"\n--- PARTIAL STDERR ---\n{e.stderr if e.stderr else '(none)'}\n"
                    f"{'='*70}\n"
                )
                
                return TestResult(
                    nodeid=nodeid,
                    outcome='failed',
                    duration=duration,
                    longrepr=timeout_info,
                    timeout=True,
                    worker_id=worker_id
                )
        
        finally:
            try:
                os.unlink(junit_path)
            except:
                pass
    
    def _parse_junit_result(self, junit_path: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Parse JUnit XML for outcome, skip reason, and failure message.
        
        Returns:
            (outcome, skip_reason, failure_message) tuple
        """
        try:
            if not os.path.exists(junit_path):
                return 'unknown', None, None
            
            tree = ET.parse(junit_path)
            root = tree.getroot()
            
            for testcase in root.findall('.//testcase'):
                skipped = testcase.find('skipped')
                if skipped is not None:
                    reason = skipped.get('message') or skipped.text or "Skipped"
                    return 'skipped', reason, None
                
                failure = testcase.find('failure')
                if failure is not None:
                    fail_msg = failure.get('message', '')
                    fail_text = failure.text or ''
                    return 'failed', None, f"{fail_msg}\n{fail_text}".strip()
                
                error = testcase.find('error')
                if error is not None:
                    err_msg = error.get('message', '')
                    err_text = error.text or ''
                    return 'failed', None, f"{err_msg}\n{err_text}".strip()
                
                return 'passed', None, None
            
            return 'unknown', None, None
        except Exception:
            return 'unknown', None, None
    
    def submit_tests(self, items: List):
        """Slice and distribute tests to workers."""
        self._worker_slices = slice_tests_to_workers(items, self.num_workers)
        self._item_map = {item.nodeid: item for item in items}
        
        # Print distribution info
        print(f"\n📊 Test distribution across {self.num_workers} workers:")
        for i, slice_items in enumerate(self._worker_slices):
            print(f"   Worker {i}: {len(slice_items)} tests")
    
    def start(self):
        """Start all worker threads with their pre-assigned slices."""
        for worker_id in range(self.num_workers):
            test_slice = self._worker_slices[worker_id] if worker_id < len(self._worker_slices) else []
            if not test_slice:
                continue
                
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_id, test_slice),
                name=f"fkit-worker-{worker_id}",
                daemon=True
            )
            self._workers.append(thread)
            thread.start()
    
    def wait_for_completion(self):
        """Wait for all workers to complete their slices."""
        for thread in self._workers:
            thread.join()
    
    def shutdown(self):
        """Force shutdown all workers."""
        self._shutdown.set()
        for thread in self._workers:
            thread.join(timeout=1.0)
    
    @property
    def stats(self):
        with self._lock:
            return dict(self._stats)


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
        print(f"✅ Completed {stats['tests_run']} tests")
        print(f"   Passed: {stats['tests_passed']}, Failed: {stats['tests_failed']}, "
              f"Skipped: {stats['tests_skipped']}")
        if stats['crashes'] > 0:
            print(f"   💥 Crashes: {stats['crashes']}")
        if stats['timeouts'] > 0:
            print(f"   ⏱️  Timeouts: {stats['timeouts']}")
        if stats['gpu_errors'] > 0:
            retries = stats.get('retries', 0)
            redistributed = stats.get('redistributed', 0)
            extra_parts = []
            if retries:
                extra_parts.append(f"retries: {retries}")
            if redistributed:
                extra_parts.append(f"redistributed: {redistributed}")
            extra = f" ({', '.join(extra_parts)})" if extra_parts else ""
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
