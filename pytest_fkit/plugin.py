"""
pytest-fkit plugin: Isolate test crashes and convert them to ERROR results

Inspired by fkitpy - when tests crash (SIGABRT, SIGSEGV, etc.),
catch them and report as normal pytest errors instead of killing the entire run.

This plugin runs each test in a subprocess to isolate crashes.
Supports parallel workers with GPU affinity for multi-GPU systems.
Uses dynamic work queue scheduling - tests are assigned to the first available worker.
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
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
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


def detect_gpus() -> GPUInfo:
    """Detect available GPUs (AMD or NVIDIA)."""
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
    max_retries: int = 1  # Allow 1 retry on GPU errors


class DynamicWorkerPool:
    """
    Pool of workers with dynamic work queue scheduling.
    
    Tests are not pre-assigned to workers. Instead:
    1. All tests go into a shared queue
    2. Workers pull tests from the queue as they become available
    3. If a worker encounters a GPU error, the test can be retried on another worker
    4. Failed workers are marked and work continues with remaining workers
    """
    
    def __init__(self, num_workers: int, gpu_allocations: List[str], 
                 gpu_vendor: str, timeout: int, result_callback: Callable):
        self.num_workers = num_workers
        self.gpu_allocations = gpu_allocations
        self.gpu_vendor = gpu_vendor
        self.timeout = timeout
        self.result_callback = result_callback
        
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
        """Get GPU environment variables for a worker."""
        gpu_ids = self.gpu_allocations[worker_id] if worker_id < len(self.gpu_allocations) else ""
        
        env_vars = {}
        if gpu_ids:
            if self.gpu_vendor == 'amd':
                env_vars['HIP_VISIBLE_DEVICES'] = gpu_ids
                env_vars['ROCR_VISIBLE_DEVICES'] = gpu_ids
                env_vars['CUDA_VISIBLE_DEVICES'] = gpu_ids
            elif self.gpu_vendor == 'nvidia':
                env_vars['CUDA_VISIBLE_DEVICES'] = gpu_ids
            env_vars['FKIT_WORKER_ID'] = str(worker_id)
            env_vars['FKIT_GPU_IDS'] = gpu_ids
        
        return env_vars
    
    def _is_gpu_error(self, result: TestResult, stderr: str = "") -> bool:
        """Detect if a failure was due to GPU issues."""
        gpu_error_patterns = [
            'CUDA out of memory',
            'CUDA error',
            'HIP error',
            'ROCm error',
            'GPU memory',
            'hipErrorNoBinaryForGpu',
            'hipErrorOutOfMemory',
            'NCCL error',
            'device-side assert',
            'no GPU',
            'GPU not found',
            'cudaErrorNoDevice',
            'hipErrorNoDevice',
        ]
        
        check_text = (result.longrepr or "") + stderr
        return any(pattern.lower() in check_text.lower() for pattern in gpu_error_patterns)
    
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
                
                # Check for GPU errors
                if result.outcome == 'failed' and self._is_gpu_error(result):
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
                              f"(attempt {work_item.retry_count + 1})")
                        self.work_queue.put(work_item)
                        self.work_queue.task_done()
                        continue
                else:
                    # Reset error count on success
                    with self._lock:
                        self._worker_error_counts[worker_id] = 0
                
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
                outcome, skip_reason = self._parse_junit_result(junit_path)
                
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
                        f"\n--- STDOUT ---\n{result.stdout}\n"
                        f"\n--- STDERR ---\n{result.stderr}\n"
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
                    # Normal failure
                    fail_info = f"\n--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}"
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
        """Parse JUnit XML for outcome and skip reason."""
        import xml.etree.ElementTree as ET
        
        try:
            if not os.path.exists(junit_path):
                return 'unknown', None
            
            tree = ET.parse(junit_path)
            root = tree.getroot()
            
            for testcase in root.findall('.//testcase'):
                skipped = testcase.find('skipped')
                if skipped is not None:
                    reason = skipped.get('message') or skipped.text or "Skipped"
                    return 'skipped', reason
                
                if testcase.find('failure') is not None or testcase.find('error') is not None:
                    return 'failed', None
                
                return 'passed', None
            
            return 'unknown', None
        except Exception:
            return 'unknown', None
    
    def submit_tests(self, items: List):
        """Submit tests to the work queue."""
        for item in items:
            work_item = WorkItem(nodeid=item.nodeid, item=item)
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


class CrashIsolationPlugin:
    """Plugin that runs tests in subprocess workers to catch crashes."""
    
    def __init__(self, config):
        self.config = config
        self.timeout = config.getoption("--fkit-timeout")
        self.gpus_per_worker = config.getoption("--fkit-gpus-per-worker")
        
        # Parse worker count
        workers_opt = config.getoption("--fkit-workers")
        
        # Detect GPUs
        self.gpu_info = detect_gpus()
        
        if workers_opt == 'auto':
            # Auto-detect based on GPUs
            if self.gpu_info.count > 0:
                self.num_workers = max(1, self.gpu_info.count // self.gpus_per_worker)
            else:
                # No GPUs - use CPU count
                import multiprocessing
                self.num_workers = max(1, multiprocessing.cpu_count() // 2)
        else:
            self.num_workers = max(1, int(workers_opt))
        
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
        
        # Print configuration
        if self.gpu_info.count > 0:
            print(f"\n🚀 pytest-fkit: {self.num_workers} workers, "
                  f"{self.gpu_info.count} {self.gpu_info.vendor.upper()} GPUs, "
                  f"{self.gpus_per_worker} GPU(s)/worker")
            print(f"   GPU allocations: {self.gpu_allocations}")
            print(f"   Dynamic scheduling: tests assigned to first available worker")
        else:
            print(f"\n🚀 pytest-fkit: {self.num_workers} workers (no GPU detected)")
    
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
        """Override test loop for parallel execution with dynamic scheduling."""
        if not self._parallel_mode:
            return None
        
        if not self._collected_items:
            return None
        
        print(f"\n🔄 Running {len(self._collected_items)} tests across {self.num_workers} workers "
              f"(dynamic scheduling)...\n")
        
        # Create worker pool with callback
        self.worker_pool = DynamicWorkerPool(
            num_workers=self.num_workers,
            gpu_allocations=self.gpu_allocations,
            gpu_vendor=self.gpu_info.vendor,
            timeout=self.timeout,
            result_callback=self._result_callback
        )
        
        # Start workers
        self.worker_pool.start()
        
        # Submit all tests to the queue
        self.worker_pool.submit_tests(self._collected_items)
        
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
            print(f"   🎮 GPU errors: {stats['gpu_errors']} (retries: {stats['retries']})")
        if stats['workers_failed'] > 0:
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
                result_callback=lambda i, r: None  # No-op callback
            )
        
        # Run test directly
        result = self.worker_pool._run_test(item.nodeid, worker_id=0, 
                                            gpu_env=self.worker_pool._get_gpu_env_vars(0))
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
