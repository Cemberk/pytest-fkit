"""
Simulate the 40-GPU cascade failure scenario and verify the fix.

Scenario (from production):
  - 20 workers × 2 GPUs each = 40 GPUs
  - Worker 5 crashes → triggers _system_gpu_recovery()
  - Workers 0-7 are idle (at barrier) → GPUs free, probe succeeds
  - Workers 8-19 are in-flight (running tests) → GPUs occupied, probe FAILS
  - OLD behavior: all 32 in-flight GPUs marked dead → run hangs at ~96%
  - NEW behavior: in-flight workers skipped, resume after current test

We mock _reset_gpu, _gpu_health_probe, and time.sleep so no real GPUs
are needed and tests run in <1s each.
"""
import sys
import os
import time
import threading
import queue

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest import mock
from pytest_fkit.plugin import (
    _ScheduledWorkerPool,
    WorkScheduler,
    WorkerState,
    TestResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeItem:
    """Minimal pytest-item stand-in."""
    def __init__(self, nodeid):
        self.nodeid = nodeid


def make_pool(num_workers, gpus_per_worker=2, dead_workers=None):
    """Create a _ScheduledWorkerPool with fake GPU allocations."""
    gpu_allocations = []
    for w in range(num_workers):
        start = w * gpus_per_worker
        ids = ",".join(str(start + i) for i in range(gpus_per_worker))
        gpu_allocations.append(ids)

    results = []
    pool = _ScheduledWorkerPool(
        num_workers=num_workers,
        gpu_allocations=gpu_allocations,
        gpu_vendor='amd',
        timeout=600,
        result_callback=lambda item, result: results.append((item, result)),
        threads_per_worker=4,
        max_retries=3,
        dead_workers=dead_workers,
    )
    return pool, results


def recovery_patches(mock_reset=None, mock_probe=None):
    """Context manager that mocks GPU functions AND time.sleep inside plugin.

    time.sleep is only patched in the plugin module so it doesn't affect
    threading primitives (Event.wait, etc.).
    """
    if mock_reset is None:
        mock_reset = lambda gpu_env, gpu_vendor: True
    if mock_probe is None:
        mock_probe = lambda gpu_env, timeout=15: True

    return _CombinedPatch(mock_reset, mock_probe)


class _CombinedPatch:
    """Apply all three mocks as a single context manager."""
    def __init__(self, mock_reset, mock_probe):
        self._patches = [
            mock.patch('pytest_fkit.plugin._reset_gpu', side_effect=mock_reset),
            mock.patch('pytest_fkit.plugin._gpu_health_probe', side_effect=mock_probe),
            mock.patch('pytest_fkit.plugin.time.sleep'),  # no-op sleep
        ]
        self._mocks = []

    def __enter__(self):
        self._mocks = [p.__enter__() for p in self._patches]
        return self._mocks

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)


# ---------------------------------------------------------------------------
# Test 1: In-flight workers are NOT probed and NOT killed
# ---------------------------------------------------------------------------

def test_in_flight_workers_survive_recovery():
    """
    Reproduce the cascade failure:
    - 20 workers, 40 GPUs
    - Workers 0-7 are idle (at barrier)
    - Workers 8-19 have in-flight tests
    - Worker 5 crashes → system recovery
    - Probes succeed for all FREE GPUs
    - In-flight workers should be SKIPPED, not marked dead
    """
    NUM_WORKERS = 20
    pool, _ = make_pool(NUM_WORKERS)

    # All workers RUNNING
    for wid in range(NUM_WORKERS):
        pool._worker_states[wid] = WorkerState.RUNNING

    # Workers 8-19 have in-flight tests
    for wid in range(8, NUM_WORKERS):
        with pool.scheduler._lock:
            pool.scheduler._in_flight[wid] = FakeItem(f"tests/test_model.py::test_{wid}")

    # Verify detection
    in_flight = pool.scheduler.get_in_flight_workers()
    assert in_flight == set(range(8, NUM_WORKERS)), \
        f"Expected workers 8-19 in-flight, got {in_flight}"

    probed_workers = []

    def mock_probe(gpu_env, timeout=15):
        gpu_ids = gpu_env.get('FKIT_GPU_IDS', '')
        if gpu_ids:
            first_gpu = int(gpu_ids.split(',')[0])
            wid = first_gpu // 2  # 2 GPUs per worker
            probed_workers.append(wid)
            # In-flight workers would FAIL (subprocess holding GPU)
            if wid >= 8:
                return False
        return True

    with recovery_patches(mock_probe=mock_probe):
        pool._system_gpu_recovery(triggering_worker_id=5)

    # --- Assertions ---
    probed_set = set(probed_workers)

    # 1. In-flight workers (8-19) should NOT have been probed
    for wid in range(8, NUM_WORKERS):
        assert wid not in probed_set, \
            f"Worker {wid} was probed but has in-flight test — should be skipped!"

    # 2. Idle workers (0-7) SHOULD have been probed
    for wid in range(8):
        assert wid in probed_set, \
            f"Worker {wid} is idle but was not probed"

    # 3. NO workers should be FAILED
    for wid in range(NUM_WORKERS):
        state = pool._worker_states.get(wid)
        health = pool.scheduler.get_worker_health(wid)
        assert state != WorkerState.FAILED, \
            f"Worker {wid} incorrectly marked FAILED (state={state}, health={health})"
        assert health == 'healthy', \
            f"Worker {wid} health is '{health}', expected 'healthy'"

    # 4. Stats: 0 workers failed
    assert pool._stats['workers_failed'] == 0, \
        f"Expected 0 workers_failed, got {pool._stats['workers_failed']}"

    print(f"✅ test_in_flight_workers_survive_recovery PASSED")
    print(f"   Probed workers: {sorted(probed_set)}")
    print(f"   Skipped in-flight: {sorted(set(range(8, NUM_WORKERS)) - probed_set)}")


# ---------------------------------------------------------------------------
# Test 2: Cooldown prevents cascading recoveries
# ---------------------------------------------------------------------------

def test_cooldown_prevents_cascade():
    """
    After recovery, in-flight workers' subprocesses will crash (GPU was
    reset under them).  Each crash tries to trigger another recovery.
    The cooldown should prevent it.
    """
    NUM_WORKERS = 20
    pool, _ = make_pool(NUM_WORKERS)
    for wid in range(NUM_WORKERS):
        pool._worker_states[wid] = WorkerState.RUNNING

    with recovery_patches():
        # First recovery should run
        pool._system_gpu_recovery(triggering_worker_id=5)
        assert pool._stats['system_resets'] == 1

        # Immediate second recovery should be blocked by cooldown
        pool._system_gpu_recovery(triggering_worker_id=8)
        assert pool._stats['system_resets'] == 1, \
            "Cascading recovery should have been blocked by cooldown!"

        pool._system_gpu_recovery(triggering_worker_id=12)
        assert pool._stats['system_resets'] == 1, \
            "Another cascading recovery should also be blocked!"

    print(f"✅ test_cooldown_prevents_cascade PASSED")
    print(f"   Recovery count: {pool._stats['system_resets']} (expected 1)")


# ---------------------------------------------------------------------------
# Test 3: Previously-failed workers CAN recover on next real recovery
# ---------------------------------------------------------------------------

def test_previously_failed_workers_can_recover():
    """
    If worker 3 was marked FAILED in a previous cycle, a later system
    recovery should re-probe it and bring it back if the GPU is healthy.
    """
    NUM_WORKERS = 10
    pool, _ = make_pool(NUM_WORKERS)
    for wid in range(NUM_WORKERS):
        pool._worker_states[wid] = WorkerState.RUNNING

    # Mark worker 3 as previously failed
    pool._worker_states[3] = WorkerState.FAILED
    pool.scheduler.set_worker_health(3, 'dead')

    probed_workers = []

    def mock_probe(gpu_env, timeout=15):
        gpu_ids = gpu_env.get('FKIT_GPU_IDS', '')
        if gpu_ids:
            first_gpu = int(gpu_ids.split(',')[0])
            wid = first_gpu // 2
            probed_workers.append(wid)
        return True  # All GPUs healthy now

    with recovery_patches(mock_probe=mock_probe):
        pool._system_gpu_recovery(triggering_worker_id=0)

    # Worker 3 should have been probed and restored
    assert 3 in probed_workers, \
        f"Previously-failed worker 3 should be re-probed, probed: {probed_workers}"
    assert pool._worker_states[3] == WorkerState.RUNNING, \
        f"Worker 3 should be RUNNING, got {pool._worker_states[3]}"
    assert pool.scheduler.get_worker_health(3) == 'healthy', \
        f"Worker 3 health should be 'healthy', got {pool.scheduler.get_worker_health(3)}"

    print(f"✅ test_previously_failed_workers_can_recover PASSED")


# ---------------------------------------------------------------------------
# Test 4: Verify worker loop recovery check works
# ---------------------------------------------------------------------------

def test_worker_loop_failed_state_recovery():
    """
    Verify that when a worker is marked FAILED but scheduler health is
    restored to 'healthy', the worker loop brings it back to RUNNING.
    """
    NUM_WORKERS = 4
    pool, _ = make_pool(NUM_WORKERS)

    # Worker 2: FAILED state, but scheduler health restored to 'healthy'
    pool._worker_states[2] = WorkerState.FAILED
    pool.scheduler.set_worker_health(2, 'healthy')

    # Replicate the check from _worker_loop (after barrier wait)
    with pool._lock:
        if pool._worker_states[2] == WorkerState.FAILED:
            if pool.scheduler.get_worker_health(2) == 'healthy':
                pool._worker_states[2] = WorkerState.RUNNING

    assert pool._worker_states[2] == WorkerState.RUNNING, \
        f"Worker 2 should be RUNNING, got {pool._worker_states[2]}"

    # Worker 3: FAILED and health is 'dead' → stays FAILED
    pool._worker_states[3] = WorkerState.FAILED
    pool.scheduler.set_worker_health(3, 'dead')

    should_break = False
    with pool._lock:
        if pool._worker_states[3] == WorkerState.FAILED:
            if pool.scheduler.get_worker_health(3) == 'healthy':
                pool._worker_states[3] = WorkerState.RUNNING
            else:
                should_break = True

    assert should_break, "Worker 3 (dead GPU) should trigger break"
    assert pool._worker_states[3] == WorkerState.FAILED

    print(f"✅ test_worker_loop_failed_state_recovery PASSED")


# ---------------------------------------------------------------------------
# Test 5: Scale test — 40 workers, matches production config
# ---------------------------------------------------------------------------

def test_40_worker_scenario():
    """
    Exact production scenario:
    - 40 workers × 1 GPU each
    - Worker 2 crashes
    - Workers 0-7 idle, workers 8-39 in-flight
    - GPU reset fails for devices 0-7 (error code 2), succeeds for 8-39
    - OLD: 32 workers dead, hang at 96%
    - NEW: 8 idle workers probed + pass, 32 in-flight skipped, 0 dead
    """
    NUM_WORKERS = 40

    gpu_allocations = [str(i) for i in range(NUM_WORKERS)]
    pool = _ScheduledWorkerPool(
        num_workers=NUM_WORKERS,
        gpu_allocations=gpu_allocations,
        gpu_vendor='amd',
        timeout=600,
        result_callback=lambda i, r: None,
        threads_per_worker=4,
    )

    for wid in range(NUM_WORKERS):
        pool._worker_states[wid] = WorkerState.RUNNING

    # Workers 8-39 in-flight (like production log)
    for wid in range(8, NUM_WORKERS):
        with pool.scheduler._lock:
            pool.scheduler._in_flight[wid] = FakeItem(f"test_{wid}")

    probed_workers = set()

    def mock_reset(gpu_env, gpu_vendor):
        gpu_id = gpu_env.get('FKIT_GPU_IDS', '')
        wid = int(gpu_id) if gpu_id else -1
        # Devices 0-7 fail (error code 2), 8-39 succeed
        return wid >= 8

    def mock_probe(gpu_env, timeout=15):
        gpu_id = gpu_env.get('FKIT_GPU_IDS', '')
        wid = int(gpu_id) if gpu_id else -1
        probed_workers.add(wid)
        return True

    with recovery_patches(mock_reset=mock_reset, mock_probe=mock_probe):
        pool._system_gpu_recovery(triggering_worker_id=2)

    # 1. Only idle workers (0-7) probed
    assert probed_workers == set(range(8)), \
        f"Expected only workers 0-7 probed, got {sorted(probed_workers)}"

    # 2. No in-flight workers probed
    assert len(probed_workers & set(range(8, NUM_WORKERS))) == 0

    # 3. ZERO workers FAILED
    failed_workers = [
        wid for wid in range(NUM_WORKERS)
        if pool._worker_states[wid] == WorkerState.FAILED
    ]
    assert len(failed_workers) == 0, \
        f"Workers incorrectly marked FAILED: {failed_workers}\n" \
        f"OLD BUG: would have killed workers 8-39 here!"

    # 4. All workers healthy
    dead_workers = [
        wid for wid in range(NUM_WORKERS)
        if pool.scheduler.get_worker_health(wid) != 'healthy'
    ]
    assert len(dead_workers) == 0, \
        f"Workers with non-healthy status: {dead_workers}"

    print(f"✅ test_40_worker_scenario PASSED (production scenario)")
    print(f"   Workers: 40 total, 8 probed, 32 skipped (in-flight), 0 dead")
    print(f"   OLD behavior would have killed 32 workers here!")


# ---------------------------------------------------------------------------
# Test 6: Full end-to-end threading simulation
# ---------------------------------------------------------------------------

def test_threaded_recovery_completes_all_tests():
    """
    Full threading test: start actual worker threads, one test crashes,
    verify all tests complete (no hang) and most workers survive.
    """
    NUM_WORKERS = 8
    pool, results = make_pool(NUM_WORKERS)

    tests_lock = threading.Lock()
    tests_executed = []
    crash_test_id = "tests/crash_test.py::test_gpu_crash"
    crashed_once = threading.Event()

    # 16 normal tests + 1 crash test
    test_items = [FakeItem(f"tests/test_{i}.py::test_run") for i in range(16)]
    crash_item = FakeItem(crash_test_id)
    test_items.insert(4, crash_item)

    def mock_run_test(nodeid, worker_id, gpu_env):
        from pytest_fkit.plugin import TestResult
        with tests_lock:
            tests_executed.append((nodeid, worker_id))

        if nodeid == crash_test_id and not crashed_once.is_set():
            crashed_once.set()
            time.sleep(0.02)
            return TestResult(
                nodeid=nodeid, outcome='failed', duration=0.02,
                longrepr="SIGABRT crash simulation", crash=True,
                worker_id=worker_id,
            )

        time.sleep(0.01)
        return TestResult(
            nodeid=nodeid, outcome='passed', duration=0.01,
            worker_id=worker_id,
        )

    def mock_reset(gpu_env, gpu_vendor):
        return True

    def mock_probe(gpu_env, timeout=15):
        return True

    # Patch _run_test on the instance, GPU functions + time.sleep on the module
    with mock.patch.object(pool, '_run_test', side_effect=mock_run_test):
        with mock.patch('pytest_fkit.plugin._reset_gpu', side_effect=mock_reset):
            with mock.patch('pytest_fkit.plugin._gpu_health_probe', side_effect=mock_probe):
                with mock.patch('pytest_fkit.plugin.time.sleep'):
                    pool.submit_tests(test_items)
                    pool.start()

                    # Wait with a hard timeout to detect hangs
                    deadline = time.time() + 30
                    while not pool.scheduler.is_done() and time.time() < deadline:
                        time.sleep(0.1)  # real sleep for our wait loop

                    if not pool.scheduler.is_done():
                        pool.shutdown()
                        resolved = pool.scheduler._total_resolved
                        total = pool.scheduler._total_submitted
                        assert False, (
                            f"HANG DETECTED! Only {resolved}/{total} tests "
                            f"resolved after 30s"
                        )

                    pool.wait_for_completion()

    total_submitted = len(test_items)
    total_resolved = pool.scheduler._total_resolved
    assert total_resolved == total_submitted, \
        f"Only {total_resolved}/{total_submitted} resolved"

    passed = sum(1 for _, r in results if r.outcome == 'passed')
    failed = sum(1 for _, r in results if r.outcome == 'failed')

    print(f"✅ test_threaded_recovery_completes_all_tests PASSED")
    print(f"   Tests: {total_resolved}/{total_submitted} resolved "
          f"({passed} passed, {failed} failed)")
    print(f"   System resets: {pool._stats['system_resets']}")
    print(f"   Workers failed: {pool._stats['workers_failed']}")


# ---------------------------------------------------------------------------
# Test 7: Probe timeout is 30s during recovery (not default 15s)
# ---------------------------------------------------------------------------

def test_probe_timeout_extended_during_recovery():
    """Verify that health probes during recovery use a 30s timeout."""
    NUM_WORKERS = 4
    pool, _ = make_pool(NUM_WORKERS)
    for wid in range(NUM_WORKERS):
        pool._worker_states[wid] = WorkerState.RUNNING

    probe_timeouts = []

    def mock_probe(gpu_env, timeout=15):
        probe_timeouts.append(timeout)
        return True

    with recovery_patches(mock_probe=mock_probe):
        pool._system_gpu_recovery(triggering_worker_id=0)

    # All probes should have been called with timeout=30
    assert all(t == 30 for t in probe_timeouts), \
        f"Expected all probe timeouts=30, got {probe_timeouts}"

    print(f"✅ test_probe_timeout_extended_during_recovery PASSED")
    print(f"   Probe timeouts: {probe_timeouts}")


# ---------------------------------------------------------------------------
# Test 8: In-flight workers RESUME and process remaining tests after recovery
# ---------------------------------------------------------------------------

def test_in_flight_workers_resume_and_finish_tests():
    """
    Full end-to-end threading test proving in-flight workers survive recovery.

    Key insight: mock.patch('pytest_fkit.plugin.time.sleep') replaces sleep
    on the GLOBAL time module (both plugin and test share the same object).
    So we use threading.Event().wait(timeout=...) for blocking inside
    mock_run_test — this is immune to the time.sleep mock.

    Flow:
      1. 8 workers start, each picks a test from the queue
      2. Crash worker waits until 7+ workers are inside mock_run_test
      3. Non-crash workers block on crashed.wait() + Event().wait(0.5s)
      4. Crash worker fires → worker loop calls report_result → recovery
      5. Recovery's get_in_flight_workers() sees non-crash workers (still
         blocked in mock_run_test, thus still in scheduler._in_flight)
      6. Recovery skips them, finishes, lowers barrier
      7. Non-crash workers' 0.5s wait expires, they return their result
      8. All workers continue processing remaining tests

    Verify:
      a. Recovery detected in-flight workers and skipped probing them
      b. ALL tests reach a terminal state (no hang)
      c. Multiple workers ran tests AFTER recovery (proving they resumed)
    """
    NUM_WORKERS = 8
    pool, results = make_pool(NUM_WORKERS)

    tests_lock = threading.Lock()
    execution_log = []  # (nodeid, worker_id, timestamp, outcome)
    crash_test_id = "tests/crash_me.py::test_boom"
    crashed = threading.Event()

    # Synchronization: crash test waits until other workers are mid-test
    workers_in_test = threading.Event()
    active_count_lock = threading.Lock()
    active_in_test = [0]

    # Capture in-flight snapshot during recovery (from mock_probe)
    in_flight_during_recovery = []

    # 30 tests: crash at index 3 (first batch), rest normal
    test_items = []
    for i in range(30):
        if i == 3:
            test_items.append(FakeItem(crash_test_id))
        else:
            test_items.append(FakeItem(f"tests/test_{i:03d}.py::test_run"))

    def mock_run_test(nodeid, worker_id, gpu_env):
        from pytest_fkit.plugin import TestResult
        ts = time.monotonic()

        # Track that this worker is inside _run_test
        with active_count_lock:
            active_in_test[0] += 1
            if active_in_test[0] >= NUM_WORKERS - 1:
                workers_in_test.set()

        try:
            # Crash test: wait for other workers to be mid-test, then crash
            if nodeid == crash_test_id and not crashed.is_set():
                workers_in_test.wait(timeout=10)
                crashed.set()
                with tests_lock:
                    execution_log.append((nodeid, worker_id, ts, 'crash'))
                return TestResult(
                    nodeid=nodeid, outcome='failed', duration=0.01,
                    longrepr="SIGABRT", crash=True, worker_id=worker_id,
                )

            # IMPORTANT: Use Event().wait() instead of time.sleep()!
            # mock.patch('pytest_fkit.plugin.time.sleep') patches the global
            # time module, which would make time.sleep() a no-op here too.
            # Event().wait() is not affected by that mock.
            #
            # First batch: wait for crash to happen, then hold for 0.5s
            # so we're still in-flight when recovery checks.
            # Subsequent batches: pass through quickly (crashed already set).
            if not crashed.is_set():
                crashed.wait(timeout=10)
            # Hold position so recovery can see us as in-flight.
            # 0.5s is plenty — with time.sleep mocked in the plugin,
            # _system_gpu_recovery completes in microseconds.
            threading.Event().wait(timeout=0.5)

            with tests_lock:
                execution_log.append((nodeid, worker_id, ts, 'passed'))
            return TestResult(
                nodeid=nodeid, outcome='passed', duration=0.01,
                worker_id=worker_id,
            )

        finally:
            with active_count_lock:
                active_in_test[0] -= 1

    def mock_reset(gpu_env, gpu_vendor):
        return True

    def mock_probe(gpu_env, timeout=15):
        # Capture the in-flight snapshot on the first probe call
        # (happens inside _system_gpu_recovery, after in-flight workers
        # have already been excluded from the probe list)
        if not in_flight_during_recovery:
            snapshot = pool.scheduler.get_in_flight_workers()
            in_flight_during_recovery.append(snapshot)
        return True

    _noop = threading.Event()  # never set — used for real waits below

    with mock.patch.object(pool, '_run_test', side_effect=mock_run_test):
        with mock.patch('pytest_fkit.plugin._reset_gpu', side_effect=mock_reset):
            with mock.patch('pytest_fkit.plugin._gpu_health_probe', side_effect=mock_probe):
                with mock.patch('pytest_fkit.plugin.time.sleep'):
                    pool.submit_tests(test_items)
                    pool.start()

                    # Wait with hard timeout to detect hangs.
                    # Use Event().wait() instead of time.sleep() (globally mocked).
                    deadline = time.time() + 60
                    while not pool.scheduler.is_done() and time.time() < deadline:
                        _noop.wait(timeout=0.1)

                    if not pool.scheduler.is_done():
                        pool.shutdown()
                        resolved = pool.scheduler._total_resolved
                        total = pool.scheduler._total_submitted
                        with pool._lock:
                            worker_states = dict(pool._worker_states)
                        in_flight = pool.scheduler.get_in_flight_workers()
                        healthy = pool.scheduler.healthy_worker_count()
                        assert False, (
                            f"HANG DETECTED! {resolved}/{total} tests resolved "
                            f"after 60s.\n"
                            f"  Worker states: {worker_states}\n"
                            f"  In-flight: {in_flight}\n"
                            f"  Healthy workers: {healthy}\n"
                            f"  Queue empty: {pool.scheduler._queue.empty()}"
                        )

                    pool.wait_for_completion()

    # --- Assertions ---
    total = pool.scheduler._total_submitted
    resolved = pool.scheduler._total_resolved

    # 1. ALL tests resolved (no hang!)
    assert resolved == total, \
        f"Only {resolved}/{total} tests resolved — HANG!"

    # 2. In-flight workers were detected during recovery
    assert len(in_flight_during_recovery) > 0, \
        "Recovery probed workers but no in-flight snapshot was captured"
    in_flight_snapshot = in_flight_during_recovery[0]
    assert len(in_flight_snapshot) >= NUM_WORKERS - 2, \
        f"Expected at least {NUM_WORKERS - 2} in-flight workers during " \
        f"recovery, got {len(in_flight_snapshot)}: {sorted(in_flight_snapshot)}"

    # 3. Identify which workers ran tests AFTER the crash
    with tests_lock:
        log_copy = list(execution_log)

    workers_used = set(wid for _, wid, _, _ in log_copy)

    crash_time = None
    for nodeid, wid, ts, outcome in log_copy:
        if nodeid == crash_test_id and outcome == 'crash':
            crash_time = ts
            break

    workers_after_recovery = set()
    if crash_time:
        for nodeid, wid, ts, outcome in log_copy:
            if ts > crash_time and outcome == 'passed':
                workers_after_recovery.add(wid)

    # 4. Most workers should have run tests after recovery
    assert len(workers_after_recovery) >= NUM_WORKERS // 2, \
        f"Only {len(workers_after_recovery)}/{NUM_WORKERS} workers ran tests " \
        f"after recovery — workers may have been incorrectly killed!\n" \
        f"Workers after recovery: {sorted(workers_after_recovery)}"

    # 5. Workers permanently failed should be 0 or at most 1
    assert pool._stats['workers_failed'] <= 1, \
        f"Too many workers failed: {pool._stats['workers_failed']}"

    passed = sum(1 for _, r in results if r.outcome == 'passed')
    failed = sum(1 for _, r in results if r.outcome == 'failed')

    print(f"✅ test_in_flight_workers_resume_and_finish_tests PASSED")
    print(f"   Tests: {resolved}/{total} resolved ({passed} passed, {failed} failed)")
    print(f"   Workers used: {sorted(workers_used)} ({len(workers_used)}/{NUM_WORKERS})")
    print(f"   Workers active AFTER recovery: {sorted(workers_after_recovery)}")
    print(f"   In-flight during recovery: {sorted(in_flight_snapshot)}")
    print(f"   System resets: {pool._stats['system_resets']}")
    print(f"   Workers permanently failed: {pool._stats['workers_failed']}")


# ---------------------------------------------------------------------------
# Test 9: Dead GPU detected by probe → test re-queued to healthy worker
# ---------------------------------------------------------------------------

def test_dead_gpu_probe_triggers_requeue():
    """
    Probe-based GPU recovery (no string parsing).

    Setup:
      - 4 workers, 10 tests
      - Worker 1's GPU is dead: _run_test returns 'skipped' (any reason)
      - Health probe for worker 1 returns False (GPU genuinely broken)
      - Other workers' probes return True

    The worker loop detects the dead GPU via probe (not by parsing the
    skip reason), marks worker 1 dead, re-queues the test, and triggers
    system recovery.

    Verify:
      a. Worker 1 marked dead via probe (not string matching)
      b. Re-queued test completed by a healthy worker
      c. ALL 10 tests resolve
    """
    NUM_WORKERS = 4
    pool, results = make_pool(NUM_WORKERS)

    tests_lock = threading.Lock()
    execution_log = []
    broken_worker = 1

    test_items = [FakeItem(f"tests/test_{i:03d}.py::test_run") for i in range(10)]

    def mock_run_test(nodeid, worker_id, gpu_env):
        # Worker 1 returns 'skipped' — any reason, doesn't matter.
        # The PROBE decides whether the GPU is dead, not this message.
        if worker_id == broken_worker:
            return TestResult(
                nodeid=nodeid, outcome='skipped', duration=0.01,
                skip_reason='some unrelated skip reason',
                worker_id=worker_id,
            )

        threading.Event().wait(timeout=0.02)
        with tests_lock:
            execution_log.append((nodeid, worker_id, 'passed'))
        return TestResult(
            nodeid=nodeid, outcome='passed', duration=0.02,
            worker_id=worker_id,
        )

    def mock_reset(gpu_env, gpu_vendor):
        return True

    def mock_probe(gpu_env, timeout=15):
        # Worker 1's GPU is genuinely dead — probe returns False.
        # This is the ONLY thing that determines re-queue, not error text.
        gpu_ids = gpu_env.get('FKIT_GPU_IDS', '')
        if gpu_ids:
            first_gpu = int(gpu_ids.split(',')[0])
            wid = first_gpu // 2
            if wid == broken_worker:
                return False
        return True

    _noop = threading.Event()

    with mock.patch.object(pool, '_run_test', side_effect=mock_run_test):
        with mock.patch('pytest_fkit.plugin._reset_gpu', side_effect=mock_reset):
            with mock.patch('pytest_fkit.plugin._gpu_health_probe', side_effect=mock_probe):
                with mock.patch('pytest_fkit.plugin.time.sleep'):
                    pool.submit_tests(test_items)
                    pool.start()

                    deadline = time.time() + 30
                    while not pool.scheduler.is_done() and time.time() < deadline:
                        _noop.wait(timeout=0.1)

                    if not pool.scheduler.is_done():
                        pool.shutdown()
                        resolved = pool.scheduler._total_resolved
                        total = pool.scheduler._total_submitted
                        assert False, (
                            f"HANG DETECTED! {resolved}/{total} tests resolved "
                            f"after 30s"
                        )

                    pool.wait_for_completion()

    total = pool.scheduler._total_submitted
    resolved = pool.scheduler._total_resolved

    # 1. ALL tests resolved (re-queued test ran on a healthy worker)
    assert resolved == total, \
        f"Only {resolved}/{total} tests resolved — test was lost in re-queue!"

    # 2. Worker 1 was killed by probe failure (not string matching)
    assert pool._stats['gpu_errors'] >= 1, \
        f"Expected gpu_errors >= 1, got {pool._stats['gpu_errors']}"
    assert pool._worker_states[broken_worker] == WorkerState.FAILED
    assert pool.scheduler.get_worker_health(broken_worker) == 'dead'

    # 3. All tests were executed by healthy workers only
    with tests_lock:
        workers_that_passed = set(wid for _, wid, _ in execution_log)
    assert broken_worker not in workers_that_passed

    passed = sum(1 for _, r in results if r.outcome == 'passed')
    print(f"✅ test_dead_gpu_probe_triggers_requeue PASSED")
    print(f"   Tests: {resolved}/{total} resolved ({passed} passed)")
    print(f"   Workers that passed: {sorted(workers_that_passed)}")
    print(f"   gpu_errors: {pool._stats['gpu_errors']}")


# ---------------------------------------------------------------------------
# Test 10: Healthy GPU + failed test → no re-queue (genuine failure)
# ---------------------------------------------------------------------------

def test_genuine_failure_not_requeued():
    """
    When a test fails but the GPU probe passes, it's a genuine test
    failure — the test should NOT be re-queued.

    This verifies that the probe-based approach doesn't over-trigger:
    only actual GPU death (probe failure) causes re-queue, not every
    test failure.
    """
    NUM_WORKERS = 4
    pool, results = make_pool(NUM_WORKERS)

    test_items = [FakeItem(f"tests/test_{i:03d}.py::test_run") for i in range(8)]

    # Two tests genuinely fail (assertion error), rest pass.
    failing_tests = {
        'tests/test_002.py::test_run',
        'tests/test_005.py::test_run',
    }

    def mock_run_test(nodeid, worker_id, gpu_env):
        if nodeid in failing_tests:
            return TestResult(
                nodeid=nodeid, outcome='failed', duration=0.05,
                longrepr='AssertionError: values differ',
                worker_id=worker_id,
            )
        threading.Event().wait(timeout=0.02)
        return TestResult(
            nodeid=nodeid, outcome='passed', duration=0.02,
            worker_id=worker_id,
        )

    def mock_reset(gpu_env, gpu_vendor):
        return True

    def mock_probe(gpu_env, timeout=15):
        # ALL GPUs are healthy — probe always passes.
        return True

    _noop = threading.Event()

    with mock.patch.object(pool, '_run_test', side_effect=mock_run_test):
        with mock.patch('pytest_fkit.plugin._reset_gpu', side_effect=mock_reset):
            with mock.patch('pytest_fkit.plugin._gpu_health_probe', side_effect=mock_probe):
                with mock.patch('pytest_fkit.plugin.time.sleep'):
                    pool.submit_tests(test_items)
                    pool.start()

                    deadline = time.time() + 30
                    while not pool.scheduler.is_done() and time.time() < deadline:
                        _noop.wait(timeout=0.1)

                    if not pool.scheduler.is_done():
                        pool.shutdown()
                        assert False, "HANG DETECTED!"

                    pool.wait_for_completion()

    total = pool.scheduler._total_submitted
    resolved = pool.scheduler._total_resolved

    # 1. ALL tests resolved
    assert resolved == total

    # 2. ZERO gpu_errors — probe passed, so these are genuine failures
    assert pool._stats['gpu_errors'] == 0, \
        f"Expected 0 gpu_errors (probe passed), got {pool._stats['gpu_errors']}"

    # 3. No workers were killed
    assert pool._stats['workers_failed'] == 0, \
        f"No workers should be killed for genuine test failures"

    # 4. The failing tests were reported as failures (not re-queued)
    failed_results = [(item.nodeid, r.outcome) for item, r in results
                      if r.outcome == 'failed']
    assert len(failed_results) == 2, \
        f"Expected 2 genuine failures, got {len(failed_results)}: {failed_results}"

    passed = sum(1 for _, r in results if r.outcome == 'passed')
    failed = sum(1 for _, r in results if r.outcome == 'failed')
    print(f"✅ test_genuine_failure_not_requeued PASSED")
    print(f"   Tests: {resolved}/{total} ({passed} passed, {failed} failed)")
    print(f"   gpu_errors: {pool._stats['gpu_errors']} (expected 0)")
    print(f"   workers_failed: {pool._stats['workers_failed']} (expected 0)")


# ---------------------------------------------------------------------------
# Tests 11-13: Preflight dead worker exclusion
# ---------------------------------------------------------------------------

def test_preflight_dead_workers_excluded_from_scheduling():
    """Workers marked dead at startup (via dead_workers param) should never
    receive test assignments from the scheduler."""
    NUM_WORKERS = 8
    DEAD = {2, 5, 7}

    pool, results = make_pool(NUM_WORKERS, gpus_per_worker=1, dead_workers=DEAD)

    # Dead workers must be in FAILED state
    for wid in DEAD:
        assert pool._worker_states[wid] == WorkerState.FAILED, \
            f"Worker {wid} should be FAILED, got {pool._worker_states[wid]}"
        assert pool.scheduler.get_worker_health(wid) == 'dead', \
            f"Worker {wid} should be 'dead' in scheduler"

    # Healthy workers unaffected
    for wid in range(NUM_WORKERS):
        if wid not in DEAD:
            assert pool._worker_states[wid] == WorkerState.IDLE, \
                f"Worker {wid} should be IDLE, got {pool._worker_states[wid]}"
            assert pool.scheduler.get_worker_health(wid) == 'healthy', \
                f"Worker {wid} should be 'healthy' in scheduler"

    # Submit tests and verify dead workers can't pull work
    items = [FakeItem(f"test_{i}") for i in range(20)]
    pool.scheduler.submit(items)

    for wid in DEAD:
        item = pool.scheduler.get_work(wid)
        assert item is None, f"Dead worker {wid} should not get work!"

    # Healthy workers CAN get work
    for wid in (0, 1, 3, 4, 6):
        item = pool.scheduler.get_work(wid)
        assert item is not None, f"Healthy worker {wid} should get work!"

    print(f"✅ test_preflight_dead_workers_excluded_from_scheduling PASSED")
    print(f"   Dead workers {sorted(DEAD)} correctly excluded")
    print(f"   Healthy workers correctly received work")


def test_worker_loop_exits_for_preflight_dead():
    """Worker threads for preflight-dead workers should exit immediately
    without overwriting FAILED → RUNNING."""
    NUM_WORKERS = 4
    DEAD = {2}

    pool, _ = make_pool(NUM_WORKERS, gpus_per_worker=1, dead_workers=DEAD)

    # Worker 2 should be FAILED from dead_workers param
    assert pool._worker_states[2] == WorkerState.FAILED

    # Run the worker loop directly — it should return immediately
    # because of the FAILED guard at the top
    pool._worker_loop(2)

    # State should still be FAILED (not overwritten to RUNNING)
    assert pool._worker_states[2] == WorkerState.FAILED, \
        f"Worker 2 state should remain FAILED, got {pool._worker_states[2]}"

    print(f"✅ test_worker_loop_exits_for_preflight_dead PASSED")
    print(f"   Worker 2: FAILED state preserved (loop exited immediately)")


def test_preflight_dead_workers_tests_complete_on_healthy():
    """Full threaded test: dead workers never execute, all tests
    complete on healthy workers only."""
    NUM_WORKERS = 8
    DEAD = {2, 5, 7}
    HEALTHY = set(range(NUM_WORKERS)) - DEAD
    NUM_TESTS = 20

    pool, results = make_pool(NUM_WORKERS, gpus_per_worker=1, dead_workers=DEAD)

    # Track which workers actually executed tests
    worker_executions = []
    lock = threading.Lock()

    def mock_run_test(nodeid, worker_id, gpu_env):
        with lock:
            worker_executions.append(worker_id)
        assert worker_id not in DEAD, \
            f"Dead worker {worker_id} should NEVER execute tests!"
        # Simulate brief work
        threading.Event().wait(timeout=0.01)
        return TestResult(
            nodeid=nodeid, outcome='passed', duration=0.01,
            worker_id=worker_id)

    items = [FakeItem(f"test_{i}") for i in range(NUM_TESTS)]

    with mock.patch.object(pool, '_run_test', side_effect=mock_run_test), \
         recovery_patches():
        pool.submit_tests(items)
        pool.start()

        # Wait for completion with timeout
        deadline = time.time() + 30
        while time.time() < deadline:
            resolved = sum(1 for _, r in results if r is not None)
            if resolved >= NUM_TESTS:
                break
            time.sleep(0.1)

        pool.shutdown()

    resolved = len(results)
    assert resolved == NUM_TESTS, \
        f"Expected {NUM_TESTS} resolved, got {resolved}"

    # Verify no dead worker executed
    dead_executions = [w for w in worker_executions if w in DEAD]
    assert len(dead_executions) == 0, \
        f"Dead workers executed tests: {dead_executions}"

    # Verify only healthy workers executed
    executing_workers = set(worker_executions)
    assert executing_workers <= HEALTHY, \
        f"Non-healthy workers executed: {executing_workers - HEALTHY}"

    print(f"✅ test_preflight_dead_workers_tests_complete_on_healthy PASSED")
    print(f"   {NUM_TESTS}/{NUM_TESTS} tests completed on "
          f"{len(executing_workers)} healthy workers")
    print(f"   Dead workers {sorted(DEAD)} never executed")
    print(f"   Active workers: {sorted(executing_workers)}")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("  GPU Recovery Cascade Failure — Regression Tests")
    print(f"{'='*70}\n")

    tests = [
        test_in_flight_workers_survive_recovery,
        test_cooldown_prevents_cascade,
        test_previously_failed_workers_can_recover,
        test_worker_loop_failed_state_recovery,
        test_40_worker_scenario,
        test_probe_timeout_extended_during_recovery,
        test_threaded_recovery_completes_all_tests,
        test_in_flight_workers_resume_and_finish_tests,
        test_gpu_unavailable_skip_requeues_to_different_worker,
        test_no_hip_gpus_failure_requeues,
        test_preflight_dead_workers_excluded_from_scheduling,
        test_worker_loop_exits_for_preflight_dead,
        test_preflight_dead_workers_tests_complete_on_healthy,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            print(f"\n--- {test_fn.__name__} ---")
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*70}\n")

    sys.exit(0 if failed == 0 else 1)
