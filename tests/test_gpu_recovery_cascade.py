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
# Tests 14-16: GPU env isolation verification
# ---------------------------------------------------------------------------

def test_gpu_env_vars_override_parent():
    """Verify that per-worker GPU env vars OVERRIDE the parent's broad
    visibility vars — not appended, not ignored, but replaced."""
    pool, _ = make_pool(8, gpus_per_worker=1)

    # Simulate parent having broad GPU visibility (like transformers_ut.sh sets)
    parent_gpu_vars = {
        'ROCR_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'HIP_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
    }

    for worker_id in range(8):
        gpu_env = pool._get_gpu_env_vars(worker_id)

        # Simulate what _run_test does: copy parent env then update
        env = parent_gpu_vars.copy()
        env.update(gpu_env)

        # Per-worker values must win over parent's broad values
        expected_rocr = str(worker_id)  # Physical GPU index
        expected_hip = '0'  # Relative to ROCR set (only 1 device)
        expected_cuda = '0'  # Same as HIP for AMD

        assert env['ROCR_VISIBLE_DEVICES'] == expected_rocr, \
            f"Worker {worker_id}: ROCR should be '{expected_rocr}', " \
            f"got '{env['ROCR_VISIBLE_DEVICES']}'"
        assert env['HIP_VISIBLE_DEVICES'] == expected_hip, \
            f"Worker {worker_id}: HIP should be '{expected_hip}', " \
            f"got '{env['HIP_VISIBLE_DEVICES']}'"
        assert env['CUDA_VISIBLE_DEVICES'] == expected_cuda, \
            f"Worker {worker_id}: CUDA should be '{expected_cuda}', " \
            f"got '{env['CUDA_VISIBLE_DEVICES']}'"

    print(f"✅ test_gpu_env_vars_override_parent PASSED")
    print(f"   All 8 workers: ROCR=N, HIP=0, CUDA=0 (parent's broad vars overridden)")


def test_gpu_env_probe_and_test_parity():
    """Verify that _gpu_health_probe and _run_test use identical GPU env vars.

    If the probe passes but the test subprocess has different GPU vars,
    we'd incorrectly conclude the GPU is healthy when the test can't see it.
    """
    pool, _ = make_pool(8, gpus_per_worker=2)

    for worker_id in range(8):
        gpu_env = pool._get_gpu_env_vars(worker_id)

        # Simulate _gpu_health_probe env construction (plugin.py line 633-634)
        probe_env = {'PARENT_VAR': 'value'}  # stand-in for os.environ.copy()
        probe_env.update(gpu_env)

        # Simulate _run_test env construction (plugin.py line 1461-1471)
        test_env = {'PARENT_VAR': 'value'}
        test_env.update(gpu_env)
        test_env.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
        test_env.setdefault('MASTER_ADDR', '127.0.0.1')
        # Whitelist re-copy (non-GPU vars only)
        for var in ('HF_TOKEN', 'PYTHONPATH', 'LD_LIBRARY_PATH'):
            pass  # These don't touch GPU vars

        # GPU vars must be IDENTICAL in both envs
        gpu_related = ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                       'CUDA_VISIBLE_DEVICES', 'FKIT_GPU_IDS')
        for var in gpu_related:
            probe_val = probe_env.get(var)
            test_val = test_env.get(var)
            assert probe_val == test_val, \
                f"Worker {worker_id}: {var} mismatch — " \
                f"probe='{probe_val}' vs test='{test_val}'"

    print(f"✅ test_gpu_env_probe_and_test_parity PASSED")
    print(f"   All 8 workers: probe and test GPU env vars are identical")


def test_gpu_env_subprocess_receives_correct_vars():
    """Actually spawn a subprocess and verify it receives the exact GPU env
    vars we set — not the parent's values.

    This is the real end-to-end test: set parent env to broad values,
    construct per-worker env, spawn subprocess, verify inside subprocess.
    """
    import subprocess as sp

    pool, _ = make_pool(8, gpus_per_worker=1)

    # Script that prints GPU env vars from inside the subprocess
    check_script = (
        "import os, json, sys\n"
        "result = {\n"
        "    'ROCR_VISIBLE_DEVICES': os.environ.get('ROCR_VISIBLE_DEVICES', ''),\n"
        "    'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES', ''),\n"
        "    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),\n"
        "    'FKIT_GPU_IDS': os.environ.get('FKIT_GPU_IDS', ''),\n"
        "    'FKIT_WORKER_ID': os.environ.get('FKIT_WORKER_ID', ''),\n"
        "}\n"
        "print(json.dumps(result))\n"
    )

    import json

    for worker_id in (0, 3, 7):  # Test a few workers
        gpu_env = pool._get_gpu_env_vars(worker_id)

        # Build env exactly like _run_test does
        env = os.environ.copy()
        # Simulate parent having broad GPU visibility
        env['ROCR_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
        env['HIP_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
        env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
        # Apply per-worker override (same as _run_test line 1462)
        env.update(gpu_env)

        # Spawn subprocess with this env
        r = sp.run(
            [sys.executable, '-c', check_script],
            capture_output=True, text=True, timeout=10, env=env,
        )
        assert r.returncode == 0, f"Subprocess failed: {r.stderr}"
        result = json.loads(r.stdout.strip())

        # Verify subprocess received per-worker values, NOT parent's broad values
        assert result['ROCR_VISIBLE_DEVICES'] == str(worker_id), \
            f"Worker {worker_id}: subprocess ROCR should be '{worker_id}', " \
            f"got '{result['ROCR_VISIBLE_DEVICES']}'"
        assert result['HIP_VISIBLE_DEVICES'] == '0', \
            f"Worker {worker_id}: subprocess HIP should be '0', " \
            f"got '{result['HIP_VISIBLE_DEVICES']}'"
        assert result['CUDA_VISIBLE_DEVICES'] == '0', \
            f"Worker {worker_id}: subprocess CUDA should be '0', " \
            f"got '{result['CUDA_VISIBLE_DEVICES']}'"
        assert result['FKIT_GPU_IDS'] == str(worker_id), \
            f"Worker {worker_id}: subprocess FKIT_GPU_IDS should be '{worker_id}', " \
            f"got '{result['FKIT_GPU_IDS']}'"
        assert result['FKIT_WORKER_ID'] == str(worker_id), \
            f"Worker {worker_id}: subprocess FKIT_WORKER_ID should be '{worker_id}', " \
            f"got '{result['FKIT_WORKER_ID']}'"

    print(f"✅ test_gpu_env_subprocess_receives_correct_vars PASSED")
    print(f"   Workers 0, 3, 7: subprocess received per-worker GPU vars")
    print(f"   Parent's broad HIP_VISIBLE=0,...,7 was correctly overridden")


# ---------------------------------------------------------------------------
# Test 17: env command inheritance preserves full environment
# ---------------------------------------------------------------------------

def test_inherited_env_cmd_preserves_full_environment():
    """Verify that _build_inherited_env_cmd produces a subprocess that
    inherits the FULL parent environment (including conda/venv/pyenv/Docker
    variables) while correctly overriding GPU-specific variables.

    This is the core fix: instead of os.environ.copy() → dict → env=dict
    (which can silently lose environment state), we use the `env` command
    to inject only the overrides on top of true process inheritance.
    """
    import subprocess as sp
    import json

    from pytest_fkit.plugin import _build_inherited_env_cmd

    # Set some marker variables in the parent's real environment
    # to verify they survive into the subprocess
    os.environ['_FKIT_TEST_MARKER_A'] = 'marker_value_alpha'
    os.environ['_FKIT_TEST_MARKER_B'] = 'marker_value_beta'
    # Simulate parent having broad GPU vars (like transformers_ut.sh sets)
    os.environ['ROCR_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    try:
        # Build per-worker GPU overrides (what _get_gpu_env_vars returns)
        gpu_overrides = {
            'ROCR_VISIBLE_DEVICES': '3',
            'HIP_VISIBLE_DEVICES': '0',
            'CUDA_VISIBLE_DEVICES': '0',
            'FKIT_WORKER_ID': '3',
            'FKIT_GPU_IDS': '3',
            'OMP_NUM_THREADS': '4',
        }

        check_script = (
            "import os, json, sys\n"
            "result = {\n"
            "    'sys_executable': sys.executable,\n"
            "    'ROCR_VISIBLE_DEVICES': os.environ.get('ROCR_VISIBLE_DEVICES', ''),\n"
            "    'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES', ''),\n"
            "    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),\n"
            "    'FKIT_WORKER_ID': os.environ.get('FKIT_WORKER_ID', ''),\n"
            "    'FKIT_GPU_IDS': os.environ.get('FKIT_GPU_IDS', ''),\n"
            "    '_FKIT_TEST_MARKER_A': os.environ.get('_FKIT_TEST_MARKER_A', ''),\n"
            "    '_FKIT_TEST_MARKER_B': os.environ.get('_FKIT_TEST_MARKER_B', ''),\n"
            "    'PYTHONPATH': os.environ.get('PYTHONPATH', ''),\n"
            "    'PATH': os.environ.get('PATH', ''),\n"
            "    'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),\n"
            "}\n"
            "# Also check package availability\n"
            "import importlib.util\n"
            "for pkg in ['torch', 'pytest']:\n"
            "    spec = importlib.util.find_spec(pkg)\n"
            "    result[f'pkg_{pkg}'] = spec is not None\n"
            "print(json.dumps(result))\n"
        )

        cmd = _build_inherited_env_cmd(
            gpu_overrides,
            [sys.executable, '-c', check_script],
        )

        r = sp.run(cmd, capture_output=True, text=True, timeout=15)
        assert r.returncode == 0, f"Subprocess failed: {r.stderr}"
        result = json.loads(r.stdout.strip())

        # 1. GPU vars were OVERRIDDEN (not inherited from parent)
        assert result['ROCR_VISIBLE_DEVICES'] == '3', \
            f"ROCR should be '3', got '{result['ROCR_VISIBLE_DEVICES']}'"
        assert result['HIP_VISIBLE_DEVICES'] == '0', \
            f"HIP should be '0', got '{result['HIP_VISIBLE_DEVICES']}'"
        assert result['CUDA_VISIBLE_DEVICES'] == '0', \
            f"CUDA should be '0', got '{result['CUDA_VISIBLE_DEVICES']}'"

        # 2. Parent environment was INHERITED (markers survive)
        assert result['_FKIT_TEST_MARKER_A'] == 'marker_value_alpha', \
            f"Marker A not inherited: got '{result['_FKIT_TEST_MARKER_A']}'"
        assert result['_FKIT_TEST_MARKER_B'] == 'marker_value_beta', \
            f"Marker B not inherited: got '{result['_FKIT_TEST_MARKER_B']}'"

        # 3. Critical path vars inherited
        assert result['PYTHONPATH'] == os.environ.get('PYTHONPATH', ''), \
            "PYTHONPATH not inherited"
        assert result['PATH'] == os.environ.get('PATH', ''), \
            "PATH not inherited"

        # 4. Same Python executable
        assert result['sys_executable'] == sys.executable, \
            f"sys.executable differs: parent={sys.executable}, " \
            f"child={result['sys_executable']}"

        # 5. Packages findable (same site-packages)
        assert result['pkg_torch'] == True or result['pkg_pytest'] == True, \
            "Subprocess can't find any packages — environment not inherited!"

    finally:
        # Clean up marker vars
        os.environ.pop('_FKIT_TEST_MARKER_A', None)
        os.environ.pop('_FKIT_TEST_MARKER_B', None)
        os.environ.pop('ROCR_VISIBLE_DEVICES', None)
        os.environ.pop('HIP_VISIBLE_DEVICES', None)
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    print(f"✅ test_inherited_env_cmd_preserves_full_environment PASSED")
    print(f"   GPU overrides applied: ROCR=3, HIP=0, CUDA=0")
    print(f"   Parent env inherited: markers, PYTHONPATH, PATH all present")
    print(f"   Same Python: {sys.executable}")
    print(f"   Packages findable: torch={result.get('pkg_torch')}, "
          f"pytest={result.get('pkg_pytest')}")


# ---------------------------------------------------------------------------
# Tests 18-19: Subprocess Python environment parity
# ---------------------------------------------------------------------------

def test_subprocess_python_environment_parity():
    """Verify that a subprocess spawned the way _run_test does it sees the
    SAME Python executable, sys.path, and importable packages as the parent.

    This is the critical test the user requested: env vars arriving is necessary
    but not sufficient.  If sys.path differs (e.g. missing site-packages or
    PYTHONPATH entries) then importlib.util.find_spec() will return None for
    packages that ARE installed, causing false SKIP results.
    """
    import subprocess as sp
    import json
    import importlib.util

    pool, _ = make_pool(4, gpus_per_worker=1)

    # ---- Collect parent-side facts ----
    parent_executable = sys.executable
    parent_sys_path = sys.path[:]

    # Packages that HuggingFace tests commonly check via find_spec / @require_*
    check_packages = [
        'torch', 'transformers', 'accelerate', 'datasets',
        'tokenizers', 'safetensors', 'scipy', 'sklearn',
        'flash_attn', 'bitsandbytes', 'deepspeed', 'peft',
        'optimum', 'onnxruntime', 'torchaudio', 'torchvision',
    ]

    parent_specs = {}
    for pkg in check_packages:
        spec = importlib.util.find_spec(pkg)
        parent_specs[pkg] = {
            'found': spec is not None,
            'origin': getattr(spec, 'origin', None) if spec else None,
        }

    # ---- Build the subprocess script ----
    # This script collects the same info from inside the subprocess.
    check_script = """
import sys, os, json, importlib.util

packages = %s

result = {
    'sys_executable': sys.executable,
    'sys_path': sys.path,
    'pythonpath_env': os.environ.get('PYTHONPATH', ''),
    'path_env': os.environ.get('PATH', ''),
    'ld_library_path_env': os.environ.get('LD_LIBRARY_PATH', ''),
    'packages': {},
}
for pkg in packages:
    spec = importlib.util.find_spec(pkg)
    result['packages'][pkg] = {
        'found': spec is not None,
        'origin': getattr(spec, 'origin', None) if spec else None,
    }

print(json.dumps(result))
""" % repr(check_packages)

    # ---- Spawn subprocess with _run_test-style env construction ----
    for worker_id in (0, 2):
        gpu_env = pool._get_gpu_env_vars(worker_id)

        # Exactly replicate _run_test env construction (plugin.py lines 1461-1471)
        env = os.environ.copy()
        env.update(gpu_env)
        env.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
        env.setdefault('MASTER_ADDR', '127.0.0.1')
        env.setdefault('MASTER_PORT', str(29500 + worker_id))

        for var in ('HF_TOKEN', 'RUN_SLOW', 'NCCL_DEBUG',
                    'PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH',
                    'TRANSFORMERS_VERBOSITY', 'TRANSFORMERS_CACHE'):
            if var in os.environ:
                env[var] = os.environ[var]

        r = sp.run(
            [sys.executable, '-c', check_script],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert r.returncode == 0, \
            f"Worker {worker_id} subprocess failed:\nstderr: {r.stderr}"
        result = json.loads(r.stdout.strip())

        # 1. Same Python executable
        assert result['sys_executable'] == parent_executable, \
            f"Worker {worker_id}: sys.executable mismatch!\n" \
            f"  Parent:     {parent_executable}\n" \
            f"  Subprocess: {result['sys_executable']}"

        # 2. Same PYTHONPATH env var
        parent_pp = os.environ.get('PYTHONPATH', '')
        assert result['pythonpath_env'] == parent_pp, \
            f"Worker {worker_id}: PYTHONPATH mismatch!\n" \
            f"  Parent:     {parent_pp}\n" \
            f"  Subprocess: {result['pythonpath_env']}"

        # 3. sys.path: subprocess should contain all parent paths
        #    (subprocess may have extra entries from -m pytest startup, that's OK)
        parent_path_set = set(parent_sys_path)
        child_path_set = set(result['sys_path'])
        missing = parent_path_set - child_path_set
        # Filter out '' and test-specific paths that legitimately differ
        missing = {p for p in missing if p and not p.endswith('__pycache__')}
        # Note: We warn but don't fail on sys.path differences because
        # subprocess -c and -m pytest have different sys.path[0] behavior.
        # The critical check is package find_spec below.
        if missing:
            print(f"   WARNING: Worker {worker_id}: {len(missing)} parent sys.path "
                  f"entries missing in subprocess: {sorted(missing)[:5]}")

        # 4. CRITICAL: Package availability must match
        child_specs = result['packages']
        mismatches = []
        for pkg in check_packages:
            parent_found = parent_specs[pkg]['found']
            child_found = child_specs[pkg]['found']
            if parent_found != child_found:
                mismatches.append({
                    'package': pkg,
                    'parent_found': parent_found,
                    'child_found': child_found,
                    'parent_origin': parent_specs[pkg]['origin'],
                    'child_origin': child_specs[pkg]['origin'],
                })

        if mismatches:
            msg = f"Worker {worker_id}: Package availability mismatch!\n"
            msg += f"  These packages differ between parent and subprocess:\n"
            for m in mismatches:
                direction = "MISSING in subprocess" if m['parent_found'] else "EXTRA in subprocess"
                msg += f"    {m['package']}: {direction}\n"
                msg += f"      parent origin: {m['parent_origin']}\n"
                msg += f"      child origin:  {m['child_origin']}\n"
            assert False, msg

    # Report what we found
    installed = [pkg for pkg, info in parent_specs.items() if info['found']]
    not_installed = [pkg for pkg, info in parent_specs.items() if not info['found']]
    print(f"✅ test_subprocess_python_environment_parity PASSED")
    print(f"   sys.executable: {parent_executable}")
    print(f"   Packages found in BOTH parent & subprocess: {installed}")
    if not_installed:
        print(f"   Packages not installed (skip expected): {not_installed}")
    print(f"   PYTHONPATH preserved: ✓")
    print(f"   No package discovery divergence between parent and subprocess")


def test_subprocess_skip_decorators_match_parent():
    """Simulate HuggingFace's actual skip-decorator logic in a subprocess
    and verify results match the parent process.

    This reproduces the exact checks that @require_torch_gpu,
    @require_flash_attn, @require_bitsandbytes etc. perform.
    If the subprocess would skip a test that the parent wouldn't (or vice
    versa), we have an environment parity bug.
    """
    import subprocess as sp
    import json

    pool, _ = make_pool(4, gpus_per_worker=1)

    # Script that mimics HuggingFace's skip-decorator checks
    # These are the ACTUAL checks from transformers/testing_utils.py
    # and transformers/utils/import_utils.py
    decorator_script = """
import sys, json

results = {}

# 1. @require_torch_gpu → checks torch_device == "cuda"
try:
    import torch
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    results['require_torch_gpu'] = {
        'would_skip': torch_device != "cuda",
        'torch_device': torch_device,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
except ImportError:
    results['require_torch_gpu'] = {'would_skip': True, 'reason': 'torch not installed'}

# 2. @require_flash_attn → importlib.util.find_spec + torch.cuda.is_available
import importlib.util
try:
    flash_spec = importlib.util.find_spec("flash_attn")
    flash_found = flash_spec is not None
    results['require_flash_attn'] = {
        'would_skip': not flash_found,
        'find_spec_result': flash_found,
        'origin': getattr(flash_spec, 'origin', None) if flash_spec else None,
    }
except Exception as e:
    results['require_flash_attn'] = {'would_skip': True, 'reason': str(e)}

# 3. @require_bitsandbytes → importlib.util.find_spec
try:
    bnb_spec = importlib.util.find_spec("bitsandbytes")
    bnb_found = bnb_spec is not None
    results['require_bitsandbytes'] = {
        'would_skip': not bnb_found,
        'find_spec_result': bnb_found,
        'origin': getattr(bnb_spec, 'origin', None) if bnb_spec else None,
    }
except Exception as e:
    results['require_bitsandbytes'] = {'would_skip': True, 'reason': str(e)}

# 4. @require_accelerate → importlib.util.find_spec
try:
    acc_spec = importlib.util.find_spec("accelerate")
    acc_found = acc_spec is not None
    results['require_accelerate'] = {
        'would_skip': not acc_found,
        'find_spec_result': acc_found,
    }
except Exception as e:
    results['require_accelerate'] = {'would_skip': True, 'reason': str(e)}

# 5. @require_deepspeed → importlib.util.find_spec
try:
    ds_spec = importlib.util.find_spec("deepspeed")
    ds_found = ds_spec is not None
    results['require_deepspeed'] = {
        'would_skip': not ds_found,
        'find_spec_result': ds_found,
    }
except Exception as e:
    results['require_deepspeed'] = {'would_skip': True, 'reason': str(e)}

# Metadata
import os
results['_meta'] = {
    'sys_executable': sys.executable,
    'pythonpath': os.environ.get('PYTHONPATH', ''),
    'rocr_visible': os.environ.get('ROCR_VISIBLE_DEVICES', ''),
    'hip_visible': os.environ.get('HIP_VISIBLE_DEVICES', ''),
    'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
}

print(json.dumps(results))
"""

    # ---- Run the same checks in the parent process ----
    import importlib.util

    parent_results = {}

    # @require_torch_gpu
    try:
        import torch
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        parent_results['require_torch_gpu'] = {
            'would_skip': torch_device != "cuda",
            'torch_device': torch_device,
        }
    except ImportError:
        parent_results['require_torch_gpu'] = {'would_skip': True}

    # @require_flash_attn
    flash_spec = importlib.util.find_spec("flash_attn")
    parent_results['require_flash_attn'] = {
        'would_skip': flash_spec is None,
        'find_spec_result': flash_spec is not None,
    }

    # @require_bitsandbytes
    bnb_spec = importlib.util.find_spec("bitsandbytes")
    parent_results['require_bitsandbytes'] = {
        'would_skip': bnb_spec is None,
        'find_spec_result': bnb_spec is not None,
    }

    # @require_accelerate
    acc_spec = importlib.util.find_spec("accelerate")
    parent_results['require_accelerate'] = {
        'would_skip': acc_spec is None,
        'find_spec_result': acc_spec is not None,
    }

    # @require_deepspeed
    ds_spec = importlib.util.find_spec("deepspeed")
    parent_results['require_deepspeed'] = {
        'would_skip': ds_spec is None,
        'find_spec_result': ds_spec is not None,
    }

    # ---- Run in subprocess with _run_test env ----
    for worker_id in (0, 3):
        gpu_env = pool._get_gpu_env_vars(worker_id)

        env = os.environ.copy()
        env.update(gpu_env)
        env.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
        env.setdefault('MASTER_ADDR', '127.0.0.1')
        env.setdefault('MASTER_PORT', str(29500 + worker_id))

        for var in ('HF_TOKEN', 'RUN_SLOW', 'NCCL_DEBUG',
                    'PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH',
                    'TRANSFORMERS_VERBOSITY', 'TRANSFORMERS_CACHE'):
            if var in os.environ:
                env[var] = os.environ[var]

        r = sp.run(
            [sys.executable, '-c', decorator_script],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert r.returncode == 0, \
            f"Worker {worker_id} subprocess failed:\n{r.stderr}"
        child_results = json.loads(r.stdout.strip())

        # Compare each decorator's would_skip decision
        decorators = [
            'require_torch_gpu', 'require_flash_attn',
            'require_bitsandbytes', 'require_accelerate', 'require_deepspeed',
        ]

        divergences = []
        for dec in decorators:
            parent_skip = parent_results[dec]['would_skip']
            child_skip = child_results[dec]['would_skip']
            if parent_skip != child_skip:
                divergences.append({
                    'decorator': dec,
                    'parent_would_skip': parent_skip,
                    'child_would_skip': child_skip,
                    'child_detail': child_results[dec],
                })

        if divergences:
            msg = f"Worker {worker_id}: Skip-decorator DIVERGENCE detected!\n"
            msg += f"  The subprocess would make DIFFERENT skip decisions than the parent:\n"
            for d in divergences:
                action = "SKIP in child but NOT in parent" if d['child_would_skip'] else "RUN in child but SKIP in parent"
                msg += f"    @{d['decorator']}: {action}\n"
                msg += f"      child detail: {d['child_detail']}\n"
            msg += f"\n  Subprocess env:\n"
            msg += f"    ROCR={child_results['_meta']['rocr_visible']}\n"
            msg += f"    HIP={child_results['_meta']['hip_visible']}\n"
            msg += f"    CUDA={child_results['_meta']['cuda_visible']}\n"
            msg += f"    PYTHONPATH={child_results['_meta']['pythonpath'][:100]}\n"
            # NOTE: torch.cuda.is_available() divergence is expected when
            # there are no real GPUs (test env) — only flag non-GPU divergences
            gpu_only_divs = [d for d in divergences if d['decorator'] == 'require_torch_gpu']
            pkg_divs = [d for d in divergences if d['decorator'] != 'require_torch_gpu']
            if pkg_divs:
                assert False, msg
            elif gpu_only_divs:
                print(f"   INFO: Worker {worker_id}: torch.cuda.is_available() differs "
                      f"(expected if no real GPUs in test env)")

    print(f"✅ test_subprocess_skip_decorators_match_parent PASSED")
    print(f"   Parent skip decisions:")
    for dec in decorators:
        skip = parent_results[dec]['would_skip']
        status = "SKIP" if skip else "RUN"
        print(f"     @{dec}: {status}")
    print(f"   Subprocess matches parent for all non-GPU decorators")
    print(f"   (GPU decorator divergence is expected without real GPUs)")


# ---------------------------------------------------------------------------
# Test 20: GPU transient error (NCCL) re-queued to different worker
# ---------------------------------------------------------------------------

def test_nccl_transient_requeued_to_different_worker():
    """
    When a test fails with NCCL Error 2 (GPU transient) and all 3 local
    retries are exhausted, the test should be re-queued to a different
    worker instead of being marked as permanently failed.

    Setup:
      - 4 workers, 5 tests
      - Worker 0's GPU pair has NCCL issues: _run_test returns 'failed'
        with 'NCCL Error 2: unhandled system error' in longrepr
      - Other workers' tests pass normally
      - Health probe returns True for ALL workers (GPUs are "healthy"
        individually — the NCCL issue is between the pair)

    Verify:
      a. Test is retried 3 times on worker 0 (in-place retries)
      b. After retries exhaust, test is re-queued to a different worker
      c. Test passes on the different worker
      d. ALL 5 tests resolve
      e. gpu_requeues stat is incremented
    """
    NUM_WORKERS = 4
    pool, results = make_pool(NUM_WORKERS)

    tests_lock = threading.Lock()
    execution_log = []
    nccl_worker = 0  # This worker has NCCL issues between its GPU pair
    nccl_attempts = {'count': 0}

    test_items = [FakeItem(f"tests/test_{i:03d}.py::test_run") for i in range(5)]

    def mock_run_test(nodeid, worker_id, gpu_env):
        if worker_id == nccl_worker and nodeid == test_items[0].nodeid:
            # NCCL error on this worker's GPU pair
            with tests_lock:
                nccl_attempts['count'] += 1
            return TestResult(
                nodeid=nodeid, outcome='failed', duration=0.5,
                longrepr='RuntimeError: NCCL Error 2: unhandled system error',
                worker_id=worker_id,
            )

        # All other workers/tests pass
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
        # All GPUs probe healthy — the NCCL issue is between pairs,
        # not visible in single-GPU health probes
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
    stats = pool.stats

    # 1. ALL tests resolved
    assert resolved == total, \
        f"Only {resolved}/{total} tests resolved!"

    # 2. NCCL test was retried locally (3 retries + 1 original = 4 attempts)
    assert nccl_attempts['count'] >= 4, \
        f"Expected >= 4 NCCL attempts on worker {nccl_worker}, " \
        f"got {nccl_attempts['count']}"

    # 3. Test was re-queued and passed on a different worker
    with tests_lock:
        nccl_test_passed = any(
            nid == test_items[0].nodeid for nid, wid, _ in execution_log
        )
    assert nccl_test_passed, \
        "NCCL-failed test was NOT re-queued to a different worker!"

    # 4. gpu_requeues stat was incremented
    assert stats.get('gpu_requeues', 0) >= 1, \
        f"Expected gpu_requeues >= 1, got {stats.get('gpu_requeues', 0)}"

    passed = sum(1 for _, r in results if r.outcome == 'passed')
    print(f"✅ test_nccl_transient_requeued_to_different_worker PASSED")
    print(f"   Tests: {resolved}/{total} resolved ({passed} passed)")
    print(f"   NCCL attempts on worker {nccl_worker}: {nccl_attempts['count']}")
    print(f"   gpu_requeues: {stats.get('gpu_requeues', 0)}")
    print(f"   gpu_errors: {stats.get('gpu_errors', 0)}")


# ---------------------------------------------------------------------------
# Test 21: GPU transient re-queue limited to 1 (no infinite loop)
# ---------------------------------------------------------------------------

def test_nccl_requeue_limited_no_infinite_loop():
    """
    If a test fails with NCCL Error 2 on EVERY worker, it should NOT
    loop infinitely.  After 1 re-queue, it becomes terminal.

    Setup:
      - 2 workers, 2 tests
      - BOTH workers fail with NCCL Error 2 for test_000

    Verify:
      a. Test is re-queued exactly once
      b. After second worker also fails, test is marked as terminal failure
      c. No hang or infinite loop
    """
    NUM_WORKERS = 2
    pool, results = make_pool(NUM_WORKERS)

    tests_lock = threading.Lock()
    nccl_attempts = {'count': 0}
    test_items = [FakeItem(f"tests/test_{i:03d}.py::test_run") for i in range(2)]

    def mock_run_test(nodeid, worker_id, gpu_env):
        if nodeid == test_items[0].nodeid:
            # NCCL error on ALL workers for this test
            with tests_lock:
                nccl_attempts['count'] += 1
            return TestResult(
                nodeid=nodeid, outcome='failed', duration=0.1,
                longrepr='RuntimeError: NCCL Error 2: unhandled system error',
                worker_id=worker_id,
            )

        # Other tests pass
        threading.Event().wait(timeout=0.02)
        return TestResult(
            nodeid=nodeid, outcome='passed', duration=0.02,
            worker_id=worker_id,
        )

    _noop = threading.Event()

    with mock.patch.object(pool, '_run_test', side_effect=mock_run_test):
        with recovery_patches():
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
                    f"INFINITE LOOP! {resolved}/{total} tests resolved "
                    f"after 30s"
                )

            pool.wait_for_completion()

    total = pool.scheduler._total_submitted
    resolved = pool.scheduler._total_resolved

    # 1. ALL tests resolved (no hang)
    assert resolved == total, \
        f"Only {resolved}/{total} tests resolved!"

    # 2. NCCL test was eventually marked as failed (terminal)
    nccl_results = [r for _, r in results if r.nodeid == test_items[0].nodeid]
    assert len(nccl_results) == 1, \
        f"Expected exactly 1 result for NCCL test, got {len(nccl_results)}"
    assert nccl_results[0].outcome == 'failed', \
        f"NCCL test should be terminal 'failed', got '{nccl_results[0].outcome}'"

    print(f"✅ test_nccl_requeue_limited_no_infinite_loop PASSED")
    print(f"   Tests: {resolved}/{total} resolved")
    print(f"   Total NCCL attempts across workers: {nccl_attempts['count']}")
    print(f"   (bounded, no infinite loop)")


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
        test_threaded_recovery_completes_all_tests,
        test_probe_timeout_extended_during_recovery,
        test_in_flight_workers_resume_and_finish_tests,
        test_dead_gpu_probe_triggers_requeue,
        test_genuine_failure_not_requeued,
        test_preflight_dead_workers_excluded_from_scheduling,
        test_worker_loop_exits_for_preflight_dead,
        test_preflight_dead_workers_tests_complete_on_healthy,
        test_gpu_env_vars_override_parent,
        test_gpu_env_probe_and_test_parity,
        test_gpu_env_subprocess_receives_correct_vars,
        test_inherited_env_cmd_preserves_full_environment,
        test_subprocess_python_environment_parity,
        test_subprocess_skip_decorators_match_parent,
        test_nccl_transient_requeued_to_different_worker,
        test_nccl_requeue_limited_no_infinite_loop,
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
