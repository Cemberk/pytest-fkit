"""
Stack Frame Capture: Python and native frame tracing.

Based on patterns from pystack - provides:
- Call stack snapshots during execution
- Thread state tracking (GIL, GC status)
- Native vs Python frame correlation
- Frame arguments and locals capture
- Memory context

Can use pystack if available, falls back to pure Python introspection.
"""

import inspect
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Try to import pystack for enhanced functionality
try:
    from pystack.engine import get_process_threads
    from pystack.types import PyThread, PyFrame, NativeFrame
    PYSTACK_AVAILABLE = True
except ImportError:
    PYSTACK_AVAILABLE = False


@dataclass
class FrameLocation:
    """Source location of a frame."""
    filename: str
    lineno: int
    function: str
    module: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "lineno": self.lineno,
            "function": self.function,
            "module": self.module,
        }

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno} in {self.function}"


@dataclass
class StackFrame:
    """A captured stack frame."""
    location: FrameLocation
    arguments: Dict[str, str] = field(default_factory=dict)
    locals: Dict[str, str] = field(default_factory=dict)
    is_native: bool = False
    is_entry: bool = False
    depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location.to_dict(),
            "arguments": self.arguments,
            "locals_count": len(self.locals),
            "is_native": self.is_native,
            "depth": self.depth,
        }

    def to_metrics(self) -> Dict[str, float]:
        """Numeric metrics for PySR."""
        return {
            "frame_depth": float(self.depth),
            "frame_locals_count": float(len(self.locals)),
            "frame_args_count": float(len(self.arguments)),
        }


@dataclass
class ThreadState:
    """State of a Python thread."""
    tid: int
    name: str
    frames: List[StackFrame]
    holds_gil: bool = False
    is_gc_collecting: bool = False
    is_daemon: bool = False
    is_alive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tid": self.tid,
            "name": self.name,
            "frame_count": len(self.frames),
            "holds_gil": self.holds_gil,
            "is_gc_collecting": self.is_gc_collecting,
            "is_daemon": self.is_daemon,
            "is_alive": self.is_alive,
            "top_frame": self.frames[0].to_dict() if self.frames else None,
        }

    def to_metrics(self) -> Dict[str, float]:
        """Numeric metrics for PySR."""
        return {
            "thread_frame_count": float(len(self.frames)),
            "thread_holds_gil": 1.0 if self.holds_gil else 0.0,
            "thread_is_gc": 1.0 if self.is_gc_collecting else 0.0,
        }

    @property
    def call_stack(self) -> List[str]:
        """Get call stack as list of function names."""
        return [f.location.function for f in self.frames]


@dataclass
class StackSnapshot:
    """Complete snapshot of all thread stacks."""
    timestamp: float
    threads: List[ThreadState]
    current_thread_id: int
    active_thread_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "thread_count": len(self.threads),
            "current_thread_id": self.current_thread_id,
            "active_thread_count": self.active_thread_count,
            "threads": [t.to_dict() for t in self.threads],
        }

    def to_metrics(self) -> Dict[str, float]:
        """Aggregate metrics for PySR."""
        total_frames = sum(len(t.frames) for t in self.threads)
        max_depth = max((len(t.frames) for t in self.threads), default=0)
        return {
            "snapshot_thread_count": float(len(self.threads)),
            "snapshot_total_frames": float(total_frames),
            "snapshot_max_depth": float(max_depth),
            "snapshot_active_threads": float(self.active_thread_count),
        }

    def get_thread(self, tid: int) -> Optional[ThreadState]:
        """Get thread by ID."""
        for t in self.threads:
            if t.tid == tid:
                return t
        return None

    def get_current_thread(self) -> Optional[ThreadState]:
        """Get the current thread's state."""
        return self.get_thread(self.current_thread_id)


def _safe_repr(value: Any, max_len: int = 100) -> str:
    """Safely get repr of a value."""
    try:
        r = repr(value)
        if len(r) > max_len:
            return r[:max_len] + "..."
        return r
    except Exception:
        return "<repr failed>"


def _extract_frame_info(
    frame: Any,
    depth: int = 0,
    capture_locals: bool = False,
    capture_args: bool = True,
) -> StackFrame:
    """Extract information from a Python frame object."""
    code = frame.f_code

    location = FrameLocation(
        filename=code.co_filename,
        lineno=frame.f_lineno,
        function=code.co_name,
        module=frame.f_globals.get("__name__"),
    )

    arguments = {}
    if capture_args:
        # Extract function arguments
        try:
            arginfo = inspect.getargvalues(frame)
            for arg in arginfo.args:
                if arg in arginfo.locals:
                    arguments[arg] = _safe_repr(arginfo.locals[arg])
        except Exception:
            pass

    locals_dict = {}
    if capture_locals:
        try:
            for name, value in frame.f_locals.items():
                if not name.startswith("_"):
                    locals_dict[name] = _safe_repr(value)
        except Exception:
            pass

    return StackFrame(
        location=location,
        arguments=arguments,
        locals=locals_dict,
        is_native=False,
        depth=depth,
    )


def capture_thread_stack(
    thread_id: Optional[int] = None,
    capture_locals: bool = False,
    max_depth: int = 100,
) -> ThreadState:
    """
    Capture stack for a specific thread or current thread.

    Args:
        thread_id: Thread ID to capture (None for current)
        capture_locals: Whether to capture local variables
        max_depth: Maximum stack depth to capture
    """
    if thread_id is None:
        thread_id = threading.current_thread().ident

    # Get all thread frames
    frames_dict = sys._current_frames()

    if thread_id not in frames_dict:
        return ThreadState(
            tid=thread_id,
            name="unknown",
            frames=[],
            is_alive=False,
        )

    # Find thread info
    thread_name = "unknown"
    is_daemon = False
    for t in threading.enumerate():
        if t.ident == thread_id:
            thread_name = t.name
            is_daemon = t.daemon
            break

    # Walk the frame stack
    frames = []
    frame = frames_dict[thread_id]
    depth = 0

    while frame is not None and depth < max_depth:
        frames.append(_extract_frame_info(frame, depth, capture_locals))
        frame = frame.f_back
        depth += 1

    return ThreadState(
        tid=thread_id,
        name=thread_name,
        frames=frames,
        is_daemon=is_daemon,
        is_alive=True,
    )


def capture_all_stacks(
    capture_locals: bool = False,
    max_depth: int = 100,
) -> StackSnapshot:
    """
    Capture stack snapshots for all threads.

    Args:
        capture_locals: Whether to capture local variables
        max_depth: Maximum stack depth per thread
    """
    timestamp = time.time()
    frames_dict = sys._current_frames()
    current_tid = threading.current_thread().ident

    # Build thread info map
    thread_info = {}
    for t in threading.enumerate():
        thread_info[t.ident] = (t.name, t.daemon, t.is_alive())

    threads = []
    for tid, frame in frames_dict.items():
        name, is_daemon, is_alive = thread_info.get(tid, ("unknown", False, True))

        # Walk frame stack
        frames = []
        depth = 0
        current_frame = frame
        while current_frame is not None and depth < max_depth:
            frames.append(_extract_frame_info(current_frame, depth, capture_locals))
            current_frame = current_frame.f_back
            depth += 1

        threads.append(ThreadState(
            tid=tid,
            name=name,
            frames=frames,
            is_daemon=is_daemon,
            is_alive=is_alive,
        ))

    return StackSnapshot(
        timestamp=timestamp,
        threads=threads,
        current_thread_id=current_tid,
        active_thread_count=threading.active_count(),
    )


def capture_with_pystack(
    pid: Optional[int] = None,
    capture_locals: bool = False,
) -> Optional[StackSnapshot]:
    """
    Capture using pystack for enhanced native frame support.

    Requires pystack to be installed and may require root/ptrace permissions.
    """
    if not PYSTACK_AVAILABLE:
        return None

    if pid is None:
        pid = os.getpid()

    try:
        pystack_threads = get_process_threads(pid, stop_process=False, locals=capture_locals)
    except Exception:
        return None

    timestamp = time.time()
    threads = []

    for pt in pystack_threads:
        frames = []
        depth = 0

        # Python frames
        for pf in pt.all_frames:
            location = FrameLocation(
                filename=pf.code.filename,
                lineno=pf.code.location.lineno if pf.code.location else 0,
                function=pf.code.scope,
            )
            frames.append(StackFrame(
                location=location,
                arguments=dict(pf.arguments) if pf.arguments else {},
                locals=dict(pf.locals) if pf.locals else {},
                is_native=False,
                depth=depth,
            ))
            depth += 1

        # Native frames (if available)
        if hasattr(pt, 'native_frames') and pt.native_frames:
            for nf in pt.native_frames:
                location = FrameLocation(
                    filename=nf.path or nf.library,
                    lineno=nf.linenumber,
                    function=nf.symbol,
                )
                frames.append(StackFrame(
                    location=location,
                    is_native=True,
                    depth=depth,
                ))
                depth += 1

        threads.append(ThreadState(
            tid=pt.tid,
            name=pt.name or f"Thread-{pt.tid}",
            frames=frames,
            holds_gil=pt.holds_the_gil > 0,
            is_gc_collecting=pt.is_gc_collecting > 0,
        ))

    return StackSnapshot(
        timestamp=timestamp,
        threads=threads,
        current_thread_id=threading.current_thread().ident,
        active_thread_count=len(threads),
    )


class StackTracer:
    """
    Stack tracer for continuous or on-demand stack sampling.

    Usage:
        tracer = StackTracer()

        # Capture single snapshot
        snapshot = tracer.capture()

        # Sample periodically
        tracer.start_sampling(interval_ms=10)
        # ... do work ...
        samples = tracer.stop_sampling()

        # Analyze hot paths
        hot_paths = tracer.get_hot_paths()
    """

    def __init__(
        self,
        capture_locals: bool = False,
        use_pystack: bool = True,
        max_depth: int = 100,
    ):
        self.capture_locals = capture_locals
        self.use_pystack = use_pystack and PYSTACK_AVAILABLE
        self.max_depth = max_depth

        self.snapshots: List[StackSnapshot] = []
        self._sampling = False
        self._sample_thread: Optional[threading.Thread] = None

    def capture(self) -> StackSnapshot:
        """Capture a single stack snapshot."""
        if self.use_pystack:
            snapshot = capture_with_pystack(capture_locals=self.capture_locals)
            if snapshot:
                self.snapshots.append(snapshot)
                return snapshot

        snapshot = capture_all_stacks(
            capture_locals=self.capture_locals,
            max_depth=self.max_depth,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def start_sampling(self, interval_ms: int = 10):
        """Start periodic stack sampling."""
        if self._sampling:
            return

        self._sampling = True

        def sample_loop():
            while self._sampling:
                self.capture()
                time.sleep(interval_ms / 1000.0)

        self._sample_thread = threading.Thread(target=sample_loop, daemon=True)
        self._sample_thread.start()

    def stop_sampling(self) -> List[StackSnapshot]:
        """Stop sampling and return collected snapshots."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)
            self._sample_thread = None
        return self.snapshots

    def get_hot_paths(self, top_n: int = 10) -> List[Tuple[Tuple[str, ...], int]]:
        """
        Analyze samples to find hot paths (frequently occurring call stacks).

        Returns list of (call_stack, count) tuples.
        """
        from collections import Counter

        path_counts: Counter = Counter()

        for snapshot in self.snapshots:
            for thread in snapshot.threads:
                # Create tuple of function names as path signature
                path = tuple(f.location.function for f in thread.frames[:10])
                if path:
                    path_counts[path] += 1

        return path_counts.most_common(top_n)

    def get_function_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics per function across all samples.

        Returns dict of function -> {count, avg_depth, locations}
        """
        from collections import defaultdict

        stats = defaultdict(lambda: {"count": 0, "depths": [], "locations": set()})

        for snapshot in self.snapshots:
            for thread in snapshot.threads:
                for frame in thread.frames:
                    func = frame.location.function
                    stats[func]["count"] += 1
                    stats[func]["depths"].append(frame.depth)
                    stats[func]["locations"].add(
                        f"{frame.location.filename}:{frame.location.lineno}"
                    )

        # Compute averages
        result = {}
        for func, data in stats.items():
            result[func] = {
                "count": data["count"],
                "avg_depth": sum(data["depths"]) / len(data["depths"]) if data["depths"] else 0,
                "locations": list(data["locations"]),
            }

        return result

    def get_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics for PySR."""
        if not self.snapshots:
            return {}

        total_threads = sum(len(s.threads) for s in self.snapshots)
        total_frames = sum(
            sum(len(t.frames) for t in s.threads)
            for s in self.snapshots
        )
        max_depth = max(
            max((len(t.frames) for t in s.threads), default=0)
            for s in self.snapshots
        )

        return {
            "stack_sample_count": float(len(self.snapshots)),
            "stack_total_threads": float(total_threads),
            "stack_total_frames": float(total_frames),
            "stack_max_depth": float(max_depth),
            "stack_avg_threads": total_threads / len(self.snapshots),
        }

    def clear(self):
        """Clear collected snapshots."""
        self.snapshots = []
