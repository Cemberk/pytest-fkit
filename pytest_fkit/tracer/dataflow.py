"""
Dataflow Tracking: Computation DAG and value provenance tracing.

Based on patterns from redun - provides:
- Call node tracking with content-addressable hashing
- Value provenance/lineage tracking
- Computation DAG building and visualization
- Dataflow walking for dependency analysis
- Cross-correlation with stack frames and metrics
"""

import hashlib
import inspect
import time
import threading
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")


def compute_value_hash(value: Any) -> str:
    """Compute content-addressable hash for any value."""
    try:
        # Use repr for hashable representation
        content = repr(value)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    except Exception:
        # Fallback to id-based hash
        return hashlib.sha256(str(id(value)).encode()).hexdigest()[:16]


def compute_call_hash(
    func_name: str,
    args_hash: str,
    kwargs_hash: str,
) -> str:
    """Compute hash for a function call."""
    content = f"{func_name}:{args_hash}:{kwargs_hash}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class NodeType(Enum):
    """Types of nodes in the dataflow graph."""
    VALUE = "value"
    CALL = "call"
    ARGUMENT = "argument"
    RESULT = "result"


@dataclass
class ValueNode:
    """A value in the computation DAG."""
    value_hash: str
    value_type: str
    preview: str
    timestamp: float = field(default_factory=time.time)
    parent_call: Optional[str] = None  # call_hash that produced this
    child_calls: List[str] = field(default_factory=list)  # calls that use this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "value",
            "value_hash": self.value_hash,
            "value_type": self.value_type,
            "preview": self.preview,
            "timestamp": self.timestamp,
            "parent_call": self.parent_call,
            "child_calls": self.child_calls,
        }


@dataclass
class ArgumentNode:
    """An argument in a function call."""
    arg_name: str
    arg_position: int
    value_hash: str
    call_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "argument",
            "arg_name": self.arg_name,
            "arg_position": self.arg_position,
            "value_hash": self.value_hash,
            "call_hash": self.call_hash,
        }


@dataclass
class CallNode:
    """A function call in the computation DAG."""
    call_hash: str
    func_name: str
    func_module: Optional[str]
    args_hash: str
    kwargs_hash: str
    result_hash: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    parent_call: Optional[str] = None
    child_calls: List[str] = field(default_factory=list)
    arguments: List[ArgumentNode] = field(default_factory=list)
    source_location: Optional[str] = None

    # Correlation IDs for cross-dimensional analysis
    stack_frame_id: Optional[str] = None
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "call",
            "call_hash": self.call_hash,
            "func_name": self.func_name,
            "func_module": self.func_module,
            "args_hash": self.args_hash,
            "kwargs_hash": self.kwargs_hash,
            "result_hash": self.result_hash,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "parent_call": self.parent_call,
            "child_calls": self.child_calls,
            "arguments": [a.to_dict() for a in self.arguments],
            "source_location": self.source_location,
            "stack_frame_id": self.stack_frame_id,
            "trace_id": self.trace_id,
        }

    def to_metrics(self) -> Dict[str, float]:
        """Numeric metrics for PySR."""
        return {
            "call_duration_ms": self.duration_ms,
            "call_arg_count": float(len(self.arguments)),
            "call_child_count": float(len(self.child_calls)),
            "call_depth": 0.0,  # Will be computed by tracker
        }


@dataclass
class DataflowEdge:
    """An edge in the dataflow graph."""
    src_hash: str
    src_type: NodeType
    dest_hash: Optional[str]
    dest_type: Optional[NodeType]
    edge_type: str  # "produces", "consumes", "calls", "returns"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_hash": self.src_hash,
            "src_type": self.src_type.value,
            "dest_hash": self.dest_hash,
            "dest_type": self.dest_type.value if self.dest_type else None,
            "edge_type": self.edge_type,
        }


class ExecutionContext:
    """Context for a workflow execution."""

    def __init__(self, execution_id: Optional[str] = None):
        self.id = execution_id or hashlib.sha256(
            str(time.time()).encode()
        ).hexdigest()[:16]
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.root_call: Optional[str] = None
        self.tags: Dict[str, str] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "root_call": self.root_call,
            "tags": self.tags,
            "duration_ms": (
                (self.end_time - self.start_time) * 1000
                if self.end_time else None
            ),
        }


class DataflowTracker:
    """
    Tracks computation dataflow for provenance and dependency analysis.

    Usage:
        tracker = DataflowTracker()

        # Wrap functions to track calls
        @tracker.track
        def my_computation(x, y):
            return x + y

        result = my_computation(1, 2)

        # Get provenance of result
        lineage = tracker.get_value_lineage(result)

        # Walk upstream dataflow
        for edge in tracker.walk_upstream(result):
            print(edge)
    """

    def __init__(self):
        self.values: Dict[str, ValueNode] = {}
        self.calls: Dict[str, CallNode] = {}
        self.edges: List[DataflowEdge] = []
        self.executions: Dict[str, ExecutionContext] = {}

        self._current_execution: Optional[ExecutionContext] = None
        self._call_stack: List[str] = []  # Stack of call_hashes
        self._lock = threading.RLock()

        # Weak reference to track value -> hash mapping
        self._value_refs: Dict[int, str] = {}

    def start_execution(self, tags: Optional[Dict[str, str]] = None) -> ExecutionContext:
        """Start a new execution context."""
        with self._lock:
            ctx = ExecutionContext()
            if tags:
                ctx.tags.update(tags)
            self.executions[ctx.id] = ctx
            self._current_execution = ctx
            return ctx

    def end_execution(self) -> Optional[ExecutionContext]:
        """End current execution context."""
        with self._lock:
            if self._current_execution:
                self._current_execution.end_time = time.time()
                ctx = self._current_execution
                self._current_execution = None
                return ctx
            return None

    def register_value(
        self,
        value: Any,
        parent_call: Optional[str] = None,
    ) -> ValueNode:
        """Register a value in the DAG."""
        with self._lock:
            value_hash = compute_value_hash(value)

            if value_hash not in self.values:
                try:
                    preview = repr(value)[:200]
                except Exception:
                    preview = f"<{type(value).__name__}>"

                node = ValueNode(
                    value_hash=value_hash,
                    value_type=type(value).__name__,
                    preview=preview,
                    parent_call=parent_call,
                )
                self.values[value_hash] = node

                # Track weak reference
                try:
                    self._value_refs[id(value)] = value_hash
                except Exception:
                    pass

            return self.values[value_hash]

    def register_call(
        self,
        func: Callable,
        args: Tuple,
        kwargs: Dict,
        trace_id: Optional[str] = None,
    ) -> CallNode:
        """Register a function call in the DAG."""
        with self._lock:
            func_name = getattr(func, "__name__", str(func))
            func_module = getattr(func, "__module__", None)

            # Hash arguments
            args_hash = compute_value_hash(args)
            kwargs_hash = compute_value_hash(tuple(sorted(kwargs.items())))
            call_hash = compute_call_hash(func_name, args_hash, kwargs_hash)

            # Get source location
            source_location = None
            try:
                source_file = inspect.getfile(func)
                source_lines = inspect.getsourcelines(func)
                source_location = f"{source_file}:{source_lines[1]}"
            except Exception:
                pass

            # Create argument nodes
            arguments = []
            sig = None
            try:
                sig = inspect.signature(func)
            except Exception:
                pass

            params = list(sig.parameters.keys()) if sig else []

            for i, arg in enumerate(args):
                arg_name = params[i] if i < len(params) else f"arg{i}"
                arg_hash = compute_value_hash(arg)
                arguments.append(ArgumentNode(
                    arg_name=arg_name,
                    arg_position=i,
                    value_hash=arg_hash,
                    call_hash=call_hash,
                ))

                # Register argument value
                self.register_value(arg)

                # Add edge: value -> call
                self.edges.append(DataflowEdge(
                    src_hash=arg_hash,
                    src_type=NodeType.VALUE,
                    dest_hash=call_hash,
                    dest_type=NodeType.CALL,
                    edge_type="consumes",
                ))

            for key, val in kwargs.items():
                arg_hash = compute_value_hash(val)
                arguments.append(ArgumentNode(
                    arg_name=key,
                    arg_position=-1,
                    value_hash=arg_hash,
                    call_hash=call_hash,
                ))

                self.register_value(val)
                self.edges.append(DataflowEdge(
                    src_hash=arg_hash,
                    src_type=NodeType.VALUE,
                    dest_hash=call_hash,
                    dest_type=NodeType.CALL,
                    edge_type="consumes",
                ))

            # Create call node
            parent_call = self._call_stack[-1] if self._call_stack else None

            node = CallNode(
                call_hash=call_hash,
                func_name=func_name,
                func_module=func_module,
                args_hash=args_hash,
                kwargs_hash=kwargs_hash,
                parent_call=parent_call,
                arguments=arguments,
                source_location=source_location,
                trace_id=trace_id,
            )

            self.calls[call_hash] = node

            # Link to parent
            if parent_call and parent_call in self.calls:
                self.calls[parent_call].child_calls.append(call_hash)
                self.edges.append(DataflowEdge(
                    src_hash=parent_call,
                    src_type=NodeType.CALL,
                    dest_hash=call_hash,
                    dest_type=NodeType.CALL,
                    edge_type="calls",
                ))

            # Set as root if first call
            if self._current_execution and not self._current_execution.root_call:
                self._current_execution.root_call = call_hash

            # Push onto call stack
            self._call_stack.append(call_hash)

            return node

    def complete_call(
        self,
        call_hash: str,
        result: Any,
        duration_ms: float,
    ) -> Optional[CallNode]:
        """Complete a call with its result."""
        with self._lock:
            if call_hash not in self.calls:
                return None

            node = self.calls[call_hash]
            result_hash = compute_value_hash(result)

            node.result_hash = result_hash
            node.duration_ms = duration_ms

            # Register result value
            value_node = self.register_value(result, parent_call=call_hash)

            # Add edge: call -> result
            self.edges.append(DataflowEdge(
                src_hash=call_hash,
                src_type=NodeType.CALL,
                dest_hash=result_hash,
                dest_type=NodeType.VALUE,
                edge_type="produces",
            ))

            # Pop from call stack
            if self._call_stack and self._call_stack[-1] == call_hash:
                self._call_stack.pop()

            return node

    def track(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to track function calls."""
        def wrapper(*args, **kwargs) -> T:
            start = time.time()
            call_node = self.register_call(func, args, kwargs)

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                self.complete_call(call_node.call_hash, result, duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                # Register exception as result
                self.complete_call(
                    call_node.call_hash,
                    f"Exception: {type(e).__name__}: {e}",
                    duration_ms,
                )
                raise

        return wrapper

    def get_value_lineage(self, value: Any) -> List[CallNode]:
        """Get the lineage of calls that produced a value."""
        value_hash = compute_value_hash(value)
        lineage = []

        if value_hash in self.values:
            node = self.values[value_hash]
            current_call = node.parent_call

            while current_call and current_call in self.calls:
                call = self.calls[current_call]
                lineage.append(call)
                current_call = call.parent_call

        return lineage

    def walk_upstream(
        self,
        value: Any,
        max_depth: int = 100,
    ) -> Iterator[DataflowEdge]:
        """Walk upstream through the dataflow from a value."""
        value_hash = compute_value_hash(value)
        visited: Set[str] = set()
        queue = [value_hash]
        depth = 0

        while queue and depth < max_depth:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for edge in self.edges:
                if edge.dest_hash == current:
                    yield edge
                    if edge.src_hash not in visited:
                        queue.append(edge.src_hash)

            depth += 1

    def walk_downstream(
        self,
        value: Any,
        max_depth: int = 100,
    ) -> Iterator[DataflowEdge]:
        """Walk downstream through the dataflow from a value."""
        value_hash = compute_value_hash(value)
        visited: Set[str] = set()
        queue = [value_hash]
        depth = 0

        while queue and depth < max_depth:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for edge in self.edges:
                if edge.src_hash == current:
                    yield edge
                    if edge.dest_hash and edge.dest_hash not in visited:
                        queue.append(edge.dest_hash)

            depth += 1

    def get_call_depth(self, call_hash: str) -> int:
        """Get the depth of a call in the execution tree."""
        depth = 0
        current = call_hash

        while current in self.calls:
            parent = self.calls[current].parent_call
            if not parent:
                break
            depth += 1
            current = parent

        return depth

    def get_hot_calls(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the calls with highest total duration."""
        call_times: Dict[str, float] = {}

        for call in self.calls.values():
            key = f"{call.func_module}.{call.func_name}" if call.func_module else call.func_name
            call_times[key] = call_times.get(key, 0) + call.duration_ms

        sorted_calls = sorted(
            call_times.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_calls[:top_n]

    def get_call_graph(self) -> Dict[str, Any]:
        """Get the full call graph structure."""
        return {
            "calls": {h: c.to_dict() for h, c in self.calls.items()},
            "values": {h: v.to_dict() for h, v in self.values.items()},
            "edges": [e.to_dict() for e in self.edges],
            "executions": {
                eid: e.to_dict() for eid, e in self.executions.items()
            },
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics for PySR."""
        if not self.calls:
            return {}

        total_duration = sum(c.duration_ms for c in self.calls.values())
        max_depth = max(
            (self.get_call_depth(h) for h in self.calls.keys()),
            default=0,
        )

        return {
            "dataflow_call_count": float(len(self.calls)),
            "dataflow_value_count": float(len(self.values)),
            "dataflow_edge_count": float(len(self.edges)),
            "dataflow_total_duration_ms": total_duration,
            "dataflow_max_depth": float(max_depth),
            "dataflow_avg_duration_ms": (
                total_duration / len(self.calls) if self.calls else 0
            ),
        }

    def clear(self):
        """Clear all tracked data."""
        with self._lock:
            self.values.clear()
            self.calls.clear()
            self.edges.clear()
            self.executions.clear()
            self._call_stack.clear()
            self._value_refs.clear()


def visualize_dataflow(
    tracker: DataflowTracker,
    value: Any,
    max_lines: int = 50,
) -> str:
    """
    Generate text visualization of value's dataflow.

    Format similar to redun's dataflow visualization:

    value <-- func1(arg1, arg2)
          <-- func2(arg3)
      arg1 = <hash> preview
      arg2 = <hash> preview
    """
    lines = []
    value_hash = compute_value_hash(value)

    if value_hash not in tracker.values:
        return "Value not tracked"

    value_node = tracker.values[value_hash]
    lines.append(f"value = <{value_hash[:8]}> {value_node.preview[:60]}")
    lines.append("")

    # Walk upstream and build visualization
    visited: Set[str] = set()

    def visualize_call(call_hash: str, indent: int = 0) -> None:
        if call_hash in visited or len(lines) > max_lines:
            return
        visited.add(call_hash)

        if call_hash not in tracker.calls:
            return

        call = tracker.calls[call_hash]
        prefix = " " * indent

        # Call line
        args_str = ", ".join(a.arg_name for a in call.arguments)
        lines.append(f"{prefix}value <-- <{call_hash[:8]}> {call.func_name}({args_str})")

        # Argument lines
        for arg in call.arguments:
            if arg.value_hash in tracker.values:
                val = tracker.values[arg.value_hash]
                lines.append(
                    f"{prefix}  {arg.arg_name:12} = <{arg.value_hash[:8]}> {val.preview[:40]}"
                )

        lines.append("")

        # Recurse to parent
        if call.parent_call:
            visualize_call(call.parent_call, indent + 2)

    if value_node.parent_call:
        visualize_call(value_node.parent_call)
    else:
        lines.append("value <-- origin")

    return "\n".join(lines[:max_lines])


class TrackedExecution:
    """
    Context manager for tracking an execution with dataflow.

    Usage:
        tracker = DataflowTracker()

        with TrackedExecution(tracker, tags={"version": "1.0"}) as exec:
            # All tracked calls within this block are associated
            result = tracked_function()

        print(exec.duration_ms)
    """

    def __init__(
        self,
        tracker: DataflowTracker,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.tracker = tracker
        self.tags = tags or {}
        self.execution: Optional[ExecutionContext] = None

    def __enter__(self) -> ExecutionContext:
        self.execution = self.tracker.start_execution(self.tags)
        return self.execution

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.tracker.end_execution()
