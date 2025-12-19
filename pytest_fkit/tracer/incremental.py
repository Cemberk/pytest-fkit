"""
Incremental Computation Layer for on-the-fly formula discovery.

Provides:
- Content-addressable caching with dependency tracking
- Memoization decorators for expensive computations
- Incremental data accumulation for PySR integration
- Hash-based cache invalidation

Based on patterns from redun (Insitro) and mandala projects.
"""

import ast
import functools
import hashlib
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")
Func = TypeVar("Func", bound=Callable)


class CacheScope(Enum):
    """Cache scope levels (from redun)."""

    NONE = "NONE"  # No caching
    SESSION = "SESSION"  # Cache within current session only
    PERSISTENT = "PERSISTENT"  # Cache across sessions (file-backed)


class CacheResult(Enum):
    """Types of cache hits/misses."""

    HIT = "HIT"  # Cache hit
    MISS = "MISS"  # Cache miss
    STALE = "STALE"  # Cached but invalidated by dependency change


def compute_hash(content: Any, normalize: bool = True) -> str:
    """
    Compute content-aware hash.

    For code strings, normalizes to be insensitive to comments/formatting.
    For other objects, uses pickle serialization.

    Args:
        content: Content to hash
        normalize: Whether to normalize code (remove comments, etc.)

    Returns:
        SHA256 hash string (first 16 chars)
    """
    if isinstance(content, str):
        if normalize:
            try:
                # Parse and normalize AST for Python code
                tree = ast.parse(content)
                # Remove docstrings
                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
                    ):
                        if (
                            node.body
                            and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)
                        ):
                            node.body = node.body[1:]
                normalized = ast.dump(tree)
            except SyntaxError:
                # Not valid Python, use raw content
                normalized = content
        else:
            normalized = content
        data = normalized.encode()
    else:
        try:
            data = pickle.dumps(content)
        except Exception:
            data = str(content).encode()

    return hashlib.sha256(data).hexdigest()[:16]


def compute_args_hash(*args, **kwargs) -> str:
    """Compute hash of function arguments."""
    try:
        data = pickle.dumps((args, tuple(sorted(kwargs.items()))))
    except Exception:
        data = str((args, kwargs)).encode()
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class CacheEntry:
    """A cached computation result."""

    key: str
    value: Any
    hash: str
    timestamp: float
    dependencies: Set[str] = field(default_factory=set)
    ttl: Optional[float] = None  # Time-to-live in seconds

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class DependencyTracker:
    """
    Track dependencies for incremental computation.

    Implements dependency tracking similar to redun's task graph,
    enabling smart cache invalidation when upstream data changes.

    Usage:
        tracker = DependencyTracker()

        # Register computation with dependencies
        tracker.register("analysis", deps=["data_v1"], hash="abc123")

        # When data changes, mark dependents as dirty
        tracker.mark_dirty("data_v1")

        # Check if cached result is still valid
        if not tracker.is_valid("analysis", current_hash):
            # Recompute...
    """

    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = {}
        self.hashes: Dict[str, str] = {}
        self.dirty: Set[str] = set()

    def register(self, name: str, deps: List[str], content_hash: str) -> None:
        """Register a computation with its dependencies."""
        self.dependencies[name] = set(deps)
        self.hashes[name] = content_hash
        # Clear dirty flag when freshly registered
        self.dirty.discard(name)

    def mark_dirty(self, name: str) -> None:
        """Mark a node and all its dependents as dirty."""
        self.dirty.add(name)
        # Propagate to dependents
        for node, deps in self.dependencies.items():
            if name in deps and node not in self.dirty:
                self.mark_dirty(node)

    def is_valid(self, name: str, current_hash: str) -> bool:
        """Check if a cached result is still valid."""
        if name in self.dirty:
            return False
        if name not in self.hashes:
            return False
        return self.hashes[name] == current_hash

    def get_stale_nodes(self) -> List[str]:
        """Get all nodes that need recomputation."""
        return list(self.dirty)

    def get_dependents(self, name: str) -> List[str]:
        """Get all nodes that depend on the given node."""
        dependents = []
        for node, deps in self.dependencies.items():
            if name in deps:
                dependents.append(node)
        return dependents

    def clear(self) -> None:
        """Clear all tracking data."""
        self.dependencies.clear()
        self.hashes.clear()
        self.dirty.clear()


class IncrementalCache:
    """
    Incremental computation cache with LRU eviction and TTL.

    Designed for streaming data scenarios where new data arrives
    continuously and we want to incrementally update computations.

    Usage:
        cache = IncrementalCache(max_size=1000, default_ttl=3600)

        # Cache expensive computation
        result = cache.get_or_compute(
            "my_analysis",
            compute_fn=lambda: expensive_analysis(data),
            deps=["data_hash"],
        )

        # When data changes
        cache.invalidate("data_hash")

        # Next call will recompute
        result = cache.get_or_compute(...)
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU
        self._tracker = DependencyTracker()

        # Stats
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Tuple[Optional[Any], CacheResult]:
        """
        Get cached value if valid.

        Returns:
            Tuple of (value or None, CacheResult)
        """
        if key not in self._cache:
            self.misses += 1
            return None, CacheResult.MISS

        entry = self._cache[key]

        # Check TTL
        if entry.is_expired():
            del self._cache[key]
            self._access_order.remove(key)
            self.misses += 1
            return None, CacheResult.STALE

        # Check dependency validity
        if key in self._tracker.dirty:
            self.misses += 1
            return None, CacheResult.STALE

        # Update access order for LRU
        self._access_order.remove(key)
        self._access_order.append(key)

        self.hits += 1
        return entry.value, CacheResult.HIT

    def set(
        self,
        key: str,
        value: Any,
        content_hash: str,
        deps: Optional[List[str]] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            content_hash: Hash of computation inputs
            deps: Dependencies for invalidation tracking
            ttl: Time-to-live (overrides default)
        """
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        entry = CacheEntry(
            key=key,
            value=value,
            hash=content_hash,
            timestamp=time.time(),
            dependencies=set(deps or []),
            ttl=ttl if ttl is not None else self.default_ttl,
        )

        self._cache[key] = entry
        self._access_order.append(key)

        # Register with dependency tracker
        if deps:
            self._tracker.register(key, deps, content_hash)

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        deps: Optional[List[str]] = None,
        ttl: Optional[float] = None,
    ) -> T:
        """
        Get cached value or compute and cache.

        This is the primary interface for incremental computation.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            deps: Dependencies for invalidation
            ttl: Time-to-live

        Returns:
            Cached or computed value
        """
        value, result = self.get(key)
        if result == CacheResult.HIT:
            return value

        # Compute fresh value
        value = compute_fn()

        # Cache it
        content_hash = compute_hash((key, deps))
        self.set(key, value, content_hash, deps, ttl)

        return value

    def invalidate(self, key: str) -> int:
        """
        Invalidate a key and all its dependents.

        Returns:
            Number of entries invalidated
        """
        self._tracker.mark_dirty(key)
        stale = self._tracker.get_stale_nodes()

        # Remove stale entries from cache
        count = 0
        for k in stale:
            if k in self._cache:
                del self._cache[k]
                self._access_order.remove(k)
                count += 1

        return count

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._access_order.clear()
        self._tracker.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "stale_nodes": len(self._tracker.dirty),
        }


def memoize(
    cache: Optional[IncrementalCache] = None,
    key_fn: Optional[Callable[..., str]] = None,
    ttl: Optional[float] = None,
) -> Callable[[Func], Func]:
    """
    Decorator for memoizing functions with incremental cache.

    Usage:
        cache = IncrementalCache()

        @memoize(cache=cache, ttl=3600)
        def expensive_analysis(data):
            ...

    Args:
        cache: IncrementalCache instance (creates new if None)
        key_fn: Custom function to generate cache key from args
        ttl: Time-to-live for cached results
    """
    _cache = cache or IncrementalCache()

    def decorator(fn: Func) -> Func:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{fn.__module__}.{fn.__name__}:{compute_args_hash(*args, **kwargs)}"

            return _cache.get_or_compute(
                key=key,
                compute_fn=lambda: fn(*args, **kwargs),
                ttl=ttl,
            )

        # Attach cache for access
        wrapper._cache = _cache  # type: ignore
        return wrapper  # type: ignore

    return decorator


class IncrementalDataAccumulator:
    """
    Accumulate data incrementally for streaming analysis.

    Designed for PySR integration where we want to:
    1. Collect numeric data points over time
    2. Trigger formula discovery when enough data is collected
    3. Invalidate formulas when new data significantly changes patterns

    Usage:
        accumulator = IncrementalDataAccumulator(
            batch_size=100,
            on_batch=lambda data: run_pysr(data),
        )

        # Add data points as they come
        for row in stream:
            accumulator.add(row)  # Triggers batch processing automatically
    """

    def __init__(
        self,
        batch_size: int = 100,
        on_batch: Optional[Callable[[List[Dict[str, float]]], Any]] = None,
        cache: Optional[IncrementalCache] = None,
    ):
        self.batch_size = batch_size
        self.on_batch = on_batch
        self.cache = cache or IncrementalCache()

        self._data: List[Dict[str, float]] = []
        self._batch_count = 0
        self._last_batch_hash: Optional[str] = None
        self._batch_results: List[Any] = []

    def add(self, row: Dict[str, float]) -> Optional[Any]:
        """
        Add a data point.

        Returns batch result if batch was triggered, None otherwise.
        """
        self._data.append(row)

        if len(self._data) >= self.batch_size:
            return self._process_batch()
        return None

    def add_many(self, rows: List[Dict[str, float]]) -> List[Any]:
        """Add multiple data points, returning any batch results."""
        results = []
        for row in rows:
            result = self.add(row)
            if result is not None:
                results.append(result)
        return results

    def _process_batch(self) -> Optional[Any]:
        """Process accumulated batch."""
        if not self._data:
            return None

        batch = self._data.copy()
        self._data = []
        self._batch_count += 1

        # Compute batch hash for cache key
        batch_hash = compute_hash(batch)

        # Check if batch is significantly different from last
        cache_key = f"batch_{self._batch_count}"

        if self.on_batch:
            result = self.cache.get_or_compute(
                key=cache_key,
                compute_fn=lambda: self.on_batch(batch),
                deps=[batch_hash],
            )
            self._batch_results.append(result)
            self._last_batch_hash = batch_hash
            return result

        return batch

    def flush(self) -> Optional[Any]:
        """Process any remaining data as final batch."""
        if self._data:
            return self._process_batch()
        return None

    def get_all_data(self) -> List[Dict[str, float]]:
        """Get all accumulated data (including unprocessed)."""
        return self._data.copy()

    def get_batch_results(self) -> List[Any]:
        """Get all batch processing results."""
        return self._batch_results.copy()

    def reset(self) -> None:
        """Reset accumulator state."""
        self._data = []
        self._batch_count = 0
        self._last_batch_hash = None
        self._batch_results = []
