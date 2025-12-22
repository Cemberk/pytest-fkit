"""
Refactor Expert: Pinpoints code locations and commits for blame analysis.

Provides deep analysis of execution traces to identify:
- Specific file:line locations causing bottlenecks
- Git commits responsible for problematic code
- Code patterns that need refactoring
- Stack layers responsible for issues

This expert helps answer: "Where in the code should we look to fix this?"
"""

import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .base import (
    ExpertPerspective,
    PerspectiveInsight,
    BlameTarget,
    SeverityLevel,
)

if TYPE_CHECKING:
    from ..correlation import CorrelationPoint
    from ..stack import StackFrame, FrameLocation


class CodeLayer(Enum):
    """Layers of the code stack."""
    APPLICATION = "application"      # User's application code
    LIBRARY = "library"              # Third-party libraries
    FRAMEWORK = "framework"          # ML frameworks (torch, tf, jax)
    RUNTIME = "runtime"              # Python runtime, C extensions
    SYSTEM = "system"                # OS-level, kernel


class RefactorCategory(Enum):
    """Categories of refactoring suggestions."""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    CONCURRENCY = "concurrency"
    STRUCTURE = "structure"
    ALGORITHM = "algorithm"


@dataclass
class CodeLocation:
    """
    A specific location in code.

    Represents a file:line that may be responsible for a bottleneck.
    """
    file_path: str
    line_number: int
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    layer: CodeLayer = CodeLayer.APPLICATION

    # Blame attribution
    occurrence_count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0

    # Git info (populated lazily)
    commit_hash: Optional[str] = None
    commit_author: Optional[str] = None
    commit_date: Optional[str] = None
    commit_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "module_name": self.module_name,
            "layer": self.layer.value,
            "occurrence_count": self.occurrence_count,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "commit_hash": self.commit_hash,
            "commit_author": self.commit_author,
            "commit_date": self.commit_date,
            "commit_message": self.commit_message,
        }

    @property
    def short_path(self) -> str:
        """Get shortened file path for display."""
        return os.path.basename(self.file_path)

    @property
    def location_str(self) -> str:
        """Get file:line string."""
        return f"{self.file_path}:{self.line_number}"

    def __hash__(self):
        return hash((self.file_path, self.line_number))

    def __eq__(self, other):
        if not isinstance(other, CodeLocation):
            return False
        return self.file_path == other.file_path and self.line_number == other.line_number


@dataclass
class CommitBlame:
    """
    Git blame information for code.
    """
    commit_hash: str
    author: str
    author_email: Optional[str] = None
    date: str = ""
    message: str = ""

    # Aggregated stats
    locations: List[CodeLocation] = field(default_factory=list)
    total_duration_ms: float = 0.0
    occurrence_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit_hash": self.commit_hash,
            "author": self.author,
            "author_email": self.author_email,
            "date": self.date,
            "message": self.message[:100] if self.message else "",
            "location_count": len(self.locations),
            "total_duration_ms": self.total_duration_ms,
            "occurrence_count": self.occurrence_count,
        }

    def to_blame_target(self) -> BlameTarget:
        return BlameTarget(
            target_type="commit",
            name=self.commit_hash[:8],
            confidence=0.7,
            commit_hash=self.commit_hash,
            impact_pct=(self.total_duration_ms / 1000) if self.total_duration_ms > 0 else 0,
            occurrence_count=self.occurrence_count,
            evidence=[
                f"Author: {self.author}",
                f"Date: {self.date}",
                f"Message: {self.message[:50]}..." if len(self.message) > 50 else f"Message: {self.message}",
                f"Affects {len(self.locations)} locations",
            ],
        )


@dataclass
class RefactorSuggestion:
    """
    A specific refactoring suggestion.
    """
    location: CodeLocation
    category: RefactorCategory
    title: str
    description: str
    priority: int = 0  # 0 = highest
    estimated_impact_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location.to_dict(),
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "estimated_impact_pct": self.estimated_impact_pct,
        }


class GitBlameHelper:
    """
    Helper for git blame operations.

    Caches blame results to avoid repeated git calls.
    """

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or self._find_repo_root()
        self._blame_cache: Dict[Tuple[str, int], CommitBlame] = {}

    def _find_repo_root(self) -> Optional[str]:
        """Find git repository root."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def is_available(self) -> bool:
        """Check if git is available."""
        return self.repo_root is not None

    @lru_cache(maxsize=1000)
    def blame_line(self, file_path: str, line_number: int) -> Optional[CommitBlame]:
        """
        Get git blame for a specific line.

        Returns CommitBlame with author, date, and commit info.
        """
        if not self.is_available():
            return None

        # Check cache
        cache_key = (file_path, line_number)
        if cache_key in self._blame_cache:
            return self._blame_cache[cache_key]

        try:
            # Make path relative to repo root if absolute
            if os.path.isabs(file_path) and self.repo_root:
                try:
                    file_path = os.path.relpath(file_path, self.repo_root)
                except ValueError:
                    pass  # Different drives on Windows

            # Run git blame for specific line
            cmd = [
                "git", "blame",
                "-L", f"{line_number},{line_number}",
                "--porcelain",
                file_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_root,
            )

            if result.returncode != 0:
                return None

            blame = self._parse_porcelain_blame(result.stdout)
            if blame:
                self._blame_cache[cache_key] = blame

            return blame

        except Exception:
            return None

    def _parse_porcelain_blame(self, output: str) -> Optional[CommitBlame]:
        """Parse git blame porcelain output."""
        lines = output.strip().split("\n")
        if not lines:
            return None

        # First line is commit hash and line info
        first_line = lines[0].split()
        if not first_line:
            return None

        commit_hash = first_line[0]
        author = ""
        author_email = ""
        date = ""
        message = ""

        for line in lines[1:]:
            if line.startswith("author "):
                author = line[7:]
            elif line.startswith("author-mail "):
                author_email = line[12:].strip("<>")
            elif line.startswith("author-time "):
                import datetime
                try:
                    ts = int(line[12:])
                    date = datetime.datetime.fromtimestamp(ts).isoformat()
                except Exception:
                    pass
            elif line.startswith("summary "):
                message = line[8:]

        return CommitBlame(
            commit_hash=commit_hash,
            author=author,
            author_email=author_email,
            date=date,
            message=message,
        )

    def blame_file(self, file_path: str) -> Dict[int, CommitBlame]:
        """
        Get git blame for entire file.

        Returns dict mapping line number to CommitBlame.
        """
        if not self.is_available():
            return {}

        try:
            if os.path.isabs(file_path) and self.repo_root:
                try:
                    file_path = os.path.relpath(file_path, self.repo_root)
                except ValueError:
                    pass

            cmd = ["git", "blame", "--porcelain", file_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.repo_root,
            )

            if result.returncode != 0:
                return {}

            return self._parse_full_blame(result.stdout)

        except Exception:
            return {}

    def _parse_full_blame(self, output: str) -> Dict[int, CommitBlame]:
        """Parse full file git blame output."""
        blames: Dict[int, CommitBlame] = {}
        current_commit = None
        current_line = 0
        commit_info: Dict[str, Dict[str, str]] = {}

        for line in output.split("\n"):
            # New commit/line entry
            match = re.match(r"^([a-f0-9]{40})\s+(\d+)\s+(\d+)", line)
            if match:
                current_commit = match.group(1)
                current_line = int(match.group(3))  # final line number
                if current_commit not in commit_info:
                    commit_info[current_commit] = {}
                continue

            if current_commit:
                if line.startswith("author "):
                    commit_info[current_commit]["author"] = line[7:]
                elif line.startswith("author-mail "):
                    commit_info[current_commit]["author_email"] = line[12:].strip("<>")
                elif line.startswith("summary "):
                    commit_info[current_commit]["message"] = line[8:]
                elif line.startswith("author-time "):
                    import datetime
                    try:
                        ts = int(line[12:])
                        commit_info[current_commit]["date"] = datetime.datetime.fromtimestamp(ts).isoformat()
                    except Exception:
                        pass
                elif line.startswith("\t"):
                    # This is the actual line content, commit block is complete
                    if current_commit and current_line > 0:
                        info = commit_info.get(current_commit, {})
                        blames[current_line] = CommitBlame(
                            commit_hash=current_commit,
                            author=info.get("author", ""),
                            author_email=info.get("author_email"),
                            date=info.get("date", ""),
                            message=info.get("message", ""),
                        )

        return blames

    def get_commit_info(self, commit_hash: str) -> Optional[Dict[str, str]]:
        """Get detailed commit information."""
        if not self.is_available():
            return None

        try:
            cmd = [
                "git", "show",
                "--no-patch",
                "--format=%H%n%an%n%ae%n%aI%n%s",
                commit_hash,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_root,
            )

            if result.returncode != 0:
                return None

            lines = result.stdout.strip().split("\n")
            if len(lines) >= 5:
                return {
                    "hash": lines[0],
                    "author": lines[1],
                    "email": lines[2],
                    "date": lines[3],
                    "message": lines[4],
                }

        except Exception:
            pass

        return None


class RefactorExpert(ExpertPerspective):
    """
    Expert perspective for code-level refactoring analysis.

    Pinpoints specific code locations and git commits responsible for
    performance bottlenecks, helping identify exactly where to refactor.
    """

    @property
    def name(self) -> str:
        return "refactor_expert"

    @property
    def description(self) -> str:
        return "Pinpoints code locations and commits for refactoring"

    def __init__(
        self,
        enable_git_blame: bool = True,
        repo_root: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the refactor expert.

        Args:
            enable_git_blame: Whether to use git blame for commit attribution
            repo_root: Git repository root (auto-detected if None)
            exclude_patterns: Patterns for files to exclude from analysis
        """
        self.enable_git_blame = enable_git_blame
        self.git_helper = GitBlameHelper(repo_root) if enable_git_blame else None

        # Default exclude patterns (system/framework code)
        self.exclude_patterns = exclude_patterns or [
            r"/site-packages/",
            r"/lib/python",
            r"<frozen",
            r"<string>",
            r"<stdin>",
            r"/torch/",
            r"/tensorflow/",
            r"/jax/",
            r"/numpy/",
            r"__pycache__",
        ]
        self._compiled_excludes = [re.compile(p) for p in self.exclude_patterns]

    def _should_exclude(self, file_path: str) -> bool:
        """Check if file should be excluded from analysis."""
        if not file_path:
            return True
        for pattern in self._compiled_excludes:
            if pattern.search(file_path):
                return True
        return False

    def _determine_layer(self, file_path: str, module: Optional[str] = None) -> CodeLayer:
        """Determine which layer of the stack a file belongs to."""
        if not file_path:
            return CodeLayer.RUNTIME

        # Framework patterns
        framework_patterns = [
            r"/torch/", r"/tensorflow/", r"/tf/", r"/jax/",
            r"/flax/", r"/keras/", r"/transformers/",
        ]
        for p in framework_patterns:
            if re.search(p, file_path):
                return CodeLayer.FRAMEWORK

        # Library patterns
        library_patterns = [
            r"/site-packages/", r"/dist-packages/",
        ]
        for p in library_patterns:
            if re.search(p, file_path):
                return CodeLayer.LIBRARY

        # Runtime patterns
        runtime_patterns = [
            r"/lib/python", r"<frozen", r"importlib",
        ]
        for p in runtime_patterns:
            if re.search(p, file_path):
                return CodeLayer.RUNTIME

        # System patterns
        if file_path.startswith("/usr/") or file_path.startswith("/lib/"):
            return CodeLayer.SYSTEM

        return CodeLayer.APPLICATION

    def analyze(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[PerspectiveInsight]:
        """Analyze correlation points to identify refactoring targets."""
        context = context or {}

        # Collect location statistics
        location_stats = self._collect_location_stats(correlation_points, context)

        if not location_stats:
            return []

        # Calculate totals
        total_duration = sum(loc.total_duration_ms for loc in location_stats.values())
        if total_duration == 0:
            total_duration = 1

        insights = []

        # Get top locations by duration
        top_locations = sorted(
            location_stats.values(),
            key=lambda x: x.total_duration_ms,
            reverse=True,
        )[:20]

        # Generate location-based insights
        for loc in top_locations:
            impact_pct = (loc.total_duration_ms / total_duration) * 100
            if impact_pct < 0.5:
                continue

            # Enrich with git blame if available
            if self.enable_git_blame and self.git_helper and self.git_helper.is_available():
                self._enrich_with_blame(loc)

            insight = self._create_location_insight(loc, impact_pct, total_duration)
            insights.append(insight)

        # Add commit-based insights if git is available
        if self.enable_git_blame and self.git_helper and self.git_helper.is_available():
            commit_insights = self._analyze_commits(top_locations, total_duration)
            insights.extend(commit_insights)

        # Add layer analysis
        layer_insight = self._create_layer_insight(location_stats, total_duration)
        if layer_insight:
            insights.append(layer_insight)

        # Generate refactoring suggestions
        suggestions = self._generate_suggestions(top_locations, total_duration)
        if suggestions:
            insights.append(self._create_suggestions_insight(suggestions))

        return insights

    def _collect_location_stats(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Dict[str, Any],
    ) -> Dict[str, CodeLocation]:
        """Collect statistics about code locations."""
        stats: Dict[str, CodeLocation] = {}

        for point in correlation_points:
            # Analyze from stack snapshot
            if point.stack_snapshot:
                for thread in point.stack_snapshot.threads:
                    for frame in thread.frames[:20]:  # Analyze top 20 frames
                        self._record_location(stats, frame.location, point.duration_ms)

        # Analyze from dataflow tracker
        dataflow_tracker = context.get("dataflow_tracker")
        if dataflow_tracker:
            for call in dataflow_tracker.calls.values():
                if call.source_location:
                    # Parse "file:line" format
                    parts = call.source_location.rsplit(":", 1)
                    if len(parts) == 2:
                        try:
                            self._record_location_str(
                                stats,
                                parts[0],
                                int(parts[1]),
                                call.func_name,
                                call.func_module,
                                call.duration_ms,
                            )
                        except ValueError:
                            pass

        return stats

    def _record_location(
        self,
        stats: Dict[str, CodeLocation],
        location: "FrameLocation",
        duration_ms: float,
    ) -> None:
        """Record a frame location in statistics."""
        if self._should_exclude(location.filename):
            return

        self._record_location_str(
            stats,
            location.filename,
            location.lineno,
            location.function,
            location.module,
            duration_ms,
        )

    def _record_location_str(
        self,
        stats: Dict[str, CodeLocation],
        file_path: str,
        line_number: int,
        func_name: Optional[str],
        module_name: Optional[str],
        duration_ms: float,
    ) -> None:
        """Record a location string in statistics."""
        if self._should_exclude(file_path):
            return

        key = f"{file_path}:{line_number}"

        if key not in stats:
            stats[key] = CodeLocation(
                file_path=file_path,
                line_number=line_number,
                function_name=func_name,
                module_name=module_name,
                layer=self._determine_layer(file_path, module_name),
            )

        loc = stats[key]
        loc.occurrence_count += 1
        loc.total_duration_ms += duration_ms
        loc.avg_duration_ms = loc.total_duration_ms / loc.occurrence_count

    def _enrich_with_blame(self, location: CodeLocation) -> None:
        """Enrich location with git blame information."""
        if not self.git_helper:
            return

        blame = self.git_helper.blame_line(location.file_path, location.line_number)
        if blame:
            location.commit_hash = blame.commit_hash
            location.commit_author = blame.author
            location.commit_date = blame.date
            location.commit_message = blame.message

    def _create_location_insight(
        self,
        location: CodeLocation,
        impact_pct: float,
        total_duration: float,
    ) -> PerspectiveInsight:
        """Create insight for a specific code location."""
        severity = self._calculate_severity(impact_pct)

        blame_targets = [
            BlameTarget(
                target_type="code_location",
                name=f"{location.short_path}:{location.line_number}",
                confidence=min(0.9, impact_pct / 50 + 0.4),
                file_path=location.file_path,
                line_number=location.line_number,
                function_name=location.function_name,
                commit_hash=location.commit_hash,
                impact_pct=impact_pct,
                occurrence_count=location.occurrence_count,
                evidence=[
                    f"Called {location.occurrence_count} times",
                    f"Total: {location.total_duration_ms:.1f}ms",
                    f"Avg: {location.avg_duration_ms:.2f}ms per call",
                    f"Layer: {location.layer.value}",
                ],
            )
        ]

        # Add commit blame target if available
        if location.commit_hash:
            blame_targets.append(BlameTarget(
                target_type="commit",
                name=location.commit_hash[:8],
                confidence=0.7,
                commit_hash=location.commit_hash,
                file_path=location.file_path,
                line_number=location.line_number,
                evidence=[
                    f"Author: {location.commit_author}",
                    f"Date: {location.commit_date}",
                    f"Message: {location.commit_message[:50]}..." if location.commit_message else "",
                ],
            ))

        description_parts = [
            f"Location: {location.file_path}:{location.line_number}",
            f"Function: {location.function_name or 'unknown'}",
            f"Layer: {location.layer.value}",
            f"Impact: {impact_pct:.1f}% of total time",
            f"Called: {location.occurrence_count} times",
            f"Total duration: {location.total_duration_ms:.1f}ms",
        ]

        if location.commit_hash:
            description_parts.extend([
                f"Commit: {location.commit_hash[:8]}",
                f"Author: {location.commit_author}",
            ])

        return PerspectiveInsight(
            perspective_name=self.name,
            category="hotspot",
            severity=severity,
            summary=f"{location.short_path}:{location.line_number} - {impact_pct:.1f}% of time",
            description="\n".join(description_parts),
            blame_targets=blame_targets,
            suggestions=self._get_location_suggestions(location),
            metrics={
                "location_duration_ms": location.total_duration_ms,
                "location_impact_pct": impact_pct,
                "location_call_count": float(location.occurrence_count),
                "location_avg_ms": location.avg_duration_ms,
            },
            tags={
                "file": location.short_path,
                "function": location.function_name or "",
                "layer": location.layer.value,
            },
        )

    def _calculate_severity(self, impact_pct: float) -> SeverityLevel:
        """Calculate severity based on impact."""
        if impact_pct > 30:
            return SeverityLevel.CRITICAL
        elif impact_pct > 15:
            return SeverityLevel.HIGH
        elif impact_pct > 5:
            return SeverityLevel.MEDIUM
        elif impact_pct > 1:
            return SeverityLevel.LOW
        return SeverityLevel.INFO

    def _get_location_suggestions(self, location: CodeLocation) -> List[str]:
        """Get refactoring suggestions for a location."""
        suggestions = []

        if location.layer == CodeLayer.APPLICATION:
            if location.occurrence_count > 100:
                suggestions.append("Consider caching or memoization")
            if location.avg_duration_ms > 10:
                suggestions.append("Profile this function for optimization opportunities")

        elif location.layer == CodeLayer.LIBRARY:
            suggestions.append("Consider alternatives or caching library calls")

        elif location.layer == CodeLayer.FRAMEWORK:
            suggestions.append("Review if all framework features are necessary")
            suggestions.append("Consider using framework-specific optimizations")

        return suggestions

    def _analyze_commits(
        self,
        locations: List[CodeLocation],
        total_duration: float,
    ) -> List[PerspectiveInsight]:
        """Analyze commits responsible for bottleneck code."""
        commit_stats: Dict[str, CommitBlame] = {}

        for loc in locations:
            if not loc.commit_hash:
                continue

            if loc.commit_hash not in commit_stats:
                commit_stats[loc.commit_hash] = CommitBlame(
                    commit_hash=loc.commit_hash,
                    author=loc.commit_author or "",
                    date=loc.commit_date or "",
                    message=loc.commit_message or "",
                )

            blame = commit_stats[loc.commit_hash]
            blame.locations.append(loc)
            blame.total_duration_ms += loc.total_duration_ms
            blame.occurrence_count += loc.occurrence_count

        if not commit_stats:
            return []

        # Sort by impact
        sorted_commits = sorted(
            commit_stats.values(),
            key=lambda x: x.total_duration_ms,
            reverse=True,
        )[:5]

        insights = []
        for commit in sorted_commits:
            impact_pct = (commit.total_duration_ms / total_duration) * 100
            if impact_pct < 1:
                continue

            insights.append(PerspectiveInsight(
                perspective_name=self.name,
                category="commit_blame",
                severity=self._calculate_severity(impact_pct),
                summary=f"Commit {commit.commit_hash[:8]} by {commit.author}: {impact_pct:.1f}%",
                description=f"Commit: {commit.commit_hash}\n"
                           f"Author: {commit.author}\n"
                           f"Date: {commit.date}\n"
                           f"Message: {commit.message}\n"
                           f"Affects {len(commit.locations)} locations\n"
                           f"Total impact: {commit.total_duration_ms:.1f}ms",
                blame_targets=[commit.to_blame_target()],
                suggestions=[
                    "Review this commit for optimization opportunities",
                    f"Focus on {len(commit.locations)} affected locations",
                ],
                metrics={
                    "commit_duration_ms": commit.total_duration_ms,
                    "commit_impact_pct": impact_pct,
                    "commit_location_count": float(len(commit.locations)),
                },
                tags={
                    "commit": commit.commit_hash[:8],
                    "author": commit.author,
                },
            ))

        return insights

    def _create_layer_insight(
        self,
        location_stats: Dict[str, CodeLocation],
        total_duration: float,
    ) -> Optional[PerspectiveInsight]:
        """Create insight about time distribution across layers."""
        layer_times: Dict[CodeLayer, float] = defaultdict(float)

        for loc in location_stats.values():
            layer_times[loc.layer] += loc.total_duration_ms

        if not layer_times:
            return None

        # Build distribution string
        dist_lines = []
        blame_targets = []
        for layer in [CodeLayer.APPLICATION, CodeLayer.LIBRARY, CodeLayer.FRAMEWORK, CodeLayer.RUNTIME]:
            duration = layer_times.get(layer, 0)
            pct = (duration / total_duration) * 100 if total_duration > 0 else 0
            if duration > 0:
                dist_lines.append(f"  {layer.value}: {pct:.1f}% ({duration:.1f}ms)")
                blame_targets.append(BlameTarget(
                    target_type="layer",
                    name=layer.value,
                    confidence=0.8,
                    impact_pct=pct,
                ))

        return PerspectiveInsight(
            perspective_name=self.name,
            category="layer_analysis",
            severity=SeverityLevel.INFO,
            summary="Time distribution across code layers",
            description="Stack layer breakdown:\n" + "\n".join(dist_lines),
            blame_targets=blame_targets,
            suggestions=[
                "Focus optimization on application layer for direct impact",
                "Consider framework alternatives for framework-heavy workloads",
            ],
            metrics={
                f"layer_{layer.value}_pct": (layer_times.get(layer, 0) / total_duration) * 100
                for layer in CodeLayer
            },
        )

    def _generate_suggestions(
        self,
        locations: List[CodeLocation],
        total_duration: float,
    ) -> List[RefactorSuggestion]:
        """Generate specific refactoring suggestions."""
        suggestions = []

        for i, loc in enumerate(locations[:10]):
            impact_pct = (loc.total_duration_ms / total_duration) * 100

            # High call count suggests caching opportunity
            if loc.occurrence_count > 50 and loc.layer == CodeLayer.APPLICATION:
                suggestions.append(RefactorSuggestion(
                    location=loc,
                    category=RefactorCategory.PERFORMANCE,
                    title="Consider memoization/caching",
                    description=f"Called {loc.occurrence_count} times with avg {loc.avg_duration_ms:.2f}ms",
                    priority=i,
                    estimated_impact_pct=impact_pct * 0.5,  # Conservative estimate
                ))

            # High average time per call
            if loc.avg_duration_ms > 100:
                suggestions.append(RefactorSuggestion(
                    location=loc,
                    category=RefactorCategory.ALGORITHM,
                    title="Optimize slow operation",
                    description=f"Average {loc.avg_duration_ms:.1f}ms per call is high",
                    priority=i,
                    estimated_impact_pct=impact_pct * 0.3,
                ))

        return suggestions

    def _create_suggestions_insight(
        self,
        suggestions: List[RefactorSuggestion],
    ) -> PerspectiveInsight:
        """Create insight from refactoring suggestions."""
        description_lines = []
        for s in suggestions[:5]:
            description_lines.append(
                f"- {s.title} at {s.location.short_path}:{s.location.line_number}"
            )

        return PerspectiveInsight(
            perspective_name=self.name,
            category="refactor_suggestions",
            severity=SeverityLevel.MEDIUM,
            summary=f"{len(suggestions)} refactoring opportunities identified",
            description="Top suggestions:\n" + "\n".join(description_lines),
            blame_targets=[
                BlameTarget(
                    target_type="code_location",
                    name=f"{s.location.short_path}:{s.location.line_number}",
                    file_path=s.location.file_path,
                    line_number=s.location.line_number,
                    confidence=0.6,
                    impact_pct=s.estimated_impact_pct,
                )
                for s in suggestions[:5]
            ],
            suggestions=[s.title for s in suggestions[:5]],
            metrics={
                "suggestion_count": float(len(suggestions)),
                "total_estimated_impact": sum(s.estimated_impact_pct for s in suggestions),
            },
        )

    def get_hotspot_report(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Optional[Dict[str, Any]] = None,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """Generate a comprehensive hotspot report."""
        location_stats = self._collect_location_stats(correlation_points, context or {})
        total_duration = sum(loc.total_duration_ms for loc in location_stats.values())

        # Get top locations
        top_locations = sorted(
            location_stats.values(),
            key=lambda x: x.total_duration_ms,
            reverse=True,
        )[:top_n]

        # Enrich with blame
        if self.enable_git_blame and self.git_helper:
            for loc in top_locations:
                self._enrich_with_blame(loc)

        return {
            "total_locations": len(location_stats),
            "total_tracked_duration_ms": total_duration,
            "git_available": self.git_helper.is_available() if self.git_helper else False,
            "hotspots": [
                {
                    **loc.to_dict(),
                    "impact_pct": (loc.total_duration_ms / total_duration * 100)
                    if total_duration > 0 else 0,
                }
                for loc in top_locations
            ],
            "layer_breakdown": {
                layer.value: sum(
                    loc.total_duration_ms
                    for loc in location_stats.values()
                    if loc.layer == layer
                )
                for layer in CodeLayer
            },
        }
