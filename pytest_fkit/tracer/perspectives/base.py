"""
Expert Perspective Base: Foundation for specialized analysis viewpoints.

Each perspective provides domain-specific analysis of execution traces,
identifying bottlenecks and issues from their expert viewpoint.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..correlation import CorrelationPoint, BottleneckAnalysis


class SeverityLevel(Enum):
    """Severity of an identified issue."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class BlameTarget:
    """
    A specific target to blame for a bottleneck.

    Can be a framework, library, file, function, or code location.
    """
    # What type of target (framework, library, file, function, commit)
    target_type: str

    # Name/identifier of the target
    name: str

    # Confidence that this target is responsible (0-1)
    confidence: float

    # Optional specifics
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    commit_hash: Optional[str] = None
    function_name: Optional[str] = None
    module_path: Optional[str] = None

    # Evidence for the blame
    evidence: List[str] = field(default_factory=list)

    # Impact metrics
    impact_pct: float = 0.0  # What percentage of time is attributed
    occurrence_count: int = 0  # How many times seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_type": self.target_type,
            "name": self.name,
            "confidence": self.confidence,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "commit_hash": self.commit_hash,
            "function_name": self.function_name,
            "module_path": self.module_path,
            "evidence": self.evidence,
            "impact_pct": self.impact_pct,
            "occurrence_count": self.occurrence_count,
        }

    def __str__(self) -> str:
        loc = ""
        if self.file_path:
            loc = f" at {self.file_path}"
            if self.line_number:
                loc += f":{self.line_number}"
        return f"{self.target_type}:{self.name}{loc} ({self.confidence:.0%} confidence)"


@dataclass
class PerspectiveInsight:
    """
    An insight from an expert perspective.

    Represents actionable analysis output from a perspective.
    """
    # Which perspective generated this
    perspective_name: str

    # Category of insight
    category: str

    # Severity level
    severity: SeverityLevel

    # Short summary
    summary: str

    # Detailed description
    description: str

    # Specific targets to blame
    blame_targets: List[BlameTarget] = field(default_factory=list)

    # Suggested fixes
    suggestions: List[str] = field(default_factory=list)

    # Related correlation point IDs
    correlation_ids: List[str] = field(default_factory=list)

    # Metrics supporting the insight
    metrics: Dict[str, float] = field(default_factory=dict)

    # Tags for filtering
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perspective": self.perspective_name,
            "category": self.category,
            "severity": self.severity.value,
            "summary": self.summary,
            "description": self.description,
            "blame_targets": [t.to_dict() for t in self.blame_targets],
            "suggestions": self.suggestions,
            "correlation_ids": self.correlation_ids,
            "metrics": self.metrics,
            "tags": self.tags,
        }

    def to_metrics(self) -> Dict[str, float]:
        """Convert to numeric metrics for PySR."""
        base_metrics = {
            f"insight_{self.category}_severity": float(
                ["info", "low", "medium", "high", "critical"].index(self.severity.value)
            ),
            f"insight_{self.category}_blame_count": float(len(self.blame_targets)),
            f"insight_{self.category}_suggestion_count": float(len(self.suggestions)),
        }
        base_metrics.update(self.metrics)
        return base_metrics


class ExpertPerspective(ABC):
    """
    Abstract base class for expert perspectives.

    Each perspective provides specialized analysis of execution traces
    from a particular viewpoint (e.g., framework expert, refactoring aid).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this perspective."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this perspective analyzes."""
        pass

    @abstractmethod
    def analyze(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[PerspectiveInsight]:
        """
        Analyze correlation points from this perspective.

        Args:
            correlation_points: List of multi-dimensional correlation points
            context: Additional context (dataflow tracker, stack tracer, etc.)

        Returns:
            List of insights with blame targets and suggestions
        """
        pass

    def get_blame_targets(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[BlameTarget]:
        """
        Extract blame targets from correlation points.

        Override this for more specific blame analysis.
        """
        insights = self.analyze(correlation_points, context)
        targets = []
        for insight in insights:
            targets.extend(insight.blame_targets)
        return targets

    def get_summary(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get summary of analysis from this perspective."""
        insights = self.analyze(correlation_points, context)
        return {
            "perspective": self.name,
            "insight_count": len(insights),
            "severity_distribution": {
                sev.value: sum(1 for i in insights if i.severity == sev)
                for sev in SeverityLevel
            },
            "categories": list(set(i.category for i in insights)),
            "total_blame_targets": sum(len(i.blame_targets) for i in insights),
        }


class PerspectiveRegistry:
    """
    Registry for expert perspectives.

    Manages multiple perspectives and coordinates analysis across them.
    """

    def __init__(self):
        self._perspectives: Dict[str, ExpertPerspective] = {}

    def register(self, perspective: ExpertPerspective) -> None:
        """Register a perspective."""
        self._perspectives[perspective.name] = perspective

    def unregister(self, name: str) -> Optional[ExpertPerspective]:
        """Unregister a perspective by name."""
        return self._perspectives.pop(name, None)

    def get(self, name: str) -> Optional[ExpertPerspective]:
        """Get a perspective by name."""
        return self._perspectives.get(name)

    def list_perspectives(self) -> List[str]:
        """List registered perspective names."""
        return list(self._perspectives.keys())

    def analyze_all(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[PerspectiveInsight]]:
        """
        Run analysis from all registered perspectives.

        Returns:
            Dict mapping perspective name to list of insights
        """
        results = {}
        for name, perspective in self._perspectives.items():
            try:
                results[name] = perspective.analyze(correlation_points, context)
            except Exception as e:
                # Log error but continue with other perspectives
                results[name] = [
                    PerspectiveInsight(
                        perspective_name=name,
                        category="error",
                        severity=SeverityLevel.INFO,
                        summary=f"Analysis failed: {e}",
                        description=str(e),
                    )
                ]
        return results

    def get_all_blame_targets(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[BlameTarget]:
        """Get blame targets from all perspectives."""
        all_targets = []
        for perspective in self._perspectives.values():
            try:
                targets = perspective.get_blame_targets(correlation_points, context)
                all_targets.extend(targets)
            except Exception:
                pass
        return all_targets

    def get_summary(
        self,
        correlation_points: List["CorrelationPoint"],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get combined summary from all perspectives."""
        all_insights = self.analyze_all(correlation_points, context)

        total_insights = sum(len(insights) for insights in all_insights.values())
        all_blame_targets = self.get_all_blame_targets(correlation_points, context)

        # Aggregate severity distribution
        severity_dist = {sev.value: 0 for sev in SeverityLevel}
        for insights in all_insights.values():
            for insight in insights:
                severity_dist[insight.severity.value] += 1

        return {
            "perspective_count": len(self._perspectives),
            "total_insights": total_insights,
            "severity_distribution": severity_dist,
            "total_blame_targets": len(all_blame_targets),
            "perspectives": {
                name: {
                    "insight_count": len(insights),
                    "categories": list(set(i.category for i in insights)),
                }
                for name, insights in all_insights.items()
            },
        }

    def clear(self) -> None:
        """Clear all registered perspectives."""
        self._perspectives.clear()
