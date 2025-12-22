"""
Expert Perspectives: Multi-dimensional analysis from specialized viewpoints.

Provides:
- FrameworkExpert: Identifies which frameworks are causing bottlenecks
- RefactorExpert: Pinpoints specific code locations and commits to blame

Usage:
    from pytest_fkit.tracer.perspectives import (
        PerspectiveRegistry,
        FrameworkExpert,
        RefactorExpert,
    )

    # Register perspectives
    registry = PerspectiveRegistry()
    registry.register(FrameworkExpert())
    registry.register(RefactorExpert())

    # Analyze correlation points
    insights = registry.analyze_all(correlation_points)
"""

from .base import (
    ExpertPerspective,
    PerspectiveInsight,
    PerspectiveRegistry,
    BlameTarget,
    SeverityLevel,
)
from .framework import (
    FrameworkExpert,
    FrameworkInfo,
    FrameworkBlame,
    FRAMEWORK_PATTERNS,
)
from .refactor import (
    RefactorExpert,
    CodeLocation,
    CommitBlame,
    RefactorSuggestion,
)

__all__ = [
    # Base
    "ExpertPerspective",
    "PerspectiveInsight",
    "PerspectiveRegistry",
    "BlameTarget",
    "SeverityLevel",
    # Framework expert
    "FrameworkExpert",
    "FrameworkInfo",
    "FrameworkBlame",
    "FRAMEWORK_PATTERNS",
    # Refactor expert
    "RefactorExpert",
    "CodeLocation",
    "CommitBlame",
    "RefactorSuggestion",
]
