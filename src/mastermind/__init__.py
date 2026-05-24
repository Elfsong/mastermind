"""Mastermind experiment plumbing.

The modules in this package provide the stable data surface shared by
baselines, retrieval, SFT, and strategy-level RL.
"""

from .config import Manifest, load_manifest
from .rollout import MilestoneSummary, RolloutRecord, VerificationSummary
from .tasks import TaskMetadata

__all__ = [
    "Manifest",
    "MilestoneSummary",
    "RolloutRecord",
    "TaskMetadata",
    "VerificationSummary",
    "load_manifest",
]
