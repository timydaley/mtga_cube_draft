"""Drafter — abstract interface for any pick-policy.

Plan §3.2. Every bot (random, heuristic, neural) implements `pick(obs)`,
returning a card index that must be present in `obs["pack"]`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

type DraftObs = dict[str, Any]


class Drafter(ABC):
    """A pick-policy."""

    @abstractmethod
    def pick(self, obs: DraftObs) -> int:
        """Return a card index that is in `obs["pack"]`."""

    def reset(self) -> None:  # noqa: B027
        """Called by the env between drafts. Default no-op."""
