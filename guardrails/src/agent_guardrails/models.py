"""Optional Hugging Face backends for the published checkpoints.

Nothing here downloads a model until ``score`` is called. Tests inject
``loader`` and never touch the network.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("agent_guardrails.models")

_UNSAFE = frozenset(
    {
        "malicious",
        "jailbreak",
        "injection",
        "unsafe",
        "toxic",
        "hate",
        "violence",
        "sexual",
        "self-harm",
        "self_harm",
        "crime",
        "label_1",
    }
)


class HuggingFaceTextScorer:
    """Sequence classifier. The score is the probability of an unsafe label."""

    uses_model = True

    def __init__(
        self,
        model_id: str,
        *,
        loader: Callable[[str], Callable[[str], float]] | None = None,
    ) -> None:
        """Defer loading ``model_id`` until the first score."""
        self.model_id = model_id
        self._loader = loader or load_text_classifier
        self._score: Callable[[str], float] | None = None

    def score(self, text: str) -> float:
        """Return the unsafe-label probability for ``text``."""
        if self._score is None:
            self._score = self._loader(self.model_id)
        return float(self._score(text))


class UnavailablePolicyScorer:
    """Try ``primary``. On failure, deny or fall back to the builtin scorer."""

    def __init__(self, primary: Any, fallback: Any, on_unavailable: str) -> None:
        """Try ``primary`` and apply ``on_unavailable`` after the first failure."""
        self._primary = primary
        self._fallback = fallback
        self._on_unavailable = on_unavailable
        self._failed = False
        self.uses_model = bool(getattr(primary, "uses_model", True))

    def score(self, text: str) -> float:
        """Return the primary score, or the fallback policy score."""
        if self._failed:
            return self._fallback_score(text)
        try:
            return float(self._primary.score(text))
        except Exception:
            logger.warning(
                "classifier unavailable, policy=%s",
                self._on_unavailable,
                exc_info=True,
            )
            self._failed = True
            return self._fallback_score(text)

    def _fallback_score(self, text: str) -> float:
        if self._on_unavailable == "deny":
            return 1.0
        return float(self._fallback.score(text))


def load_text_classifier(model_id: str) -> Callable[[str], float]:
    """Load a Hugging Face text-classification pipeline for ``model_id``."""
    from transformers import pipeline

    classifier = pipeline("text-classification", model=model_id, truncation=True)

    def score(text: str) -> float:
        rows = classifier(text[:2000])
        if isinstance(rows, dict):
            rows = [rows]
        best = 0.0
        for row in rows:
            label = str(row.get("label", "")).lower().replace(" ", "_")
            probability = float(row.get("score", 0.0))
            if label in _UNSAFE:
                best = max(best, probability)
        return best

    return score
