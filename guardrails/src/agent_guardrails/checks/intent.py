"""Intent head. This is the only guardrail the host trains.

The published encoder is ``BAAI/bge-small-en-v1.5``. ``HashingEncoder`` is the
portable stand-in with the same logistic-regression head.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re

from agent_guardrails.types import IntentPrediction

logger = logging.getLogger("agent_guardrails.intent")

_TOKEN = re.compile(r"[a-z0-9]+")


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sigmoid(value: float) -> float:
    if value >= 20:
        return 1.0
    if value <= -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


class HashingEncoder:
    """Signed feature hashing over tokens and token pairs."""

    def __init__(self, dimensions: int = 256) -> None:
        """Hash tokens into ``dimensions`` signed features."""
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        """Return an L2-normalized feature vector for ``text``."""
        tokens = _TOKEN.findall(text.lower())
        features = list(tokens)
        features.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
        vector = [0.0] * self.dimensions
        for token in features:
            digest = hashlib.blake2s(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class BgeEncoder:
    """Sentence-transformers encoder for ``BAAI/bge-small-en-v1.5``."""

    def __init__(self, model_id: str) -> None:
        """Load the sentence-transformers checkpoint named ``model_id``."""
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self._model = SentenceTransformer(model_id)

    def encode(self, text: str) -> list[float]:
        """Return a normalized embedding for ``text``."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


class LogisticIntentClassifier:
    """One-versus-rest logistic regression on a fixed encoder."""

    def __init__(self, encoder: HashingEncoder | BgeEncoder | None = None) -> None:
        """Use ``encoder``, or the portable hashing encoder when it is omitted."""
        self.encoder = encoder if encoder is not None else HashingEncoder()
        self._labels: list[str] = []
        self._weights: dict[str, list[float]] = {}
        self._bias: dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        """True after :meth:`fit` has seen at least one label."""
        return bool(self._labels)

    def fit(self, texts: list[str], labels: list[str]) -> LogisticIntentClassifier:
        """Fit on labeled queries. Returns self so hosts can chain the call."""
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts and labels must be the same non-empty length")
        encoded = [self.encoder.encode(text) for text in texts]
        labels_found = sorted(set(labels))
        width = len(encoded[0])
        self._labels = labels_found
        self._weights = {label: [0.0] * width for label in labels_found}
        self._bias = {label: 0.0 for label in labels_found}
        learning_rate = 0.8
        for _epoch in range(80):
            for vector, label in zip(encoded, labels):
                for name in labels_found:
                    target = 1.0 if name == label else 0.0
                    logit = self._bias[name] + _dot(self._weights[name], vector)
                    error = _sigmoid(logit) - target
                    self._weights[name] = [
                        weight - learning_rate * error * feature
                        for weight, feature in zip(self._weights[name], vector)
                    ]
                    self._bias[name] -= learning_rate * error
        return self

    def predict(self, text: str) -> IntentPrediction:
        """Return the top label and its softmax confidence."""
        if not self.is_fitted:
            return IntentPrediction(label="unknown", confidence=0.0, read_only=False)
        vector = self.encoder.encode(text)
        logits = {
            label: self._bias[label] + _dot(self._weights[label], vector)
            for label in self._labels
        }
        peak = max(logits.values())
        exp_scores = {label: math.exp(value - peak) for label, value in logits.items()}
        total = sum(exp_scores.values()) or 1.0
        probabilities = {label: score / total for label, score in exp_scores.items()}
        label = max(probabilities, key=lambda name: probabilities[name])
        return IntentPrediction(
            label=label,
            confidence=probabilities[label],
            read_only=False,
        )


def build_encoder(embedding: str, model_id: str) -> HashingEncoder | BgeEncoder:
    """Return the requested encoder, falling back to hashing when BGE is absent."""
    if embedding == "bge":
        try:
            return BgeEncoder(model_id)
        except Exception:
            logger.warning(
                "bge encoder unavailable, using hashing encoder", exc_info=True
            )
    return HashingEncoder()
