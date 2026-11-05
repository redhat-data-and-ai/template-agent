"""DVA Query Classifier - embedded DiffusionGemma classification.

One MaaS LLM call does NER + decomposition + intent classification + product identification.
Config files (routing guide, classify rules, product tables) are loaded from disk.
"""

from deep_agent.aegra.classifier.classify import (
    ClassifyResult,
    SubQuery,
    classify_query,
)
from deep_agent.aegra.classifier.runnable import ClassifierRunnable

__all__ = ["ClassifierRunnable", "ClassifyResult", "SubQuery", "classify_query"]
