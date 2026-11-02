"""Configuration and file loading for the DVA classifier.

All config files are loaded from the classifier config directory:
  config/agent/classifier/routing_guide.txt
  config/agent/classifier/classify_config.json
  config/agent/classifier/product_tables.json
  config/agent/classifier/unified_prompt.txt

Environment variables:
  CLASSIFIER_CONFIG_DIR  - Override path to classifier config directory
  MAAS_API_KEY           - MaaS Enterprise API key (required)
  MAAS_BASE_URL          - MaaS endpoint (default: https://models.corp.redhat.com/oai/v1)
  MAAS_MODEL_ID          - Model (default: RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic)
  MAAS_TIMEOUT           - Request timeout in seconds (default: 15)
  MAAS_VERIFY_SSL        - TLS verification (default: true)
  MAAS_MAX_TOKENS        - Max output tokens (default: 8192)
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(os.environ.get("CLASSIFIER_CONFIG_DIR", "config/agent/classifier"))

# MaaS settings
MAAS_API_KEY = os.environ.get("MAAS_API_KEY", "")
MAAS_BASE_URL = os.environ.get("MAAS_BASE_URL", "https://models.corp.redhat.com/oai/v1").rstrip("/")
MAAS_MODEL_ID = os.environ.get("MAAS_MODEL_ID", "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic")
MAAS_TIMEOUT = float(os.environ.get("MAAS_TIMEOUT", "15"))
MAAS_VERIFY_SSL = os.environ.get("MAAS_VERIFY_SSL", "true").lower() != "false"
MAAS_MAX_TOKENS = int(os.environ.get("MAAS_MAX_TOKENS", "8192"))


@lru_cache(maxsize=1)
def load_routing_guide() -> str:
    """Load the routing guide system prompt from routing_guide.txt."""
    path = _CONFIG_DIR / "routing_guide.txt"
    if not path.exists():
        logger.warning("routing_guide.txt not found at %s", path)
        return ""
    content = path.read_text(encoding="utf-8")
    logger.info("Loaded routing_guide.txt (%d chars)", len(content))
    return content


@lru_cache(maxsize=1)
def load_unified_prompt() -> str:
    """Load the unified task suffix from unified_prompt.txt."""
    path = _CONFIG_DIR / "unified_prompt.txt"
    if not path.exists():
        logger.warning("unified_prompt.txt not found at %s", path)
        return ""
    content = path.read_text(encoding="utf-8")
    logger.info("Loaded unified_prompt.txt (%d chars)", len(content))
    return content


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """Build the full system prompt: routing guide + unified task."""
    return load_routing_guide() + "\n" + load_unified_prompt()


@lru_cache(maxsize=1)
def load_product_tables() -> dict[str, list[dict]]:
    """Load product_tables.json (product -> tables mapping)."""
    path = _CONFIG_DIR / "product_tables.json"
    if not path.exists():
        logger.warning("product_tables.json not found at %s", path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Loaded product_tables.json: %d products", len(data))
    return data


@lru_cache(maxsize=1)
def load_classify_config() -> dict:
    """Load classify_config.json (fallback keywords, entity types)."""
    path = _CONFIG_DIR / "classify_config.json"
    if not path.exists():
        logger.warning("classify_config.json not found at %s", path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Loaded classify_config.json v%s", data.get("version", "?"))
    return data
