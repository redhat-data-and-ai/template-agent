"""Load classifier YAML and read the MaaS key from the environment.

Files (all YAML), shipped beside this module so a config mount cannot drop them:
  deep_agent/aegra/classifier/data/product_catalog.yaml
  deep_agent/aegra/classifier/data/product_tables.yaml
  deep_agent/aegra/classifier/data/logic.yaml

Environment variables, read on each call:
  MAAS_API_KEY            - MaaS Enterprise API key (required for the LLM path)
  MAAS_BASE_URL           - default https://models.corp.redhat.com/oai/v1
  MAAS_MODEL_ID           - default RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic
  MAAS_TIMEOUT            - seconds, default 30
  MAAS_VERIFY_SSL         - default true
  MAAS_MAX_TOKENS         - default 8192
  CLASSIFICATION_DATA_DIR - override the classification YAML directory
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class MaasSettings:
    """MaaS connection settings taken from the process environment."""

    api_key: str
    base_url: str
    model_id: str
    timeout: float
    verify_ssl: bool
    max_tokens: int


def get_maas_settings() -> MaasSettings:
    """Read MaaS settings now. Does not cache the API key."""
    base_url = os.environ.get(
        "MAAS_BASE_URL", "https://models.corp.redhat.com/oai/v1"
    ).rstrip("/")
    return MaasSettings(
        api_key=os.environ.get("MAAS_API_KEY", ""),
        base_url=base_url,
        model_id=os.environ.get(
            "MAAS_MODEL_ID",
            "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic",
        ),
        timeout=float(os.environ.get("MAAS_TIMEOUT", "30")),
        verify_ssl=os.environ.get("MAAS_VERIFY_SSL", "true").lower() != "false",
        max_tokens=int(os.environ.get("MAAS_MAX_TOKENS", "8192")),
    )


def classification_dir() -> Path:
    """Directory that holds the classification YAML files."""
    override = os.environ.get("CLASSIFICATION_DATA_DIR", "").strip()
    if override:
        return Path(override)
    return _DEFAULT_DIR


@lru_cache(maxsize=8)
def _load_yaml(path_str: str) -> dict:
    path = Path(path_str)
    if not path.is_file():
        logger.warning("Classification YAML missing: %s", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        logger.warning("Classification YAML is not a mapping: %s", path)
        return {}
    logger.info("Loaded %s (%d keys)", path.name, len(data))
    return data


def load_logic() -> dict:
    """Classification rules and prompt text from logic.yaml."""
    return _load_yaml(str(classification_dir() / "logic.yaml"))


def load_product_catalog() -> dict:
    """Product catalog from product_catalog.yaml."""
    return _load_yaml(str(classification_dir() / "product_catalog.yaml"))


def load_product_tables() -> dict[str, list[dict]]:
    """Product name to table list, from product_tables.yaml."""
    data = _load_yaml(str(classification_dir() / "product_tables.yaml"))
    tables: dict[str, list[dict]] = {}
    for name, rows in data.items():
        if isinstance(rows, list):
            tables[str(name)] = [row for row in rows if isinstance(row, dict)]
    return tables


def load_system_prompt() -> str:
    """Routing guide plus the unified task, both stored in logic.yaml."""
    prompts = load_logic().get("prompts") or {}
    guide = str(prompts.get("routing_guide") or "").strip()
    task = str(prompts.get("task") or "").strip()
    if guide and task:
        return f"{guide}\n\n{task}\n"
    return guide or task


def intent_classes() -> frozenset[str]:
    """Intent labels declared in logic.yaml."""
    raw = load_logic().get("intent_classes") or []
    classes = frozenset(str(item).strip() for item in raw if str(item).strip())
    if classes:
        return classes
    return frozenset(
        ["data-query", "unstructured", "catalog", "chart", "direct-response"]
    )


def entity_types() -> frozenset[str]:
    """Entity type names declared in logic.yaml."""
    raw = load_logic().get("entity_types") or {}
    if isinstance(raw, dict) and raw:
        return frozenset(str(key) for key in raw)
    return frozenset(
        [
            "person",
            "rh_product",
            "dv_product",
            "metric",
            "time_period",
            "geography",
            "organization",
            "account",
            "document",
            "sku",
        ]
    )
