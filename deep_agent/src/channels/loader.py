"""Load and cache channels.yaml.

Separate from `config.py` so the models can be imported — by the config
validator, by tests — without touching the filesystem.
"""

from __future__ import annotations

import yaml

from deep_agent.src.channels.config import ChannelsFileConfig
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_cache: tuple[float, ChannelsFileConfig] | None = None


def load_channels() -> ChannelsFileConfig:
    """Current channel config.

    Cached on the file's mtime rather than loaded once: a channel is the
    agent's public surface, and turning one off should not need a pod
    restart.
    """
    global _cache

    from deep_agent.src.agent.config.paths import config_file

    path = config_file("channels")
    if path is None:
        return ChannelsFileConfig()

    mtime = path.stat().st_mtime
    if _cache is not None and _cache[0] == mtime:
        return _cache[1]

    try:
        raw = yaml.safe_load(path.read_text()) or {}
        config = ChannelsFileConfig(**raw)
    except Exception:
        # Fail closed. A malformed channels.yaml must not leave the
        # previous config serving, nor open an unverified endpoint.
        logger.exception("channels.yaml is invalid — all channels disabled")
        config = ChannelsFileConfig()

    _cache = (mtime, config)
    return config


def reset_cache() -> None:
    """Drop the cache. For tests."""
    global _cache
    _cache = None
