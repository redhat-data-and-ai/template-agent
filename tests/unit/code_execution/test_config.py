"""Tests for CodeExecutionConfig."""

from __future__ import annotations

import pytest


class TestCodeExecutionConfigDefaults:
    def test_defaults(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        cfg = CodeExecutionConfig()
        assert cfg.enabled is False
        assert cfg.max_timeout_seconds == 60
        assert cfg.max_code_length == 50_000
        assert cfg.max_output_bytes == 1_048_576
        assert "python" in cfg.images
        assert "shell" in cfg.images
        assert "node" in cfg.images
        assert cfg.images["python"] == "python:3.12-slim"
        assert cfg.entrypoints["python"] == ["python", "-c"]
        assert cfg.entrypoints["shell"] == ["bash", "-c"]
        assert cfg.entrypoints["node"] == ["node", "-e"]
        assert cfg.resource_requests == {"cpu": "100m", "memory": "128Mi"}
        assert cfg.resource_limits == {"cpu": "500m", "memory": "256Mi"}
        assert cfg.tmp_size_limit == "64Mi"
        assert cfg.job_ttl_after_finished == 30
        assert cfg.pod_poll_interval_seconds == 1.0
        assert cfg.pod_poll_timeout_seconds == 120.0

    def test_from_dict(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        cfg = CodeExecutionConfig.model_validate(
            {
                "enabled": True,
                "max_timeout_seconds": 120,
                "images": {"python": "my-registry/python:3.12"},
            }
        )
        assert cfg.enabled is True
        assert cfg.max_timeout_seconds == 120
        assert cfg.images["python"] == "my-registry/python:3.12"

    def test_supported_languages(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        cfg = CodeExecutionConfig()
        assert cfg.supported_languages == {
            "python",
            "python-ds",
            "python-ml",
            "shell",
            "node",
        }

    def test_new_feature_defaults(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        cfg = CodeExecutionConfig()
        assert cfg.network_access == "deny"
        assert cfg.max_concurrent_per_org == 3
        assert cfg.queue_timeout_seconds == 30.0
        assert cfg.max_input_file_size == 1_048_576
        assert cfg.cost_tracking_enabled is False
        assert cfg.streaming_enabled is False

    def test_custom_image_variants(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        cfg = CodeExecutionConfig()
        assert "python-ds" in cfg.images
        assert "python-ml" in cfg.images
        assert cfg.entrypoints["python-ds"] == ["python", "-c"]
        assert cfg.entrypoints["python-ml"] == ["python", "-c"]


class TestCodeExecutionConfigValidation:
    def test_timeout_min(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        with pytest.raises(Exception):
            CodeExecutionConfig(max_timeout_seconds=4)

    def test_timeout_max(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        with pytest.raises(Exception):
            CodeExecutionConfig(max_timeout_seconds=301)

    def test_poll_interval_min(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        with pytest.raises(Exception):
            CodeExecutionConfig(pod_poll_interval_seconds=0.1)

    def test_concurrent_limit_min(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        with pytest.raises(Exception):
            CodeExecutionConfig(max_concurrent_per_org=0)

    def test_queue_timeout_min(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        with pytest.raises(Exception):
            CodeExecutionConfig(queue_timeout_seconds=0.5)

    def test_network_access_values(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        for val in ("deny", "allow_internet", "per_execution"):
            cfg = CodeExecutionConfig(network_access=val)
            assert cfg.network_access == val

    def test_network_access_invalid(self):
        from deep_agent.src.code_execution.config import CodeExecutionConfig

        with pytest.raises(Exception):
            CodeExecutionConfig(network_access="invalid")
