"""Guards on the usage write.

The SQL failed on every call from the day the feature shipped —
`jsonb_build_object(%(model)s, …)` uses the parameter as an object
key, where Postgres cannot infer a type and answers "could not
determine data type of parameter $3". It was caught, logged at
debug, and the Usage tab reported zero tokens as fact.

A unit test cannot run Postgres, so these assert the two properties
that would have made it visible: every parameter is cast, and a
failure is loud.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.analytics import usage


class TestEveryParameterIsCast:
    def test_no_bare_placeholder_survives(self):
        # Inference happens to work for some positions and not for
        # others; relying on which is how this broke.
        bare = re.findall(r"%\(\w+\)s(?!::)", usage._RECORD_SQL)
        assert bare == [], f"uncast parameters in _RECORD_SQL: {bare}"

    def test_the_model_key_is_text(self):
        # The position that actually failed.
        assert "jsonb_build_object(%(model)s::text" in usage._RECORD_SQL


class TestFailuresAreLoud:
    @pytest.mark.asyncio
    async def test_the_first_failure_warns(self, caplog):
        usage._failures = 0
        pool = MagicMock()
        pool.connection.side_effect = RuntimeError("boom")
        with patch.object(usage, "_get_pool", new=AsyncMock(return_value=pool)):
            with caplog.at_level("WARNING"):
                await usage.record_llm_usage("r1", "m", 1, 1)

        assert any("Failed to record usage" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_missing_run_id_is_not_a_failure(self):
        # Evals and warmups have no run row; that is not an error.
        usage._failures = 0
        with patch.object(usage, "_get_pool", new=AsyncMock()) as pool:
            await usage.record_llm_usage("", "m", 1, 1)
        pool.assert_not_awaited()
        assert usage._failures == 0
