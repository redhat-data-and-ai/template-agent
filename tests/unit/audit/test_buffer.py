"""Unit tests for audit in-memory buffer."""

from unittest.mock import patch

from deep_agent.src.audit.buffer import drain, enqueue


class TestAuditBuffer:
    def test_enqueue_and_drain(self):
        with patch("deep_agent.src.audit.buffer.settings") as mock_settings:
            mock_settings.PLATFORM_AUDIT_BUFFER_MAX = 10

            import deep_agent.src.audit.buffer as buffer_mod

            buffer_mod._queue.clear()
            buffer_mod._dropped = 0

            envelope = {"event": "platform.audit", "audit_event_type": "llm_call"}
            enqueue(envelope)
            assert drain() == [envelope]
            assert drain() == []

    def test_drops_when_full(self):
        with patch("deep_agent.src.audit.buffer.settings") as mock_settings:
            mock_settings.PLATFORM_AUDIT_BUFFER_MAX = 2

            import deep_agent.src.audit.buffer as buffer_mod

            buffer_mod._queue.clear()
            buffer_mod._dropped = 0

            enqueue({"id": 1})
            enqueue({"id": 2})
            enqueue({"id": 3})

            assert drain() == [{"id": 1}, {"id": 2}]
            assert buffer_mod._dropped == 1
