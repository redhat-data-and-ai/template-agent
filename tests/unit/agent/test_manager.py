"""Unit tests for AgentManager."""

import pytest

from template_agent.src.agent.manager import AgentManager
from template_agent.src.streaming import MessageDeduplicator, ToolCallTracker


@pytest.fixture
def agent_manager() -> AgentManager:
    """Fixture providing a fresh AgentManager instance."""
    return AgentManager()


class TestAgentManager:
    """Integration tests for AgentManager."""

    def test_manager_initialization(self, agent_manager: AgentManager):
        """Test that AgentManager initializes with correct components."""
        assert agent_manager.deduplicator is not None
        assert agent_manager.tracker is not None
        assert isinstance(agent_manager.deduplicator, MessageDeduplicator)
        assert isinstance(agent_manager.tracker, ToolCallTracker)

    def test_manager_has_all_handlers(self, agent_manager: AgentManager):
        """Test that manager has all required event handlers."""
        assert "updates" in agent_manager.handlers
        assert "messages" in agent_manager.handlers

    def test_manager_with_sso_token(self):
        """Test manager initialization with SSO token."""
        manager = AgentManager(redhat_sso_token="test_token_123")
        assert manager.redhat_sso_token == "test_token_123"

    def test_manager_components_are_independent(self):
        """Test that each manager instance has independent components."""
        manager1 = AgentManager()
        manager2 = AgentManager()

        # Each manager should have its own instances
        assert manager1.deduplicator is not manager2.deduplicator
        assert manager1.tracker is not manager2.tracker

    def test_manager_handlers_are_configured(self, agent_manager: AgentManager):
        """Test that handlers are properly configured with dependencies."""
        from template_agent.src.streaming.handlers import (
            TokenEventHandler,
            UpdateEventHandler,
        )

        # Check handler types
        assert isinstance(agent_manager.handlers["updates"], UpdateEventHandler)
        assert isinstance(agent_manager.handlers["messages"], TokenEventHandler)

        # Check that UpdateEventHandler has the manager's deduplicator
        updates_handler = agent_manager.handlers["updates"]
        assert updates_handler.deduplicator is agent_manager.deduplicator

        # Check that TokenEventHandler has the manager's tracker
        token_handler = agent_manager.handlers["messages"]
        assert token_handler.tracker is agent_manager.tracker

    def test_manager_with_langfuse_client(self):
        """Test manager initialization with Langfuse client."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        manager = AgentManager(langfuse_client=mock_client)

        assert manager.langfuse_client is mock_client

    def test_manager_without_langfuse_client(self):
        """Test manager initialization without Langfuse client."""
        manager = AgentManager()

        assert manager.langfuse_client is None
