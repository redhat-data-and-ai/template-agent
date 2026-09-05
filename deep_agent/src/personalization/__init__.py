"""User personalization: custom rules and prompt injection.

Provides per-user rule storage (Postgres-backed) and a prompt injector
that appends user rules to the agent's system prompt at graph-creation time.

Conversational memory is handled separately by the LangGraph Store via the
memory middleware (writes to /memories/ namespace).
"""

from deep_agent.src.personalization.injector import inject_rules
from deep_agent.src.personalization.models import Memory, Rule
from deep_agent.src.personalization.repository import PersonalizationRepository

__all__ = [
    "Memory",
    "Rule",
    "PersonalizationRepository",
    "inject_rules",
]
