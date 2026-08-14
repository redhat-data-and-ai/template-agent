"""User personalization: memories, custom rules, and prompt injection.

Provides per-user memory and rule storage (Postgres-backed) and a
prompt injector that prepends personalization context to the agent's
system prompt at graph-creation time.
"""

from deep_agent.src.personalization.injector import inject_personalization
from deep_agent.src.personalization.models import Memory, Rule
from deep_agent.src.personalization.repository import PersonalizationRepository

__all__ = [
    "Memory",
    "Rule",
    "PersonalizationRepository",
    "inject_personalization",
]
