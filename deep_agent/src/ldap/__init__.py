"""LDAP-based access control for the template agent.

Reads group-to-role mappings from PROMPT.md front matter, queries LDAP
for user membership, and resolves the highest-priority role. Results
are cached in Redis (with in-memory fallback) for configurable TTL.
"""

from deep_agent.src.ldap.prompt_config import (
    PRIVILEGED_ROLES,
    ROLE_HIERARCHY,
    get_prompt_access_config,
)
from deep_agent.src.ldap.service import close_ldap, resolve_user_role

__all__ = [
    "PRIVILEGED_ROLES",
    "ROLE_HIERARCHY",
    "close_ldap",
    "get_prompt_access_config",
    "resolve_user_role",
]
