"""LDAP connection settings loaded from environment variables.

Required when PROMPT.md has groups defined:
    LDAP_URL:               LDAP server URL (e.g. ldaps://ldap.example.com).
                            Base DN is derived from hostname automatically.
    LDAP_BASE_UID:          Service account uid for bind (e.g. svcacct).
                            If it contains a comma, used as-is as bind DN.
    LDAP_PASSWORD:          Service account password.

Optional (have sensible defaults):
    LDAP_GROUP_SEARCH_BASE: Override group search subtree.
                            Default: ou=adhoc,ou=managedGroups,<derived base DN>.
    LDAP_CACHE_TTL_SECONDS: Redis cache TTL for membership lookups (default: 300).
"""

from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings


class LdapSettings(BaseSettings):
    """LDAP connection configuration loaded from environment."""

    LDAP_URL: str = Field(default="")
    LDAP_BASE_UID: str = Field(default="")
    LDAP_PASSWORD: str = Field(default="", repr=False)
    LDAP_GROUP_SEARCH_BASE: str = Field(default="")
    LDAP_TLS_VERIFY: bool = Field(default=True)
    LDAP_CACHE_TTL_SECONDS: int = Field(default=300, ge=10, le=86400)

    def derive_base_dn(self) -> str:
        """Derive LDAP base DN from the URL hostname (e.g. ldap.example.com -> dc=example,dc=com)."""
        if not self.LDAP_URL:
            return ""
        try:
            hostname = urlparse(self.LDAP_URL).hostname or ""
            parts = hostname.split(".")
            if len(parts) < 2:
                return ""
            return ",".join(f"dc={p}" for p in parts[-2:])
        except Exception:
            return ""

    def get_bind_dn(self) -> str:
        """Construct the bind DN from LDAP_BASE_UID."""
        if not self.LDAP_BASE_UID:
            return ""
        if "," in self.LDAP_BASE_UID:
            return self.LDAP_BASE_UID
        base_dn = self.derive_base_dn()
        return f"uid={self.LDAP_BASE_UID},ou=users,{base_dn}"

    def get_group_search_base(self) -> str:
        """Return the group search base, defaulting to ou=adhoc,ou=managedGroups,<baseDn>."""
        if self.LDAP_GROUP_SEARCH_BASE:
            return self.LDAP_GROUP_SEARCH_BASE
        base_dn = self.derive_base_dn()
        if not base_dn:
            return ""
        return f"ou=adhoc,ou=managedGroups,{base_dn}"


ldap_settings = LdapSettings()
