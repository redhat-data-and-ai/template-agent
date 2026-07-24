package agent.authz

# Agent Authorization Policy - Local Development Placeholder
#
# This file is intentionally minimal. Production policies are loaded from
# the git repository and will completely control authorization behavior.
#
# Do not add allow/deny rules here to avoid conflicts with git policies.
# Local trajectory overrides live in trajectory_limits.rego (local cap 10 vs git 20).

import rego.v1
