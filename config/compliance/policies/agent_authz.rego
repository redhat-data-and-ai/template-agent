package agent.authz

# Agent Authorization Policy
#
# This policy evaluates agent trajectory and tool usage against fixed limits.

import rego.v1

# Default policy - deny by default
default allow := false
