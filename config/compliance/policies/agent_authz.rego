package agent.authz

# Agent Authorization Policy - Local Development Placeholder
#
# This file is intentionally minimal. Production policies are loaded from
# the git repository and will completely control authorization behavior.
#
# Do not add allow/deny rules here to avoid conflicts with git policies.

import rego.v1

# For local testing of retry mechanism, uncomment the test rules below.
# These will be ignored when git policies are loaded in production.

# Test rules for retry mechanism verification (commented by default)
# Uncomment to test locally without git policies:

# default allow := false
#
# allow if {
#     input.current_intent.action == "trajectory_validation"
# }
#
# allow if {
#     input.current_intent.action == "tool_response"
# }
#
# allow if {
#     input.current_intent.action == "llm_response"
#     not contains_banned_words
# }
#
# deny_reasons contains reason if {
#     input.current_intent.action == "llm_response"
#     contains(lower(input.current_intent.agent_message), "bmi")
#     reason := "Banned word 'BMI' found in agent response"
# }
#
# contains_banned_words if {
#     contains(lower(input.current_intent.agent_message), "bmi")
# }
