package agent.authz

import rego.v1

# Local trajectory limits for development.
# Git trajectory policy defines max_trajectory_length := 20; this file sets a
# different local cap via local_max_trajectory_length to avoid OPA complete-rule
# conflict when both policy sources are loaded.

local_max_trajectory_length := 20

deny_reasons contains msg if {
	input.current_intent.action == "trajectory_validation"
	count(input.trajectory) > local_max_trajectory_length
	msg := sprintf(
		"Local trajectory length (%d) exceeds limit (%d)",
		[count(input.trajectory), local_max_trajectory_length],
	)
}
