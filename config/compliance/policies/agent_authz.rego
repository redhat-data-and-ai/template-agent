package agent.authz

# Agent Authorization Policy
#
# This policy evaluates agent trajectory and tool usage against fixed limits.

import rego.v1

# Configuration - hardcoded defaults
max_trajectory_length := 100
enable_trajectory_limits := true

# Default policy - deny by default
default allow := false

# Allow if trajectory limits are disabled
allow if {
    not enable_trajectory_limits
}

# Allow if trajectory limits are enabled and not exceeded
allow if {
    enable_trajectory_limits
    not is_trajectory_too_long
}

# ── Trajectory Length Check ────────────────────────────────────────────

is_trajectory_too_long if {
    enable_trajectory_limits
    count(input.trajectory) > max_trajectory_length
}

# ── Denial Reasons (for debugging) ─────────────────────────────────────

denial_reasons contains reason if {
    is_trajectory_too_long
    reason := sprintf("trajectory length %d exceeds limit %d", [
        count(input.trajectory),
        max_trajectory_length
    ])
}