#!/bin/sh
# OPA with polling-based hot reload for container volumes
# Note: inotify doesn't work reliably with Docker/Podman bind mounts,
# so we use polling instead.
set -e

POLICY_DIR="/policies"
OPA_ADDR="0.0.0.0:8181"
POLL_INTERVAL=2  # seconds

# Calculate checksum of all .rego files
get_checksum() {
    find "$POLICY_DIR" -name "*.rego" -type f 2>/dev/null | sort | xargs cat 2>/dev/null | md5sum | cut -d' ' -f1
}

# Start OPA server in background
start_opa() {
    echo "Starting OPA server on $OPA_ADDR with policies from $POLICY_DIR"
    /usr/local/bin/opa run --server --addr="$OPA_ADDR" "$POLICY_DIR" &
    OPA_PID=$!
    echo "OPA started (PID: $OPA_PID)"
}

# Function to reload policies by restarting OPA
reload_opa() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Policy change detected, reloading OPA..."

    # Kill the old OPA process
    if [ -n "$OPA_PID" ] && kill -0 "$OPA_PID" 2>/dev/null; then
        kill "$OPA_PID"
        wait "$OPA_PID" 2>/dev/null || true
    fi

    # Start new OPA process
    /usr/local/bin/opa run --server --addr="$OPA_ADDR" "$POLICY_DIR" &
    OPA_PID=$!
    echo "OPA reloaded (new PID: $OPA_PID)"
}

# Cleanup handler
cleanup() {
    echo "Shutting down..."
    [ -n "$OPA_PID" ] && kill "$OPA_PID" 2>/dev/null || true
    exit 0
}

trap cleanup TERM INT

# Start OPA
start_opa
sleep 2

echo "Watching $POLICY_DIR for changes (polling every ${POLL_INTERVAL}s)..."

# Watch for changes using polling
LAST_CHECKSUM=$(get_checksum)
while true; do
    sleep "$POLL_INTERVAL"

    # Check if OPA is still running
    if ! kill -0 "$OPA_PID" 2>/dev/null; then
        echo "OPA process died unexpectedly, restarting..."
        start_opa
        LAST_CHECKSUM=$(get_checksum)
        continue
    fi

    # Check for policy changes
    CURRENT_CHECKSUM=$(get_checksum)
    if [ "$CURRENT_CHECKSUM" != "$LAST_CHECKSUM" ] && [ -n "$CURRENT_CHECKSUM" ]; then
        # Add a small delay to let file writes complete (editors may use temp files)
        sleep 0.5
        CURRENT_CHECKSUM=$(get_checksum)
        if [ -n "$CURRENT_CHECKSUM" ]; then
            reload_opa
            LAST_CHECKSUM="$CURRENT_CHECKSUM"
        fi
    fi
done
