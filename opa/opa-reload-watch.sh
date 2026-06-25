#!/bin/sh
# OPA Hot-Reload Watcher
#
# Polls /policies for *.rego changes and restarts OPA when detected.
# Uses MD5 checksums to detect modifications.

set -e

POLICY_DIR="/policies"
OPA_BIN="/usr/local/bin/opa"
OPA_ADDR="0.0.0.0:8181"
POLL_INTERVAL="${POLL_INTERVAL:-2}"  # seconds
OPA_PID=""

# Color output for logs
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*"
}

# Calculate MD5 checksum of all *.rego files
get_policies_checksum() {
    find "$POLICY_DIR" -name "*.rego" -type f 2>/dev/null | sort | xargs cat 2>/dev/null | md5sum | cut -d' ' -f1
}

# Start OPA server in background
start_opa() {
    log_info "Starting OPA server on $OPA_ADDR"
    $OPA_BIN run --server --addr="$OPA_ADDR" "$POLICY_DIR" &
    OPA_PID=$!
    log_info "OPA started with PID $OPA_PID"
}

# Stop OPA server
stop_opa() {
    if [ -n "$OPA_PID" ] && kill -0 "$OPA_PID" 2>/dev/null; then
        log_info "Stopping OPA (PID $OPA_PID)"
        kill -TERM "$OPA_PID" 2>/dev/null || true
        wait "$OPA_PID" 2>/dev/null || true
        OPA_PID=""
    fi
}

# Reload OPA by stopping and starting
reload_opa() {
    log_info "Policy change detected - reloading OPA"
    stop_opa
    sleep 0.5
    start_opa
}

# Cleanup on exit
cleanup() {
    log_info "Shutting down..."
    stop_opa
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup TERM INT

# Check if policy directory exists
if [ ! -d "$POLICY_DIR" ]; then
    log_error "Policy directory not found: $POLICY_DIR"
    exit 1
fi

# Count initial *.rego files
POLICY_COUNT=$(find "$POLICY_DIR" -name "*.rego" -type f 2>/dev/null | wc -l)
if [ "$POLICY_COUNT" -eq 0 ]; then
    log_warn "No *.rego files found in $POLICY_DIR"
else
    log_info "Found $POLICY_COUNT policy file(s) in $POLICY_DIR"
fi

# Initial checksum
LAST_CHECKSUM=$(get_policies_checksum)
log_info "Initial policy checksum: $LAST_CHECKSUM"

# Start OPA for the first time
start_opa

# Give OPA a moment to start
sleep 2

# Main watch loop
log_info "Starting file watcher (polling every ${POLL_INTERVAL}s)"
while true; do
    sleep "$POLL_INTERVAL"

    # Check if OPA is still running
    if [ -n "$OPA_PID" ] && ! kill -0 "$OPA_PID" 2>/dev/null; then
        log_error "OPA process died unexpectedly, restarting..."
        start_opa
        continue
    fi

    # Check for policy changes
    CURRENT_CHECKSUM=$(get_policies_checksum)

    if [ "$CURRENT_CHECKSUM" != "$LAST_CHECKSUM" ]; then
        log_info "Checksum changed: $LAST_CHECKSUM -> $CURRENT_CHECKSUM"
        LAST_CHECKSUM="$CURRENT_CHECKSUM"
        reload_opa
    fi
done
