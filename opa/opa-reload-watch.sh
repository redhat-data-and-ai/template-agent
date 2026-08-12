#!/bin/sh
# OPA Hot-Reload Watcher with Git Repository Support
#
# Loads policies from BOTH local files AND git repository (if configured).
# Uses sparse checkout for efficiency when subdirectory is specified.
#
# Environment Variables:
#   OPA_POLICY_GIT_REPO       - Git repository URL (optional)
#   OPA_POLICY_GIT_BRANCH     - Git branch to checkout (default: main)
#   OPA_POLICY_GIT_SUBDIR     - Subdirectory within repo (uses sparse checkout)
#   OPA_POLICY_GIT_AUTH_USER  - Git username for private repos (optional)
#   OPA_POLICY_GIT_AUTH_TOKEN - Git token/password for private repos (optional)
#   OPA_POLICY_GIT_SSL_VERIFY - Verify SSL certificates (default: true, set to false for self-signed certs)
#   OPA_POLL_INTERVAL         - Seconds between polls (default: 2)

set -e

POLICY_DIR="/policies"
GIT_CLONE_DIR="/tmp/policy-repo"
OPA_BIN="/usr/local/bin/opa"
OPA_ADDR="0.0.0.0:8181"
POLL_INTERVAL="${OPA_POLL_INTERVAL:-2}"
POLICY_GIT_REPO="${OPA_POLICY_GIT_REPO:-}"
POLICY_GIT_BRANCH="${OPA_POLICY_GIT_BRANCH:-main}"
POLICY_GIT_SUBDIR="${OPA_POLICY_GIT_SUBDIR:-}"
POLICY_GIT_AUTH_USER="${OPA_POLICY_GIT_AUTH_USER:-}"
POLICY_GIT_AUTH_TOKEN="${OPA_POLICY_GIT_AUTH_TOKEN:-}"
POLICY_GIT_SSL_VERIFY="${OPA_POLICY_GIT_SSL_VERIFY:-true}"
OPA_PID=""
USE_GIT=false
GIT_POLICY_DIR=""

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

# Configure git SSL verification
configure_git_ssl() {
    if [ "$POLICY_GIT_SSL_VERIFY" = "false" ]; then
        export GIT_SSL_NO_VERIFY=1
        log_warn "SSL certificate verification disabled for git operations"
    fi
}

# Build git URL with authentication if provided
build_git_url() {
    if [ -n "$POLICY_GIT_AUTH_USER" ] && [ -n "$POLICY_GIT_AUTH_TOKEN" ]; then
        # Extract protocol and rest of URL
        PROTO=$(echo "$POLICY_GIT_REPO" | grep -o '^https\?://')
        REST=$(echo "$POLICY_GIT_REPO" | sed 's|^https\?://||')
        echo "${PROTO}${POLICY_GIT_AUTH_USER}:${POLICY_GIT_AUTH_TOKEN}@${REST}"
    else
        echo "$POLICY_GIT_REPO"
    fi
}

# Clone or update git repository with sparse checkout support
sync_git_repo() {
    local git_url
    git_url=$(build_git_url)

    if [ ! -d "$GIT_CLONE_DIR/.git" ]; then
        log_info "Cloning policy repository: $POLICY_GIT_REPO (branch: $POLICY_GIT_BRANCH)"

        if [ -n "$POLICY_GIT_SUBDIR" ]; then
            # Use sparse checkout for subdirectory
            log_info "Using sparse checkout for subdirectory: $POLICY_GIT_SUBDIR"
            mkdir -p "$GIT_CLONE_DIR"
            cd "$GIT_CLONE_DIR" || return 1

            git_output=$(git init 2>&1) || {
                log_error "Failed to initialize git repository"
                cd /
                return 1
            }
            echo "$git_output" | grep -v "Username\|Password" || true

            git_output=$(git remote add origin "$git_url" 2>&1) || {
                log_error "Failed to add git remote"
                cd /
                return 1
            }
            echo "$git_output" | grep -v "Username\|Password" || true

            git config core.sparseCheckout true
            echo "$POLICY_GIT_SUBDIR/*" > .git/info/sparse-checkout

            git_output=$(git fetch --depth 1 origin "$POLICY_GIT_BRANCH" 2>&1) || {
                log_error "Failed to fetch from git repository"
                cd /
                return 1
            }
            echo "$git_output" | grep -v "Username\|Password" || true

            git_output=$(git checkout "$POLICY_GIT_BRANCH" 2>&1) || {
                log_error "Failed to checkout branch $POLICY_GIT_BRANCH"
                cd /
                return 1
            }
            echo "$git_output" | grep -v "Username\|Password" || true

            cd /
        else
            # Regular clone
            git_output=$(git clone --depth 1 --branch "$POLICY_GIT_BRANCH" "$git_url" "$GIT_CLONE_DIR" 2>&1) || {
                log_error "Failed to clone git repository"
                return 1
            }
            echo "$git_output" | grep -v "Username\|Password" || true
        fi

        log_info "Repository cloned successfully"
    else
        log_info "Updating policy repository from remote"
        cd "$GIT_CLONE_DIR" || return 1

        git_output=$(git fetch origin "$POLICY_GIT_BRANCH" --depth 1 2>&1) || {
            log_error "Failed to fetch updates from git repository"
            cd /
            return 1
        }
        echo "$git_output" | grep -v "Username\|Password" || true

        git_output=$(git reset --hard "origin/$POLICY_GIT_BRANCH" 2>&1) || {
            log_error "Failed to reset to origin/$POLICY_GIT_BRANCH"
            cd /
            return 1
        }
        echo "$git_output" | grep -v "Username\|Password" || true

        cd /
        log_info "Repository updated successfully"
    fi

    # Verify policy directory exists
    if [ ! -d "$GIT_POLICY_DIR" ]; then
        log_error "Git policy directory not found: $GIT_POLICY_DIR"
        log_error "Expected directory: $GIT_POLICY_DIR"
        if [ -d "$GIT_CLONE_DIR" ]; then
            log_error "Git clone dir exists, listing contents:"
            ls -la "$GIT_CLONE_DIR" || true
        fi
        return 1
    fi

    # Count git policies
    local git_count
    git_count=$(find "$GIT_POLICY_DIR" -name "*.rego" -type f 2>/dev/null | wc -l)
    log_info "Found $git_count policy file(s) in git repository"

    return 0
}

# Calculate MD5 checksum of all policies (local + git)
get_policies_checksum() {
    local local_hash git_hash combined

    # Local policies checksum
    local_hash=$(find "$POLICY_DIR" -name "*.rego" -type f 2>/dev/null | sort | xargs cat 2>/dev/null | md5sum | cut -d' ' -f1)

    if [ "$USE_GIT" = true ] && [ -d "$GIT_POLICY_DIR" ]; then
        # Git policies checksum + commit hash
        git_hash=$(find "$GIT_POLICY_DIR" -name "*.rego" -type f 2>/dev/null | sort | xargs cat 2>/dev/null | md5sum | cut -d' ' -f1)

        if [ -d "$GIT_CLONE_DIR/.git" ]; then
            cd "$GIT_CLONE_DIR"
            local commit_hash
            commit_hash=$(git rev-parse HEAD 2>/dev/null || echo "none")
            cd /
            combined="${local_hash}-${git_hash}-${commit_hash}"
        else
            combined="${local_hash}-${git_hash}"
        fi
        echo "$combined"
    else
        echo "$local_hash"
    fi
}

# Start OPA server with multiple policy directories
start_opa() {
    local opa_dirs="$POLICY_DIR"

    if [ "$USE_GIT" = true ] && [ -d "$GIT_POLICY_DIR" ]; then
        opa_dirs="$opa_dirs $GIT_POLICY_DIR"
        log_info "Starting OPA server on $OPA_ADDR with local and git policies"
    else
        log_info "Starting OPA server on $OPA_ADDR with local policies only"
    fi

    # shellcheck disable=SC2086
    $OPA_BIN run --server --addr="$OPA_ADDR" $opa_dirs &
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

# Initialize local policy directory
if [ ! -d "$POLICY_DIR" ]; then
    log_error "Local policy directory not found: $POLICY_DIR"
    exit 1
fi

# Check if git mode is enabled
if [ -n "$POLICY_GIT_REPO" ]; then
    USE_GIT=true
    log_info "Git mode enabled: $POLICY_GIT_REPO"

    # Configure SSL verification
    configure_git_ssl

    # Set git policy directory
    if [ -n "$POLICY_GIT_SUBDIR" ]; then
        GIT_POLICY_DIR="$GIT_CLONE_DIR/$POLICY_GIT_SUBDIR"
    else
        GIT_POLICY_DIR="$GIT_CLONE_DIR"
    fi

    # Initial git sync
    if sync_git_repo; then
        log_info "Initial git sync completed"
    else
        log_error "Failed to sync git repository"
        exit 1
    fi
else
    log_info "Local file mode (no git repository configured)"
fi

# Count initial local policies
LOCAL_COUNT=$(find "$POLICY_DIR" -name "*.rego" -type f 2>/dev/null | wc -l)
log_info "Found $LOCAL_COUNT local policy file(s)"

if [ "$LOCAL_COUNT" -eq 0 ] && [ "$USE_GIT" != true ]; then
    log_warn "No policies found in local directory and no git repository configured"
fi

# Initial checksum
LAST_CHECKSUM=$(get_policies_checksum)
log_info "Initial policy checksum: $LAST_CHECKSUM"

# Start OPA for the first time
start_opa

# Give OPA a moment to start
sleep 2

# Main watch loop
if [ "$USE_GIT" = true ]; then
    log_info "Starting watch loop: local + git sync (polling every ${POLL_INTERVAL}s)"
else
    log_info "Starting watch loop: local files only (polling every ${POLL_INTERVAL}s)"
fi

while true; do
    sleep "$POLL_INTERVAL"

    # Check if OPA is still running
    if [ -n "$OPA_PID" ] && ! kill -0 "$OPA_PID" 2>/dev/null; then
        log_error "OPA process died unexpectedly, restarting..."
        start_opa
        continue
    fi

    # Sync git repo if in git mode
    if [ "$USE_GIT" = true ]; then
        if ! sync_git_repo; then
            log_warn "Git sync failed, will retry on next poll"
            continue
        fi
    fi

    # Check for policy changes (local + git)
    CURRENT_CHECKSUM=$(get_policies_checksum)

    if [ "$CURRENT_CHECKSUM" != "$LAST_CHECKSUM" ]; then
        log_info "Checksum changed: $LAST_CHECKSUM -> $CURRENT_CHECKSUM"
        LAST_CHECKSUM="$CURRENT_CHECKSUM"
        reload_opa
    fi
done
