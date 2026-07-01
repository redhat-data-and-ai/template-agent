#!/bin/bash
# Verification script for per-user policy settings implementation

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Per-User Policy Settings Implementation Verification             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_mark="${GREEN}✓${NC}"
cross_mark="${RED}✗${NC}"
warning="${YELLOW}⚠${NC}"

# Track results
PASSED=0
FAILED=0

check_file() {
    local file=$1
    local description=$2

    if [ -f "$file" ]; then
        echo -e "${check_mark} ${description}: ${file}"
        ((PASSED++))
    else
        echo -e "${cross_mark} ${description}: ${file} NOT FOUND"
        ((FAILED++))
    fi
}

check_content() {
    local file=$1
    local pattern=$2
    local description=$3

    if [ -f "$file" ] && grep -q "$pattern" "$file"; then
        echo -e "${check_mark} ${description}"
        ((PASSED++))
    else
        echo -e "${cross_mark} ${description}"
        ((FAILED++))
    fi
}

echo "1. Checking Created Files..."
echo "─────────────────────────────────────────────────────────────────────"

check_file "deep_agent/src/policy/__init__.py" "Policy module init"
check_file "deep_agent/src/policy/models.py" "Policy models"
check_file "deep_agent/src/policy/repository.py" "Policy repository"
check_file "deep_agent/src/policy/api.py" "Policy API"
check_file "scripts/test_policy_settings.py" "Test script"
check_file "docs/POLICY_SETTINGS.md" "Full documentation"
check_file "docs/POLICY_QUICK_START.md" "Quick start guide"

echo ""
echo "2. Checking Modified Files..."
echo "─────────────────────────────────────────────────────────────────────"

check_file "config/agent/compliance/policies/agent_authz.rego" "OPA policy file"
check_content "config/agent/compliance/policies/agent_authz.rego" "input.user_settings" \
    "Policy uses input.user_settings"
check_content "deep_agent/src/infrastructure/rego_trajectory_middleware.py" "_get_user_settings" \
    "Middleware fetches user settings"
check_content "deep_agent/aegra/startup.py" "PolicySettingsRepository" \
    "Startup initializes policy table"
check_content "deep_agent/aegra/feedback.py" "policy_router" \
    "API routes registered"

echo ""
echo "3. Checking Database Schema..."
echo "─────────────────────────────────────────────────────────────────────"

check_content "deep_agent/src/policy/repository.py" "CREATE TABLE IF NOT EXISTS user_policy_settings" \
    "Database table definition exists"
check_content "deep_agent/src/policy/repository.py" "settings JSONB NOT NULL" \
    "JSONB column for settings"

echo ""
echo "4. Checking API Endpoints..."
echo "─────────────────────────────────────────────────────────────────────"

check_content "deep_agent/src/policy/api.py" "GET /api/v1/policy/settings/{user_id}" \
    "GET user settings endpoint"
check_content "deep_agent/src/policy/api.py" "PUT /api/v1/policy/settings/{user_id}" \
    "PUT user settings endpoint"
check_content "deep_agent/src/policy/api.py" "DELETE /api/v1/policy/settings/{user_id}" \
    "DELETE user settings endpoint"
check_content "deep_agent/src/policy/api.py" "GET /api/v1/policy/defaults" \
    "GET defaults endpoint"

echo ""
echo "5. Checking Rego Policy..."
echo "─────────────────────────────────────────────────────────────────────"

check_content "config/agent/compliance/policies/agent_authz.rego" "config := input.user_settings if" \
    "User settings conditional"
check_content "config/agent/compliance/policies/agent_authz.rego" "max_trajectory_length" \
    "Trajectory length setting"
check_content "config/agent/compliance/policies/agent_authz.rego" "blocked_tools" \
    "Blocked tools setting"
check_content "config/agent/compliance/policies/agent_authz.rego" "denial_reasons" \
    "Denial reasons for debugging"

echo ""
echo "6. Checking Middleware..."
echo "─────────────────────────────────────────────────────────────────────"

check_content "deep_agent/src/infrastructure/rego_trajectory_middleware.py" "_settings_cache" \
    "Settings cache exists"
check_content "deep_agent/src/infrastructure/rego_trajectory_middleware.py" "invalidate_cache" \
    "Cache invalidation method"
check_content "deep_agent/src/infrastructure/rego_trajectory_middleware.py" "_get_user_id" \
    "User ID extraction"

echo ""
echo "7. Checking Documentation..."
echo "─────────────────────────────────────────────────────────────────────"

check_content "docs/POLICY_SETTINGS.md" "Architecture" \
    "Architecture section"
check_content "docs/POLICY_SETTINGS.md" "API Endpoints" \
    "API documentation"
check_content "docs/POLICY_QUICK_START.md" "5-Minute Setup" \
    "Quick start guide"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Results"
echo "════════════════════════════════════════════════════════════════════"
echo -e "  Passed: ${GREEN}${PASSED}${NC}"
echo -e "  Failed: ${RED}${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Start services: make dev"
    echo "  2. Run tests: python scripts/test_policy_settings.py"
    echo "  3. Read docs: docs/POLICY_QUICK_START.md"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo ""
    echo "Please review the implementation."
    exit 1
fi
