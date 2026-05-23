#!/bin/bash
# Start mock MCP server for local development and testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Run 'make install' first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

echo "Starting Mock MCP Server..."
echo "This provides stub implementations of:"
echo "  - calculate_bmi"
echo "  - validate_email"
echo "  - send_email"
echo "  - search_web"
echo ""
echo "Server will run on http://localhost:5001"
echo "Press Ctrl+C to stop"
echo ""

python tests/mocks/mock_mcp_server.py
