#!/usr/bin/env python3
"""Test script for per-user policy settings.

This script demonstrates:
1. Setting custom policy settings for a user
2. Querying OPA with user settings
3. Verifying that different users get different policy evaluations
"""

import asyncio
import httpx


async def test_opa_evaluation():
    """Test OPA policy evaluation with and without user settings."""

    print("=" * 70)
    print("Testing OPA Policy Evaluation with Per-User Settings")
    print("=" * 70)

    opa_url = "http://localhost:8181/v1/data/agent/authz"

    # Test 1: User with no custom settings (uses defaults)
    print("\n1. User with NO custom settings (uses OPA defaults):")
    print("-" * 70)

    payload_default = {
        "input": {
            "trajectory": [
                {"type": "agent_action", "tools": [{"name": "search_web"}]},
                {"type": "tool_response", "name": "search_web", "status": "completed"}
            ],
            "current_intent": {
                "action": "tool_call",
                "name": "delete_file",
                "args": {"path": "/tmp/test.txt"}
            },
            "user_settings": None  # No custom settings
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(opa_url, json=payload_default, timeout=5.0)
            result = response.json()
            print(f"   Request: user_settings=None, tool=delete_file")
            print(f"   Response: {result}")
            print(f"   Result: {'✓ ALLOWED' if result['result'].get('allow') else '✗ DENIED'}")
    except Exception as exc:
        print(f"   ✗ ERROR: {exc}")

    # Test 2: User with strict custom settings
    print("\n2. User with STRICT custom settings:")
    print("-" * 70)

    payload_strict = {
        "input": {
            "trajectory": [
                {"type": "agent_action", "tools": [{"name": "search_web"}]},
                {"type": "tool_response", "name": "search_web", "status": "completed"}
            ],
            "current_intent": {
                "action": "tool_call",
                "name": "delete_file",
                "args": {"path": "/tmp/test.txt"}
            },
            "user_settings": {
                "max_trajectory_length": 50,
                "blocked_tools": ["delete_file", "exec_shell"],
                "enable_tool_restrictions": True
            }
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(opa_url, json=payload_strict, timeout=5.0)
            result = response.json()
            print(f"   Request: blocked_tools=['delete_file', 'exec_shell'], tool=delete_file")
            print(f"   Response: {result}")
            print(f"   Result: {'✓ ALLOWED' if result['result'].get('allow') else '✗ DENIED'}")
            if not result['result'].get('allow'):
                reasons = result['result'].get('denial_reasons', [])
                print(f"   Denial reasons: {reasons}")
    except Exception as exc:
        print(f"   ✗ ERROR: {exc}")

    # Test 3: User with strict settings but allowed tool
    print("\n3. User with STRICT settings but ALLOWED tool:")
    print("-" * 70)

    payload_allowed = {
        "input": {
            "trajectory": [
                {"type": "agent_action", "tools": [{"name": "search_web"}]},
            ],
            "current_intent": {
                "action": "tool_call",
                "name": "search_web",
                "args": {}
            },
            "user_settings": {
                "max_trajectory_length": 50,
                "blocked_tools": ["delete_file", "exec_shell"],
                "enable_tool_restrictions": True
            }
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(opa_url, json=payload_allowed, timeout=5.0)
            result = response.json()
            print(f"   Request: blocked_tools=['delete_file', 'exec_shell'], tool=search_web")
            print(f"   Response: {result}")
            print(f"   Result: {'✓ ALLOWED' if result['result'].get('allow') else '✗ DENIED'}")
    except Exception as exc:
        print(f"   ✗ ERROR: {exc}")

    # Test 4: Trajectory length limit
    print("\n4. User with TRAJECTORY LENGTH limit:")
    print("-" * 70)

    long_trajectory = [
        {"type": "agent_action", "tools": [{"name": f"tool_{i}"}]}
        for i in range(60)
    ]

    payload_trajectory = {
        "input": {
            "trajectory": long_trajectory,
            "current_intent": {
                "action": "llm_request"
            },
            "user_settings": {
                "max_trajectory_length": 50,
                "enable_trajectory_limits": True
            }
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(opa_url, json=payload_trajectory, timeout=5.0)
            result = response.json()
            print(f"   Request: trajectory_length=60, max_allowed=50")
            print(f"   Response: {result}")
            print(f"   Result: {'✓ ALLOWED' if result['result'].get('allow') else '✗ DENIED'}")
            if not result['result'].get('allow'):
                reasons = result['result'].get('denial_reasons', [])
                print(f"   Denial reasons: {reasons}")
    except Exception as exc:
        print(f"   ✗ ERROR: {exc}")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


async def test_api_endpoints():
    """Test the policy settings API endpoints."""

    print("\n" + "=" * 70)
    print("Testing Policy Settings API")
    print("=" * 70)

    api_base = "http://localhost:5002/api/v1/policy"

    # Test 1: Get defaults
    print("\n1. Get default settings:")
    print("-" * 70)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_base}/defaults", timeout=5.0)
            print(f"   GET /api/v1/policy/defaults")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
    except Exception as exc:
        print(f"   ✗ ERROR: {exc}")

    # Test 2: Get user settings (should be empty initially)
    print("\n2. Get user settings (no custom settings yet):")
    print("-" * 70)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_base}/settings/test-user-123", timeout=5.0)
            print(f"   GET /api/v1/policy/settings/test-user-123")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
    except Exception as exc:
        print(f"   ✗ ERROR: {exc}")

    # Test 3: Set custom settings
    print("\n3. Set custom settings for user:")
    print("-" * 70)

    custom_settings = {
        "settings": {
            "max_trajectory_length": 25,
            "blocked_tools": ["delete_file", "exec_shell", "write_file"],
            "enable_tool_restrictions": True,
            "default_allow": False
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{api_base}/settings/test-user-123",
                json=custom_settings,
                timeout=5.0
            )
            print(f"   PUT /api/v1/policy/settings/test-user-123")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
    except Exception as exc:
        print(f"   ✗ ERROR: {exc}")

    # Test 4: Get user settings again (should have custom settings now)
    print("\n4. Get user settings (after update):")
    print("-" * 70)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_base}/settings/test-user-123", timeout=5.0)
            print(f"   GET /api/v1/policy/settings/test-user-123")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
    except Exception as exc:
        print(f"   ✗ ERROR: {exc}")

    # Test 5: List all users with custom settings
    print("\n5. List all users with custom settings:")
    print("-" * 70)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_base}/settings", timeout=5.0)
            print(f"   GET /api/v1/policy/settings")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
    except Exception as exc:
        print(f"   ✗ ERROR: {exc}")

    print("\n" + "=" * 70)
    print("API Test Complete")
    print("=" * 70)


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Per-User Policy Settings Test Suite" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")

    print("\nPrerequisites:")
    print("  1. OPA is running at http://localhost:8181")
    print("  2. Agent is running at http://localhost:5002")
    print("  3. PostgreSQL is running with template_agent database")

    input("\nPress Enter to start OPA tests (or Ctrl+C to cancel)...")
    await test_opa_evaluation()

    input("\nPress Enter to start API tests (or Ctrl+C to cancel)...")
    await test_api_endpoints()

    print("\n✓ All tests complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
