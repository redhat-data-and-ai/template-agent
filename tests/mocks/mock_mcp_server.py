#!/usr/bin/env python3
"""Mock MCP server for testing.

Provides stub implementations of the tools required by the agent:
- calculate_bmi: Returns mock BMI calculation
- validate_email: Basic email format validation
- send_email: Simulates email sending (always succeeds)
- search_web: Returns mock health tips

Uses FastMCP with Streamable HTTP transport so the agent's MCP client
can connect via the standard MCP protocol at /mcp.
"""

import json
import re
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Mock MCP Server", host="0.0.0.0", port=5001)

# Mock health tips by BMI category
HEALTH_TIPS = {
    "Underweight": [
        "Focus on nutrient-dense foods with healthy fats and proteins",
        "Consider increasing meal frequency with healthy snacks",
        "Consult with a healthcare provider for personalized guidance",
    ],
    "Normal": [
        "Maintain a balanced diet with whole grains, lean proteins, and vegetables",
        "Aim for 150 minutes of moderate aerobic activity per week",
        "Stay hydrated and get adequate sleep for optimal health",
    ],
    "Overweight": [
        "Focus on portion control and mindful eating habits",
        "Incorporate regular physical activity into your daily routine",
        "Consider working with a registered dietitian for personalized nutrition advice",
    ],
    "Obese": [
        "Consult with a healthcare provider for a comprehensive health assessment",
        "Set realistic, sustainable goals for gradual weight management",
        "Focus on building healthy habits rather than quick fixes",
    ],
}


@mcp.tool()
def calculate_bmi(height_cm: float, weight_kg: float) -> str:
    """Calculate BMI (Body Mass Index) from height and weight.

    Args:
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms
    """
    if height_cm <= 0 or weight_kg <= 0:
        return json.dumps(
            {"success": False, "error": "Height and weight must be positive values"}
        )

    height_m = height_cm / 100
    bmi = weight_kg / (height_m**2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return json.dumps(
        {
            "success": True,
            "bmi": round(bmi, 1),
            "category": category,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
        }
    )


@mcp.tool()
def validate_email(email: str) -> str:
    """Validate email address format.

    Args:
        email: Email address to validate
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    is_valid = bool(re.match(pattern, email))

    return json.dumps(
        {
            "success": True,
            "valid": is_valid,
            "email": email,
            "message": "Valid email format" if is_valid else "Invalid email format",
        }
    )


@mcp.tool()
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email (mock - always succeeds).

    Args:
        recipient: Email recipient
        subject: Email subject
        body: Email body (HTML or plain text)
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, recipient):
        return json.dumps(
            {"success": False, "error": f"Invalid recipient email: {recipient}"}
        )

    return json.dumps(
        {
            "success": True,
            "recipient": recipient,
            "subject": subject,
            "message": f"Email sent successfully to {recipient}",
            "message_id": f"mock-{hash(recipient + subject)}",
        }
    )


@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for health tips (mock - returns predefined tips).

    Args:
        query: Search query
    """
    query_lower = query.lower()
    category = None

    if "underweight" in query_lower:
        category = "Underweight"
    elif "overweight" in query_lower:
        category = "Overweight"
    elif "obese" in query_lower or "obesity" in query_lower:
        category = "Obese"
    elif "normal" in query_lower:
        category = "Normal"

    tips = HEALTH_TIPS.get(category, HEALTH_TIPS["Normal"])

    return json.dumps(
        {
            "success": True,
            "query": query,
            "category": category,
            "results": [
                {"title": f"Health Tip {i + 1}", "snippet": tip}
                for i, tip in enumerate(tips)
            ],
        }
    )


if __name__ == "__main__":
    print("Starting Mock MCP Server on http://localhost:5001/mcp")
    print("Available tools: calculate_bmi, validate_email, send_email, search_web")
    mcp.run(transport="streamable-http")
