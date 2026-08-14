#!/usr/bin/env python3
"""Mock MCP server for testing.

Provides stub implementations of the tools required by the agent:
- calculate_bmi: Returns mock BMI calculation
- validate_email: Basic email format validation
- send_email: Simulates email sending (always succeeds)
- search_web: Returns mock health tips

This allows agent evals to run without requiring the full template-mcp-server.
"""

import json
import re
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock MCP Server")

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


def calculate_bmi_value(height_cm: float, weight_kg: float) -> Dict[str, Any]:
    """Calculate BMI and determine category.

    Args:
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms

    Returns:
        Dict with bmi, category, and message
    """
    if height_cm <= 0 or weight_kg <= 0:
        return {
            "success": False,
            "error": "Height and weight must be positive values",
        }

    height_m = height_cm / 100
    bmi = weight_kg / (height_m**2)

    # Determine category
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return {
        "success": True,
        "bmi": round(bmi, 1),
        "category": category,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
    }


def validate_email_address(email: str) -> Dict[str, Any]:
    """Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        Dict with valid flag and message
    """
    # Basic email regex
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    is_valid = bool(re.match(pattern, email))

    return {
        "success": True,
        "valid": is_valid,
        "email": email,
        "message": "Valid email format" if is_valid else "Invalid email format",
    }


def send_email_mock(recipient: str, subject: str, body: str) -> Dict[str, Any]:
    """Mock email sending (always succeeds).

    Args:
        recipient: Email recipient
        subject: Email subject
        body: Email body (HTML or plain text)

    Returns:
        Dict with success flag and message
    """
    # Validate recipient email
    validation = validate_email_address(recipient)
    if not validation["valid"]:
        return {
            "success": False,
            "error": f"Invalid recipient email: {recipient}",
        }

    return {
        "success": True,
        "recipient": recipient,
        "subject": subject,
        "message": f"Email sent successfully to {recipient}",
        "message_id": f"mock-{hash(recipient + subject)}",
    }


def search_web_mock(query: str) -> Dict[str, Any]:
    """Mock web search for health tips.

    Args:
        query: Search query (should contain BMI category)

    Returns:
        Dict with search results (health tips)
    """
    # Extract category from query
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

    # Get tips for category
    tips = HEALTH_TIPS.get(category, HEALTH_TIPS["Normal"])

    return {
        "success": True,
        "query": query,
        "category": category,
        "results": [
            {"title": f"Health Tip {i + 1}", "snippet": tip}
            for i, tip in enumerate(tips)
        ],
    }


# MCP Tool definitions
TOOLS = [
    {
        "name": "calculate_bmi",
        "description": "Calculate BMI (Body Mass Index) from height and weight",
        "inputSchema": {
            "type": "object",
            "properties": {
                "height_cm": {
                    "type": "number",
                    "description": "Height in centimeters",
                },
                "weight_kg": {
                    "type": "number",
                    "description": "Weight in kilograms",
                },
            },
            "required": ["height_cm", "weight_kg"],
        },
    },
    {
        "name": "validate_email",
        "description": "Validate email address format",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Email address to validate",
                },
            },
            "required": ["email"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email (mock - always succeeds)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Email recipient",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject",
                },
                "body": {
                    "type": "string",
                    "description": "Email body (HTML or plain text)",
                },
            },
            "required": ["recipient", "subject", "body"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for health tips (mock - returns predefined tips)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["query"],
        },
    },
]


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Mock MCP Server"}


@app.get("/mcp/tools")
async def list_tools():
    """List available MCP tools."""
    return {"tools": TOOLS}


@app.post("/mcp/tools/{tool_name}")
async def call_tool(tool_name: str, request: Request):
    """Execute an MCP tool."""
    body = await request.json()
    arguments = body.get("arguments", {})

    # Route to appropriate tool implementation
    if tool_name == "calculate_bmi":
        result = calculate_bmi_value(
            arguments.get("height_cm"),
            arguments.get("weight_kg"),
        )
    elif tool_name == "validate_email":
        result = validate_email_address(arguments.get("email"))
    elif tool_name == "send_email":
        result = send_email_mock(
            arguments.get("recipient"),
            arguments.get("subject"),
            arguments.get("body"),
        )
    elif tool_name == "search_web":
        result = search_web_mock(arguments.get("query"))
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"Tool not found: {tool_name}"},
        )

    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    print("Starting Mock MCP Server on http://localhost:5001")
    print("Available tools: calculate_bmi, validate_email, send_email, search_web")

    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
