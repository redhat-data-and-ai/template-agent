"""Mock tools for skill testing."""

from typing import Annotated


def calculate_bmi(
    height_cm: Annotated[float, "Height in centimeters"],
    weight_kg: Annotated[float, "Weight in kilograms"],
) -> float:
    """Calculate BMI from height and weight."""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m**2)
    return round(bmi, 2)


def search_web(query: Annotated[str, "Search query"]) -> str:
    """Mock web search returning category-specific health tips."""
    query_lower = query.lower()

    # Return tips based on BMI category mentioned in query
    if "underweight" in query_lower:
        return """
        Health tips for underweight individuals:
        1. Eat nutrient-dense foods like nuts, whole grains, and lean proteins
        2. Increase meal frequency to 5-6 smaller meals per day
        3. Add healthy fats like avocados and olive oil to your diet
        """

    elif "overweight" in query_lower:
        return """
        Health tips for overweight individuals:
        1. Focus on gradual, sustainable weight loss of 0.5-1 kg per week
        2. Reduce portion sizes and avoid processed foods
        3. Increase physical activity to 150-300 minutes per week
        """

    elif "obese" in query_lower:
        return """
        Health tips for obese individuals:
        1. Consult a healthcare provider or registered dietitian for guidance
        2. Start with small, achievable goals like 10-minute daily walks
        3. Build a support system with family, friends, or support groups
        """

    else:  # Normal weight
        return """
        Health tips for normal weight individuals:
        1. Continue balanced nutrition with variety of whole foods
        2. Maintain regular physical activity (150 minutes moderate exercise per week)
        3. Stay hydrated with 8+ glasses of water daily
        """


def send_email(
    recipient: Annotated[str, "Email address"],
    subject: Annotated[str, "Email subject"],
    body: Annotated[str, "Email body (HTML)"],
) -> str:
    """Mock email sending."""
    return f"Email sent successfully to {recipient}"


# Tool registry for easy lookup
MOCK_TOOLS = {
    "calculate_bmi": calculate_bmi,
    "search_web": search_web,
    "send_email": send_email,
}
