"""Adapters for external library formats.

This package contains adapters that convert between external library formats
and our internal schema. Each adapter module is named after the library it
adapts (e.g., langchain.py for LangChain).
"""

from .langchain import convert_message_content_to_string, langchain_to_chat_message

__all__ = [
    "langchain_to_chat_message",
    "convert_message_content_to_string",
]
