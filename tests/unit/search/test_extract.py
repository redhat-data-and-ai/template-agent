"""Extracting searchable text out of LangGraph checkpoint blobs.

The extractor is the part most likely to silently return nothing — a
format change upstream produces an empty index, not an error, and an
empty index looks exactly like "no results". These tests pin the shape.
"""

from __future__ import annotations

import msgpack
import pytest

from deep_agent.src.search.index import MAX_BODY_CHARS, derive_title, extract_text


def blob(*messages: dict) -> bytes:
    """A blob in LangGraph's serde shape: [module, class, message_dict]."""
    return msgpack.packb(
        [["langchain_core.messages", "HumanMessage", m] for m in messages]
    )


class TestExtract:
    def test_human_and_ai_text_is_indexed(self):
        text = extract_text(
            blob(
                {"type": "human", "content": "how do I deploy"},
                {"type": "ai", "content": "run make deploy"},
            )
        )
        assert "how do I deploy" in text
        assert "run make deploy" in text

    def test_tool_messages_are_not_indexed(self):
        """Tool payloads are mostly JSON; indexing them makes every query
        match on a shared key name."""
        text = extract_text(
            blob({"type": "tool", "content": '{"status": "deploy ok"}'})
        )
        assert text == ""

    def test_multimodal_content_contributes_its_text_blocks(self):
        text = extract_text(
            blob(
                {
                    "type": "human",
                    "content": [
                        {"type": "text", "text": "what is in this image"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                    ],
                }
            )
        )
        assert "what is in this image" in text
        assert "data:" not in text

    def test_empty_content_is_skipped(self):
        assert extract_text(blob({"type": "human", "content": "  "})) == ""

    def test_a_corrupt_blob_yields_nothing_rather_than_raising(self):
        """One bad thread must not fail the whole reindex."""
        assert extract_text(b"not msgpack at all") == ""

    def test_an_unexpected_structure_yields_nothing(self):
        assert extract_text(msgpack.packb({"not": "a list"})) == ""

    def test_short_items_are_skipped(self):
        assert extract_text(msgpack.packb([["too", "short"]])) == ""

    def test_an_unknown_ext_type_does_not_lose_the_message(self):
        """Gemini thought signatures are binary ext payloads.

        Raising on one would drop an entire conversation from the index
        over a field nobody wants to search.
        """
        packed = msgpack.packb(
            [
                [
                    "m",
                    "HumanMessage",
                    {
                        "type": "human",
                        "content": "findable text",
                        "signature": msgpack.ExtType(42, b"\x00\x01"),
                    },
                ]
            ]
        )
        assert "findable text" in extract_text(packed)

    def test_body_is_capped(self):
        text = extract_text(
            blob({"type": "human", "content": "x" * (MAX_BODY_CHARS * 2)})
        )
        assert len(text) <= MAX_BODY_CHARS


class TestTitle:
    def test_the_first_non_empty_line_becomes_the_title(self):
        assert derive_title("\n\n  deploy the agent  \nmore") == "deploy the agent"

    def test_an_empty_body_has_no_title(self):
        assert derive_title("   \n  ") == ""

    def test_titles_are_bounded(self):
        assert len(derive_title("y" * 500)) == 200
