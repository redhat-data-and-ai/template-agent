"""Thread-search index: what gets extracted from a checkpoint blob.

Message text is msgpack inside `checkpoint_blobs`, so this decoder is
the whole reason search works. Its failure mode matters: a blob it
cannot read must cost one thread's entry, never the query.
"""

from __future__ import annotations

import msgpack

from deep_agent.src.search.index import MAX_BODY_CHARS, derive_title, extract_text


def _blob(messages: list[dict]) -> bytes:
    """A LangGraph serde blob: [module, class, message_dict] triples."""
    return msgpack.packb(
        [["langchain_core.messages", "Message", m] for m in messages], use_bin_type=True
    )


class TestExtractText:
    def test_pulls_human_and_ai_text(self):
        blob = _blob(
            [
                {"type": "human", "content": "the question"},
                {"type": "ai", "content": "the answer"},
            ]
        )
        assert extract_text(blob) == "the question\nthe answer"

    def test_skips_tool_messages(self):
        # Tool payloads are mostly JSON; indexing them returns noise
        # for every query that happens to share a key name.
        blob = _blob(
            [
                {"type": "tool", "content": '{"rows": [1, 2, 3]}'},
                {"type": "human", "content": "real text"},
            ]
        )
        assert extract_text(blob) == "real text"

    def test_reads_multimodal_text_blocks(self):
        blob = _blob(
            [
                {
                    "type": "human",
                    "content": [
                        {"type": "text", "text": "described"},
                        {"type": "image_url", "image_url": {"url": "data:…"}},
                    ],
                }
            ]
        )
        assert extract_text(blob) == "described"

    def test_skips_blank_content(self):
        assert extract_text(_blob([{"type": "ai", "content": "   "}])) == ""

    def test_truncates_a_very_long_thread(self):
        blob = _blob([{"type": "ai", "content": "x" * (MAX_BODY_CHARS + 500)}])
        assert len(extract_text(blob)) == MAX_BODY_CHARS

    def test_undecodable_blob_costs_one_entry_not_the_query(self):
        assert extract_text(b"not msgpack at all") == ""

    def test_tolerates_a_non_list_payload(self):
        assert extract_text(msgpack.packb({"not": "a list"}, use_bin_type=True)) == ""

    def test_skips_malformed_triples(self):
        payload = msgpack.packb(
            [["too", "short"], ["m", "c", {"type": "ai", "content": "kept"}]],
            use_bin_type=True,
        )
        assert extract_text(payload) == "kept"

    def test_skips_a_non_dict_message(self):
        payload = msgpack.packb([["m", "c", "not a dict"]], use_bin_type=True)
        assert extract_text(payload) == ""

    def test_survives_an_unknown_ext_type(self):
        # Gemini thought signatures are binary ext payloads. Raising on
        # one would lose the whole conversation from the index over a
        # field nobody searches.
        payload = msgpack.packb(
            [
                [
                    "m",
                    "c",
                    {
                        "type": "ai",
                        "content": "kept",
                        "signature": msgpack.ExtType(5, b"\x01\x02"),
                    },
                ]
            ],
            use_bin_type=True,
            default=lambda o: o,
        )
        assert "kept" in extract_text(payload)


class TestDeriveTitle:
    def test_uses_the_first_non_empty_line(self):
        assert derive_title("\n\n  Incident review  \nmore text") == "Incident review"

    def test_empty_body_has_no_title(self):
        assert derive_title("") == ""
        assert derive_title("\n   \n") == ""

    def test_truncates_a_long_first_line(self):
        assert len(derive_title("y" * 500)) == 200
