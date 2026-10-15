"""Full-text search across a user's threads.

Conversation text lives in LangGraph's `checkpoint_blobs` as msgpack, so
there is nothing for SQL to search directly. This package maintains a
derived table — `thread_search` — and keeps it current lazily, on the
read path, scoped to the searching user.
"""
