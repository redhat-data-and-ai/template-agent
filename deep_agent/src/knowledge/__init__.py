"""A retrieval corpus the agent can search.

Documents are chunked, embedded and stored in pgvector alongside the
checkpoints, so retrieval is a SQL query rather than a second service.
"""
