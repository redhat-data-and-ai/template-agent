"""Tests for audit logging.

A package marker, like every sibling. Without one, pytest imports
`test_middleware.py` as a top-level module named `test_middleware`,
so the next directory to add a file with that name collides with it
and collection fails for both.
"""
