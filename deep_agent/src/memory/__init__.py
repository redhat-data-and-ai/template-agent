"""Memory management — consolidation, decay, clustering, and scheduling.

All operations run as **background jobs** via APScheduler.
Nothing in this package runs in the request path.

Feature flag: ``MEMORY_CONSOLIDATION_ENABLED`` (+ individual layer flags).
"""
