"""FastAPI application and routing.

This package contains the FastAPI application setup, middleware configuration,
lifecycle management, and all API routes.

Modules:
    app: FastAPI application factory
    middleware: Middleware configuration
    lifecycle: Application lifecycle management
    routes: API endpoint definitions

Main exports:
    app: Configured FastAPI application instance
    create_app: Create and configure a new FastAPI application
"""

from .app import app, create_app

__all__ = ["app", "create_app"]
