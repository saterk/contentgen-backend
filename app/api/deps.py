from __future__ import annotations

import secrets

from fastapi import Request

from app.api.errors import APIError
from app.auth import verify_token
from app.pipeline.context import AppContext


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    return header[7:].strip() if header.lower().startswith("bearer ") else ""


def require_app_token(request: Request) -> str:
    """Accept a login session token (web app) or APP_API_KEY (scripts, CLI, tests).

    Returns the caller: the username, or "api_key".
    """
    token = bearer(request)
    if token:
        user = verify_token(token, request.app.state.auth_secret)
        if user:
            return user
        api_key = request.app.state.ctx.settings.app_api_key
        if api_key and secrets.compare_digest(token, api_key):  # empty APP_API_KEY = API-key access off
            return "api_key"
    raise APIError(401, "Invalid or missing API key or session", "authentication_error", "invalid_api_key")
