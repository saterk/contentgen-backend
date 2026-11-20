"""Login endpoints for the web app."""

from __future__ import annotations

import asyncio
import secrets

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.deps import require_app_token
from app.api.errors import APIError
from app.auth import issue_token, verify_password

router = APIRouter(prefix="/api/auth")


class LoginBody(BaseModel):
    username: str = Field(max_length=200)
    password: str = Field(max_length=200)


@router.post("/login")
async def login(body: LoginBody, request: Request) -> dict:
    s = request.app.state.ctx.settings
    guard = request.app.state.login_guard
    client = request.client.host if request.client else "unknown"
    guard.reset_expired(client)
    if (wait := guard.locked_for(client)) > 0:
        raise APIError(429, f"Too many failed logins. Try again in {wait} s.", "rate_limit_error", "login_locked")
    if not (s.auth_username and s.auth_password_hash):
        raise APIError(503, "Login is not configured (AUTH_USERNAME / AUTH_PASSWORD_HASH)", "server_error", "auth_not_configured")

    user_ok = secrets.compare_digest(body.username.strip().lower(), s.auth_username.lower())
    pw_ok = verify_password(body.password, s.auth_password_hash)  # always run: same timing either way
    if not (user_ok and pw_ok):
        guard.failed(client)
        await asyncio.sleep(0.5)  # slow down guessing
        raise APIError(401, "Wrong username or password", "authentication_error", "invalid_credentials")

    guard.succeeded(client)
    token, exp = issue_token(s.auth_username, request.app.state.auth_secret, int(s.auth_session_hours * 3600))
    return {"token": token, "username": s.auth_username, "expires_at": exp}


@router.get("/me")
def me(user: str = Depends(require_app_token)) -> dict:
    return {"username": user}
