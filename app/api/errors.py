"""OpenAI-shaped errors for every endpoint: {"error": {"message", "type", "code"}}."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class APIError(Exception):
    def __init__(self, status: int, message: str, type_: str = "invalid_request_error", code: str | None = None):
        self.status = status
        self.message = message
        self.type = type_
        self.code = code


def error_body(message: str, type_: str, code: str | None) -> dict:
    return {"error": {"message": message, "type": type_, "code": code}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(error_body(exc.message, exc.type, exc.code), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        msg = f"{loc}: {first.get('msg', 'invalid request')}" if loc else first.get("msg", "invalid request")
        return JSONResponse(error_body(msg, "invalid_request_error", "invalid_request"), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "invalid_api_key", 404: "not_found"}.get(exc.status_code, "http_error")
        type_ = "authentication_error" if exc.status_code == 401 else "invalid_request_error"
        return JSONResponse(error_body(str(exc.detail), type_, code), status_code=exc.status_code)
