"""FastAPI app factory.

Run:  uvicorn app.main:create_app --factory --reload --port 8000
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import auth_routes, items, openai_compat
from app.auth import LoginGuard
from app.api.deps import get_ctx
from app.api.errors import install_error_handlers
from app.config import Settings, get_settings
from app.pipeline.context import AppContext


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ctx = AppContext.build(settings)

    app = FastAPI(title="Content Creator (Module 3 prototype)", version="0.1.0")
    app.state.ctx = ctx
    app.state.auth_secret = settings.auth_secret or secrets.token_urlsafe(32)
    app.state.login_guard = LoginGuard()
    if not (settings.auth_username and settings.auth_password_hash):
        logging.getLogger("content_creator").warning("Login is not configured: set AUTH_USERNAME and AUTH_PASSWORD_HASH")
    install_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health(ctx: AppContext = Depends(get_ctx)) -> dict:
        return {
            "status": "ok" if not ctx.warnings else "degraded",
            "warnings": ctx.warnings,
            "kb_version": ctx.kb.kb_version,
            "providers": {
                "llm": f"{ctx.llm.provider_name}:{ctx.llm.model}",
                "judge": f"{ctx.judge_llm.provider_name}:{ctx.judge_llm.model}" if settings.judge_enabled else "disabled",
                "image": f"{ctx.images.provider_name}:{ctx.images.model}",
                "embedding": f"{ctx.embedder.provider_name}:{ctx.embedder.model}",
                "vector_store": (
                    f"chroma:{settings.chroma_database}/{getattr(getattr(ctx.store, 'collection', None), 'name', '')}"
                    if settings.vector_store == "chroma"
                    else settings.vector_store
                ),
            },
        }

    app.include_router(auth_routes.router)
    app.include_router(openai_compat.router)
    app.include_router(items.router)
    app.mount("/images", StaticFiles(directory=settings.images_dir), name="images")
    return app
