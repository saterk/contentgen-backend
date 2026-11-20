"""Settings loaded from the environment and backend/.env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> repo root is two levels above "app"
ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: str = "mock"  # mock | openrouter | openai (any OpenAI-compatible API via LLM_BASE_URL)
    llm_model: str = "mock-llm-1"
    llm_api_key: str = ""
    llm_base_url: str = ""  # empty -> provider default
    llm_reasoning: str = "off"  # off | low | medium | high (OpenRouter reasoning models)
    llm_timeout_seconds: float = 60
    judge_enabled: bool = True  # false -> copy and image checks are rules only
    judge_model: str = ""  # empty -> same as llm_model

    # Image
    image_provider: str = "mock"  # mock | openrouter
    image_model: str = "mock-image-1"
    image_api_key: str = ""  # empty -> LLM_API_KEY
    image_base_url: str = ""  # empty -> provider default
    image_size: str = "1080x1350"
    image_variants: int = 1

    # Knowledge base
    vector_store: str = "local"  # local (JSON file) | chroma (Chroma Cloud or self-hosted)
    vector_db_url: str = ""
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""
    chroma_collection: str = "content_creator_kb"  # the embedding model is appended to the name
    chroma_host: str = ""  # empty -> Chroma Cloud; set for a self-hosted server
    embedding_provider: str = "mock"  # mock | openrouter | openai
    embedding_model: str = "mock-embed-1"
    embedding_api_key: str = ""  # empty -> LLM_API_KEY
    embedding_base_url: str = ""  # empty -> LLM_BASE_URL / provider default
    km_match_threshold: float = 0.75
    km_ambiguity_margin: float = 0.03

    # App
    database_url: str = ""  # empty -> sqlite file in DATA_DIR
    data_dir: Path = Path("./data")  # relative paths are resolved against the repo root
    max_concurrent_jobs: int = 4
    max_pieces_per_request: int = 10
    item_timeout_seconds: float = 180
    request_timeout_seconds: float = 600
    prompt_version: str = "v1"
    app_api_key: str = ""  # for scripts and SDK calls; empty = API-key access off (the web app logs in instead)

    # Login (prototype: one fixed user)
    auth_username: str = ""
    auth_password_hash: str = ""  # python -m app.auth hash-password
    auth_secret: str = ""  # signs session tokens; empty -> random per process (logins reset on restart)
    auth_session_hours: float = 12
    cors_origins: str = "http://localhost:3000"

    # Chat defaults (used when the user doesn't say)
    default_channel: str = "line"
    default_language: str = "th"
    default_objective: str = "Invite to talk to an advisor"
    default_journey_stage: str = "awareness"
    default_style: str = "story"  # used (and stated in the summary) when the user gives no style
    default_tone: str = "warm"
    default_visual_style: str = "photo"  # photo | cartoon | 3d (see taxonomy.json visual_styles)

    # Orchestrator limits
    request_tool_call_limit: int = 6
    piece_tool_call_limit: int = 3

    # Copy length limits (characters)
    headline_max_chars: int = 60
    body_max_chars: int = 400
    cta_max_chars: int = 30

    # Variety check: pieces whose headline + image prompt similarity is above this are regenerated
    variety_threshold: float = 0.9

    # Caps for client-supplied generation params
    max_temperature: float = 1.0
    max_tokens_cap: int = 4096

    # ---- derived paths -------------------------------------------------
    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else (ROOT_DIR / p).resolve()

    @property
    def kb_dir(self) -> Path:
        return self.data_path / "kb"

    @property
    def kb_source_dir(self) -> Path:
        return self.data_path / "kb_source"

    @property
    def images_dir(self) -> Path:
        return self.data_path / "images"

    @property
    def prompts_dir(self) -> Path:
        return BACKEND_DIR / "prompts"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_path / 'app.db').as_posix()}"

    @property
    def image_wh(self) -> tuple[int, int]:
        w, h = self.image_size.lower().split("x")
        return int(w), int(h)

    @property
    def judge_model_name(self) -> str:
        return self.judge_model or self.llm_model

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
