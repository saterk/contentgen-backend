"""Live smoke test against the providers configured in the backend/.env.

Skipped unless RUN_LIVE=1. Uses a temporary data folder, so the dev database is untouched.

    RUN_LIVE=1 .venv/Scripts/python -m pytest tests/test_live.py -s
"""

from __future__ import annotations

import json
import os

import pytest
from sqlmodel import Session

from app.config import BACKEND_DIR, Settings
from app.kb.ingest import ingest_with_settings
from app.models import ContentItem
from app.pipeline.context import AppContext
from tests.conftest import of_type, run

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call real providers")


def test_live_free_form_request(data_dir):
    s = Settings(_env_file=BACKEND_DIR / ".env", data_dir=data_dir, image_provider="mock", image_size="216x270",
                 vector_store="local")  # never write test data to the cloud KB
    if s.llm_provider == "mock":
        pytest.skip("LLM_PROVIDER is mock in .env")
    ingest_with_settings(s, kb_version="live")
    ctx = AppContext.build(s)
    events = run(ctx, "Need 2 reassuring, educational single-image ads with a CTA about health insurance "
                      "for first jobbers.", mode="content-creator-brief")
    req = of_type(events, "requirement")[0].data["request"]
    assert req["product_id"] == "health" and req["segment_id"] == "first_jobber"
    assert req["profile"]["style"] == "educational" and req["profile"]["tone"] == "reassuring"
    done = of_type(events, "done")[0].data
    assert done["summary"]["total"] == 2
    with Session(ctx.engine) as sess:
        briefs = [json.loads(sess.get(ContentItem, i).brief_json) for i in done["item_ids"]]
    assert all(set(b["benefit_fact_ids"]) <= {f"HLT-F0{i}" for i in range(1, 6)} for b in briefs)
    print(json.dumps(done, indent=2), *(b["headline"] for b in briefs), sep="\n")
