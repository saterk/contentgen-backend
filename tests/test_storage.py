"""Image storage (local + R2 with a fake S3 client) and database URL handling."""

from __future__ import annotations

from app.db import normalise_url
from app.storage import LocalImageStore, R2ImageStore, configure_r2_cors


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.cors = None

    def put_object(self, Bucket, Key, Body, ContentType, **kw):
        self.objects[f"{Bucket}/{Key}"] = (Body, ContentType)

    def generate_presigned_url(self, op, Params, ExpiresIn):
        assert op == "get_object"
        return f"https://acc.r2.cloudflarestorage.com/{Params['Bucket']}/{Params['Key']}?X-Amz-Expires={ExpiresIn}&X-Amz-Signature=x"

    def put_bucket_cors(self, Bucket, CORSConfiguration):
        self.cors = (Bucket, CORSConfiguration)


def test_local_store(tmp_path):
    s = LocalImageStore(tmp_path / "images")
    s.save("a.jpg", b"jpg", "image/jpeg")
    assert (tmp_path / "images" / "a.jpg").read_bytes() == b"jpg"
    assert s.url("a.jpg") == "/images/a.jpg"


def test_r2_store_uploads_keeps_local_copy_and_presigns(tmp_path):
    s3 = FakeS3()
    s = R2ImageStore(tmp_path / "cache", bucket="cc", client=s3, url_ttl_seconds=600)
    s.save("itm_1.jpg", b"jpg", "image/jpeg")
    assert s3.objects["cc/images/itm_1.jpg"] == (b"jpg", "image/jpeg")
    assert s.local_path("itm_1.jpg").read_bytes() == b"jpg", "local copy for the image judge"
    url = s.url("itm_1.jpg")
    assert url.startswith("https://") and "images/itm_1.jpg" in url and "X-Amz-Expires=600" in url


def test_r2_public_base_url(tmp_path):
    s = R2ImageStore(tmp_path, bucket="cc", client=FakeS3(), public_base_url="https://img.example.com/")
    assert s.url("x.jpg") == "https://img.example.com/images/x.jpg"


def test_r2_cors(tmp_path):
    s3 = FakeS3()
    configure_r2_cors(R2ImageStore(tmp_path, bucket="cc", client=s3), ["https://app.vercel.app"])
    bucket, cfg = s3.cors
    assert bucket == "cc" and cfg["CORSRules"][0]["AllowedOrigins"] == ["https://app.vercel.app"]


def test_r2_requires_credentials(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="R2_ACCOUNT_ID"):
        R2ImageStore(tmp_path, bucket="cc")


def test_pipeline_saves_through_the_store_and_api_returns_store_urls(client):
    """With a fake R2 store in place, generated images go to R2 and the item API returns R2 URLs."""
    from tests.conftest import FULL_REQUEST

    ctx = client.app.state.ctx
    s3 = FakeS3()
    ctx.storage = R2ImageStore(ctx.settings.images_dir, bucket="cc", client=s3)
    body = client.post("/v1/chat/completions", json={
        "model": "content-creator", "messages": [{"role": "user", "content": FULL_REQUEST.format(n=1)}],
    }).json()
    item_id = next(e["item_id"] for e in body["x_events"] if e["type"] == "item_ready")
    item = client.get(f"/api/items/{item_id}").json()
    assert len(s3.objects) == 1
    assert item["images"][0]["url"].startswith("https://acc.r2.cloudflarestorage.com/cc/images/")


def test_database_url_normalisation():
    assert normalise_url("postgres://u:p@h/db?sslmode=require") == "postgresql+psycopg://u:p@h/db?sslmode=require"
    assert normalise_url("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalise_url("sqlite:///x.db") == "sqlite:///x.db"


def test_allow_local_storage_false_refuses_local_stores(tmp_path):
    import pytest

    from tests.conftest import make_settings

    with pytest.raises(ValueError, match="DATABASE_URL is not set.*VECTOR_STORE=local.*IMAGE_STORE=local"):
        make_settings(tmp_path, allow_local_storage=False)
    ok = make_settings(
        tmp_path, allow_local_storage=False, database_url="postgresql://u:p@h/db",
        vector_store="chroma", image_store="r2",
    )
    assert ok.allow_local_storage is False
