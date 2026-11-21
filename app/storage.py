"""Where generated images are kept and how the browser reaches them.

- local: files in DATA_DIR/images, served by the API at /images/<key> (development).
- r2:    Cloudflare R2 (S3-compatible). The bucket stays private; the API hands out presigned
         GET URLs that expire, or public URLs when R2_PUBLIC_BASE_URL is set.

Every store also keeps a local copy in DATA_DIR/images, so the image judge can read the file.
On hosts with an ephemeral disk (e.g. Render) that copy is a short-lived cache only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ImageStore(ABC):
    name: str

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def local_path(self, key: str) -> Path:
        return self.cache_dir / key

    def save(self, key: str, data: bytes, content_type: str) -> None:
        self.local_path(key).write_bytes(data)
        self._put(key, data, content_type)

    @abstractmethod
    def _put(self, key: str, data: bytes, content_type: str) -> None: ...

    @abstractmethod
    def url(self, key: str) -> str:
        """URL the browser uses: relative to the API ("/images/…") or absolute."""


class LocalImageStore(ImageStore):
    name = "local"

    def _put(self, key: str, data: bytes, content_type: str) -> None:
        pass  # the local copy is the store

    def url(self, key: str) -> str:
        return f"/images/{key}"


class R2ImageStore(ImageStore):
    name = "r2"

    def __init__(
        self,
        cache_dir: Path,
        *,
        bucket: str,
        prefix: str = "images/",
        public_base_url: str = "",
        url_ttl_seconds: int = 21600,
        client: Any = None,
        account_id: str = "",
        access_key_id: str = "",
        secret_access_key: str = "",
    ) -> None:
        super().__init__(cache_dir)
        if client is None:
            if not (account_id and access_key_id and secret_access_key and bucket):
                raise ValueError("R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET are required for IMAGE_STORE=r2")
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name="auto",
                config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
            )
        self.client = client
        self.bucket = bucket
        self.prefix = prefix
        self.public_base_url = public_base_url.rstrip("/")
        self.url_ttl = url_ttl_seconds

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def _put(self, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(
            Bucket=self.bucket, Key=self._key(key), Body=data, ContentType=content_type,
            CacheControl="private, max-age=86400",
        )

    def url(self, key: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/{self._key(key)}"
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": self._key(key)}, ExpiresIn=self.url_ttl
        )


def configure_r2_cors(store: R2ImageStore, origins: list[str]) -> None:
    """Let the web app load images cross-origin (needed for the preview, copy and download)."""
    store.client.put_bucket_cors(
        Bucket=store.bucket,
        CORSConfiguration={
            "CORSRules": [
                {
                    "AllowedOrigins": origins,
                    "AllowedMethods": ["GET", "HEAD"],
                    "AllowedHeaders": ["*"],
                    "MaxAgeSeconds": 3600,
                }
            ]
        },
    )


def main() -> None:
    """python -m app.storage check | setup-cors <origin> [<origin> …]"""
    import sys
    import time

    from app.config import get_settings

    store = make_image_store(get_settings())
    if not isinstance(store, R2ImageStore):
        sys.exit("IMAGE_STORE is not r2")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "setup-cors":
        origins = sys.argv[2:] or get_settings().cors_origin_list
        configure_r2_cors(store, origins)
        print(f"CORS set on bucket {store.bucket} for: {', '.join(origins)}")
    elif cmd == "check":
        key = f"_check_{int(time.time())}.txt"
        store._put(key, b"ok", "text/plain")
        url = store.url(key)
        store.client.delete_object(Bucket=store.bucket, Key=store._key(key))
        print(f"R2 OK: wrote and deleted {store.bucket}/{store._key(key)}; URL style: {url.split('?')[0]}")
    else:
        sys.exit(main.__doc__)


def make_image_store(settings) -> ImageStore:
    if settings.image_store in ("local", ""):
        return LocalImageStore(settings.images_dir)
    if settings.image_store == "r2":
        return R2ImageStore(
            settings.images_dir,
            bucket=settings.r2_bucket,
            prefix=settings.r2_prefix,
            public_base_url=settings.r2_public_base_url,
            url_ttl_seconds=settings.r2_url_ttl_seconds,
            account_id=settings.r2_account_id,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
        )
    raise ValueError(f"IMAGE_STORE '{settings.image_store}' is not supported (use local or r2)")


if __name__ == "__main__":
    main()
