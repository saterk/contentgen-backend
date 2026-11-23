from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.auth import LoginGuard, hash_password, issue_token, verify_password, verify_token
from app.main import create_app
from tests.conftest import API_KEY, make_settings

USER, PASSWORD = "chaiyote", "s3cret-test"
USER2, PASSWORD2 = "mktadmin", "other-s3cret"


@pytest.fixture
def auth_client(settings):
    from app.kb.ingest import ingest_with_settings  # noqa: F401  (settings fixture already ingested)

    s = make_settings(
        settings.data_dir,
        auth_username=USER,
        auth_password_hash=hash_password(PASSWORD, iterations=1000),
        auth_users=f"{USER2}={hash_password(PASSWORD2, iterations=1000)}",
        auth_secret="test-secret",
    )
    with TestClient(create_app(s)) as c:
        yield c


def login(client, username=USER, password=PASSWORD):
    return client.post("/api/auth/login", json={"username": username, "password": password})


# ---- unit -------------------------------------------------------------------


def test_password_hash_roundtrip():
    h = hash_password("Lucky", iterations=1000)
    assert h.startswith("pbkdf2_sha256$1000$") and "Lucky" not in h
    assert verify_password("Lucky", h)
    assert not verify_password("lucky", h)
    assert not verify_password("Lucky", "garbage")
    assert hash_password("Lucky", iterations=1000) != h, "salted"


def test_token_roundtrip_expiry_and_tampering():
    token, exp = issue_token("chaiyote", "k", ttl_seconds=60, now=1000)
    assert exp == 1060
    assert verify_token(token, "k", now=1059) == "chaiyote"
    assert verify_token(token, "k", now=1061) is None, "expired"
    assert verify_token(token, "other-secret", now=1000) is None
    payload, sig = token.split(".")
    forged, _ = issue_token("admin", "attacker", ttl_seconds=60, now=1000)
    assert verify_token(forged.split(".")[0] + "." + sig, "k", now=1000) is None
    assert verify_token("not-a-token", "k") is None


def test_login_guard_locks_after_failures():
    g = LoginGuard(max_failures=3, lock_seconds=60)
    for _ in range(3):
        g.failed("1.2.3.4")
    assert g.locked_for("1.2.3.4") > 0 and g.locked_for("5.6.7.8") == 0
    g.succeeded("1.2.3.4")
    assert g.locked_for("1.2.3.4") == 0


# ---- API --------------------------------------------------------------------


def test_login_and_use_session(auth_client):
    r = login(auth_client)
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == USER and body["expires_at"] > time.time()
    h = {"Authorization": f"Bearer {body['token']}"}
    assert auth_client.get("/api/auth/me", headers=h).json() == {"username": USER}
    assert auth_client.get("/v1/models", headers=h).status_code == 200
    assert auth_client.get("/api/conversations/x/items", headers=h).status_code == 200


def test_username_is_case_insensitive(auth_client):
    assert login(auth_client, username="  Chaiyote ").status_code == 200


@pytest.mark.parametrize(("username", "password"), [(USER, "wrong"), ("someone", PASSWORD), ("", "")])
def test_wrong_credentials(auth_client, username, password):
    r = login(auth_client, username, password)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_credentials"
    assert r.json()["error"]["message"] == "Wrong username or password", "same message either way"


def test_lockout_after_repeated_failures(auth_client):
    for _ in range(5):
        login(auth_client, password="nope")
    r = login(auth_client)  # even the right password is refused while locked
    assert r.status_code == 429 and r.json()["error"]["code"] == "login_locked"


def test_protected_endpoints_reject_missing_or_bad_tokens(auth_client):
    for headers in ({}, {"Authorization": "Bearer junk"}, {"Authorization": "Bearer a.b"}):
        assert auth_client.get("/api/auth/me", headers=headers).status_code == 401
        assert auth_client.get("/v1/models", headers=headers).status_code == 401


def test_app_api_key_still_works_for_scripts(auth_client):
    h = {"Authorization": f"Bearer {API_KEY}"}
    assert auth_client.get("/v1/models", headers=h).status_code == 200
    assert auth_client.get("/api/auth/me", headers=h).json() == {"username": "api_key"}


def test_login_not_configured(client):
    r = client.post("/api/auth/login", json={"username": "a", "password": "b"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "auth_not_configured"


def test_health_stays_public(auth_client):
    assert auth_client.get("/api/health").status_code == 200


def test_empty_app_api_key_disables_key_access(settings):
    s = make_settings(settings.data_dir, app_api_key="")
    with TestClient(create_app(s)) as c:
        assert c.get("/v1/models", headers={"Authorization": "Bearer "}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer change-me"}).status_code == 401


def test_second_user_from_auth_users(auth_client):
    r = login(auth_client, "MktAdmin", PASSWORD2)
    assert r.status_code == 200 and r.json()["username"] == USER2
    h = {"Authorization": f"Bearer {r.json()['token']}"}
    assert auth_client.get("/api/auth/me", headers=h).json() == {"username": USER2}
    assert login(auth_client, USER2, PASSWORD).status_code == 401, "each user has their own password"
    assert login(auth_client, USER, PASSWORD2).status_code == 401


def test_login_users_parsing(settings):
    from app.config import login_users

    s = make_settings(settings.data_dir, auth_username="Alice", auth_password_hash="h1",
                      auth_users=" bob=h2 , carol=pbkdf2_sha256$1$a$b ,broken, =h3")
    assert login_users(s) == {"alice": "h1", "bob": "h2", "carol": "pbkdf2_sha256$1$a$b"}


def test_token_of_removed_user_is_rejected(auth_client):
    tok, _ = issue_token("ghost", "test-secret", 60)
    assert auth_client.get("/api/auth/me", headers={"Authorization": f"Bearer {tok}"}).status_code == 401
