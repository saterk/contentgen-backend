"""Prototype login: one fixed user, password stored as a PBKDF2 hash, signed session tokens.

    python -m app.auth hash-password      # prints an AUTH_PASSWORD_HASH value for .env

Tokens are `<base64url(payload)>.<base64url(hmac)>` with payload {"u": username, "exp": unix_time},
signed with AUTH_SECRET. No server-side session store is needed.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import json
import secrets
import sys
import threading
import time
from dataclasses import dataclass

PBKDF2_ITERATIONS = 310_000


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# ---- passwords -------------------------------------------------------------


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt, expected = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), _unb64(salt), int(iters))
        return hmac.compare_digest(dk, _unb64(expected))
    except (ValueError, TypeError):
        return False


# ---- session tokens --------------------------------------------------------


def issue_token(username: str, secret: str, ttl_seconds: int, now: float | None = None) -> tuple[str, int]:
    exp = int((now or time.time()) + ttl_seconds)
    payload = _b64(json.dumps({"u": username, "exp": exp}, separators=(",", ":")).encode())
    sig = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}", exp


def verify_token(token: str, secret: str, now: float | None = None) -> str | None:
    """Return the username for a valid, unexpired token, else None."""
    try:
        payload, sig = token.split(".")
        expected = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(_unb64(payload))
        if int(data["exp"]) < (now or time.time()):
            return None
        return str(data["u"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


# ---- brute-force guard -----------------------------------------------------


@dataclass
class LoginGuard:
    """Blocks a client for `lock_seconds` after `max_failures` failed logins in a row."""

    max_failures: int = 5
    lock_seconds: int = 300

    def __post_init__(self) -> None:
        self._fails: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def locked_for(self, key: str) -> int:
        with self._lock:
            count, until = self._fails.get(key, (0, 0.0))
            return max(0, int(until - time.time())) if count >= self.max_failures else 0

    def failed(self, key: str) -> None:
        with self._lock:
            count, _ = self._fails.get(key, (0, 0.0))
            count += 1
            self._fails[key] = (count, time.time() + self.lock_seconds if count >= self.max_failures else 0.0)

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)

    def reset_expired(self, key: str) -> None:
        with self._lock:
            count, until = self._fails.get(key, (0, 0.0))
            if count >= self.max_failures and until <= time.time():
                self._fails.pop(key, None)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "hash-password":
        print(__doc__)
        sys.exit(1)
    pw = getpass.getpass("Password: ")
    if pw != getpass.getpass("Repeat: "):
        sys.exit("Passwords don't match")
    print(f"AUTH_PASSWORD_HASH={hash_password(pw)}")
    print(f"AUTH_SECRET={secrets.token_urlsafe(32)}  # only if you don't have one yet")


if __name__ == "__main__":
    main()
