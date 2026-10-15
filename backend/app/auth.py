import hashlib
import hmac
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import Cookie, HTTPException, Response
from pydantic import BaseModel, Field

from .config import COOKIE_SECURE
from .db import connection

SESSION_DAYS = 30
COOKIE_NAME = "pagesutra_session"


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)


def normalized_email(email: str) -> str:
    value = email.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
        raise HTTPException(400, "Enter a valid email address.")
    return value


def password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, salt, digest = stored.split("$")
        if algorithm != "scrypt":
            return False
        candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1)
        return hmac.compare_digest(candidate, bytes.fromhex(digest))
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(response: Response, user_id: str) -> None:
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
    with connection() as db:
        db.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                   (token_hash(token), user_id, expires.isoformat()))
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_DAYS * 86400,
                        httponly=True, secure=COOKIE_SECURE, samesite="lax", path="/")


def current_user(pagesutra_session: str | None = Cookie(default=None)) -> dict:
    if not pagesutra_session:
        raise HTTPException(401, "Sign in to continue.")
    with connection() as db:
        row = db.execute(
            "SELECT u.id, u.email FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (token_hash(pagesutra_session), datetime.now(timezone.utc).isoformat()),
        ).fetchone()
    if not row:
        raise HTTPException(401, "Your session has expired. Sign in again.")
    return dict(row)


def register(credentials: Credentials, response: Response) -> dict:
    email = normalized_email(credentials.email)
    user_id = str(uuid.uuid4())
    hashed = password_hash(credentials.password)
    with connection() as db:
        if db.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            raise HTTPException(409, "An account with this email already exists.")
        first_user = not db.execute("SELECT 1 FROM users LIMIT 1").fetchone()
        db.execute("INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
                   (user_id, email, hashed))
        if first_user:
            db.execute("UPDATE documents SET owner_id = ? WHERE owner_id IS NULL", (user_id,))
            db.execute("UPDATE conversations SET owner_id = ? WHERE owner_id IS NULL", (user_id,))
    start_session(response, user_id)
    return {"id": user_id, "email": email}


def login(credentials: Credentials, response: Response) -> dict:
    email = normalized_email(credentials.email)
    with connection() as db:
        row = db.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    if not row or not verify_password(credentials.password, row["password_hash"]):
        raise HTTPException(401, "Incorrect email or password.")
    start_session(response, row["id"])
    return {"id": row["id"], "email": row["email"]}


def logout(response: Response, pagesutra_session: str | None = Cookie(default=None)) -> None:
    if pagesutra_session:
        with connection() as db:
            db.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(pagesutra_session),))
    response.delete_cookie(COOKIE_NAME, path="/")
