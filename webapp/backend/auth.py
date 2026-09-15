from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from webapp.backend.config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET

_BCRYPT_MAX_BYTES = 72  # bcrypt truncates beyond this; reject instead


def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > _BCRYPT_MAX_BYTES:
        raise ValueError(f"Password must be at most {_BCRYPT_MAX_BYTES} bytes")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise ValueError("Invalid or expired token")
    sub = payload.get("sub")
    if sub is None:
        raise ValueError("Invalid token payload")
    return int(sub)
