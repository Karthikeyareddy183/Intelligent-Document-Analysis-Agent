"""JWT authentication middleware for Supabase Auth (ES256 / HS256)."""

import jwt
from jwt import PyJWKClient
from fastapi import Request, HTTPException

from core.config import get_settings
from core.logger import setup_logger

logger = setup_logger(__name__)

# Cache the JWKS client per Supabase project
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        config = get_settings()
        jwks_url = f"{config.SUPABASE_URL}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(jwks_url, cache_keys=True)
        logger.info("JWKS client initialized", extra={"url": jwks_url})
    return _jwks_client


async def get_current_user(request: Request) -> str:
    """FastAPI dependency — extract and verify user_id from Supabase JWT.

    Supports both ES256 (asymmetric) and HS256 (symmetric) Supabase projects.
    Returns the user_id (sub claim) from a valid JWT token.
    Raises HTTP 401 if token missing/invalid/expired.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization token")

    token = auth_header.split(" ", 1)[1]

    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "HS256")

        if alg == "ES256":
            # Asymmetric: fetch public key from Supabase JWKS
            jwks_client = _get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256"],
                audience="authenticated",
            )
        else:
            # Symmetric: use JWT secret from config
            config = get_settings()
            if not config.SUPABASE_JWT_SECRET:
                raise HTTPException(status_code=500, detail="JWT secret not configured")
            payload = jwt.decode(
                token,
                config.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
            )

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub claim")
        return user_id

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning("JWT validation failed", extra={"error": str(e)})
        raise HTTPException(status_code=401, detail="Invalid token")
