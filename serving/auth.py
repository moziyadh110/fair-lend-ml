"""API-key authentication for the decision-making endpoints. A lending
decision service with no auth at all was flagged as a real gap (not a
nitpick) — anyone could hit /predict. A single shared API key is the
proportionate fix for this project's scope; a real deployment would want
per-caller keys / OAuth2, which is a bigger, separate project.
"""
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided_key: str | None = Security(_api_key_header)) -> None:
    expected_key = os.environ.get("API_KEY")
    if not expected_key:
        # No key configured — fail closed rather than silently running unauthenticated.
        raise HTTPException(status_code=503, detail="API_KEY is not configured on the server")
    if provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
