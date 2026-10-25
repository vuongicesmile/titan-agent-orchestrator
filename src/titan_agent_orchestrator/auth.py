"""Xác thực user context đáng tin cậy do Titan Control Plane chuyển tiếp."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from .models import Principal, SignedContext
from .settings import settings


def _decode_context(value: str) -> Principal:
    try:
        encoded, supplied_signature = value.rsplit(".", 1)
        expected_signature = hmac.new(
            settings.context_signing_secret.encode(), encoded.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise ValueError("signature")
        padded = encoded + "=" * (-len(encoded) % 4)
        context = SignedContext.model_validate_json(base64.urlsafe_b64decode(padded))
        if context.expires_at <= datetime.now(UTC):
            raise ValueError("expired")
        return Principal(
            org_id=context.org_id,
            user_id=context.user_id,
            permissions=context.permissions,
        )
    except (ValueError, UnicodeError, json.JSONDecodeError):
        raise HTTPException(status_code=401, detail="Titan context is invalid") from None


def current_principal(
    titan_context: Annotated[str | None, Header(alias="X-Titan-Context")] = None,
) -> Principal:
    if titan_context and settings.context_signing_secret:
        principal = _decode_context(titan_context)
    elif settings.allow_dev_context:
        principal = Principal(
            org_id="demo",
            user_id="local-developer",
            permissions=frozenset({"agent:chat", "agent:execute"}),
        )
    else:
        raise HTTPException(status_code=401, detail="Signed Titan context is required")
    if "agent:chat" not in principal.permissions:
        raise HTTPException(status_code=403, detail="agent:chat permission is required")
    return principal


PrincipalDep = Annotated[Principal, Depends(current_principal)]
