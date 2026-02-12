import base64
import hmac
from hashlib import sha256


def validate_turn_signature(req, secret: str):
    if not secret:
        return {"error": "TURN_HMAC_SECRET must be set"}, 500

    signature = req.headers.get("X-Turn-Hook-Signature")
    if not signature:
        return {"error": "X-Turn-Hook-Signature header required"}, 401

    digest = hmac.new(secret.encode(), req.get_data(), sha256).digest()
    expected = base64.b64encode(digest).decode()

    if not hmac.compare_digest(expected, signature):
        return {"error": "invalid hook signature"}, 401

    return None
