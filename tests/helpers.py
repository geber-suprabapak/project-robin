from datetime import datetime, timedelta, timezone

import jwt


TEST_JWT_SECRET = "test-secret"
TEST_USER_ID = "550e8400-e29b-41d4-a716-446655440000"


def make_token(
    secret: str = TEST_JWT_SECRET,
    user_id: str = TEST_USER_ID,
    expired: bool = False,
) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=-1 if expired else 15)
    return jwt.encode(
        {
            "sub": user_id,
            "aud": "authenticated",
            "exp": expires_at,
        },
        secret,
        algorithm="HS256",
    )
