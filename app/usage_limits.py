"""Daily usage caps (global + per-user) for CV processing, backed by Redis.

Reuses the same REDIS_URL Celery already talks to, but through its own
`redis.Redis` connection - not tied to Celery's broker/backend usage.

Note on what "count" means here: each CV processed costs roughly two LLM
calls in this project's current design (extraction in extractor.py, then
explanation in matcher.py's explain_match). These counters track CVs
processed, not raw LLM calls - a deliberate simplification, not an
oversight. If per-call cost tracking is ever needed, this is the place
to change it.
"""

import logging
import os
from datetime import datetime, timezone

import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GLOBAL_DAILY_CV_CAP = int(os.getenv("GLOBAL_DAILY_CV_CAP", "30"))
USER_DAILY_CV_CAP = int(os.getenv("USER_DAILY_CV_CAP", "10"))
SIGNUP_RATE_LIMIT_PER_HOUR = int(os.getenv("SIGNUP_RATE_LIMIT_PER_HOUR", "3"))

SECONDS_PER_DAY = 24 * 60 * 60
SECONDS_PER_HOUR = 60 * 60

_redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


class UsageLimitExceeded(Exception):
    """Raised when processing `count` more CVs would exceed the global or
    per-user daily cap. `scope` lets callers give a global-cap message a
    different tone than a per-user-cap message."""

    def __init__(self, message: str, scope: str):
        super().__init__(message)
        self.scope = scope  # "global" or "user"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def check_and_increment_usage(user_id: int, count: int = 1) -> None:
    """Check both the global and per-user daily CV caps BEFORE incrementing
    either counter, and only increment if both would stay under cap.

    Must be called before the extraction call in /match and before queuing
    candidates in /candidates/batch - the whole point is to stop the LLM
    call from happening at all once a cap is hit, not to count after the
    fact.
    """
    if count <= 0:
        return

    today = _today()
    global_key = f"usage:global:{today}"
    user_key = f"usage:user:{user_id}:{today}"

    current_global = int(_redis_client.get(global_key) or 0)
    if current_global + count > GLOBAL_DAILY_CV_CAP:
        logger.warning(
            "Global daily CV cap would be exceeded: %d + %d > %d",
            current_global, count, GLOBAL_DAILY_CV_CAP,
        )
        raise UsageLimitExceeded(
            f"The system-wide daily limit of {GLOBAL_DAILY_CV_CAP} CVs has been reached. "
            "Please try again after midnight UTC.",
            scope="global",
        )

    current_user = int(_redis_client.get(user_key) or 0)
    if current_user + count > USER_DAILY_CV_CAP:
        logger.warning(
            "User %d daily CV cap would be exceeded: %d + %d > %d",
            user_id, current_user, count, USER_DAILY_CV_CAP,
        )
        raise UsageLimitExceeded(
            f"You've reached your daily limit of {USER_DAILY_CV_CAP} CVs. "
            "Please try again after midnight UTC.",
            scope="user",
        )

    _increment_with_expiry(global_key, count)
    _increment_with_expiry(user_key, count)
    logger.info("Usage recorded for user %d: +%d CV(s)", user_id, count)


def _increment_with_expiry(key: str, count: int) -> None:
    # INCRBY on a missing key starts from 0, so a fresh key's result equals
    # `count` exactly once - that's the "first increment of the day" for
    # this key, generalized from a bare INCR to handle batch-sized jumps.
    new_value = _redis_client.incrby(key, count)
    if new_value == count:
        _redis_client.expire(key, SECONDS_PER_DAY)


class SignupRateLimitExceeded(Exception):
    """Raised when an IP has hit SIGNUP_RATE_LIMIT_PER_HOUR signups already
    this hour - a blunt deterrent against scripted mass account creation."""


def check_and_increment_signup_rate(ip: str) -> None:
    """Same increment-then-expire-on-first-write pattern as the CV usage
    counters above, just keyed by hour instead of day."""
    hour_bucket = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    key = f"signup_rate:{ip}:{hour_bucket}"

    current = int(_redis_client.get(key) or 0)
    if current + 1 > SIGNUP_RATE_LIMIT_PER_HOUR:
        logger.warning("Signup rate limit exceeded for IP %s", ip)
        raise SignupRateLimitExceeded(
            "Too many signups from this network in the last hour. Please try again later."
        )

    new_value = _redis_client.incr(key)
    if new_value == 1:
        _redis_client.expire(key, SECONDS_PER_HOUR)
