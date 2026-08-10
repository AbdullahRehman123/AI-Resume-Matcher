"""Sends account-verification emails via Resend.

Whether this gets called at all is decided by the caller (/auth/signup in
main.py checks RESEND_API_KEY before calling) - this module doesn't decide
whether to fall back to the dev-mode verification_link, it just reports
success/failure of the actual send.
"""

import logging
import os

import resend

logger = logging.getLogger(__name__)

FROM_EMAIL = os.getenv("FROM_EMAIL", "onboarding@resend.dev")


def send_verification_email(to_email: str, verification_link: str) -> bool:
    """Send the verification link via Resend. Returns True on success,
    False on any failure - a flaky email provider shouldn't crash signup,
    so the caller falls back to showing the link directly instead of us
    raising here."""
    resend.api_key = os.getenv("RESEND_API_KEY")

    html = f"""
    <div style="font-family: sans-serif; max-width: 480px; margin: 0 auto;">
      <h2>Verify your HireLens account</h2>
      <p>Click the button below to verify your email and start using HireLens.</p>
      <p>
        <a href="{verification_link}"
           style="display:inline-block; padding:10px 20px; background:#2f5d50;
                  color:#fff; text-decoration:none; border-radius:6px;">
          Verify email
        </a>
      </p>
      <p>Or paste this link into your browser: {verification_link}</p>
    </div>
    """

    try:
        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": to_email,
            "subject": "Verify your HireLens account",
            "html": html,
        })
        logger.info("Verification email sent to %s via Resend", to_email)
        return True
    except Exception:
        logger.exception("Failed to send verification email to %s via Resend", to_email)
        return False
