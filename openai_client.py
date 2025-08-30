import os
import logging
from openai import OpenAI
try:
    from openai import RateLimitError, APIStatusError
except Exception:
    RateLimitError = tuple()
    APIStatusError = tuple()

logger = logging.getLogger(__name__)

CORE_PROMPT = os.getenv("OPENAI_CORE_PROMPT", "").strip()
ROLE_PROMPT = os.getenv("OPENAI_ROLE_PROMPT", "").strip()
FALLBACK_GENERIC   = "処理中に問題が発生しました。恐れ入りますが、もう一度入力していただけますか？"
FALLBACK_RATE      = "ただいま少し混み合っているため、すぐに応答できませんでした。時間をおいてから、もう一度お試しください。"
FALLBACK_SENSITIVE = ""


class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

    def _is_rate_limit(self, err: Exception) -> bool:
        try:
            if isinstance(err, RateLimitError):
                return True
        except Exception:
            pass

        try:
            if isinstance(err, APIStatusError) and getattr(err, "status_code", None) == 429:
                return True
        except Exception:
            pass

        msg = (str(err) or "").lower()
        if "429" in msg or "rate limit" in msg or "too many requests" in msg or "Too Many Requests" in msg:
            return True
        if hasattr(err, "status_code") and getattr(err, "status_code") == 429:
            return True
        return False

    def _friendly_fallback(self, err: Exception | None, kind: str = "generic") -> str:
        if err and self._is_rate_limit(err):
            return FALLBACK_RATE
        if kind == "sensitive":
            return FALLBACK_SENSITIVE
        if kind == "rate":
            return FALLBACK_RATE
        msg = (str(err) or "").lower() if err else ""
        if ("policy" in msg or ("content" in msg and "filter" in msg)):
            return FALLBACK_SENSITIVE
        if ("rate" in msg or "429" in msg or "timeout" in msg or "temporarily" in msg or "503" in msg):
            return FALLBACK_RATE
        return FALLBACK_GENERIC

    def _compose_messages(self, user_message: str):
        messages = [{"role": "system", "content": CORE_PROMPT}]
        if ROLE_PROMPT:
            messages.append({"role": "system", "content": ROLE_PROMPT})

        messages.append({"role": "user", "content": user_message})
        return messages

    def get_reply(self, user_message: str, previous_response_id: str=None):
        try:
            messages = self._compose_messages(user_message)

            if previous_response_id:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    previous_response_id=previous_response_id
                )
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    store=True
                )

            ai_message  = response.output_text.strip()
            response_id = response.id
            return ai_message, response_id

        except Exception as e:
            logger.exception("🔥 OpenAIClient#get_reply failed")
            if self._is_rate_limit(e):
                return self._friendly_fallback(e, "rate"), None
            return self._friendly_fallback(e), None
