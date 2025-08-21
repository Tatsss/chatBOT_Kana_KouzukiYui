import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

CORE_PROMPT = os.getenv("OPENAI_CORE_PROMPT", "").strip()
ROLE_PROMPT = os.getenv("OPENAI_ROLE_PROMPT", "").strip()

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

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

        except Exception:
            logger.exception("🔥 OpenAIClient#get_reply failed")
            return "エラーが発生しました。", None
