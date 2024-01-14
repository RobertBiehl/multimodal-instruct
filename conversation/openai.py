import re

from . import ConversationBatchService


class OpenAIBatchService(ConversationBatchService):
    MAX_CONCURRENT_REQUESTS = 16

    def __init__(self, model: str, openai_base_url: str = None):
        name = re.sub(r'\W+', '_', f"{model}{openai_base_url or ''}").lower()
        super().__init__(f"openai_{name}")
        from openai import OpenAI
        self.client = OpenAI(base_url=openai_base_url)
        self.model = model

    def process(self, messages):
        result = self.client.chat.completions.create(model=self.model, messages=messages)
        return result.choices[0].message.content
