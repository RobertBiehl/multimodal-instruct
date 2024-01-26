import re

import transformers

from . import ConversationBatchService


class HuggingfaceBatchService(ConversationBatchService):
    MAX_CONCURRENT_REQUESTS = 1

    def __init__(self, model: str):
        name = re.sub(r'\W+', '_', model).lower()
        super().__init__(f"huggingface_{name}")
        self._pipe = transformers.pipeline("conversational", model)

    def process(self, messages):
        return self._pipe(messages).generated_responses[-1]
