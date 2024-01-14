import re

import transformers

from . import ConversationBatchService


class LLaMACppBatchService(ConversationBatchService):
    MAX_CONCURRENT_REQUESTS = 1

    def __init__(self, model: str):
        name = re.sub(r'\W+', '_', model).lower()
        super().__init__(f"huggingface_{name}")
        from llama_cpp import Llama
        self.llm = Llama(model_path=model, n_ctx=4096)

    def process(self, messages):
        res = self.llm.create_chat_completion(messages=messages)
        response = res["choices"][0]["message"]["content"]
        return response