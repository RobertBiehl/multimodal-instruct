import asyncio
import dataclasses
import json
import logging
import os
import re
from typing import Any, Optional
import requests

from thirdparty.openai_api_request_parallel_processor import process_api_requests_from_iterator, StatusTracker
from . import ConversationBatchService


@dataclasses.dataclass
class OpenAILimits:
    max_requests_per_minute: int
    max_tokens_per_minute: int
    max_attempts: int


class OpenAIBatchService(ConversationBatchService):
    MAX_CONCURRENT_REQUESTS = 16

    def __init__(self, model: str, openai_base_url: str, token_encoding_name: str = "cl100k_base"):
        name = re.sub(r'\W+', '_', f"{model}{openai_base_url or ''}").lower().strip("_")
        openai_base_url = "https://api.openai.com/v1/" if openai_base_url is None else openai_base_url

        super().__init__(f"openai_{name}")

        self.model = model

        self.headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }

        self.openai_base_url = openai_base_url
        self.chat_completions_url = f"{openai_base_url}chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.openrouter_limits_url = f"{openai_base_url}auth/key"

        self.max_batch_size = 256
        self.batch = []

        self.limits = self._get_limits()

        self.token_encoding_name = token_encoding_name

        self.status_tracker: Optional[StatusTracker] = None

    def _get_limits(self) -> OpenAILimits:
        default = OpenAILimits(
            max_requests_per_minute=int(3_000 * 0.5),
            max_tokens_per_minute=int(250_000 * 0.5),
            max_attempts=5
        )
        if "openrouter" in self.openai_base_url:
            def convert_to_seconds(time_str):
                if time_str.endswith('s'):
                    return int(time_str[:-1])
                elif time_str.endswith('m'):
                    return int(time_str[:-1]) * 60
                else:
                    raise ValueError("Unsupported time format")

            response = requests.get(
                url=self.openrouter_limits_url, headers=self.headers
            )

            try:
                data = response.json()
                data = data["data"]
                return OpenAILimits(
                    max_requests_per_minute=int(
                        data["rate_limit"]["requests"] / convert_to_seconds(data["rate_limit"]["interval"]) * 60),
                    max_tokens_per_minute=default.max_tokens_per_minute,
                    max_attempts=default.max_attempts
                )
            except Exception:
                pass

        return default

    async def _submit_internal(self, request_id: str, messages, *vargs, **kwargs):
        response = await self.cache_get(request_id)
        if response:
            # print(f"Cache hit {request_id}")
            await self._on_complete(request_id, response, True, *vargs, **kwargs)
        else:
            # # print(f"Processing {request_id} ({self._num_in_progress} / {self.MAX_CONCURRENT_REQUESTS} jobs)")
            self.batch.append({
                "model": self.model,
                "messages": messages,
                "max_tokens": 1024,
                "metadata": {
                    "request_id": request_id,
                    "vargs": vargs,
                    "kwargs": kwargs,
                }

            })
            if len(self.batch) > self.max_batch_size:
                await self._commit()

    async def _commit(self):
        async def cb(err: Exception, request: Any, response, metadata):
            if err:
                self._on_error(metadata["request_id"], err, request, *metadata["vargs"], **metadata["kwargs"])
            else:
                result = response["choices"][0]["message"]["content"]
                await self._on_complete(metadata["request_id"], result, False, *metadata["vargs"], **metadata["kwargs"])

        num_items = len(self.batch)
        self._num_submitted += num_items
        self._num_in_progress += num_items

        requests_iterator = iter(self.batch)
        self.batch = []

        limits = self.limits
        self.status_tracker = StatusTracker()
        await process_api_requests_from_iterator(
            requests=requests_iterator,
            callback=cb,
            request_url=self.chat_completions_url,
            api_key=self.api_key,
            max_requests_per_minute=limits.max_requests_per_minute,
            max_tokens_per_minute=limits.max_tokens_per_minute,
            token_encoding_name=self.token_encoding_name,
            max_attempts=limits.max_attempts,
            logging_level=logging.WARNING,
            status_tracker=self.status_tracker
        )

    async def finish(self):
        if self.batch:
            await self._commit()

    async def _on_submit(self):
        pass

    async def _on_complete(self, request_id: str, result, from_cache: bool, *vargs, **kwargs):
        self._num_in_progress -= 1
        self._num_complete += 1

        if not from_cache:
            await self.cache_set(request_id, result)
        else:
            self._num_complete_from_cache += 1

        if self._on_result:
            self._on_result(request_id, result, *vargs, **kwargs)

    async def process(self, messages):
        pass

    @property
    def num_in_progress(self) -> int:
        return self.status_tracker.num_tasks_in_progress if self.status_tracker else 0

    @property
    def num_temp_failed(self) -> int:
        return self.status_tracker.num_api_errors + self.status_tracker.num_rate_limit_errors if self.status_tracker else 0

    @property
    def num_failed(self) -> int:
        return self.status_tracker.num_tasks_failed if self.status_tracker else 0
    # @property
    # def num_completed(self) -> int:
    #     return self.status_tracker.num_tasks_succeeded if self.status_tracker else 0