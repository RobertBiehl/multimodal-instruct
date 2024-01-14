import asyncio
from typing import Dict, Any, Callable, Optional, List, Coroutine

import shelve

caches = {}


class ConversationBatchService:
    MAX_CONCURRENT_REQUESTS = 1

    def __init__(self, name: str):
        self._num_submitted = 0
        self._num_in_progress = 0
        self._num_complete = 0
        self._num_failed = 0

        self._failed_requests = []

        self._results: Dict[str, Any] = {}

        self._on_result: Optional[Callable[[str, Any, ...], None]] = None

        self.cache = caches.get(name, shelve.open(f"cache/response_cache_{name}"))

        self.jobs: List[Coroutine] = []

        self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

    def is_finished(self) -> bool:
        return self._num_submitted == self._num_complete + self._num_failed

    async def finish(self, tasks: Optional[List[asyncio.Task]] = None):
        while not self.is_finished() and tasks:
            await asyncio.gather(*tasks)
            # await asyncio.gather(*self.jobs)

    @property
    def failed_requests(self):
        return self._failed_requests

    def process(self, messages):
        raise NotImplementedError

    async def submit(self, request_id: str, messages, *vargs, **kwargs):
        await self._on_submit()
        task = asyncio.create_task(self._submit_internal(request_id, messages, *vargs, **kwargs))
        return task

    async def _submit_internal(self, request_id: str, messages, *vargs, **kwargs):
        if request_id in self.cache:
            # print(f"Cache hit {request_id}")
            response = self.cache[request_id]
        else:
            # print(f"Processing {request_id} ({self._num_in_progress} / {self.MAX_CONCURRENT_REQUESTS} jobs)")
            response = self.process(messages)
            self.cache[request_id] = response
        self._on_complete(request_id, response, *vargs, **kwargs)

    async def _on_submit(self):
        # if self._num_in_progress >= self.MAX_CONCURRENT_REQUESTS:
        #     print(f"Waiting ({self._num_in_progress} / {self.MAX_CONCURRENT_REQUESTS} jobs)")
        await self.semaphore.acquire()
        self._num_submitted += 1
        self._num_in_progress += 1

    def _on_complete(self, request_id: str, result, *vargs, **kwargs):
        self._results[request_id] = result
        self._num_in_progress -= 1
        self._num_complete += 1

        self.semaphore.release()
        if self._on_result:
            self._on_result(request_id, result, *vargs, **kwargs)

    def _on_error(self, request):
        self._num_in_progress -= 1
        self._num_failed += 1
        self._failed_requests.append(request)

    def set_on_result(self, cb: Callable[[str, Any, ...], None]) -> None:
        self._on_result = cb
