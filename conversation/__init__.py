import asyncio
import logging
from typing import Dict, Any, Callable, Optional, List, Coroutine

import diskcache

caches: Dict[str, diskcache.Cache] = {}


class ConversationBatchService:
    MAX_CONCURRENT_REQUESTS = 1

    def __init__(self, name: str):
        self._num_submitted = 0
        self._num_in_progress = 0
        self._num_complete = 0
        self._num_failed = 0
        self._num_complete_from_cache = 0

        self._failed_requests = []

        self._on_result: Optional[Callable[[str, Any, ...], None]] = None

        self.cache_lock = asyncio.Lock()
        self.cache = caches.get(name, diskcache.Cache(f"cache/response_cache_{name}"))

        self.jobs: List[Coroutine] = []

        self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

    def __del__(self):
        self.cache.close()

    async def cache_get(self, key: str) -> Any:
        await self.cache_lock.acquire()
        res = self.cache.get(key, None)
        self.cache_lock.release()
        return res

    async def cache_set(self, key: str, obj: Any) -> None:
        await self.cache_lock.acquire()
        self.cache.set(key, obj)
        self.cache_lock.release()

    def is_finished(self) -> bool:
        return self._num_submitted == self._num_complete + self._num_failed

    async def finish(self, tasks: Optional[List[asyncio.Task]] = None):
        while not self.is_finished() and tasks:

            while self._failed_requests:
                (request_id, messages, vargs, kwargs) = self._failed_requests.pop()
                await self.submit(request_id, messages, *vargs, **kwargs)

            await asyncio.gather(*tasks)
            # await asyncio.gather(*self.jobs)

    @property
    def failed_requests(self):
        return self._failed_requests

    async def process(self, messages):
        raise NotImplementedError

    async def submit(self, request_id: str, messages, *vargs, **kwargs):
        await self._on_submit()
        task = asyncio.create_task(self._submit_internal(request_id, messages, *vargs, **kwargs))
        return task

    async def _submit_internal(self, request_id: str, messages, *vargs, **kwargs):
        from_cache = False
        error: Optional[Exception] = None
        response = await self.cache_get(request_id)
        if response:
            from_cache = True
        else:
            # print(f"Processing {request_id} ({self._num_in_progress} / {self.MAX_CONCURRENT_REQUESTS} jobs)")
            response, error = await self.process(messages)
        if error:
            self._on_error(request_id, error, messages, *vargs, **kwargs)
        else:
            await self._on_complete(request_id, response, from_cache, *vargs, **kwargs)

    async def _on_submit(self):
        # if self._num_in_progress >= self.MAX_CONCURRENT_REQUESTS:
        #     print(f"Waiting ({self._num_in_progress} / {self.MAX_CONCURRENT_REQUESTS} jobs)")
        while self._num_in_progress >= self.MAX_CONCURRENT_REQUESTS:
            await asyncio.sleep(0.1)

        await self.semaphore.acquire()
        self._num_submitted += 1
        self._num_in_progress += 1

    async def _on_complete(self, request_id: str, result, from_cache: bool, *vargs, **kwargs):
        self._num_in_progress -= 1
        self._num_complete += 1

        self.semaphore.release()

        if not from_cache:
            await self.cache_set(request_id, result)
        else:
            self._num_complete_from_cache += 1

        if self._on_result:
            self._on_result(request_id, result, *vargs, **kwargs)

    def _on_error(self, request_id, error, request, *vargs, **kwargs):
        self._num_in_progress -= 1
        self._num_failed += 1
        self._failed_requests.append((request_id, request, vargs, kwargs))

    def set_on_result(self, cb: Callable[[str, Any, ...], None]) -> None:
        self._on_result = cb

    @property
    def num_in_progress(self) -> int:
        return self._num_in_progress

    @property
    def num_temp_failed(self) -> int:
        return self._num_failed

    @property
    def num_failed(self) -> int:
        return self._num_failed

    @property
    def num_completed(self) -> int:
        return self._num_complete

    @property
    def num_completed_from_cache(self) -> int:
        return self._num_complete_from_cache
