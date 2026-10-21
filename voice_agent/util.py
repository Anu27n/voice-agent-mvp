from __future__ import annotations

import asyncio
from typing import AsyncIterator, TypeVar

T = TypeVar("T")


async def aiter_timeout(source: AsyncIterator[T], first_s: float, idle_s: float) -> AsyncIterator[T]:
    """Re-yield ``source``, raising TimeoutError if the first item takes longer
    than ``first_s`` or any later gap exceeds ``idle_s``. Closes ``source`` on exit."""
    iterator = source.__aiter__()
    limit = first_s
    try:
        while True:
            try:
                async with asyncio.timeout(limit):
                    item = await anext(iterator)
            except StopAsyncIteration:
                return
            yield item
            limit = idle_s
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()
