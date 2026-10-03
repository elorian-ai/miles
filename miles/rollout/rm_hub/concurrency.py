"""Bounded asynchronous wrappers for Miles custom reward functions."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from miles.utils import async_utils
from miles.utils import types as miles_types


Result = TypeVar("Result")


def async_reward(
    scorer: Callable[..., Awaitable[Result]], *, concurrency: int
) -> Callable[..., Awaitable[Result | list[Result]]]:
    """Wrap an async scorer as a Miles hook with a shared in-flight limit.

    Create one hook per worker/event loop. Single and batch calls share its
    limit; batches preserve order and use Miles' standard error handling.
    Args, kwargs, and results pass through without conversion.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def reward_func(
        args: argparse.Namespace,
        sample_or_samples: miles_types.Sample | list[miles_types.Sample],
        **kwargs: Any,
    ) -> Result | list[Result]:
        if isinstance(sample_or_samples, list):
            return cast(
                list[Result],
                await async_utils.gather_and_raise_first(
                    [reward_func(args, sample, **kwargs) for sample in sample_or_samples]
                ),
            )
        async with semaphore:
            return await scorer(args, sample_or_samples, **kwargs)

    return reward_func
