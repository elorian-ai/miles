"""Contracts for shared concurrency, cancellation, and native Miles dispatch."""

from __future__ import annotations

import argparse
import asyncio
from unittest import mock

import pytest
import torch

from miles.rollout import on_policy_distillation, rm_hub
from miles.rollout.rm_hub import concurrency
from miles.utils import function_registry
from miles.utils import types as miles_types


@pytest.mark.asyncio
async def test_miles_dispatch_preserves_arguments_and_results() -> None:
    payload = {"score": 0.5, "opaque": [1, 2]}
    scorer = mock.AsyncMock(return_value=payload)
    hook = concurrency.async_reward(scorer, concurrency=2)
    args = argparse.Namespace(custom_rm_path="test.reward", rm_type="")
    sample = miles_types.Sample()
    with function_registry.function_registry.temporary("test.reward", hook):
        assert await rm_hub.async_rm(args, sample, evaluation=True) is payload
        scorer.assert_awaited_once_with(args, sample, evaluation=True)
        values = await rm_hub.batched_async_rm(args, [sample, sample])
        assert len(values) == 2
        assert all(value is payload for value in values)
        assert await rm_hub.batched_async_rm(args, []) == []


@pytest.mark.asyncio
async def test_concurrency_is_shared_across_batches_and_single_calls() -> None:
    occupied = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0

    async def score(args: argparse.Namespace, sample: miles_types.Sample) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            occupied.set()
        try:
            await release.wait()
            return sample.index
        finally:
            active -= 1

    hook = concurrency.async_reward(score, concurrency=2)
    samples = [miles_types.Sample(index=index) for index in range(5)]
    args = argparse.Namespace()
    calls = asyncio.gather(hook(args, samples[:2]), hook(args, samples[2:4]), hook(args, samples[4]))
    await asyncio.wait_for(occupied.wait(), timeout=1)
    release.set()
    assert await asyncio.wait_for(calls, timeout=1) == [[0, 1], [2, 3], 4]
    assert peak == 2
    assert active == 0


@pytest.mark.asyncio
async def test_batch_failure_waits_for_peers_and_releases_capacity() -> None:
    started = asyncio.Event()
    failed = asyncio.Event()
    release = asyncio.Event()
    stopped = asyncio.Event()

    async def score(args: argparse.Namespace, sample: miles_types.Sample) -> int:
        if sample.index == 0:
            await started.wait()
            failed.set()
            raise RuntimeError("scoring failed")
        if sample.index == 1:
            started.set()
            await release.wait()
            stopped.set()
        return sample.index

    hook = concurrency.async_reward(score, concurrency=2)
    args = argparse.Namespace()
    task = asyncio.create_task(hook(args, [miles_types.Sample(index=0), miles_types.Sample(index=1)]))
    await asyncio.wait_for(failed.wait(), timeout=1)
    assert not task.done()
    release.set()
    with pytest.raises(RuntimeError, match="scoring failed"):
        await asyncio.wait_for(task, timeout=1)
    assert stopped.is_set()
    assert await asyncio.wait_for(
        hook(args, [miles_types.Sample(index=2), miles_types.Sample(index=3)]), timeout=1
    ) == [2, 3]


@pytest.mark.asyncio
async def test_external_cancellation_releases_capacity() -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    async def score(args: argparse.Namespace, sample: miles_types.Sample) -> int:
        if sample.index == 0:
            started.set()
            await never.wait()
        return sample.index

    hook = concurrency.async_reward(score, concurrency=1)
    args = argparse.Namespace()
    task = asyncio.create_task(hook(args, miles_types.Sample(index=0)))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.wait_for(hook(args, miles_types.Sample(index=1)), timeout=1) == 1


@pytest.mark.asyncio
async def test_native_miles_opd_payload_and_postprocessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"meta_info": {"input_token_logprobs": [[None, 1], [-0.25, 2], [-0.5, 3]]}}
    monkeypatch.setattr(on_policy_distillation, "_post_json", mock.AsyncMock(return_value=payload))
    hook = concurrency.async_reward(on_policy_distillation.reward_func, concurrency=2)
    args = argparse.Namespace(
        custom_rm_path="test.opd",
        reward_key=None,
        rm_url="https://teacher.invalid/generate",
    )
    samples = [miles_types.Sample(tokens=[1, 2, 3], response_length=2)]
    with function_registry.function_registry.temporary("test.opd", hook):
        await rm_hub.batched_async_rm(args, samples, inplace_set_reward_field=True)
    assert samples[0].reward is payload
    assert on_policy_distillation.post_process_rewards(args, samples) == ([0.0], [0.0])
    torch.testing.assert_close(samples[0].teacher_log_probs, torch.tensor([-0.25, -0.5]))


@pytest.mark.parametrize("limit", [0, -1])
def test_concurrency_must_be_positive(limit: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        concurrency.async_reward(mock.AsyncMock(), concurrency=limit)
