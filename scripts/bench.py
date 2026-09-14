#!/usr/bin/env python3
"""Concurrency benchmark for the Chatterbox Flash TTS server.

For each level in --levels, fires that many simultaneous synthesis requests
and reports wall time, per-request latency, and throughput — so the scaling
behavior as concurrency grows is visible at a glance.

Usage:
    .venv/bin/python scripts/bench.py --audio voice.mp3 --levels 1,2,4,8
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

DEFAULT_TEXT = (
    "Lucy is expressive, curious, romantic, idealistic, and emotionally sincere. "
    "She wants to feel grown-up, but her feelings are often bigger and faster "
    "than her judgment."
)


def _mime(path: str) -> str:
    if path.endswith((".wav", ".wave")):
        return "audio/wav"
    if path.endswith(".mp3"):
        return "audio/mpeg"
    return "application/octet-stream"


async def one_request(
    client: httpx.AsyncClient, base: str, text: str, audio: bytes, mime: str
) -> float:
    t0 = time.perf_counter()
    resp = await client.post(
        f"{base}/v1/syntheses",
        data={"text": text},
        files={"reference_audio": ("reference", audio, mime)},
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return elapsed


def summarize(level: int, latencies: list[float], wall: float) -> None:
    print(
        f"  {level:>3} req:  wall {wall:7.2f}s   avg {statistics.mean(latencies):6.2f}s   "
        f"min {min(latencies):6.2f}s   max {max(latencies):6.2f}s   {level / wall:5.2f} req/s"
    )


async def run(args: argparse.Namespace) -> None:
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    audio = open(args.audio, "rb").read()
    mime = _mime(args.audio)
    timeout = httpx.Timeout(args.timeout)

    async with httpx.AsyncClient(timeout=timeout) as client:
        warm = await one_request(client, args.url, args.text, audio, mime)
        print(f"  warmup:      {warm:7.2f}s   (not counted)")
        for level in levels:
            t0 = time.perf_counter()
            latencies = list(
                await asyncio.gather(
                    *(
                        one_request(client, args.url, args.text, audio, mime)
                        for _ in range(level)
                    )
                )
            )
            summarize(level, latencies, time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", required=True, help="reference voice file")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--levels", default="1,2,4,8", help="comma list, e.g. 1,2,4,8")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
