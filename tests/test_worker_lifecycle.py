import asyncio

from fastapi.testclient import TestClient

from conftest import LoaderFactory, make_config, make_wav_bytes
from flash_server import FakeTTSModel, WorkerPool, create_app


def _pool(max_workers, min_spare, ttl=300):
    loader = LoaderFactory()
    pool = WorkerPool(
        loader,
        max_workers=max_workers,
        min_spare_workers=min_spare,
        keep_alive_ttl_seconds=ttl,
    )
    return pool, loader


def test_startup_loads_exactly_min_spare():
    async def scenario():
        pool, loader = _pool(3, 2)
        await pool.initialize()
        assert pool.model_state == "ready"
        assert pool.loaded == 2
        assert pool.loader_calls == 2
        counts = pool.counts()
        assert counts["idle"] == 2 and counts["busy"] == 0 and counts["available"] == 2

    asyncio.run(scenario())


def test_idle_worker_reused_before_new_load():
    async def scenario():
        pool, loader = _pool(3, 1)
        await pool.initialize()
        worker = await pool.acquire()
        assert pool.loader_calls == 1  # reused the baseline worker
        assert pool.loaded == 1
        pool.release(worker, ok=True)

    asyncio.run(scenario())


def test_concurrent_demand_expands_to_max():
    async def scenario():
        pool, loader = _pool(3, 1)
        await pool.initialize()
        workers = await asyncio.gather(
            pool.acquire(), pool.acquire(), pool.acquire()
        )
        assert pool.loaded == 3
        assert pool.loader_calls == 3
        for worker in workers:
            pool.release(worker, ok=True)

    asyncio.run(scenario())


def test_beyond_max_waits_then_executes():
    async def scenario():
        pool, _ = _pool(1, 1)
        await pool.initialize()
        first = await pool.acquire()
        acquired = asyncio.Event()

        async def second():
            worker = await pool.acquire()
            acquired.set()
            pool.release(worker, ok=True)

        task = asyncio.create_task(second())
        await asyncio.sleep(0.05)
        assert not acquired.is_set()  # blocked: at capacity
        pool.release(first, ok=True)
        await asyncio.wait_for(acquired.wait(), timeout=1.0)
        assert acquired.is_set()
        assert pool.loaded == 1

    asyncio.run(scenario())


def test_idle_above_min_unload_after_ttl():
    async def scenario():
        pool, _ = _pool(3, 1, ttl=0)
        await pool.initialize()
        workers = await asyncio.gather(
            pool.acquire(), pool.acquire(), pool.acquire()
        )
        for worker in workers:
            pool.release(worker, ok=True)
        assert pool.loaded == 3
        unloaded = await pool.reap_once()
        assert unloaded == 2
        assert pool.loaded == 1  # minimum spare survives

    asyncio.run(scenario())


def test_request_after_reap_loads_new_worker():
    async def scenario():
        pool, loader = _pool(2, 1, ttl=0)
        await pool.initialize()
        a = await pool.acquire()
        b = await pool.acquire()
        pool.release(a, ok=True)
        pool.release(b, ok=True)
        await pool.reap_once()  # down to 1 idle
        assert pool.loaded == 1
        first = await pool.acquire()  # reuses the idle worker
        instances_before = pool.loader_calls
        second = await pool.acquire()  # no idle -> a fresh worker loads
        assert pool.loaded == 2
        assert pool.loader_calls == instances_before + 1
        pool.release(first, ok=True)
        pool.release(second, ok=True)

    asyncio.run(scenario())


def test_status_tracks_transitions():
    async def scenario():
        pool, _ = _pool(2, 1)
        assert pool.model_state == "loading"
        await pool.initialize()
        assert pool.status()["workers"]["loaded"] == 1
        assert pool.status()["workers"]["idle"] == 1
        worker = await pool.acquire()
        assert pool.counts()["busy"] == 1 and pool.counts()["idle"] == 0
        pool.release(worker, ok=True)
        assert pool.counts()["idle"] == 1

    asyncio.run(scenario())


def test_generation_failure_drops_worker_and_restores_baseline(make_client, wav_bytes):
    with make_client(model_kwargs={"fail_generate": True}) as client:
        resp = client.post(
            "/v1/syntheses",
            data={"text": "hi"},
            files={"reference_audio": ("ref.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 500, resp.text
        assert resp.json()["code"] == "synthesis_failed"
        pool = client.app.state.pool
        # The failed worker was dropped; the loader still works, so baseline
        # is restored and the pool stays ready.
        assert pool.model_state == "ready"
        assert pool.loaded == 1


def test_pool_initialization_failure_is_503(tmp_path):
    def bad_loader():
        raise RuntimeError("weights unavailable")

    cfg = make_config(tmp_path, max_workers=1, min_spare_workers=1)
    app = create_app(cfg=cfg, model_loader=bad_loader)
    with TestClient(app) as client:
        status = client.get("/status")
        assert status.status_code == 503
        assert status.json()["model_state"] == "failed"

        resp = client.post(
            "/v1/syntheses",
            data={"text": "hi"},
            files={"reference_audio": ("ref.wav", make_wav_bytes(), "audio/wav")},
        )
        assert resp.status_code == 503
        assert resp.json()["code"] == "unavailable"
