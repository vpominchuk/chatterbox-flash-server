"""FastAPI application factory, routes, and server entrypoint for Flash TTS."""
from __future__ import annotations

import functools
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .config import Config
from .errors import (
    FlashError,
    InvalidReferenceAudioError,
    InvalidTextError,
    SynthesisNotFoundError,
    UploadTooLargeError,
)
from .model import ChatterboxModel, GenerationSettings
from .pool import WorkerPool
from .schemas import SynthesisResponse
from .service import SynthesisService
from .storage import Store

MAX_TEXT_LENGTH = 5000


def make_real_loader(cfg: Config):
    # functools.partial is picklable (a closure over cfg is not), which the
    # spawn-based worker processes require.
    return functools.partial(
        ChatterboxModel,
        cfg.model_repo,
        cfg.device,
        cfg.dtype,
        cfg.block_size,
        backend=cfg.backend,
    )


def _create_app(
    cfg: Config,
    pool: WorkerPool,
    store: Store,
    service: SynthesisService,
    reap_interval: float,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await pool.initialize()
        reaper = asyncio.create_task(pool.reap_loop(interval=reap_interval))
        try:
            yield
        finally:
            reaper.cancel()
            await pool.shutdown()

    app = FastAPI(
        title="Chatterbox Flash TTS",
        version="0.1.0",
        description="Synchronous zero-shot English TTS API for Chatterbox Flash.",
        lifespan=lifespan,
    )
    app.state.cfg = cfg
    app.state.pool = pool
    app.state.service = service

    # ---- structured error handling --------------------------------------
    @app.exception_handler(FlashError)
    async def _flash_error_handler(request: Request, exc: FlashError):
        return JSONResponse(status_code=exc.status, content=exc.body())

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        errors = [
            {k: v for k, v in err.items() if k != "ctx"} for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "code": "invalid_request",
                "message": "request validation failed",
                "details": {"errors": errors},
            },
        )

    # ---- routes ----------------------------------------------------------
    @app.post(
        "/v1/syntheses",
        status_code=status.HTTP_201_CREATED,
        response_model=SynthesisResponse,
    )
    async def create_synthesis(
        text: str = Form(...),
        reference_audio: UploadFile = File(...),
        exaggeration: float = Form(0.5, ge=0, le=1),
        normalize_text: bool = Form(True),
        num_steps: int = Form(10, ge=1, le=50),
        temperature: float = Form(0.2, ge=0, le=2),
        time_shift_tau: float = Form(0.5, ge=0, le=1),
        omnivoice_schedule_t_shift: float = Form(0.5, ge=0, le=1),
        cfg_scale: float = Form(1.0, ge=0, le=3),
        position_temperature: float = Form(5.0, ge=0, le=10),
        max_speech_tokens: int | None = Form(None, ge=1, le=4096),
        n_cfm_timesteps: int = Form(2, ge=1, le=10),
    ) -> SynthesisResponse:
        if not text or not text.strip():
            raise InvalidTextError("text must be a non-empty string")
        if len(text) > MAX_TEXT_LENGTH:
            raise InvalidTextError(
                f"text must be at most {MAX_TEXT_LENGTH} characters"
            )

        audio_bytes = await reference_audio.read()
        if not audio_bytes:
            raise InvalidReferenceAudioError("reference_audio is empty")
        if len(audio_bytes) > cfg.max_upload_bytes:
            raise UploadTooLargeError(
                f"reference_audio exceeds the {cfg.max_upload_bytes} byte limit"
            )

        settings = GenerationSettings(
            exaggeration=exaggeration,
            normalize_text=normalize_text,
            num_steps=num_steps,
            temperature=temperature,
            time_shift_tau=time_shift_tau,
            omnivoice_schedule_t_shift=omnivoice_schedule_t_shift,
            cfg_scale=cfg_scale,
            position_temperature=position_temperature,
            max_speech_tokens=max_speech_tokens,
            n_cfm_timesteps=n_cfm_timesteps,
        )
        record = await service.synthesize(text, audio_bytes, settings)
        return SynthesisResponse.from_record(record)

    @app.get("/v1/syntheses/{synthesis_id}", response_model=SynthesisResponse)
    async def get_synthesis(synthesis_id: str) -> SynthesisResponse:
        record = store.get(synthesis_id)
        if record is None:
            raise SynthesisNotFoundError("synthesis not found")
        return SynthesisResponse.from_record(record)

    @app.get("/v1/audio/{synthesis_id}.wav")
    async def get_audio(synthesis_id: str) -> FileResponse:
        record = store.get(synthesis_id)
        if record is None:
            raise SynthesisNotFoundError("synthesis not found")
        return FileResponse(record.audio_path, media_type="audio/wav")

    @app.get("/status")
    async def get_status() -> JSONResponse:
        body = {
            "model_state": pool.model_state,
            "workers": pool.counts(),
            "runtime": {
                "backend": cfg.backend,
                "device": cfg.device,
                "dtype": cfg.dtype,
            },
        }
        if pool.model_state == "failed":
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body)
        return JSONResponse(status_code=status.HTTP_200_OK, content=body)

    return app


def create_app(
    cfg: Config | None = None,
    model_loader=None,
    pool: WorkerPool | None = None,
    store: Store | None = None,
    reap_interval: float = 1.0,
) -> FastAPI:
    """Build the FastAPI app.

    ``model_loader`` is a zero-arg callable returning a :class:`TTSModel`;
    inject a fake loader for tests. ``pool`` may be injected to drive
    lifecycle tests directly.
    """
    cfg = cfg or Config.from_env()
    if pool is None:
        loader = model_loader or make_real_loader(cfg)
        pool = WorkerPool(
            loader,
            max_workers=cfg.max_workers,
            min_spare_workers=cfg.min_spare_workers,
            keep_alive_ttl_seconds=cfg.keep_alive_ttl_seconds,
        )
    if store is None:
        store = Store(cfg.output_dir)
    service = SynthesisService(pool, store, cfg)
    return _create_app(cfg, pool, store, service, reap_interval)


def run() -> None:  # pragma: no cover - console entrypoint
    import uvicorn

    from . import env

    cfg = Config.from_env()
    app = create_app(cfg)
    uvicorn.run(
        app,
        host=env.get_env_str("HOST", "127.0.0.1"),
        port=env.get_env_int("PORT", 8000),
    )


if __name__ == "__main__":  # pragma: no cover
    run()
