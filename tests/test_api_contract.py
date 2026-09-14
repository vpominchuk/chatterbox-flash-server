from conftest import make_wav_bytes


def test_valid_request_returns_completed_metadata_and_audio(make_client, wav_bytes):
    with make_client() as client:
        resp = client.post(
            "/v1/syntheses",
            data={"text": "Hello, world.", "num_steps": "20", "temperature": "1.5"},
            files={"reference_audio": ("ref.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["status"] == "completed"
        assert body["text"] == "Hello, world."
        assert body["audio_url"].startswith("/v1/audio/")
        assert body["audio_url"].endswith(".wav")
        assert body["sample_rate_hz"] == 24000
        assert body["duration_seconds"] > 0
        # Caller overrides are reflected; un-set fields fall back to defaults.
        assert body["settings"]["num_steps"] == 20
        assert body["settings"]["temperature"] == 1.5
        assert body["settings"]["exaggeration"] == 0.5
        synthesis_id = body["id"]

        metadata = client.get(f"/v1/syntheses/{synthesis_id}")
        assert metadata.status_code == 200
        assert metadata.json()["id"] == synthesis_id
        assert metadata.json()["settings"]["num_steps"] == 20

        audio = client.get(body["audio_url"])
        assert audio.status_code == 200
        assert audio.headers["content-type"].startswith("audio/wav")
        assert len(audio.content) > 0


def test_defaults_are_effective_defaults(make_client, wav_bytes):
    with make_client() as client:
        body = client.post(
            "/v1/syntheses",
            data={"text": "Hi"},
            files={"reference_audio": ("ref.wav", wav_bytes, "audio/wav")},
        ).json()
        expected = {
            "exaggeration": 0.5,
            "normalize_text": True,
            "num_steps": 10,
            "temperature": 0.2,
            "time_shift_tau": 0.5,
            "omnivoice_schedule_t_shift": 0.5,
            "cfg_scale": 1.0,
            "position_temperature": 5.0,
            "n_cfm_timesteps": 2,
        }
        for key, value in expected.items():
            assert body["settings"][key] == value, key
        assert "max_speech_tokens" not in body["settings"]


def test_unknown_ids_return_404(make_client):
    with make_client() as client:
        assert client.get("/v1/syntheses/does-not-exist").status_code == 404
        assert client.get("/v1/audio/does-not-exist.wav").status_code == 404
        body = client.get("/v1/syntheses/does-not-exist").json()
        assert body["code"] == "not_found"


def test_status_reports_ready_pool(make_client):
    with make_client() as client:
        resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_state"] == "ready"
        assert body["workers"]["configured"] == 2
        assert body["workers"]["loaded"] == 1
        assert body["workers"]["min_spare"] == 1
        assert body["workers"]["idle"] == 1
        assert body["runtime"] == {"backend": "auto", "device": "cpu", "dtype": "bfloat16"}
