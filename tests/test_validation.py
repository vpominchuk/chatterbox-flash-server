from conftest import make_wav_bytes

AUDIO = ("ref.wav", make_wav_bytes(), "audio/wav")


def test_missing_text_is_422(make_client):
    with make_client() as client:
        resp = client.post("/v1/syntheses", files={"reference_audio": AUDIO})
        assert resp.status_code == 422
        assert resp.json()["code"] == "invalid_request"


def test_blank_text_is_422(make_client):
    with make_client() as client:
        resp = client.post(
            "/v1/syntheses", data={"text": "   "}, files={"reference_audio": AUDIO}
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "invalid_text"


def test_oversized_text_is_422(make_client):
    with make_client() as client:
        resp = client.post(
            "/v1/syntheses",
            data={"text": "a" * 5001},
            files={"reference_audio": AUDIO},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "invalid_text"


def test_max_length_text_is_accepted(make_client, wav_bytes):
    with make_client() as client:
        resp = client.post(
            "/v1/syntheses",
            data={"text": "a" * 5000},
            files={"reference_audio": ("ref.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 201, resp.text


def test_missing_audio_is_422(make_client):
    with make_client() as client:
        resp = client.post("/v1/syntheses", data={"text": "hi"})
        assert resp.status_code == 422
        assert resp.json()["code"] == "invalid_request"


def test_empty_audio_is_422(make_client):
    with make_client() as client:
        resp = client.post(
            "/v1/syntheses",
            data={"text": "hi"},
            files={"reference_audio": ("ref.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "invalid_audio"


def test_undecodable_audio_is_422(make_client):
    with make_client() as client:
        resp = client.post(
            "/v1/syntheses",
            data={"text": "hi"},
            files={"reference_audio": ("ref.wav", b"this is not audio", "audio/wav")},
        )
        assert resp.status_code == 422
        assert resp.json()["code"] == "invalid_audio"


def test_oversized_upload_is_413(make_client):
    with make_client(max_upload_bytes=1024) as client:
        resp = client.post(
            "/v1/syntheses",
            data={"text": "hi"},
            files={"reference_audio": ("ref.wav", b"\x00" * 2048, "audio/wav")},
        )
        assert resp.status_code == 413
        assert resp.json()["code"] == "upload_too_large"


def test_invalid_setting_range_is_422(make_client):
    with make_client() as client:
        for field, value in [
            ("num_steps", "0"),
            ("num_steps", "51"),
            ("temperature", "3.0"),
            ("exaggeration", "1.5"),
            ("cfg_scale", "-0.1"),
        ]:
            resp = client.post(
                "/v1/syntheses",
                data={"text": "hi", field: value},
                files={"reference_audio": AUDIO},
            )
            assert resp.status_code == 422, (field, value, resp.text)
            assert resp.json()["code"] == "invalid_request"
