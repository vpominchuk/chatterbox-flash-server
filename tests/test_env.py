import pytest

from flash_server import env
from flash_server.config import Config
from flash_server.errors import ConfigError


@pytest.fixture(autouse=True)
def _clear_env_cache():
    env.clear_cache()
    yield
    env.clear_cache()


def _write(path, text):
    path.write_text(text)


def test_precedence_system_over_local_over_env(tmp_path, monkeypatch):
    _write(tmp_path / ".env.local", "FLASH_A=local\nFLASH_C=local_c\n")
    _write(tmp_path / ".env", "FLASH_A=env\nFLASH_B=env_b\nFLASH_C=env_c\n")
    monkeypatch.setenv("FLASH_A", "sys_a")
    monkeypatch.setenv("FLASH_B", "sys_b")
    monkeypatch.delenv("FLASH_C", raising=False)
    monkeypatch.delenv("FLASH_D", raising=False)
    assert env.get_env("FLASH_A", base_dir=tmp_path) == "sys_a"      # system beats files
    assert env.get_env("FLASH_B", base_dir=tmp_path) == "sys_b"      # system beats .env
    assert env.get_env("FLASH_C", base_dir=tmp_path) == "local_c"    # not in system -> .env.local
    assert env.get_env("FLASH_D", "dflt", base_dir=tmp_path) == "dflt"


def test_key_only_in_system_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASH_ONLY_SYS", "from_system")
    assert env.get_env("FLASH_ONLY_SYS", base_dir=tmp_path) == "from_system"


def test_key_nowhere_returns_default(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASH_NOPE", raising=False)
    assert env.get_env("FLASH_NOPE", "default", base_dir=tmp_path) == "default"


def test_present_empty_in_local_beats_env_when_not_in_system(tmp_path, monkeypatch):
    _write(tmp_path / ".env.local", "FLASH_E=\n")
    _write(tmp_path / ".env", "FLASH_E=from_env\n")
    monkeypatch.delenv("FLASH_E", raising=False)
    assert env.get_env("FLASH_E", base_dir=tmp_path) == ""


def test_system_beats_present_file_value(tmp_path, monkeypatch):
    _write(tmp_path / ".env.local", "FLASH_G=\n")
    monkeypatch.setenv("FLASH_G", "sys_g")
    assert env.get_env("FLASH_G", base_dir=tmp_path) == "sys_g"


def test_typed_helpers(tmp_path, monkeypatch):
    _write(tmp_path / ".env", "FLASH_N=42\nFLASH_F=1.5\n")
    for key in ("FLASH_N", "FLASH_F", "FLASH_NOPE_I"):
        monkeypatch.delenv(key, raising=False)
    assert env.get_env_int("FLASH_N", base_dir=tmp_path) == 42
    assert env.get_env_float("FLASH_F", base_dir=tmp_path) == 1.5
    assert env.get_env_int("FLASH_NOPE_I", 7, base_dir=tmp_path) == 7


def test_bad_int_raises_value_error(tmp_path, monkeypatch):
    _write(tmp_path / ".env", "FLASH_N=notanint\n")
    monkeypatch.delenv("FLASH_N", raising=False)
    with pytest.raises(ValueError):
        env.get_env_int("FLASH_N", base_dir=tmp_path)


def test_config_from_env_resolves_files(tmp_path, monkeypatch):
    _write(tmp_path / ".env.local", "DEVICE=cpu-local\nMAX_WORKERS=4\n")
    _write(tmp_path / ".env", "MODEL_REPO=myorg/repo\nDTYPE=bf16\n")
    for key in ("DEVICE", "MAX_WORKERS", "MODEL_REPO", "DTYPE", "MIN_SPARE_WORKERS"):
        monkeypatch.delenv(key, raising=False)
    cfg = Config.from_env(base_dir=tmp_path)
    assert cfg.device == "cpu-local"
    assert cfg.max_workers == 4
    assert cfg.model_repo == "myorg/repo"
    assert cfg.dtype == "bf16"
    assert cfg.min_spare_workers == 1  # default: absent from files and system


def test_config_from_env_system_overrides_file(tmp_path, monkeypatch):
    _write(tmp_path / ".env", "MAX_WORKERS=2\n")
    monkeypatch.setenv("MAX_WORKERS", "6")
    monkeypatch.delenv("MIN_SPARE_WORKERS", raising=False)
    cfg = Config.from_env(base_dir=tmp_path)
    assert cfg.max_workers == 6  # system wins over the .env file


def test_config_from_env_bad_int_is_config_error(tmp_path, monkeypatch):
    _write(tmp_path / ".env", "MAX_WORKERS=abc\n")
    monkeypatch.delenv("MAX_WORKERS", raising=False)
    with pytest.raises(ConfigError):
        Config.from_env(base_dir=tmp_path)
