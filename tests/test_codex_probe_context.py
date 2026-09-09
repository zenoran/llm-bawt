import pytest

from codex_bridge.probe_context import validate_probe_context
from codex_bridge.transport import validate_auth_json


@pytest.fixture
def context(tmp_path, monkeypatch):
    home = tmp_path / 'bridge-home'
    home.mkdir()
    (home / 'auth.json').write_text('not read by preflight')
    binary = tmp_path / 'codex'
    binary.write_text('fixture')
    monkeypatch.setenv('LLM_BAWT_API_URL', 'http://test-app')
    monkeypatch.setenv('CODEX_HOME', str(home))
    monkeypatch.setenv('CODEX_AUTH_PATH', str(home / 'auth.json'))
    monkeypatch.setenv('CODEX_BIN', str(binary))
    return home, binary


def test_accepts_matching_bridge_context_without_reading_tokens(context):
    home, binary = context
    result = validate_probe_context(base_home=home, binary=str(binary))
    assert result['auth_source'] == 'configured-bridge-file'
    assert 'not read' not in str(result)


def test_rejects_unconfigured_host(context, monkeypatch):
    home, binary = context
    monkeypatch.delenv('LLM_BAWT_API_URL')
    with pytest.raises(ValueError, match='inside the configured Codex bridge'):
        validate_probe_context(base_home=home, binary=str(binary))


def test_rejects_other_home(context, tmp_path):
    _, binary = context
    with pytest.raises(ValueError, match='base home differs'):
        validate_probe_context(base_home=tmp_path / 'host-home', binary=str(binary))


def test_rejects_other_auth_path(context, tmp_path, monkeypatch):
    home, binary = context
    monkeypatch.setenv('CODEX_AUTH_PATH', str(tmp_path / 'other.json'))
    with pytest.raises(ValueError, match='auth path differs'):
        validate_probe_context(base_home=home, binary=str(binary))


def test_rejects_other_binary(context, tmp_path):
    home, _ = context
    with pytest.raises(ValueError, match='binary differs'):
        validate_probe_context(base_home=home, binary=str(tmp_path / 'other-codex'))


def test_missing_auth_guidance_does_not_send_user_to_host_login(tmp_path):
    with pytest.raises(RuntimeError) as error:
        validate_auth_json(tmp_path / 'absent')
    message = str(error.value)
    assert 'app-owned' in message
    assert "Run 'codex login'" not in message
    assert 'then restart' not in message
