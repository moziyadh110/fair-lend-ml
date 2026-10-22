"""API-key check, tested directly so it runs in CI — tests/test_serving.py
covers the same paths over HTTP but skips without a trained model artifact."""
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serving.auth import require_api_key


@pytest.fixture(autouse=True)
def configured_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "correct-key")


def test_correct_key_is_accepted():
    assert require_api_key("correct-key") is None


@pytest.mark.parametrize("provided", [None, "", "wrong-key", "correct-ke", "correct-key-extra", "corrëct-key"])
def test_wrong_or_missing_key_is_rejected(provided):
    with pytest.raises(HTTPException) as exc:
        require_api_key(provided)
    assert exc.value.status_code == 401


def test_unconfigured_server_fails_closed(monkeypatch):
    monkeypatch.delenv("API_KEY")
    with pytest.raises(HTTPException) as exc:
        require_api_key("anything")
    assert exc.value.status_code == 503
