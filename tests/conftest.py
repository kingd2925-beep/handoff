"""Shared fixtures. Every test runs against a throwaway HANDOFF_HOME and a fake user home, and any
attempt to load the real Laya model fails loudly."""
from pathlib import Path

import pytest

from handoff import paths, serve, train

FIXTURES = Path(__file__).parent / "fixtures"
REAL_SERVE_LOAD = serve._load     # kept so tests can exercise it with a fake load_laya


def _refuse_real_model(*_args, **_kwargs):
    raise AssertionError("tests must never load the real Laya model")


def point_home_at(monkeypatch, home: Path) -> None:
    """Re-point every handoff path constant at `home`."""
    layout = {
        "HOME": home,
        "PROMPTS": home / "data" / "prompts.jsonl",
        "LABELS": home / "data" / "labels.jsonl",
        "HEAD": home / "model" / "router.pt",
        "REPORT": home / "model" / "report.json",
        "LOG": home / "logs" / "server.log",
        "TOKEN": home / "token",
        "PIDFILE": home / "service.pid",
        "MARKER": home / ".handoff-home",
    }
    for name, value in layout.items():
        monkeypatch.setattr(paths, name, value)


@pytest.fixture(autouse=True)
def handoff_home(tmp_path, monkeypatch):
    """Point every handoff path and Path.home() into tmp_path; block the real model loaders."""
    home = tmp_path / "handoff_home"
    user_home = tmp_path / "user_home"
    user_home.mkdir()
    monkeypatch.setenv("HANDOFF_HOME", str(home))
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: user_home))
    point_home_at(monkeypatch, home)
    monkeypatch.setattr(train, "load_laya", _refuse_real_model)
    monkeypatch.setattr(serve, "_load", _refuse_real_model)
    monkeypatch.setattr(serve.Handler, "secret", None)
    return home


@pytest.fixture
def user_home():
    return Path.home()


@pytest.fixture
def fixtures_dir():
    return FIXTURES
