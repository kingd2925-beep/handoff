import stat
from pathlib import Path

import pytest

from conftest import point_home_at
from handoff import paths


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_ensure_creates_private_home_with_marker(handoff_home):
    target = paths.ensure(paths.PROMPTS)

    assert target == paths.PROMPTS
    assert paths.MARKER.exists()
    assert _mode(handoff_home) == 0o700 and _mode(paths.PROMPTS.parent) == 0o700


@pytest.mark.parametrize("which", ["user home", "root", "tmp"])
def test_check_home_refuses_shared_folders(monkeypatch, which):
    home = {"user home": Path.home(), "root": Path("/"), "tmp": Path("/tmp")}[which]
    point_home_at(monkeypatch, home)

    with pytest.raises(SystemExit, match="not a dedicated folder"):
        paths.ensure(paths.PROMPTS)


def test_check_home_refuses_user_home_given_through_a_symlink(monkeypatch, tmp_path):
    link = tmp_path / "link-to-home"
    link.symlink_to(Path.home())
    point_home_at(monkeypatch, link)

    with pytest.raises(SystemExit, match="not a dedicated folder"):
        paths.ensure(paths.PROMPTS)


def test_check_home_refuses_a_non_empty_folder_without_marker(monkeypatch, tmp_path):
    folder = tmp_path / "projects"
    folder.mkdir()
    (folder / "thesis.docx").write_text("important")
    before = _mode(folder)
    point_home_at(monkeypatch, folder)

    with pytest.raises(SystemExit, match="already has other files"):
        paths.ensure(paths.PROMPTS)
    assert _mode(folder) == before                 # never chmodded: the folder is untouched
    assert sorted(p.name for p in folder.iterdir()) == ["thesis.docx"]


def test_check_home_accepts_an_empty_existing_folder(monkeypatch, tmp_path):
    folder = tmp_path / "fresh"
    folder.mkdir()
    point_home_at(monkeypatch, folder)

    paths.ensure(paths.PROMPTS)

    assert paths.MARKER.exists()


def test_check_home_accepts_its_own_folder_on_later_runs(handoff_home):
    paths.ensure(paths.PROMPTS).write_text("x\n")

    paths.ensure(paths.LABELS)   # non-empty now, but it carries the marker

    assert paths.LABELS.parent.exists()


def test_private_sets_0600_and_ignores_missing_files(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    f.chmod(0o644)

    paths.private(f)
    paths.private(tmp_path / "missing")

    assert _mode(f) == 0o600


def test_token_is_created_once_with_0600_and_reused(handoff_home):
    first = paths.token()
    second = paths.token()

    assert first == second
    assert len(first) == 48 and all(c in "0123456789abcdef" for c in first)
    assert paths.TOKEN.parent == handoff_home
    assert _mode(paths.TOKEN) == 0o600
    assert _mode(handoff_home) == 0o700


def test_token_strips_whitespace_from_an_edited_file(handoff_home):
    paths.ensure(paths.TOKEN).write_text("  abc123\n")

    assert paths.token() == "abc123"
