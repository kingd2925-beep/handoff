import json

import pytest
import torch

from conftest import REAL_SERVE_LOAD
from handoff import paths, serve, train


def _write(path, rows, extra_lines=()):
    paths.ensure(path).write_text("".join(json.dumps(r) + "\n" for r in rows) + "".join(extra_lines))


def test_load_labelled_last_label_wins_and_never_duplicates(handoff_home):
    # Arrange
    _write(paths.PROMPTS, [{"id": "a", "text": "rename a var"}, {"id": "b", "text": "design a ledger"}])
    _write(paths.LABELS, [{"id": "a", "label": "strong"}, {"id": "b", "label": "strong"},
                          {"id": "a", "label": "cheap"}])

    # Act
    rows = train.load_labelled(paths.PROMPTS, paths.LABELS)

    # Assert
    assert sorted(rows) == [("design a ledger", 1), ("rename a var", 0)]


def test_load_labelled_skips_unknown_ids_bad_labels_and_corrupt_lines(handoff_home):
    _write(paths.PROMPTS, [{"id": "a", "text": "rename a var"}, {"no_id": True}], ['{"id": "half-writ'])
    _write(paths.LABELS, [{"id": "zz", "label": "cheap"}, {"id": "a", "label": "maybe"},
                          {"id": "a", "label": "cheap"}], ["\n", "garbage\n", '{"id": "a", "lab'])

    assert train.load_labelled(paths.PROMPTS, paths.LABELS) == [("rename a var", 0)]


def test_load_labelled_ignores_an_invalid_relabel(handoff_home):
    _write(paths.PROMPTS, [{"id": "a", "text": "rename a var"}])
    _write(paths.LABELS, [{"id": "a", "label": "strong"}, {"id": "a", "label": "medium"}])

    assert train.load_labelled(paths.PROMPTS, paths.LABELS) == [("rename a var", 1)]


def test_run_refuses_too_few_labels(handoff_home):
    _write(paths.PROMPTS, [{"id": "a", "text": "rename a var"}])
    _write(paths.LABELS, [{"id": "a", "label": "cheap"}])

    with pytest.raises(SystemExit, match="at least 60"):
        train.run(say=lambda _m: None)


class _Agent:
    def predict(self, state, questions):
        return {"answers": {"hard": {"noul": 0.5}}}


def _embed(texts):
    # Feature 0 separates the classes; feature 1 is noise.
    return [[1.0 if t.startswith("hard") else -1.0, (len(t) % 5) / 5] for t in texts]


def test_run_end_to_end_writes_a_private_head_that_serve_can_load(monkeypatch, handoff_home):
    # Arrange: 40 easy + 30 hard prompts, tiny fake encoder, few epochs to stay fast
    prompts = [{"id": f"e{i}", "text": f"easy task number {i}"} for i in range(40)]
    prompts += [{"id": f"h{i}", "text": f"hard task number {i}"} for i in range(30)]
    _write(paths.PROMPTS, prompts)
    _write(paths.LABELS, [{"id": p["id"], "label": "strong" if p["id"][0] == "h" else "cheap"} for p in prompts])
    monkeypatch.setattr(train, "EPOCHS", 60)
    monkeypatch.setattr(train, "load_laya", lambda: (_Agent(), _embed))

    # Act
    report = train.run(say=lambda _m: None)

    # Assert: report card
    assert report["labelled"] == 70 and report["strong"] == 30
    assert sum(report["split"].values()) == 70 and report["split"]["test"] == 14
    assert report["test_trained_at_0.5"]["accuracy"] == 1.0
    assert json.loads(paths.REPORT.read_text()) == report
    assert oct(paths.HEAD.stat().st_mode & 0o777) == "0o600"
    assert oct(paths.REPORT.stat().st_mode & 0o777) == "0o600"

    # Assert: the saved head survives torch.load(weights_only=True) and routes sensibly
    monkeypatch.setattr(train, "load_laya", lambda: (_Agent(), _embed))
    agent, embed, head = REAL_SERVE_LOAD()
    assert set(head) == {"w", "b", "mean", "std", "gate"}
    monkeypatch.setattr(serve, "BRAIN", serve.Brain(loader=lambda: (agent, embed, head)))
    assert serve.route("hard task number 99")["route"] == "strong"
    assert serve.route("easy task number 99")["source"] == "trained"
