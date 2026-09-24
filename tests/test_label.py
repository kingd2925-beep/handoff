import json
import subprocess

import pytest

from handoff import label

ROWS = [{"id": "a1", "text": "rename a variable"},
        {"id": "b2", "text": "design a payments ledger"},
        {"id": "c3", "text": "fix a typo in the README"}]


def _reply(*labels):
    return json.dumps([{"i": n, "label": lab} for n, lab in enumerate(labels, 1)])


# ── parse_reply ───────────────────────────────────────────────────────
def test_parse_reply_maps_numbers_back_to_ids():
    out = label.parse_reply(_reply("cheap", "strong", "cheap"), ROWS)

    assert out == {"a1": "cheap", "b2": "strong", "c3": "cheap"}


def test_parse_reply_tolerates_prose_around_the_array_and_label_case():
    reply = 'Sure! Here you go:\n' + _reply("CHEAP", " Strong ", "cheap") + '\nHope that helps.'

    out = label.parse_reply(reply, ROWS)

    assert out == {"a1": "cheap", "b2": "strong", "c3": "cheap"}


def test_parse_reply_raises_when_items_are_missing():
    with pytest.raises(ValueError, match="labelled 2 of 3"):
        label.parse_reply(_reply("cheap", "strong"), ROWS)


def test_parse_reply_raises_when_there_is_no_array():
    with pytest.raises(ValueError, match="no JSON array"):
        label.parse_reply("I cannot label these.", ROWS)


def test_parse_reply_ignores_invalid_labels_and_out_of_range_numbers():
    reply = json.dumps([{"i": 1, "label": "cheap"}, {"i": 2, "label": "medium"}, {"i": 3, "label": "strong"},
                        {"i": 9, "label": "cheap"}, {"i": 0, "label": "strong"}])

    with pytest.raises(ValueError, match="labelled 2 of 3"):
        label.parse_reply(reply, ROWS)


def test_parse_reply_ignores_extra_invalid_items_when_all_rows_are_covered():
    reply = json.dumps([{"i": 1, "label": "cheap"}, {"i": 2, "label": "unsure"}, {"i": 2, "label": "strong"},
                        {"i": 3, "label": "cheap"}])

    assert label.parse_reply(reply, ROWS) == {"a1": "cheap", "b2": "strong", "c3": "cheap"}


@pytest.mark.parametrize("reply", ['["cheap", "strong", "cheap"]', '[{"i": null, "label": "cheap"}]', "[1, 2, 3]"])
def test_parse_reply_raises_value_error_for_malformed_items(reply):
    # A model reply with the wrong item shape must be rejected, not crash the labelling run.
    with pytest.raises(ValueError):
        label.parse_reply(reply, ROWS)


def test_build_prompt_contains_policy_instruction_and_numbered_items():
    prompt = label.build_prompt("  POLICY TEXT \n", ROWS[:2])

    assert prompt.startswith("POLICY TEXT\n\n" + label.INSTRUCTION)
    assert prompt.endswith("1. rename a variable\n2. design a payments ledger")


# ── label_with_model ──────────────────────────────────────────────────
def test_label_with_model_writes_labels_in_batches(tmp_path):
    # Arrange
    out = tmp_path / "labels.jsonl"
    prompts_seen = []

    def fake_ask(prompt):
        prompts_seen.append(prompt)
        return _reply("cheap", "strong") if len(prompts_seen) == 1 else _reply("strong")

    # Act
    n = label.label_with_model(ROWS, "policy", fake_ask, out, batch_size=2, report=lambda _m: None)

    # Assert
    assert n == 3
    assert len(prompts_seen) == 2
    assert label.read_jsonl(out) == [{"id": "a1", "label": "cheap"}, {"id": "b2", "label": "strong"},
                                     {"id": "c3", "label": "strong"}]


def test_label_with_model_skips_a_bad_batch_and_keeps_going(tmp_path):
    out = tmp_path / "labels.jsonl"
    messages = []
    replies = iter(["not json at all", _reply("cheap")])

    n = label.label_with_model(ROWS, "policy", lambda _p: next(replies), out, batch_size=2, report=messages.append)

    assert n == 1
    assert label.read_jsonl(out) == [{"id": "c3", "label": "cheap"}]
    assert messages[0].startswith("! batch 1 skipped")


def test_label_with_model_survives_a_failing_labeller(tmp_path):
    out = tmp_path / "labels.jsonl"

    def broken(_prompt):
        raise RuntimeError("claude -p failed")

    n = label.label_with_model(ROWS, "policy", broken, out, batch_size=10, report=lambda _m: None)

    assert n == 0
    assert not out.exists()


def test_label_with_model_survives_malformed_items(tmp_path):
    out = tmp_path / "labels.jsonl"

    n = label.label_with_model(ROWS, "policy", lambda _p: '["cheap"]', out, report=lambda _m: None)

    assert n == 0


# ── claude_labeller ───────────────────────────────────────────────────
def test_claude_labeller_returns_result_field(monkeypatch):
    # Arrange
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": "[{}]"}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Act
    reply = label.claude_labeller("haiku")("label these")

    # Assert
    assert reply == "[{}]"
    cmd, kwargs = calls[0]
    assert cmd == ["claude", "-p", "--output-format", "json", "--model", "haiku"]
    assert kwargs["input"] == "label these"
    assert kwargs["timeout"] == label.CLAUDE_TIMEOUT


def test_claude_labeller_raises_runtime_error_on_non_zero_exit(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **_k: subprocess.CompletedProcess(cmd, 1, stdout="", stderr=" not logged in \n"))

    with pytest.raises(RuntimeError, match="^not logged in$"):
        label.claude_labeller("haiku")("x")


def test_claude_labeller_uses_generic_message_when_stderr_is_empty(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **_k: subprocess.CompletedProcess(cmd, 2, stdout="", stderr=""))

    with pytest.raises(RuntimeError, match="claude -p failed"):
        label.claude_labeller("haiku")("x")


def test_claude_labeller_raises_on_bad_json(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **_k: subprocess.CompletedProcess(cmd, 0, stdout="oops, not json", stderr=""))

    with pytest.raises(json.JSONDecodeError):
        label.claude_labeller("haiku")("x")


# ── label_by_hand / import_csv / files ────────────────────────────────
def test_label_by_hand_follows_keys_and_stops_on_quit(tmp_path):
    out = tmp_path / "labels.jsonl"
    answers = iter(["c", " S ", "q"])
    shown = []

    def scripted_read(prompt):
        shown.append(prompt)
        return next(answers)

    rows = ROWS + [{"id": "d4", "text": "never shown"}]
    n = label.label_by_hand(rows, out, read=scripted_read)

    assert n == 2
    assert label.read_jsonl(out) == [{"id": "a1", "label": "cheap"}, {"id": "b2", "label": "strong"}]
    assert len(shown) == 3 and "fix a typo" in shown[2]


def test_label_by_hand_skip_and_unknown_keys_write_nothing(tmp_path):
    out = tmp_path / "labels.jsonl"
    answers = iter(["k", "x", ""])

    n = label.label_by_hand(ROWS, out, read=lambda _p: next(answers))

    assert n == 0
    assert label.read_jsonl(out) == []


def test_import_csv_keeps_only_known_ids_and_valid_labels(tmp_path):
    # Arrange
    src = tmp_path / "labels.csv"
    src.write_text("a1,cheap\nb2, STRONG \nzz9,cheap\nc3,maybe\nonlyonecolumn\n\n")
    out = tmp_path / "labels.jsonl"

    # Act
    n = label.import_csv(src, {"a1", "b2", "c3"}, out)

    # Assert
    assert n == 2
    assert label.read_jsonl(out) == [{"id": "a1", "label": "cheap"}, {"id": "b2", "label": "strong"}]


def test_read_jsonl_returns_empty_for_missing_file_and_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    assert label.read_jsonl(path) == []

    path.write_text('{"id": "a"}\n\n{"id": "b"}\n')

    assert label.read_jsonl(path) == [{"id": "a"}, {"id": "b"}]


def test_read_jsonl_skips_corrupt_and_non_object_lines(tmp_path):
    # e.g. a half-written last line after a crash, or a stray array line
    path = tmp_path / "labels.jsonl"
    path.write_text('{"id": "a", "label": "cheap"}\nnot json\n[1, 2]\n"a string"\n{"id": "b", "label": "strong"}\n'
                    '{"id": "c", "lab')

    assert label.read_jsonl(path) == [{"id": "a", "label": "cheap"}, {"id": "b", "label": "strong"}]


def test_append_labels_makes_the_file_private(tmp_path):
    import stat
    out = tmp_path / "labels.jsonl"

    label.append_labels(out, {"a1": "cheap"})

    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_resume_after_a_corrupt_last_line_still_skips_labelled_ids(tmp_path):
    from handoff import cli, paths
    paths.ensure(paths.PROMPTS).write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    paths.ensure(paths.LABELS).write_text('{"id": "a1", "label": "cheap"}\n{"id": "b2", "lab')

    _, todo = cli._unlabelled(None)

    assert [r["id"] for r in todo] == ["b2", "c3"]
