import json
import shutil
import stat

import pytest

from handoff import cli, label, paths


def _write_prompts(rows):
    paths.ensure(paths.PROMPTS).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


PROMPT_ROWS = [{"id": "a1", "text": "rename a variable"},
               {"id": "b2", "text": "design a payments ledger"},
               {"id": "c3", "text": "fix a typo in the README"}]


# ── parser ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("argv, cmd", [
    (["extract"], "extract"),
    (["extract", "--source", "codex"], "extract"),
    (["extract", "--file", "p.txt"], "extract"),
    (["label", "--backend", "manual", "--limit", "5"], "label"),
    (["label", "--backend", "csv", "--csv", "x.csv"], "label"),
    (["label", "--model", "sonnet", "--policy", "p.md"], "label"),
    (["train"], "train"),
    (["report"], "report"),
    (["route", "summarise", "these", "notes"], "route"),
    (["ask", '{"state": "s", "questions": {}}'], "ask"),
    (["serve"], "serve"),
    (["status"], "status"),
    (["stop"], "stop"),
])
def test_parser_accepts_every_subcommand(argv, cmd):
    args = cli.build_parser().parse_args(argv)

    assert args.cmd == cmd
    assert cmd in cli.COMMANDS


def test_parser_defaults():
    extract_args = cli.build_parser().parse_args(["extract"])
    label_args = cli.build_parser().parse_args(["label"])

    assert extract_args.source == "claude" and extract_args.file is None
    assert label_args.backend == "claude" and label_args.model == "haiku" and label_args.limit is None


@pytest.mark.parametrize("argv", [[], ["route"], ["extract", "--source", "gemini"], ["label", "--backend", "gpt"]])
def test_parser_rejects_bad_input(argv):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


def test_label_csv_backend_without_csv_file_exits():
    with pytest.raises(SystemExit, match="--csv"):
        cli.main(["label", "--backend", "csv"])


def test_default_policy_file_exists():
    assert cli.DEFAULT_POLICY.is_file()


# ── extract ───────────────────────────────────────────────────────────
def test_extract_from_file_writes_deduped_private_prompts(fixtures_dir, handoff_home):
    # Act
    cli.main(["extract", "--file", str(fixtures_dir / "prompts.txt")])

    # Assert
    rows = label.read_jsonl(paths.PROMPTS)
    assert paths.PROMPTS.parent.parent == handoff_home
    assert sorted(r["text"] for r in rows) == [
        "Draft a polite follow-up email to a client who has not paid",
        "Explain the difference between TCP and UDP simply",
        "Plan a three-week migration from Postgres 12 to 16",
    ]
    assert len({r["id"] for r in rows}) == 3
    assert _mode(paths.PROMPTS) == 0o600
    assert _mode(paths.PROMPTS.parent) == 0o700
    assert _mode(handoff_home) == 0o700


def test_extract_from_claude_history_reads_the_fake_home(fixtures_dir, user_home):
    project = user_home / ".claude" / "projects" / "demo"
    project.mkdir(parents=True)
    shutil.copy(fixtures_dir / "claude_session.jsonl", project / "one.jsonl")
    shutil.copy(fixtures_dir / "claude_session.jsonl", project / "two.jsonl")   # duplicates collapse

    cli.main(["extract"])

    texts = sorted(r["text"] for r in label.read_jsonl(paths.PROMPTS))
    assert texts == ["Refactor the billing module to use the new tax table",
                     "Summarise these meeting notes for me please"]


def test_extract_overwrites_previous_prompts(fixtures_dir):
    _write_prompts([{"id": "old", "text": "stale prompt from last run"}])

    cli.main(["extract", "--file", str(fixtures_dir / "prompts.txt")])

    assert "old" not in {r["id"] for r in label.read_jsonl(paths.PROMPTS)}


# ── label / resume ────────────────────────────────────────────────────
def test_unlabelled_without_prompts_exits():
    with pytest.raises(SystemExit, match="extract"):
        cli._unlabelled(None)


def test_unlabelled_skips_already_labelled_ids_and_applies_limit():
    _write_prompts(PROMPT_ROWS)
    label.append_labels(paths.ensure(paths.LABELS), {"b2": "strong"})

    rows, todo = cli._unlabelled(None)
    _, limited = cli._unlabelled(1)

    assert len(rows) == 3
    assert [r["id"] for r in todo] == ["a1", "c3"]
    assert [r["id"] for r in limited] == ["a1"]


def test_label_with_claude_backend_resumes_and_uses_the_given_policy(monkeypatch, tmp_path):
    # Arrange
    _write_prompts(PROMPT_ROWS)
    label.append_labels(paths.ensure(paths.LABELS), {"a1": "cheap"})
    policy = tmp_path / "policy.md"
    policy.write_text("MY POLICY")
    prompts_sent = []

    def fake_labeller(model):
        assert model == "sonnet"

        def ask(prompt):
            prompts_sent.append(prompt)
            return json.dumps([{"i": 1, "label": "strong"}, {"i": 2, "label": "cheap"}])
        return ask

    monkeypatch.setattr(label, "claude_labeller", fake_labeller)

    # Act
    cli.main(["label", "--model", "sonnet", "--policy", str(policy)])

    # Assert
    assert len(prompts_sent) == 1
    assert prompts_sent[0].startswith("MY POLICY") and "rename a variable" not in prompts_sent[0]
    assert label.read_jsonl(paths.LABELS) == [{"id": "a1", "label": "cheap"}, {"id": "b2", "label": "strong"},
                                              {"id": "c3", "label": "cheap"}]


def test_label_manual_backend_only_asks_about_unlabelled(monkeypatch):
    _write_prompts(PROMPT_ROWS)
    label.append_labels(paths.ensure(paths.LABELS), {"a1": "cheap", "b2": "strong"})
    shown = []
    by_hand = label.label_by_hand   # its `read=input` default is bound at import, so inject the fake reader
    monkeypatch.setattr(label, "label_by_hand",
                        lambda rows, out: by_hand(rows, out, read=lambda p: shown.append(p) or "s"))

    cli.main(["label", "--backend", "manual"])

    assert len(shown) == 1 and "fix a typo" in shown[0]
    assert label.read_jsonl(paths.LABELS)[-1] == {"id": "c3", "label": "strong"}


def test_label_csv_backend_imports_new_ids_without_duplicating_existing_ones(tmp_path):
    # Arrange
    _write_prompts(PROMPT_ROWS)
    label.append_labels(paths.ensure(paths.LABELS), {"a1": "cheap"})
    src = tmp_path / "mine.csv"
    src.write_text("a1,strong\nb2,strong\nzz,cheap\n")

    # Act
    cli.main(["label", "--backend", "csv", "--csv", str(src)])

    # Assert: a1 was already labelled, so it is not written a second time
    labels = label.read_jsonl(paths.LABELS)
    assert labels == [{"id": "a1", "label": "cheap"}, {"id": "b2", "label": "strong"}]


# ── report ────────────────────────────────────────────────────────────
def test_report_without_report_file_exits():
    with pytest.raises(SystemExit, match="train"):
        cli.main(["report"])


def test_report_prints_the_report_card(fixtures_dir, capsys):
    paths.ensure(paths.REPORT).write_text((fixtures_dir / "report.json").read_text())

    cli.main(["report"])

    err = capsys.readouterr().err
    assert "handoff report card — 2026-09-25 · 120 labelled prompts · test set of 24" in err
    assert "zero-shot Laya 62%" in err and "trained 88%" in err
    assert "p_strong < 0.12" in err
    assert "42% of tasks go to the cheap model · 0.0% of hard tasks leak" in err


def test_status_when_service_is_down(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_health", lambda: None)

    cli.main(["status"])

    assert json.loads(capsys.readouterr().out) == {"running": False}


# ── service client: proof, token, pidfile ─────────────────────────────
def test_health_returns_info_when_proof_matches(monkeypatch):
    from handoff import serve
    info = {"loaded": False, "proof": serve.proof(paths.token())}
    monkeypatch.setattr(cli, "_call", lambda path, body=None, timeout=0: info)

    assert cli._health() == info


def test_health_refuses_a_server_with_the_wrong_proof(monkeypatch):
    monkeypatch.setattr(cli, "_call", lambda path, body=None, timeout=0: {"loaded": True, "proof": "0" * 16})

    with pytest.raises(SystemExit, match="something else is answering"):
        cli._health()


@pytest.mark.parametrize("reply", [{"error": "missing token"}, ["not", "a", "dict"], "plain string"])
def test_health_refuses_an_impostor_reply_without_proof(monkeypatch, reply):
    monkeypatch.setattr(cli, "_call", lambda path, body=None, timeout=0: reply)

    with pytest.raises(SystemExit, match="something else is answering"):
        cli._health()


def test_health_returns_none_when_nothing_is_listening(monkeypatch):
    import urllib.error

    def refuse(path, body=None, timeout=0):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(cli, "_call", refuse)

    assert cli._health() is None


@pytest.fixture
def live_service(monkeypatch):
    """The real Handler on an ephemeral port, sharing paths.token() with the CLI, with a stub brain."""
    import threading
    from http.server import ThreadingHTTPServer

    from handoff import serve

    class Agent:
        def predict(self, state, questions):
            return {"answers": {"hard": {"noul": 0.9}}}

    monkeypatch.setattr(serve, "BRAIN", serve.Brain(loader=lambda: (Agent(), None, None)))
    monkeypatch.setattr(serve.Handler, "secret", paths.token())
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    monkeypatch.setattr(cli, "URL", f"http://127.0.0.1:{httpd.server_address[1]}")
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def test_cli_talks_to_the_real_handler_with_its_token(live_service):
    health = cli._health()
    routed = cli._call("/route", {"request": "prove the Riemann hypothesis"})

    assert health["loaded"] is False
    assert routed["route"] == "strong" and routed["source"] == "zero-shot"


def test_cli_detects_a_service_holding_a_different_token(live_service, monkeypatch):
    from handoff import serve
    monkeypatch.setattr(serve.Handler, "secret", "some-other-secret")

    with pytest.raises(SystemExit, match="something else is answering"):
        cli._health()


def test_status_prints_running_service(live_service, capsys):
    cli.main(["status"])

    assert json.loads(capsys.readouterr().out)["loaded"] is False


def _fake_ps(monkeypatch, command_line):
    import subprocess
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=command_line, stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _fake_kill(monkeypatch):
    killed = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    return killed


def test_stop_kills_the_pidfile_process_when_it_is_our_service(monkeypatch, capsys):
    import signal
    paths.ensure(paths.PIDFILE).write_text("4242\n")
    ps_calls = _fake_ps(monkeypatch, "/usr/bin/python3 -m handoff.serve\n")
    killed = _fake_kill(monkeypatch)

    cli.main(["stop"])

    assert ps_calls == [["ps", "-p", "4242", "-o", "command="]]
    assert killed == [(4242, signal.SIGTERM)]
    assert not paths.PIDFILE.exists()
    assert json.loads(capsys.readouterr().out) == {"stopped": 1}


@pytest.mark.parametrize("command_line", ["/usr/bin/python3 -m http.server 7071\n", "", "/bin/zsh\n"])
def test_stop_never_kills_a_pid_that_is_not_handoff_serve(monkeypatch, capsys, command_line):
    # A stale pidfile whose PID was reused by another program must not get that program killed.
    paths.ensure(paths.PIDFILE).write_text("4242")
    _fake_ps(monkeypatch, command_line)
    killed = _fake_kill(monkeypatch)

    cli.main(["stop"])

    assert killed == []
    assert not paths.PIDFILE.exists()
    assert json.loads(capsys.readouterr().out) == {"stopped": 0}


@pytest.mark.parametrize("content", [None, "", "not-a-pid"])
def test_stop_with_missing_or_garbage_pidfile_kills_nothing(monkeypatch, capsys, content):
    if content is not None:
        paths.ensure(paths.PIDFILE).write_text(content)
    ps_calls = _fake_ps(monkeypatch, "python -m handoff.serve")
    killed = _fake_kill(monkeypatch)

    cli.main(["stop"])

    assert ps_calls == [] and killed == []
    assert json.loads(capsys.readouterr().out) == {"stopped": 0}


def test_route_fails_open_to_strong_when_service_cannot_start(monkeypatch, capsys):
    # Arrange
    def boom():
        raise SystemExit("service did not start")
    monkeypatch.setattr(cli, "_start", boom)
    # Act
    cli.main(["route", "summarise this"])
    # Assert
    out = json.loads(capsys.readouterr().out)
    assert out["route"] == "strong" and out["source"] == "fail-open"


def test_route_fails_open_on_error_reply(monkeypatch, capsys):
    # Arrange
    monkeypatch.setattr(cli, "_start", lambda: None)
    monkeypatch.setattr(cli, "_call", lambda *a, **k: {"error": "boom"})
    # Act
    cli.main(["route", "anything"])
    # Assert
    assert json.loads(capsys.readouterr().out)["route"] == "strong"
