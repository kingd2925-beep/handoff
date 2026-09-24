import hashlib
import json
import shutil

import pytest

from handoff import extract


def test_claude_history_keeps_only_human_typed_prompts(fixtures_dir):
    # Arrange
    session = fixtures_dir / "claude_session.jsonl"

    # Act
    texts = list(extract.from_jsonl([session]))

    # Assert: assistant, meta, sidechain, tool_result, "<" / interrupt prefixes, short and junk lines all dropped
    assert texts == [
        "Refactor the billing module to use the new tax table",
        "Summarise these meeting notes for me please",
    ]


def test_codex_history_keeps_only_user_payload_messages(fixtures_dir):
    texts = list(extract.from_jsonl([fixtures_dir / "codex_session.jsonl"]))

    assert texts == ["Write a bash script that rotates my log files daily"]


def test_from_jsonl_reads_several_files_in_order(fixtures_dir):
    files = [fixtures_dir / "codex_session.jsonl", fixtures_dir / "claude_session.jsonl"]

    texts = list(extract.from_jsonl(files))

    assert texts[0] == "Write a bash script that rotates my log files daily"
    assert len(texts) == 3


def test_from_text_skips_short_blank_and_bracketed_lines(fixtures_dir):
    texts = list(extract.from_text(fixtures_dir / "prompts.txt"))

    assert texts == [
        "Explain the difference between TCP and UDP simply",
        "explain the difference between TCP and UDP simply",
        "Draft a polite follow-up email to a client who has not paid",
        "Plan a three-week migration from Postgres 12 to 16",
    ]


def test_clean_truncates_long_text_and_collapses_whitespace():
    long_text = "word  \n " * 400

    cleaned = extract._clean(long_text)

    assert len(cleaned) == extract.MAX_CHARS
    assert "  " not in cleaned and "\n" not in cleaned


@pytest.mark.parametrize("text", ["tiny", "<tag> padded out to be long enough",
                                  "Caveat: generated while running local commands",
                                  "this mentions tool_result and is long enough"])
def test_clean_rejects_noise(text):
    assert extract._clean(text) == ""


def test_dedupe_is_case_insensitive_with_stable_ids_and_deterministic_order():
    texts = ["Alpha prompt number one", "alpha PROMPT number one", "Beta prompt number two", "Gamma prompt three"]

    first = extract.dedupe(texts)
    second = extract.dedupe(texts)

    assert first == second
    assert len(first) == 3
    alpha = next(r for r in first if r["text"].lower() == "alpha prompt number one")
    assert alpha["text"] == "Alpha prompt number one"   # first spelling wins
    assert alpha["id"] == hashlib.sha1(b"alpha prompt number one").hexdigest()[:12]


def test_history_files_for_claude_ignores_nested_subagent_transcripts(user_home, fixtures_dir):
    # Arrange
    project = user_home / ".claude" / "projects" / "my-project"
    (project / "subagents").mkdir(parents=True)
    shutil.copy(fixtures_dir / "claude_session.jsonl", project / "session.jsonl")
    shutil.copy(fixtures_dir / "claude_session.jsonl", project / "subagents" / "agent.jsonl")

    # Act
    files = extract.history_files("claude")

    # Assert
    assert files == [project / "session.jsonl"]


def test_history_files_for_codex_searches_recursively(user_home, fixtures_dir):
    day = user_home / ".codex" / "sessions" / "2026" / "09" / "25"
    day.mkdir(parents=True)
    shutil.copy(fixtures_dir / "codex_session.jsonl", day / "rollout.jsonl")

    assert extract.history_files("codex") == [day / "rollout.jsonl"]


def test_history_files_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        extract.history_files("gemini")


# ── secret redaction ──────────────────────────────────────────────────
# Fake credentials are assembled at runtime so no secret-shaped literal sits in the repo.
R = extract.REDACTED
FAKE = {
    "anthropic": "sk-" + "ant-api03-" + "Ab3dEf6hIj9kLm2nOp5q",
    "openai": "sk-" + "proj-" + "A1b2C3d4E5f6G7h8I9j0",
    "github_classic": "ghp" + "_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",
    "github_oauth": "gho" + "_" + "Z9y8X7w6V5u4T3s2R1q0P9o8",
    "github_fine_grained": "github" + "_pat_" + "11ABCDEFG0123456789_abcdefghijklmnop",
    "aws": "AKIA" + "IOSFODNN7EXAMPLE",
    "google": "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6",
    "slack": "xox" + "b-" + "1234567890-abcdefghij",
    "jwt": "eyJ" + "hbGciOiJIUzI1NiJ9" + ".eyJzdWIiOiIxMjM0NTY3ODkwIn0" + ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9P",
    "long_hex": "a3f5" * 10,
}


@pytest.mark.parametrize("kind", sorted(FAKE))
def test_redact_blanks_each_credential_pattern(kind):
    secret = FAKE[kind]

    out = extract.redact(f"use this {secret} for the deploy")

    assert out == f"use this {R} for the deploy"
    assert secret not in out


def test_redact_blanks_a_whole_private_key_block():
    body = "MIIEow" + "IBAAKCAQEA" * 5
    text = f"here: -----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY----- thanks"

    assert extract.redact(text) == f"here: {R} thanks"


def test_redact_blanks_an_unterminated_private_key_to_the_end():
    text = "key -----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\nmore lines"

    assert extract.redact(text) == f"key {R}"


@pytest.mark.parametrize("text, expected", [
    ("password: hunter2hunter2", f"password: {R}"),
    ("PASSWORD=correct-horse", f"PASSWORD={R}"),
    ("api_key = abcdef123456", f"api_key = {R}"),
    ("my api-key:zzzzzzzz ok", f"my api-key:{R} ok"),
    ("secret: s3cr3t!!", f"secret: {R}"),
    ("pwd=letmein99", f"pwd={R}"),
])
def test_redact_hides_keyed_values_but_keeps_the_key_name(text, expected):
    assert extract.redact(text) == expected


@pytest.mark.parametrize("benign", [
    "Please fix the tokenizer so the password field shows a clear error message",
    "Use sk-learn and pandas to fit a quick regression on this CSV",
    "Rebase onto commit a1b2c3d and rerun the tests",
    "Explain what an AWS access key is and how to rotate one safely",
    "token: abc",
])
def test_redact_leaves_benign_text_untouched(benign):
    assert extract.redact(benign) == benign


def test_clean_redacts_before_saving(fixtures_dir):
    line = f"deploy the staging app with key {FAKE['aws']} please"

    assert extract._clean(line) == f"deploy the staging app with key {R} please"


def test_extracted_jsonl_prompt_is_redacted(tmp_path):
    record = {"type": "user", "message": {"role": "user", "content": f"my token is {FAKE['github_classic']} fix CI"}}
    session = tmp_path / "s.jsonl"
    session.write_text(json.dumps(record) + "\n")

    assert list(extract.from_jsonl([session])) == [f"my token is {R} fix CI"]
