"""Pull only the messages YOU typed out of your AI tools' local history files.

Sources: Claude Code (~/.claude/projects/*/*.jsonl, top level only — subagent transcripts are the AI talking
to itself), Codex CLI (~/.codex/sessions/**/*.jsonl) and plain text files (one prompt per line).
Nothing is uploaded. Assistant replies are never kept.
"""
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator, List

MIN_CHARS = 15
MAX_CHARS = 600
SEED = 7
SKIP_PREFIXES = ("<", "[Request interrupted", "Caveat:")
REDACTED = "[redacted]"
# Credentials people paste into chats. Redacted before anything is written to disk or sent for labelling.
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)", re.S),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),                       # Anthropic / OpenAI style
    re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                          # AWS access key id
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),                       # Google API key
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),                 # Slack
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),   # JWT
    re.compile(r"\b[a-fA-F0-9]{40,}\b"),                           # long hex secrets / hashes
]
# "password: hunter2" → keep the key name, hide the value
KEYED_SECRET = re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\b(\s*[:=]\s*)\S{6,}")


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return KEYED_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") in ("text", "input_text"))
    return ""


def _user_text(record: dict) -> str:
    """Return the human-typed text of one history record, or ''."""
    if record.get("isMeta") or record.get("isSidechain"):
        return ""
    if record.get("type") == "user":                                   # Claude Code
        return _text_of(record.get("message", {}).get("content"))
    payload = record.get("payload", record)                            # Codex CLI
    if payload.get("type") == "message" and payload.get("role") == "user":
        return _text_of(payload.get("content"))
    return ""


def _clean(text: str) -> str:
    text = " ".join(redact(text).split())
    if len(text) < MIN_CHARS or text.startswith(SKIP_PREFIXES) or "tool_result" in text:
        return ""
    return text[:MAX_CHARS]


def from_jsonl(files: Iterable[Path]) -> Iterator[str]:
    for path in files:
        with path.open(errors="ignore") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    text = _clean(_user_text(record))
                    if text:
                        yield text


def from_text(path: Path) -> Iterator[str]:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = _clean(line)
        if text:
            yield text


def history_files(source: str) -> List[Path]:
    home = Path.home()
    if source == "claude":
        return sorted((home / ".claude" / "projects").glob("*/*.jsonl"))
    if source == "codex":
        return sorted((home / ".codex" / "sessions").rglob("*.jsonl"))
    raise ValueError(f"unknown source {source!r} (use claude, codex, or --file)")


def dedupe(texts: Iterable[str]) -> List[dict]:
    seen, rows = set(), []
    for text in texts:
        key = hashlib.sha1(text.lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            rows.append({"id": key[:12], "text": text})
    random.Random(SEED).shuffle(rows)
    return rows
