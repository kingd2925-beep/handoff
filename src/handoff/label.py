"""Label your extracted prompts cheap/strong against a policy you control.

Backends:
  claude  — your own Claude Code CLI (`claude -p`) in batches; default model haiku (cheapest).
            Your prompts go to the same place they came from; nothing else is uploaded.
  manual  — you press c / s for each prompt in the terminal.
  csv     — import an existing `id,label` file.
Labelling resumes where it stopped; already-labelled ids are skipped.
"""
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, List

LABELS = ("cheap", "strong")
BATCH = 40
CLAUDE_TIMEOUT = 300
INSTRUCTION = ("Label every numbered request below as cheap or strong using the policy above. "
               'Reply with ONLY a JSON array like [{"i": 1, "label": "cheap"}], one object per request, nothing else.')


def read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file, skipping blank or corrupt lines (e.g. a half-written last line after a crash)."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_labels(path: Path, labels: Dict[str, str]) -> None:
    with path.open("a") as fh:
        for pid, label in labels.items():
            fh.write(json.dumps({"id": pid, "label": label}) + "\n")
    os.chmod(path, 0o600)


def build_prompt(policy: str, batch: List[dict]) -> str:
    items = "\n".join(f"{n}. {row['text']}" for n, row in enumerate(batch, 1))
    return f"{policy.strip()}\n\n{INSTRUCTION}\n\n{items}"


def parse_reply(reply: str, batch: List[dict]) -> Dict[str, str]:
    """Map a model reply back to ids. Raises ValueError if it can't be trusted."""
    start, end = reply.find("["), reply.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no JSON array in the reply")
    out = {}
    for item in json.loads(reply[start:end + 1]):
        if not isinstance(item, dict):
            continue
        try:
            n = int(item.get("i", 0))
        except (TypeError, ValueError, OverflowError):
            continue
        label = str(item.get("label", "")).strip().lower()
        if 1 <= n <= len(batch) and label in LABELS:
            out[batch[n - 1]["id"]] = label
    if len(out) < len(batch):
        raise ValueError(f"reply labelled {len(out)} of {len(batch)} requests")
    return out


def claude_labeller(model: str) -> Callable[[str], str]:
    def ask(prompt: str) -> str:
        done = subprocess.run(["claude", "-p", "--output-format", "json", "--model", model],
                              input=prompt, capture_output=True, text=True, timeout=CLAUDE_TIMEOUT)
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip() or "claude -p failed")
        return json.loads(done.stdout).get("result", "")
    return ask


def label_with_model(rows: List[dict], policy: str, ask: Callable[[str], str], out: Path,
                     batch_size: int = BATCH, report: Callable[[str], None] = print) -> int:
    done = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            labels = parse_reply(ask(build_prompt(policy, batch)), batch)
        except (ValueError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as err:
            report(f"! batch {i // batch_size + 1} skipped ({err}); run label again to retry")
            continue
        append_labels(out, labels)
        done += len(labels)
        report(f"✓ {done}/{len(rows)} labelled")
    return done


def label_by_hand(rows: List[dict], out: Path, read: Callable[[str], str] = input) -> int:
    done = 0
    for row in rows:
        answer = read(f"\n{row['text']}\n[c]heap / [s]trong / [k] skip / [q] quit: ").strip().lower()
        if answer == "q":
            break
        if answer in ("c", "s"):
            append_labels(out, {row["id"]: "cheap" if answer == "c" else "strong"})
            done += 1
    return done


def import_csv(path: Path, known: set, out: Path) -> int:
    labels = {}
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0] in known and row[1].strip().lower() in LABELS:
                labels[row[0]] = row[1].strip().lower()
    append_labels(out, labels)
    return len(labels)
