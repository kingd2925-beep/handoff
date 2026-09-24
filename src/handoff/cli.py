"""handoff — train a personal model router on your own AI chat history.

  handoff extract [--source claude|codex] [--file prompts.txt]   pull only YOUR messages (local)
  handoff label   [--backend claude|manual|csv] [--model haiku] [--policy policy.md] [--csv file] [--limit N]
  handoff train                                                  fit the router, write the report card
  handoff report                                                 show the report card
  handoff route "summarise these notes"                          → one JSON line: cheap | strong (fails open to strong)
  handoff gate --cheap-share 0.3                                 hand off more (or less); shows the leak it costs
  handoff ask '{"state": "...", "questions": {...}}'             any Laya question (choice/score/noul)
  handoff serve | status | stop                                  the on-demand local service
Data lives in ~/.handoff (override with HANDOFF_HOME). Nothing is uploaded except what you
choose to label with your own Claude CLI.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import extract, label, paths

URL = f"http://127.0.0.1:{os.environ.get('HANDOFF_PORT', '7071')}"
START_WAIT = 20
CALL_TIMEOUT = 240     # the first call also loads the model (~50 s)
DEFAULT_POLICY = Path(__file__).resolve().parent / "policy.md"   # shipped inside the package


def say(msg: str) -> None:
    print(msg, file=sys.stderr)


# ── data ──────────────────────────────────────────────────────────────
def cmd_extract(args) -> None:
    texts = extract.from_text(Path(args.file)) if args.file else extract.from_jsonl(extract.history_files(args.source))
    rows = extract.dedupe(texts)
    out = paths.ensure(paths.PROMPTS)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths.private(out)
    say(f"✓ {len(rows)} of your own prompts → {out}")


def _unlabelled(limit):
    rows = label.read_jsonl(paths.PROMPTS)
    if not rows:
        raise SystemExit("no prompts yet — run `handoff extract` first")
    done = {r["id"] for r in label.read_jsonl(paths.LABELS)}
    todo = [r for r in rows if r["id"] not in done]
    return rows, (todo[:limit] if limit else todo)


def cmd_label(args) -> None:
    rows, todo = _unlabelled(args.limit)
    out = paths.ensure(paths.LABELS)
    if args.backend == "csv":
        n = label.import_csv(Path(args.csv), {r["id"] for r in todo}, out)
    elif args.backend == "manual":
        n = label.label_by_hand(todo, out)
    else:
        policy = Path(args.policy or DEFAULT_POLICY).read_text()
        say(f"labelling {len(todo)} prompts with `claude -p --model {args.model}` in batches of {label.BATCH}…")
        n = label.label_with_model(todo, policy, label.claude_labeller(args.model), out, report=say)
    say(f"✓ {n} newly labelled; {len(label.read_jsonl(out))} total")


def cmd_train(_args) -> None:
    from . import train
    report = train.run(say=say)
    print_report(report)


def print_report(r: dict) -> None:
    t, z, g = r["test_trained_at_0.5"], r["test_zero_shot_at_0.5"], r["test_gate"]
    say(f"\nhandoff report card — {r['date']} · {r['labelled']} labelled prompts · test set of {t['n']} never used for any choice")
    say(f"  accuracy            zero-shot Laya {z['accuracy']:.0%}   →   trained {t['accuracy']:.0%}")
    say(f"  hard → cheap (bad)  zero-shot {z['hard_sent_cheap']}   →   trained {t['hard_sent_cheap']}  (at 0.5)")
    say(f"  confidence gate     hand off only if p_strong < {r['gate']:.2f} (picked on out-of-fold predictions, max leak {r['max_leak']:.0%})")
    say(f"  on the test set     {g['to_cheap']:.0%} of tasks go to the cheap model · {g['hard_leak']:.1%} of hard tasks leak")


def cmd_report(_args) -> None:
    if not paths.REPORT.exists():
        raise SystemExit("no report yet — run `handoff train`")
    print_report(json.loads(paths.REPORT.read_text()))


# ── service ───────────────────────────────────────────────────────────
def _call(path: str, body=None, timeout: float = CALL_TIMEOUT) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json", "X-Handoff-Token": paths.token()}
    req = urllib.request.Request(URL + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return json.loads(err.read() or b'{"error": "http error"}')


def _health():
    """Return health only if the answer comes from OUR service (it must prove it knows the token)."""
    from .serve import proof
    try:
        info = _call("/health", timeout=2)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not isinstance(info, dict) or info.get("proof") != proof(paths.token()):
        raise SystemExit(f"something else is answering on {URL} — refusing to send it your prompts")
    return info


def _start() -> None:
    if _health():
        return
    log = paths.ensure(paths.LOG).open("ab")
    paths.private(paths.LOG)
    child = subprocess.Popen([sys.executable, "-m", "handoff.serve"], stdout=log, stderr=log, start_new_session=True)
    paths.PIDFILE.write_text(str(child.pid))
    paths.private(paths.PIDFILE)
    deadline = time.time() + START_WAIT
    while time.time() < deadline:
        if _health():
            return
        time.sleep(0.5)
    raise SystemExit(f"service did not start — see {paths.LOG}")


def cmd_route(args) -> None:
    """Fail open: if anything goes wrong, the answer is 'strong' — a wrong 'strong' only costs money."""
    try:
        _start()
        out = _call("/route", {"request": " ".join(args.text)})
        if out.get("route") not in ("cheap", "strong"):
            raise ValueError(out.get("error", "bad reply"))
    except (SystemExit, OSError, ValueError, urllib.error.URLError) as err:
        out = {"route": "strong", "source": "fail-open", "error": str(err)}
    print(json.dumps(out, ensure_ascii=False))


def cmd_gate(args) -> None:
    """Move the confidence gate to hand off a chosen share of tasks (measured on out-of-fold dev data)."""
    import torch
    if not (paths.REPORT.exists() and paths.HEAD.exists()):
        raise SystemExit("train first: `handoff train`")
    report = json.loads(paths.REPORT.read_text())
    dev = report.get("dev_gate_table")
    if not dev:
        raise SystemExit("this report predates `gate` — run `handoff train` again")
    row = next((r for r in dev if r["to_cheap"] >= args.cheap_share), dev[-1])
    head = torch.load(paths.HEAD, map_location="cpu", weights_only=True)
    head["gate"] = row["cut"]
    torch.save(head, paths.HEAD)
    paths.private(paths.HEAD)
    test = next((r for r in report["test_gate_table"] if r["cut"] == row["cut"]), None)
    report.update({"gate": row["cut"], "gate_method": f"user target cheap share {args.cheap_share:.0%}"})
    if test:
        report["test_gate"] = test
    paths.REPORT.write_text(json.dumps(report, indent=2))
    say(f"gate → p_strong < {row['cut']:.3f}: ~{row['to_cheap']:.0%} of tasks to cheap, ~{row['hard_leak']:.1%} of hard tasks leak (dev)")
    if test:
        say(f"on the held-out test set: {test['to_cheap']:.0%} to cheap, {test['hard_leak']:.1%} leak")
    cmd_stop(args)   # the service reloads the new gate on next call


def cmd_ask(args) -> None:
    _start()
    print(json.dumps(_call("/ask", json.loads(args.json)), ensure_ascii=False))


def cmd_serve(_args) -> None:
    from . import serve
    say(f"handoff service on {URL} (Ctrl-C to stop)")
    serve.main()


def cmd_status(_args) -> None:
    print(json.dumps(_health() or {"running": False}))


def _our_pid():
    """The service PID from the pidfile, only if that process is still our handoff service."""
    try:
        pid = int(paths.PIDFILE.read_text().strip())
    except (OSError, ValueError):
        return None
    cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    return pid if "handoff.serve" in cmd else None


def cmd_stop(_args) -> None:
    pid = _our_pid()
    if pid:
        os.kill(pid, signal.SIGTERM)
    paths.PIDFILE.unlink(missing_ok=True)
    print(json.dumps({"stopped": 1 if pid else 0}))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="handoff", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--source", default="claude", choices=["claude", "codex"]); e.add_argument("--file")
    lb = sub.add_parser("label"); lb.add_argument("--backend", default="claude", choices=["claude", "manual", "csv"])
    lb.add_argument("--model", default="haiku"); lb.add_argument("--policy"); lb.add_argument("--csv"); lb.add_argument("--limit", type=int)
    sub.add_parser("train"); sub.add_parser("report")
    r = sub.add_parser("route"); r.add_argument("text", nargs="+")
    g = sub.add_parser("gate"); g.add_argument("--cheap-share", type=float, required=True)
    a = sub.add_parser("ask"); a.add_argument("json")
    for name in ("serve", "status", "stop"):
        sub.add_parser(name)
    return p


COMMANDS = {"extract": cmd_extract, "label": cmd_label, "train": cmd_train, "report": cmd_report, "route": cmd_route,
            "gate": cmd_gate,
            "ask": cmd_ask, "serve": cmd_serve, "status": cmd_status, "stop": cmd_stop}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "label" and args.backend == "csv" and not args.csv:
        raise SystemExit("--backend csv needs --csv <file>")
    COMMANDS[args.cmd](args)


if __name__ == "__main__":
    main()
