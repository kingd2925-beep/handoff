"""Train the router head on Laya's frozen encoder and write an honest report card.

80 % development set: 5-fold CV picks the regularisation, then 5-fold *out-of-fold* predictions pick the
confidence gate (every dev prompt is scored by a model that never saw it — far steadier than one small
validation split). 20 % test set: reported numbers only, never used for any choice.
"""
import json
import os
import random
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from . import metrics, paths

CHECKPOINT = ("convaiinnovations/laya", "multilingual")
L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
FOLDS = 5
EPOCHS = 400
MAX_LEAK = 0.05
ZERO_SHOT_Q = {"hard": {"type": "noul", "instructions":
    "Does `request` need long multi-step reasoning, careful engineering, money math, legal judgement or specialist knowledge?"}}


def load_labelled(prompts: Path, labels: Path) -> List[Tuple[str, int]]:
    """Join prompts with labels; the last label for an id wins, so re-labelling never duplicates a prompt."""
    from .label import read_jsonl
    texts = {r["id"]: r["text"] for r in read_jsonl(prompts) if "id" in r and "text" in r}
    chosen = {r["id"]: r["label"] for r in read_jsonl(labels)
              if r.get("id") in texts and r.get("label") in ("cheap", "strong")}
    return [(texts[i], 1 if lab == "strong" else 0) for i, lab in chosen.items()]


def load_laya():
    if _weights_cached():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")   # skip the network check once weights are on disk
    import laya
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    agent = laya.load(CHECKPOINT[0], subfolder=CHECKPOINT[1], device=device)
    return agent, laya.embed_fn_from_agent(agent)


def _weights_cached() -> bool:
    return (Path.home() / ".cache" / "huggingface" / "hub" / "models--convaiinnovations--laya").exists()


def fit(x, y, l2: float):
    import torch
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = y.sum().clamp(min=1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(len(y) - pos) / pos)
    opt = torch.optim.Adam([w, b], lr=0.01, weight_decay=l2)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss_fn(x @ w + b, y).backward()
        opt.step()
    return w.detach(), b.detach()


def probs(x, w, b) -> List[float]:
    import torch
    return torch.sigmoid(x @ w + b).tolist()


def pick_l2(x, y) -> float:
    idx = list(range(len(y)))
    random.Random(7).shuffle(idx)
    best, best_acc = L2_GRID[0], -1.0
    for l2 in L2_GRID:
        accs = []
        for k in range(FOLDS):
            val = set(idx[k::FOLDS])
            tr = [i for i in idx if i not in val]
            va = sorted(val)
            w, b = fit(x[tr], y[tr], l2)
            accs.append(metrics.confusion(probs(x[va], w, b), y[va].int().tolist(), 0.5)["accuracy"])
        if sum(accs) / FOLDS > best_acc:
            best, best_acc = l2, sum(accs) / FOLDS
    return best


def out_of_fold(x, y, l2: float) -> List[float]:
    """Score every row with a head trained on the other folds."""
    idx = list(range(len(y)))
    random.Random(11).shuffle(idx)
    oof = [0.0] * len(y)
    for k in range(FOLDS):
        held = sorted(set(idx[k::FOLDS]))
        rest = [i for i in idx if i not in set(held)]
        w, b = fit(x[rest], y[rest], l2)
        for i, p in zip(held, probs(x[held], w, b)):
            oof[i] = p
    return oof


def run(say: Callable[[str], None] = print) -> Dict:
    import torch
    rows = load_labelled(paths.PROMPTS, paths.LABELS)
    if len(rows) < 60 or len({r[1] for r in rows}) < 2:
        raise SystemExit("need at least 60 labelled prompts with both labels — run `handoff label` first")
    train, val, test = metrics.split3(rows)
    dev = train + val
    say(f"{len(rows)} labelled ({sum(r[1] for r in rows)} strong) → dev {len(dev)} (5-fold) · test {len(test)}")
    agent, embed = load_laya()
    say("embedding with Laya's encoder…")
    enc = lambda part: torch.tensor(embed([t for t, _ in part]), dtype=torch.float32)
    lab = lambda part: torch.tensor([float(l) for _, l in part])
    xdev, xte = enc(dev), enc(test)
    ydev, yte = lab(dev), lab(test)
    mean, std = xdev.mean(0), xdev.std(0).clamp(min=1e-6)
    xdev, xte = (xdev - mean) / std, (xte - mean) / std

    l2 = pick_l2(xdev, ydev)
    oof, ydev_list = out_of_fold(xdev, ydev, l2), [int(v) for v in ydev.tolist()]
    gate = metrics.pick_gate(oof, ydev_list, MAX_LEAK)
    w, b = fit(xdev, ydev, l2)
    p_test, y_test = probs(xte, w, b), [int(v) for v in yte.tolist()]
    zero = [agent.predict({"request": t}, ZERO_SHOT_Q)["answers"]["hard"]["noul"] for t, _ in test]

    report = {
        "date": date.today().isoformat(), "labelled": len(rows), "strong": sum(r[1] for r in rows),
        "split": {"dev": len(dev), "test": len(test)}, "gate_method": "5-fold out-of-fold on dev", "l2": l2,
        "gate": gate, "max_leak": MAX_LEAK,
        "test_trained_at_0.5": metrics.confusion(p_test, y_test, 0.5),
        "test_zero_shot_at_0.5": metrics.confusion(zero, y_test, 0.5),
        "test_gate": next((r for r in metrics.gate_table(p_test, y_test) if r["cut"] == gate),
                          {"cut": 0.0, "to_cheap": 0.0, "hard_leak": 0.0}),
        "test_gate_table": metrics.gate_table(p_test, y_test),
        "dev_gate_table": metrics.gate_table(oof, ydev_list),
    }
    torch.save({"w": w, "b": b, "mean": mean, "std": std, "gate": gate}, paths.ensure(paths.HEAD))
    paths.ensure(paths.REPORT).write_text(json.dumps(report, indent=2))
    paths.private(paths.HEAD)
    paths.private(paths.REPORT)
    return report
