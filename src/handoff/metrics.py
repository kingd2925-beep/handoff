"""Scoring maths in plain Python, so it is easy to test and read. "Positive" = needs the strong model."""
import random
from typing import Dict, List, Sequence, Tuple

GATE_CUTS = [0.005, 0.01, 0.015] + [c / 100 for c in range(2, 52, 2)]   # finer at the careful end


def confusion(p: Sequence[float], y: Sequence[int], threshold: float) -> Dict[str, float]:
    """Accuracy etc. when everything with p ≥ threshold goes to the strong model."""
    tp = sum(1 for a, b in zip(p, y) if a >= threshold and b == 1)
    tn = sum(1 for a, b in zip(p, y) if a < threshold and b == 0)
    fp = sum(1 for a, b in zip(p, y) if a >= threshold and b == 0)
    fn = sum(1 for a, b in zip(p, y) if a < threshold and b == 1)
    n = max(len(y), 1)
    return {"accuracy": round((tp + tn) / n, 3),
            "strong_recall": round(tp / max(tp + fn, 1), 3),
            "cheap_recall": round(tn / max(tn + fp, 1), 3),
            "hard_sent_cheap": fn, "easy_sent_strong": fp, "n": len(y)}


def gate_table(p: Sequence[float], y: Sequence[int]) -> List[Dict[str, float]]:
    """For each cut: share of all tasks handed to the cheap model (p < cut) and share of hard tasks leaked there."""
    hard = max(sum(y), 1)
    rows = []
    for cut in GATE_CUTS:
        below = [a < cut for a in p]
        rows.append({"cut": cut,
                     "to_cheap": round(sum(below) / max(len(p), 1), 3),
                     "hard_leak": round(sum(1 for b, t in zip(below, y) if b and t == 1) / hard, 3)})
    return rows


def pick_gate(p: Sequence[float], y: Sequence[int], max_leak: float) -> float:
    """Largest cut whose hard-task leak stays within max_leak; 0.0 means 'never hand off'."""
    safe = [row["cut"] for row in gate_table(p, y) if row["hard_leak"] <= max_leak]
    return max(safe) if safe else 0.0


def split3(rows: List[Tuple[str, int]], seed: int = 7, val: float = 0.2, test: float = 0.2):
    """Stratified train / validation / test split."""
    rng = random.Random(seed)
    parts = {"train": [], "val": [], "test": []}
    for label in (0, 1):
        group = [r for r in rows if r[1] == label]
        rng.shuffle(group)
        n_test, n_val = int(len(group) * test), int(len(group) * val)
        parts["test"] += group[:n_test]
        parts["val"] += group[n_test:n_test + n_val]
        parts["train"] += group[n_test + n_val:]
    for part in parts.values():
        rng.shuffle(part)
    return parts["train"], parts["val"], parts["test"]
