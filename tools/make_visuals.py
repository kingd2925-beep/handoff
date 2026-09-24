#!/usr/bin/env python3
"""Regenerate the README visuals from a handoff report.json.

    python tools/make_visuals.py                      # reads ~/.handoff/model/report.json
    python tools/make_visuals.py --report path.json --out docs

Writes docs/report-card.png, docs/demo.gif and docs/social-preview.png.
Every number drawn comes from report.json, so re-run this after `handoff train`.
The two `handoff route` lines in the GIF are illustrative (labelled as such);
the `handoff report` block reproduces cli.print_report() with the real numbers.

Needs Pillow (numpy optional) and ffmpeg on PATH or at ~/.local/bin/ffmpeg.
Uses macOS system fonts (SF Pro, SF Mono); pass --font-dir to point elsewhere.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── palette ───────────────────────────────────────────────────────────
BG = (17, 19, 24)            # #111318
CARD = (24, 27, 33)
CARD_EDGE = (38, 42, 50)
GRID = (44, 48, 57)
TEXT = (236, 232, 225)
MUTED = (150, 154, 162)
DIM = (104, 108, 117)
ZERO = (92, 97, 106)         # zero-shot grey
STRONG = (217, 119, 87)      # #D97757
CHEAP = (127, 209, 139)      # #7FD18B

URL = "github.com/kingd2925-beep/handoff"
FONT_DIR = Path("/System/Library/Fonts")
SS = 2                       # supersampling factor for the static PNGs


# ── fonts ─────────────────────────────────────────────────────────────
def font(size: int, weight: str = "Regular", family: str = "SFNS.ttf") -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT_DIR / family), size)
    try:
        f.set_variation_by_name(weight)
    except (OSError, ValueError):
        pass
    return f


def sans(size, weight="Regular"):
    return font(size * SS, weight)


def mono(size, weight="Regular", scale=1):
    return font(size * scale, weight, "SFNSMono.ttf")


# ── report ────────────────────────────────────────────────────────────
REQUIRED = ["date", "labelled", "split", "gate", "max_leak", "test_trained_at_0.5",
            "test_zero_shot_at_0.5", "test_gate", "test_gate_table"]


def load_report(path: Path) -> dict:
    r = json.loads(path.read_text())
    missing = [k for k in REQUIRED if k not in r]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}")
    if not r["test_gate_table"]:
        raise SystemExit(f"{path}: empty test_gate_table")
    return r


def hard_in_test(r: dict) -> int | None:
    """Hard (strong-label) tasks in the test set, derived from hard_sent_cheap / (1 - strong_recall).
    Returned only when trained and zero-shot agree, else None (then the card shows counts only)."""
    est = []
    for key in ("test_trained_at_0.5", "test_zero_shot_at_0.5"):
        m = r[key]
        if "strong_recall" in m and m["strong_recall"] < 1:
            est.append(m["hard_sent_cheap"] / (1 - m["strong_recall"]))
    if len(est) == 2 and abs(est[0] - est[1]) < 1.0:
        return round(sum(est) / 2)
    return None


def pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%"


# ── drawing helpers ───────────────────────────────────────────────────
def S(*v):
    return tuple(int(round(x * SS)) for x in v)


def card(d: ImageDraw.ImageDraw, box, radius=18, fill=CARD):
    d.rounded_rectangle(S(*box), radius=radius * SS, fill=fill, outline=CARD_EDGE, width=SS)


def text(d, xy, s, f, fill=TEXT, anchor="la"):
    d.text(S(*xy), s, font=f, fill=fill, anchor=anchor)


def text_w(s, f) -> float:
    return f.getlength(s) / SS


def dashed_vline(d, x, y0, y1, fill, dash=8, gap=6, width=2):
    y = y0
    while y < y1:
        d.line(S(x, y, x, min(y + dash, y1)), fill=fill, width=width * SS)
        y += dash + gap


def dot(d, x, y, r, fill, ring=None):
    if ring:
        d.ellipse(S(x - r - 3, y - r - 3, x + r + 3, y + r + 3), fill=ring)
    d.ellipse(S(x - r, y - r, x + r, y + r), fill=fill)


def finish(img: Image.Image, out: Path):
    w, h = img.size
    img.resize((w // SS, h // SS), Image.LANCZOS).save(out, optimize=True)


# ── 1. report card ────────────────────────────────────────────────────
def bar_row(d, x, y, label, value, frac, color, label_w=128, track=330, h=34):
    text(d, (x, y + h / 2), label, sans(21), MUTED, "lm")
    bx = x + label_w
    d.rounded_rectangle(S(bx, y, bx + track, y + h), radius=6 * SS, fill=(33, 37, 44))
    fw = max(h * 0.4, track * frac)
    d.rounded_rectangle(S(bx, y, bx + fw, y + h), radius=6 * SS, fill=color)
    text(d, (bx + track + 18, y + h / 2), value, sans(30, "Semibold"), TEXT, "lm")


def panel_accuracy(d, r, box):
    x0, y0, x1, y1 = box
    card(d, box)
    z, t = r["test_zero_shot_at_0.5"], r["test_trained_at_0.5"]
    text(d, (x0 + 32, y0 + 30), "Accuracy on the test set", sans(25, "Semibold"))
    text(d, (x0 + 32, y0 + 66), "higher is better · both at p_strong = 0.5", sans(18), DIM)
    bar_row(d, x0 + 32, y0 + 112, "zero-shot", pct(z["accuracy"]), z["accuracy"], ZERO)
    bar_row(d, x0 + 32, y0 + 162, "trained", pct(t["accuracy"]), t["accuracy"], TEXT)


def panel_hard(d, r, box):
    x0, y0, x1, y1 = box
    card(d, box)
    z, t = r["test_zero_shot_at_0.5"], r["test_trained_at_0.5"]
    n_hard = hard_in_test(r)
    denom = n_hard or max(z["hard_sent_cheap"], t["hard_sent_cheap"], 1)
    text(d, (x0 + 32, y0 + 30), "Hard tasks sent to the cheap model", sans(25, "Semibold"))
    sub = f"of {n_hard} hard tasks in the test set · lower is better" if n_hard else "lower is better"
    text(d, (x0 + 32, y0 + 66), sub, sans(18), DIM)
    bar_row(d, x0 + 32, y0 + 112, "zero-shot", str(z["hard_sent_cheap"]), z["hard_sent_cheap"] / denom, ZERO)
    bar_row(d, x0 + 32, y0 + 162, "trained", str(t["hard_sent_cheap"]), t["hard_sent_cheap"] / denom, STRONG)


def chart_axes(d, px0, py0, px1, py1, xmax, ymax):
    fx = lambda v: px0 + (px1 - px0) * v / xmax
    fy = lambda v: py1 - (py1 - py0) * v / ymax
    steps = int(round(ymax / 0.1))
    for i in range(steps + 1):
        yv = i * 0.1
        d.line(S(px0, fy(yv), px1, fy(yv)), fill=GRID, width=SS)
        text(d, (px0 - 12, fy(yv)), pct(yv), sans(17), DIM, "rm")
    for i in range(int(round(xmax / 0.1)) + 1):
        xv = i * 0.1
        text(d, (fx(xv), py1 + 14), f"{xv:.1f}", sans(17), DIM, "ma")
    return fx, fy


def polyline(d, pts, color, width=4):
    d.line([S(*p) for p in pts], fill=color, width=width * SS, joint="curve")


def gate_label(d, gx, y, r, right_edge):
    """Callout beside the gate line; flips to the left side if it would overflow the plot."""
    g = r["test_gate"]
    lines = [("gate picked on held-out folds", sans(19, "Semibold"), TEXT),
             (f"cut {r['gate']:.2f} → on test: {pct(g['to_cheap'])} to cheap", sans(17), MUTED),
             (f"{pct(g['hard_leak'], 1)} of hard tasks leak", sans(17), MUTED)]
    w = max(text_w(s, f) for s, f, _ in lines) + 32
    h = 24 + 26 * len(lines)
    x = gx + 24 if gx + 24 + w <= right_edge else gx - 24 - w
    d.rounded_rectangle(S(x, y, x + w, y + h), radius=10 * SS, fill=(31, 35, 42), outline=CARD_EDGE, width=SS)
    for i, (s, f, c) in enumerate(lines):
        text(d, (x + 16, y + 14 + 26 * i), s, f, c)


def legend(d, x, y, items):
    for color, label in items:
        d.rounded_rectangle(S(x, y - 3, x + 22, y + 3), radius=3 * SS, fill=color)
        text(d, (x + 32, y), label, sans(18), MUTED, "lm")
        x += 32 + text_w(label, sans(18)) + 30


def panel_gate(d, r, box):
    x0, y0, x1, y1 = box
    card(d, box)
    table = sorted(r["test_gate_table"], key=lambda row: row["cut"])
    text(d, (x0 + 32, y0 + 30), "The confidence gate, on the test set", sans(25, "Semibold"))
    text(d, (x0 + 32, y0 + 66), "hand a task to the cheap model only if p_strong < cut", sans(18), DIM)
    legend(d, x0 + 32, y0 + 112, [(CHEAP, "% of all tasks sent to cheap"), (STRONG, "% of hard tasks leaked")])
    xmax = max(0.5, math.ceil(table[-1]["cut"] * 10) / 10)
    top = max(max(t["to_cheap"], t["hard_leak"]) for t in table)
    ymax = math.ceil((top + 0.15) * 10) / 10          # headroom above the lines for the gate label
    px0, py0, px1, py1 = x0 + 90, y0 + 150, x1 - 36, y1 - 80
    fx, fy = chart_axes(d, px0, py0, px1, py1, xmax, ymax)
    text(d, ((px0 + px1) / 2, y1 - 36), "cut", sans(18), DIM, "ma")
    gx = fx(r["gate"])
    dashed_vline(d, gx, py0, py1, MUTED)
    polyline(d, [(fx(t["cut"]), fy(t["to_cheap"])) for t in table], CHEAP)
    polyline(d, [(fx(t["cut"]), fy(t["hard_leak"])) for t in table], STRONG)
    g = r["test_gate"]
    dot(d, gx, fy(g["to_cheap"]), 7, CHEAP, ring=CARD)
    dot(d, gx, fy(g["hard_leak"]), 7, STRONG, ring=CARD)
    gate_label(d, gx, py0 + 8, r, px1)


def render_report_card(r: dict, out: Path):
    W, H = 1400, 800
    img = Image.new("RGB", S(W, H), BG)
    d = ImageDraw.Draw(img)
    text(d, (56, 50), "handoff report card", sans(44, "Bold"))
    right = f"{r['date']} · held-out test set · n = {r['split']['test']} prompts"
    text(d, (W - 56, 72), right, sans(19), MUTED, "rm")
    panel_accuracy(d, r, (56, 128, 648, 392))
    panel_hard(d, r, (56, 412, 648, 676))
    panel_gate(d, r, (672, 128, W - 56, 676))
    caption = (f"{r['labelled']} of one person's real prompts · test set of {r['split']['test']} "
               "never used for any choice · n = 1 user")
    text(d, (W / 2, 730), caption, sans(21), MUTED, "ma")
    finish(img, out)


# ── 2. demo gif ───────────────────────────────────────────────────────
FPS = 20
TYPE_CPS = 24               # typing speed, characters per second
MONO_SIZE = 16
LINE_H = 26


def json_segments(obj: dict) -> list:
    """Colour a flat JSON object like a terminal with syntax highlighting."""
    segs = [("{", MUTED)]
    for i, (k, v) in enumerate(obj.items()):
        if i:
            segs.append((", ", MUTED))
        segs.append((f'"{k}"', DIM))
        segs.append((": ", MUTED))
        if isinstance(v, str):
            color = STRONG if v == "strong" else CHEAP if v == "cheap" else TEXT
            segs.append((f'"{v}"', color))
        else:
            segs.append((json.dumps(v), TEXT))
    segs.append(("}", MUTED))
    return segs


def report_lines(r: dict) -> list:
    """Same text as handoff.cli.print_report, split into coloured segments."""
    t, z, g = r["test_trained_at_0.5"], r["test_zero_shot_at_0.5"], r["test_gate"]
    head = (f"handoff report card — {r['date']} · {r['labelled']} labelled prompts · "
            f"test set of {t['n']} never used for any choice")
    return [
        [],
        [(head, TEXT)],
        [("  accuracy            ", MUTED), (f"zero-shot Laya {z['accuracy']:.0%}", ZERO_TXT),
         ("   →   ", MUTED), (f"trained {t['accuracy']:.0%}", TEXT)],
        [("  hard → cheap (bad)  ", MUTED), (f"zero-shot {z['hard_sent_cheap']}", ZERO_TXT),
         ("   →   ", MUTED), (f"trained {t['hard_sent_cheap']}", STRONG), ("  (at 0.5)", DIM)],
        [("  confidence gate     ", MUTED), (f"hand off only if p_strong < {r['gate']:.2f}", TEXT),
         (f" (picked on out-of-fold predictions, max leak {r['max_leak']:.0%})", DIM)],
        [("  on the test set     ", MUTED), (f"{g['to_cheap']:.0%} of tasks go to the cheap model", CHEAP),
         (" · ", MUTED), (f"{g['hard_leak']:.1%} of hard tasks leak", STRONG)],
    ]


ZERO_TXT = (160, 164, 172)

DEMO_ROUTES = [  # illustrative: realistic shape and latency, not captured output
    ("summarise these meeting notes into 5 bullets",
     {"route": "cheap", "p_strong": 0.01, "source": "trained", "ms": 61}),
    ("check if this GST pricing model breaks even",
     {"route": "strong", "p_strong": 0.93, "source": "trained", "ms": 58}),
]


def build_script(r: dict) -> list:
    """Timeline of (kind, payload, seconds). kinds: type, out, prompt, pause."""
    steps = [("pause", None, 0.8)]
    for prompt, result in DEMO_ROUTES:
        steps += [("type", f'handoff route "{prompt}"', None), ("pause", None, 0.45),
                  ("out", json_segments(result), None), ("prompt", None, None), ("pause", None, 1.6)]
    steps += [("type", "handoff report", None), ("pause", None, 0.5)]
    for line in report_lines(r):
        steps += [("out", line, None), ("pause", None, 0.12)]
    steps += [("prompt", None, None), ("pause", None, 3.0)]
    return steps


class Term:
    ROWS = 13

    def __init__(self, r: dict):
        self.f = mono(MONO_SIZE)
        self.fb = mono(MONO_SIZE, "Bold")
        self.ui = font(13, "Medium")
        self.cw = self.f.getlength("M")
        lines = [sum(len(seg[0]) for seg in ln) for ln in report_lines(r)]
        cols = max(lines + [len(f'$ handoff route "{p}"') for p, _ in DEMO_ROUTES]) + 2
        self.pad, self.bar, self.margin = 26, 38, 22
        self.W = int(self.margin * 2 + self.pad * 2 + cols * self.cw)
        self.W += self.W % 2
        self.H = self.margin * 2 + self.bar + self.pad * 2 + self.ROWS * LINE_H + 20
        self.H += self.H % 2
        self.chrome = self._chrome()

    def _chrome(self) -> Image.Image:
        img = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(img)
        m, bar_fill = self.margin, (32, 35, 42)
        d.rounded_rectangle((m, m, self.W - m, self.H - m - 20), radius=12, fill=(22, 24, 30),
                            outline=CARD_EDGE, width=1)
        d.rounded_rectangle((m, m, self.W - m, m + self.bar), radius=12, fill=bar_fill,
                            outline=CARD_EDGE, width=1)
        d.rectangle((m + 1, m + self.bar - 12, self.W - m - 1, m + self.bar), fill=bar_fill)
        d.line((m, m + self.bar, self.W - m, m + self.bar), fill=CARD_EDGE)
        for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
            cx, cy = m + 22 + i * 20, m + self.bar / 2
            d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=c)
        d.text((self.W / 2, m + self.bar / 2), "handoff — zsh", font=self.ui, fill=MUTED, anchor="mm")
        note = "simulated session · the two route lines are illustrative · report numbers are from report.json"
        d.text((self.W - m, self.H - m + 2), note, font=font(13), fill=DIM, anchor="rm")
        return img

    def frame(self, lines: list, cursor_on: bool) -> Image.Image:
        img = self.chrome.copy()
        d = ImageDraw.Draw(img)
        x0 = self.margin + self.pad
        y0 = self.margin + self.bar + self.pad - 6
        visible = lines[-self.ROWS:]
        for row, segs in enumerate(visible):
            x, y = x0, y0 + row * LINE_H
            for seg in segs:
                s, c, bold = seg[0], seg[1], len(seg) > 2
                d.text((x, y), s, font=self.fb if bold else self.f, fill=c)
                x += len(s) * self.cw
        if cursor_on and visible:
            x = x0 + sum(len(seg[0]) for seg in visible[-1]) * self.cw
            y = y0 + (len(visible) - 1) * LINE_H
            d.rectangle((x + 1, y + 1, x + self.cw - 1, y + LINE_H - 6), fill=(200, 200, 200))
        return img


def prompt_line(typed: str = "") -> list:
    return [("$ ", CHEAP, "bold"), (typed, TEXT)]


def render_frames(r: dict, term: Term) -> list:
    frames, lines, t = [], [prompt_line()], 0.0
    at_prompt = True

    def emit(n, blink):
        nonlocal t
        for _ in range(n):
            on = at_prompt and ((int(t * 2) % 2 == 0) if blink else True)
            frames.append(term.frame(lines, on))
            t += 1 / FPS

    for kind, payload, secs in build_script(r):
        if kind == "pause":
            emit(max(1, round(secs * FPS)), blink=True)
        elif kind == "type":
            acc = 0.0
            for i in range(1, len(payload) + 1):
                lines[-1] = prompt_line(payload[:i])
                acc += FPS / TYPE_CPS
                n, acc = int(acc), acc - int(acc)
                emit(n, blink=False)
            at_prompt = False
        elif kind == "out":
            lines.append(payload)
        elif kind == "prompt":
            lines.append(prompt_line())
            at_prompt = True
    return frames


def ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg") or str(Path.home() / ".local/bin/ffmpeg")
    if not Path(found).exists():
        raise SystemExit("ffmpeg not found (PATH or ~/.local/bin/ffmpeg)")
    return found


def render_demo(r: dict, out: Path):
    term = Term(r)
    frames = render_frames(r, term)
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(frames):
            fr.save(Path(tmp) / f"f{i:05d}.png")
        ff = ffmpeg_bin()
        pal = Path(tmp) / "palette.png"
        src = ["-framerate", str(FPS), "-i", str(Path(tmp) / "f%05d.png")]
        run = lambda args: subprocess.run([ff, "-y", "-loglevel", "error", *args], check=True)
        run(src + ["-vf", "palettegen=max_colors=96:stats_mode=full:reserve_transparent=0", str(pal)])
        run(src + ["-i", str(pal), "-lavfi", "paletteuse=dither=none:diff_mode=rectangle",
                   "-loop", "0", str(out)])
    return len(frames) / FPS


# ── 3. social preview ─────────────────────────────────────────────────
def bezier(p0, p1, p2, p3, n=60):
    pts = []
    for i in range(n + 1):
        u = i / n
        a, b, c, e = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + e * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + e * p3[1]))
    return pts


def arrow(d, pts, color, width=5, head=16):
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    ang = math.atan2(yb - ya, xb - xa)
    base = (xb - head * 0.7 * math.cos(ang), yb - head * 0.7 * math.sin(ang))
    keep = [p for p in pts if math.dist(p, (xb, yb)) > head * 0.7]
    polyline(d, keep + [base], color, width)
    tip = (xb, yb)
    left = (xb - head * math.cos(ang - 0.45), yb - head * math.sin(ang - 0.45))
    right = (xb - head * math.cos(ang + 0.45), yb - head * math.sin(ang + 0.45))
    d.polygon([S(*tip), S(*left), S(*right)], fill=color)


def pill(d, cx, cy, label, f, fg, bg, edge, padx=24, h=56):
    w = text_w(label, f) + padx * 2
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    d.rounded_rectangle(S(*box), radius=h / 2 * SS, fill=bg, outline=edge, width=2 * SS)
    text(d, (cx, cy), label, f, fg, "mm")
    return box


def split_graphic(d, ox, oy):
    """A prompt splitting into two arrows: strong (up) and cheap (down)."""
    fm = font(22 * SS, "Medium", "SFNSMono.ttf")
    pbox = pill(d, ox + 88, oy, "prompt", fm, TEXT, (30, 33, 40), CARD_EDGE)
    nx = ox + 250
    polyline(d, [(pbox[2], oy), (nx - 20, oy)], MUTED, 4)
    dot(d, nx, oy, 20, (30, 33, 40), ring=CARD_EDGE)
    dot(d, nx, oy, 8, TEXT)
    tx = ox + 470
    up = bezier((nx + 22, oy - 6), (nx + 110, oy - 20), (tx - 150, oy - 110), (tx - 62, oy - 110))
    dn = bezier((nx + 22, oy + 6), (nx + 110, oy + 20), (tx - 150, oy + 110), (tx - 62, oy + 110))
    arrow(d, up, STRONG)
    arrow(d, dn, CHEAP)
    fl = sans(24, "Semibold")
    pill(d, tx, oy - 110, "strong", fl, STRONG, (44, 30, 26), (98, 58, 45), padx=26)
    pill(d, tx, oy + 110, "cheap", fl, CHEAP, (27, 42, 31), (55, 96, 62), padx=26)
    text(d, (nx, oy + 38), "handoff", sans(17), DIM, "ma")


def render_social(r: dict, out: Path):
    W, H = 1280, 640
    img = Image.new("RGB", S(W, H), BG)
    d = ImageDraw.Draw(img)
    x = 88
    text(d, (x, 196), "handoff", sans(124, "Bold"), TEXT, "ls")
    text(d, (x + 4, 262), "a router trained on", sans(38), MUTED, "ls")
    text(d, (x + 4, 312), "your own AI history", sans(38), MUTED, "ls")
    tag = [("local", TEXT), ("  ·  ", DIM), ("free", TEXT), ("  ·  ", DIM), ("honest numbers", TEXT)]
    tx = x + 4
    for s, c in tag:
        text(d, (tx, 404), s, sans(27, "Medium"), c, "ls")
        tx += text_w(s, sans(27, "Medium"))
    t, z = r["test_trained_at_0.5"], r["test_zero_shot_at_0.5"]
    stat = (f"accuracy {z['accuracy']:.0%} → {t['accuracy']:.0%} on a held-out test set "
            f"· {r['labelled']} real prompts · n = 1 user")
    text(d, (x + 4, 452), stat, sans(19), DIM, "ls")
    split_graphic(d, W - 88 - 530, 300)                # right edge lands on the URL margin
    d.line(S(x, 540, W - 88, 540), fill=CARD_EDGE, width=SS)
    text(d, (W - 88, 580), URL, font(21 * SS, "Regular", "SFNSMono.ttf"), MUTED, "rm")
    text(d, (x, 580), "MIT · Python 3.10+", font(21 * SS, "Regular", "SFNSMono.ttf"), DIM, "lm")
    finish(img, out)


# ── main ──────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    global FONT_DIR
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=Path.home() / ".handoff/model/report.json")
    ap.add_argument("--out", type=Path, default=root / "docs")
    ap.add_argument("--font-dir", type=Path, default=FONT_DIR)
    ap.add_argument("--only", choices=["card", "gif", "social"], action="append")
    a = ap.parse_args(argv)
    FONT_DIR = a.font_dir
    if not a.report.exists():
        raise SystemExit(f"no report at {a.report} — run `handoff train` first")
    r = load_report(a.report)
    a.out.mkdir(parents=True, exist_ok=True)
    todo = set(a.only or ["card", "gif", "social"])
    if "card" in todo:
        render_report_card(r, a.out / "report-card.png")
    if "gif" in todo:
        secs = render_demo(r, a.out / "demo.gif")
        print(f"demo.gif: {secs:.1f} s", file=sys.stderr)
    if "social" in todo:
        render_social(r, a.out / "social-preview.png")
    for name in ("report-card.png", "demo.gif", "social-preview.png"):
        p = a.out / name
        if p.exists():
            print(f"{p}  {p.stat().st_size / 1024:.0f} KB", file=sys.stderr)


if __name__ == "__main__":
    main()
