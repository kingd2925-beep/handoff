"""Social preview cards for handoff (1280×640). Two designs:

  sorter  — hook headline + your prompts visibly splitting into "free model" and "strong model" lanes
  baton   — Charm-style: giant name, one bold object (a relay baton being handed over), one benefit line

Run: python tools/make_social.py [--only sorter|baton] [--out docs/]
Numbers come from ~/.handoff/model/report.json when present (falls back to the published run).
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 640
BG = (14, 15, 20)
INK = (244, 240, 232)
DIM = (150, 154, 166)
STRONG = (217, 119, 87)
CHEAP = (127, 209, 139)
PANEL = (27, 29, 37)
FONTS = Path("/System/Library/Fonts")
REPORT = Path.home() / ".handoff" / "model" / "report.json"
PUBLISHED = {"trained": 0.74, "zero": 0.52, "to_cheap": 0.20, "leak": 0.039}

CHEAP_PROMPTS = [("thanks, done", 0.00), ("summarise these meeting notes", 0.01),
                 ("rename the png files to lowercase", 0.01), ("translate this paragraph to Hindi", 0.00)]
STRONG_PROMPTS = [("does this pricing model break even?", 0.84), ("debug the failing payment webhook", 0.50),
                  ("audit this contract clause for risk", 0.91)]


def font(name: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    for candidate in (name, "SFNS.ttf"):
        try:
            return ImageFont.truetype(str(FONTS / candidate), size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


def sans(size, bold=False):
    f = font("SFNS.ttf", size)
    try:
        f.set_variation_by_name("Bold" if bold else "Regular")
    except (OSError, ValueError):
        pass
    return f


def mono(size):
    return font("SFNSMono.ttf", size)


def numbers():
    try:
        r = json.loads(REPORT.read_text())
        return {"trained": r["test_trained_at_0.5"]["accuracy"], "zero": r["test_zero_shot_at_0.5"]["accuracy"],
                "to_cheap": r["test_gate"]["to_cheap"], "leak": r["test_gate"]["hard_leak"]}
    except (OSError, KeyError, ValueError):
        return PUBLISHED


def glow(img, box, color, radius=40, alpha=90):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(box, 24, fill=color + (alpha,))
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(radius)))


def chip(d, xy, text, p, color, w=390):
    x, y = xy
    d.rounded_rectangle([x, y, x + w, y + 44], 12, fill=PANEL, outline=color + (160,), width=2)
    d.text((x + 16, y + 11), text, font=sans(19), fill=INK)
    tag = f"{p:.2f}"
    tw = d.textlength(tag, font=mono(17))
    d.text((x + w - tw - 14, y + 13), tag, font=mono(17), fill=color)


def sorter(out: Path):
    n = numbers()
    img = Image.new("RGBA", (W, H), BG + (255,))
    glow(img, [770, 110, 1190, 330], CHEAP, alpha=36)
    glow(img, [770, 380, 1190, 550], STRONG, alpha=36)
    d = ImageDraw.Draw(img)

    d.text((64, 58), "handoff", font=mono(26), fill=STRONG)
    d.text((64, 108), "Stop paying your", font=sans(66, True), fill=INK)
    d.text((64, 182), "best model to", font=sans(66, True), fill=INK)
    d.text((64, 256), "say “thanks”.", font=sans(66, True), fill=INK)
    d.text((66, 350), "A tiny router trained on your own Claude Code", font=sans(24), fill=DIM)
    d.text((66, 382), "history decides which prompts a free model can take.", font=sans(24), fill=DIM)

    stats = [(f"{n['to_cheap']:.0%}", "of tasks → free model"), (f"{n['leak']:.1%}", "hard ones missed"),
             ("0.1 s", "on your laptop")]
    x = 66
    for big, small in stats:
        d.text((x, 452), big, font=sans(46, True), fill=INK)
        d.text((x, 508), small, font=sans(18), fill=DIM)
        x += max(d.textlength(big, font=sans(46, True)), d.textlength(small, font=sans(18))) + 44
    d.text((66, 574), "local · free · honest report card · MIT", font=mono(18), fill=DIM)

    # the split: prompts arrive → handoff node → a bracket feeding each lane
    node, trunk_x = (690, 330), 752
    lanes = {"cheap": (130, 4, CHEAP, "→ FREE MODEL", CHEAP_PROMPTS), "strong": (396, 3, STRONG, "→ STRONG MODEL", STRONG_PROMPTS)}
    d.line([(626, 330), (node[0] - 20, 330)], fill=INK, width=4)
    for (top, count, color, title, prompts) in lanes.values():
        mid = top + 22 + (count - 1) * 26
        d.line([(node[0] + 18, node[1]), (trunk_x, mid)], fill=color, width=4)
        d.line([(trunk_x, top + 22), (trunk_x, top + 22 + (count - 1) * 52)], fill=color, width=4)
        d.text((780, top - 34), title, font=mono(20), fill=color)
        for i, (text, p) in enumerate(prompts):
            y = top + i * 52
            d.line([(trunk_x, y + 22), (780, y + 22)], fill=color, width=3)
            chip(d, (780, y), text, p, color, w=400)
    d.ellipse([node[0] - 20, node[1] - 20, node[0] + 20, node[1] + 20], fill=INK)
    d.text((node[0] - 40, node[1] + 28), "handoff", font=mono(16), fill=DIM)
    d.text((780, 560), "numbers = p_strong · hand off below 0.14", font=mono(15), fill=DIM)
    url = "github.com/kingd2925-beep/handoff"
    d.text((W - 48 - d.textlength(url, font=mono(16)), 598), url, font=mono(16), fill=DIM)
    img.convert("RGB").save(out / "social-preview-sorter.png")


def baton(out: Path):
    n = numbers()
    img = Image.new("RGBA", (W, H), (23, 20, 18, 255))
    glow(img, [640, 90, 1220, 560], STRONG, radius=90, alpha=70)
    d = ImageDraw.Draw(img)
    d.text((70, 96), "handoff", font=sans(150, True), fill=INK)
    d.text((76, 290), "Hands the easy prompts to the free model.", font=sans(30), fill=INK)
    d.text((76, 332), "Keeps the hard ones for the smart one.", font=sans(30), fill=DIM)
    d.text((76, 420), f"trained on your own AI history · {n['to_cheap']:.0%} handed off · {n['leak']:.1%} missed",
           font=mono(19), fill=STRONG)
    d.text((76, 560), "local · free · MIT", font=mono(18), fill=DIM)

    # a relay baton mid-handoff: striped cylinder at an angle, two "hands" as rounded blocks
    baton_img = Image.new("RGBA", (520, 180), (0, 0, 0, 0))
    b = ImageDraw.Draw(baton_img)
    b.rounded_rectangle([20, 60, 500, 120], 30, fill=STRONG)
    for x in range(60, 480, 70):
        b.rectangle([x, 60, x + 26, 120], fill=(245, 170, 140))
    b.rounded_rectangle([20, 60, 500, 120], 30, outline=(120, 60, 40), width=4)
    b.ellipse([470, 60, 530, 120], fill=(240, 150, 118), outline=(120, 60, 40), width=4)
    rotated = baton_img.rotate(18, expand=True, resample=Image.BICUBIC)
    img.alpha_composite(rotated, (700, 190))
    d = ImageDraw.Draw(img)
    for (x, y, color, label) in ((720, 360, CHEAP, "free model"), (1080, 200, INK, "you")):
        d.rounded_rectangle([x, y, x + 120, y + 70], 26, fill=color)
        d.text((x + 10, y + 82), label, font=mono(18), fill=DIM)
    url = "github.com/kingd2925-beep/handoff"
    d.text((W - 48 - d.textlength(url, font=mono(16)), 598), url, font=mono(16), fill=DIM)
    img.convert("RGB").save(out / "social-preview-baton.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["sorter", "baton"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("sorter", sorter), ("baton", baton)):
        if args.only in (None, name):
            fn(out)
            print(f"✓ {out / ('social-preview-' + name + '.png')}  (docs/social-preview.png = the sorter card)")


if __name__ == "__main__":
    main()
