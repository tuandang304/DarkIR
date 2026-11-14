"""Light EDA of LOL-v2 (real + synthetic) and the README dataset sizes, for the slides.

Run with the project venv from the repo root:
    venv_DarkIR/bin/python presentation/eda/eda_lolv2.py
Expects data/datasets/LOL-v2/{Real_captured,Synthetic}/{Train,Test}/{Low,Normal}.
Writes PNGs to presentation/slide/images/eda/ and prints a JSON summary.
"""
import json
from pathlib import Path

from style import BLUE as LOW, ORANGE as NORMAL, INK, MUTED, GRID  # noqa: F401  (sets rcParams)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/datasets/LOL-v2"
OUT = ROOT / "presentation/slide/images/eda"
OUT.mkdir(parents=True, exist_ok=True)
SUBSETS = {"Real": "Real_captured", "Synthetic": "Synthetic"}


def find_dir(base, name):
    """Case-insensitive child lookup (LOL-v2 mixes Train/train, Low/low)."""
    for p in base.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    raise FileNotFoundError(f"{name} not in {base}")


def pairs(subset, split):
    d = find_dir(DATA, SUBSETS[subset]) / split
    d = find_dir(d.parent, split)
    low = {p.name: p for p in find_dir(d, "Low").iterdir() if p.is_file()}
    nor = {p.name: p for p in find_dir(d, "Normal").iterdir() if p.is_file()}
    # real set names files lowXXXXX / normalXXXXX: match on the trailing id
    key = lambda n: n.replace("low", "").replace("normal", "")
    nor_by_key = {key(n): p for n, p in nor.items()}
    matched = [(low[n], nor_by_key[key(n)]) for n in sorted(low) if key(n) in nor_by_key]
    return matched, len(low), len(nor)


def luma(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return a @ np.array([0.299, 0.587, 0.114], dtype=np.float32), a


# ---------------------------------------------------------------- stats
summary, bright, pix_hist = {}, {}, {}
bins = np.linspace(0, 1, 51)  # 0.02 wide, so [:5] is exactly Y < 0.1
for subset in SUBSETS:
    bright[subset] = {"Low": [], "Normal": []}
    pix_hist[subset] = {"Low": np.zeros(50), "Normal": np.zeros(50)}
    for split in ("Train", "Test"):
        ps, n_low, n_nor = pairs(subset, split)
        sizes, modes = set(), set()
        for lp, np_ in ps:
            for tag, path in (("Low", lp), ("Normal", np_)):
                im = Image.open(path)
                sizes.add(im.size); modes.add(im.mode)
                y, _ = luma(im)
                bright[subset][tag].append(float(y.mean()))
                pix_hist[subset][tag] += np.histogram(y, bins=bins)[0]
        summary[f"{subset}/{split}"] = {
            "pairs": len(ps), "low_files": n_low, "normal_files": n_nor,
            "sizes": sorted(map(list, sizes)), "modes": sorted(modes),
        }
    lo, no = np.array(bright[subset]["Low"]), np.array(bright[subset]["Normal"])
    summary[f"{subset}/brightness"] = {
        "low_mean": round(float(lo.mean()), 3), "normal_mean": round(float(no.mean()), 3),
        "gain_median": round(float(np.median(no / np.maximum(lo, 1e-3))), 2),
        "gain_p5_p95": [round(float(v), 2) for v in np.percentile(no / np.maximum(lo, 1e-3), [5, 95])],
        "low_pixels_below_0.1": round(float(pix_hist[subset]["Low"][:5].sum() / pix_hist[subset]["Low"].sum()), 3),
    }

# ---------------------------------------------------------------- 1. dataset sizes (README)
readme = [  # name, train pairs, test pairs
    ("LOL-Blur", 10200, 1800), ("LSRW-Nikon", 3150, 20), ("LSRW-Huawei", 2450, 30),
    ("LOLv2-synth", 900, 100), ("LOLv2-real", 689, 100), ("LOL", 485, 15),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
names = [r[0] for r in readme][::-1]
for ax, col, title, color in ((axes[0], 1, "Train pairs", MUTED), (axes[1], 2, "Test pairs", MUTED)):
    vals = [r[col] for r in readme][::-1]
    bars = ax.barh(names, vals, color=LOW if col == 1 else NORMAL, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:,}", va="center", fontsize=11, color=INK)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.tick_params(axis="y", length=0)
fig.tight_layout(w_pad=3)
fig.savefig(OUT / "dataset_sizes.png"); plt.close(fig)

# ---------------------------------------------------------------- 2. sample pairs
picks = []
for subset in SUBSETS:
    ps, _, _ = pairs(subset, "Test")
    picks += [(subset, ps[i]) for i in np.linspace(0, len(ps) - 1, 3, dtype=int)[:2]]
fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 4.6))
for c, (subset, (lp, np_)) in enumerate(picks):
    for r, (tag, path) in enumerate((("Low", lp), ("Normal", np_))):
        im = Image.open(path).convert("RGB")
        y, _ = luma(im)
        ax = axes[r, c]
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{subset} · {tag} · Y={y.mean():.2f}", fontsize=10, color=INK if tag == "Normal" else MUTED)
fig.tight_layout(h_pad=0.6, w_pad=0.4)
fig.savefig(OUT / "samples.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. per-image brightness + pixel distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
hb = np.linspace(0, 0.8, 41)
for subset, ls in (("Real", "-"), ("Synthetic", "--")):
    for tag, color in (("Low", LOW), ("Normal", NORMAL)):
        ax.hist(bright[subset][tag], bins=hb, histtype="step", linewidth=2, linestyle=ls,
                color=color, label=f"{subset} · {tag}", density=True)
ax.set_xlabel("Mean brightness per image (Y, 0–1)"); ax.set_ylabel("Density")
ax.set_title("Image-level brightness", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=10)

ax = axes[1]
centers = (bins[:-1] + bins[1:]) / 2
for tag, color in (("Low", LOW), ("Normal", NORMAL)):
    h = pix_hist["Real"][tag] + pix_hist["Synthetic"][tag]
    ax.plot(centers, h / h.sum() * 100, color=color, linewidth=2, label=tag)
ax.set_yscale("log")
ax.set_xlabel("Pixel brightness (Y, 0–1)"); ax.set_ylabel("% of pixels (log)")
ax.set_title("Pixel-level distribution", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "brightness.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. low vs normal per pair (gain is not constant)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
for subset, color, marker in (("Real", LOW, "o"), ("Synthetic", NORMAL, "s")):
    ax.scatter(bright[subset]["Low"], bright[subset]["Normal"], s=14, alpha=0.55, color=color,
               marker=marker, edgecolors="none", label=subset)
xmax = max(max(bright[s]["Low"]) for s in SUBSETS) * 1.03   # show every pair, none clipped
ymax = max(max(bright[s]["Normal"]) for s in SUBSETS) * 1.03
for g in (2, 5, 10):
    y_end = min(ymax, g * xmax) * 0.97
    ax.plot([0, y_end / g], [0, y_end], color=MUTED, linewidth=1, linestyle=":")
    ax.text(y_end / g, y_end, f" ×{g}", fontsize=10, color=MUTED, va="bottom", ha="left")
ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
ax.set_xlabel("Low-light brightness"); ax.set_ylabel("Normal-light brightness")
ax.set_title("Brightness gain per pair", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=10, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "gain_scatter.png"); plt.close(fig)

print(json.dumps(summary, indent=1))
