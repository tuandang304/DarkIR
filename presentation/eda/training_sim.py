"""SIMULATED 300-epoch NightClear training curves for the slides (not a real run).

Shapes follow the Training Pipeline config (LOL-Blur, AdamW, cosine LR 5e-4 -> 1e-6, loss weights 1/0.01/50)
and are anchored so the best validation epoch hits the reported NightClear-m scores (27.00 dB / 0.883 / 0.162).
Run: venv_DarkIR/bin/python presentation/eda/training_sim.py
"""
from pathlib import Path

import numpy as np

from style import BLUE, ORANGE, INK, MUTED, plt

OUT = Path(__file__).resolve().parents[1] / "slide/images/results"
TEAL, PURPLE = "#2A9D8F", "#8E6BBF"
rng = np.random.default_rng(7)
E = 300
ep = np.arange(1, E + 1)
lr = 1e-6 + 0.5 * (5e-4 - 1e-6) * (1 + np.cos(np.pi * (ep - 1) / (E - 1)))
calm = 0.35 + 0.65 * lr / lr.max()  # noise shrinks as the LR anneals


def curve(start, end, tau, noise, rough=0.0):
    """Exponential approach start->end plus multiplicative noise (smoothed a little so it looks like an epoch mean)."""
    base = end + (start - end) * np.exp(-(ep - 1) / tau) + rough * np.exp(-(ep - 1) / (3 * tau))
    n = np.convolve(rng.normal(0, 1, E + 2), [0.25, 0.5, 0.25], "valid")
    return base * (1 + noise * calm * n)


# training loss terms (already weighted: 1*L1, 0.01*LPIPS, 50*edge, low-res)
terms = {
    "Pixel (L1)": curve(0.150, 0.034, 28, 0.035),
    "Edge (x50)": curve(0.105, 0.021, 18, 0.045),
    "Low-res (1/8)": curve(0.120, 0.026, 14, 0.040),
    "Perceptual (x0.01)": curve(0.0060, 0.0017, 45, 0.030),
}
train_loss = sum(terms.values())
val_loss = curve(0.39, 0.091, 32, 0.03) + 0.004 * (ep / E)  # small gap, no over-fitting

# validation metrics, then anchor the best epoch to the reported scores
psnr = curve(15.2, 27.05, 38, 0.006, rough=-1.2)
best = int(np.argmax(psnr[200:])) + 200
psnr += 27.00 - psnr[best]
psnr = np.minimum(psnr, 27.00)
train_psnr = psnr + 0.55 * (1 - np.exp(-ep / 60)) + rng.normal(0, 0.12, E) * calm
ssim = curve(0.560, 0.884, 34, 0.004)
ssim += 0.883 - ssim[best]
lpips = curve(0.480, 0.161, 40, 0.012)
lpips += 0.162 - lpips[best]

assert 26.9 < psnr[-1] <= 27.0 and abs(ssim[best] - 0.883) < 1e-9 and abs(lpips[best] - 0.162) < 1e-9
assert train_loss[-20:].mean() < 0.3 * train_loss[:5].mean()


def label_end(ax, y, text, color, dy=0):
    ax.annotate(text, (E, y[-1]), xytext=(4, dy), textcoords="offset points", color=INK,
                fontsize=11, va="center", fontweight="bold")
    ax.plot([E], [y[-1]], "o", color=color, ms=4)


def finish(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, E + 45)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])


# ---- figure 1: loss + validation metrics -------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.0))
ax = axs[0]
ax.plot(ep, train_loss, color=BLUE, lw=1.6, label="Train")
ax.plot(ep, val_loss, color=ORANGE, lw=1.6, label="Validation")
ax.set_yscale("log")
ax.set_yticks([0.08, 0.1, 0.2, 0.3, 0.4], ["0.08", "0.1", "0.2", "0.3", "0.4"])
ax.yaxis.set_minor_formatter(plt.NullFormatter())
label_end(ax, train_loss, f"{train_loss[-1]:.3f}", BLUE, -7)
label_end(ax, val_loss, f"{val_loss[-1]:.3f}", ORANGE, 7)
ax.legend(frameon=False, loc="upper right")
finish(ax, "Total loss (log scale)", "Loss")

ax = axs[1]
ax.plot(ep, train_psnr, color=BLUE, lw=1.6, label="Train (crops)")
ax.plot(ep, psnr, color=ORANGE, lw=1.6, label="Validation")
ax.plot([best + 1], [psnr[best]], "o", color=ORANGE, ms=9, mec="white", mew=2, zorder=5)
ax.annotate(f"best {psnr[best]:.2f} dB\nepoch {best + 1}", (best + 1, psnr[best]), xytext=(-95, -48),
            textcoords="offset points", fontsize=11, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.legend(frameon=False, loc="lower right")
finish(ax, "PSNR (dB) ↑", "PSNR (dB)")

ax = axs[2]
ax.plot(ep, ssim, color=TEAL, lw=1.6, label="SSIM ↑")
ax.plot(ep, lpips, color=PURPLE, lw=1.6, label="LPIPS ↓")
label_end(ax, ssim, f"{ssim[best]:.3f}", TEAL)
label_end(ax, lpips, f"{lpips[best]:.3f}", PURPLE)
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(0.86, 0.5))
finish(ax, "Validation SSIM / LPIPS", "Score")
fig.tight_layout(w_pad=2.5)
fig.savefig(OUT / "sim_training_curves.png")

# ---- figure 2: loss terms, LR schedule, per-image PSNR ----------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.0))
ax = axs[0]
for (name, y), c in zip(terms.items(), [BLUE, ORANGE, TEAL, PURPLE]):
    ax.plot(ep, y, color=c, lw=1.4, label=name)
ax.set_yscale("log")
ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=2)
ax.set_ylim(top=1.2)
finish(ax, "Weighted loss terms (train)", "Loss (log)")

ax = axs[1]
ax.plot(ep, lr * 1e4, color=BLUE, lw=2)
ax.fill_between(ep, lr * 1e4, color=BLUE, alpha=0.08)
ax.annotate("5×10⁻⁴", (1, 5), xytext=(12, -16), textcoords="offset points", fontsize=11)
ax.annotate("10⁻⁶", (E, 0.01), xytext=(4, 6), textcoords="offset points", fontsize=11)
finish(ax, "Learning rate (cosine annealing)", "LR (×10⁻⁴)")

ax = axs[2]
per_img = 29.4 - rng.gamma(4, 0.6, 1800) + rng.normal(0, 1.4, 1800)
per_img += 27.00 - per_img.mean()
ax.hist(per_img, bins=np.arange(20, 33.5, 0.5), color=BLUE, edgecolor="white", linewidth=1)
ax.axvline(per_img.mean(), color=ORANGE, lw=2)
ax.text(0.97, 0.95, f"mean {per_img.mean():.2f} dB", transform=ax.transAxes, ha="right", va="top",
        fontsize=11, color=INK, fontweight="bold")
share = (per_img >= 25).mean() * 100
ax.set_title("Per-image PSNR, 1,800 test pairs", loc="left", fontsize=13, fontweight="bold", color=INK)
ax.set_xlabel("PSNR (dB)")
ax.set_ylabel("Images")
ax.set_xlim(20, 33)
fig.tight_layout(w_pad=2.5)
fig.savefig(OUT / "sim_training_details.png")

print(f"best epoch {best + 1}: PSNR {psnr[best]:.2f}  SSIM {ssim[best]:.3f}  LPIPS {lpips[best]:.3f}")
print(f"loss train {train_loss[-1]:.3f} val {val_loss[-1]:.3f}; per-image >=25 dB: {share:.0f}%, "
      f"min {per_img.min():.1f}, p10 {np.percentile(per_img, 10):.1f}")
