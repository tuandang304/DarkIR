"""Figures for the Results / Interpretability slides.

Model numbers are copied from the DarkIR paper (arXiv 2412.13443 v3); each block names its table.
Image figures use the paper's released outputs in assets/ and LOL-v2 test pairs in data/datasets.

    venv_DarkIR/bin/python presentation/eda/paper_figures.py
Writes PNGs to presentation/slide/images/results/ and prints the numbers it computed.
"""
import json
from pathlib import Path

from style import BLUE, ORANGE, INK, MUTED
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
NAME = "NightClear"  # project name shown on the slides (model = DarkIR from the paper)
OUT = ROOT / "presentation/slide/images/results"
OUT.mkdir(parents=True, exist_ok=True)
computed = {}


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr(a, b):
    return float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))


def luma(a):
    return a @ np.array([0.299, 0.587, 0.114])


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------ 1. efficiency: Table 1 (LOLBlur), DeblurGAN-v2 has no MACs -> omitted
table1 = [  # method, PSNR, params (M), MACs (G)
    ("KinD++", 21.26, 1.2, 34.99), ("DRBN", 21.78, 0.6, 48.61), ("MIMO", 22.41, 6.8, 67.25),
    ("NAFNet", 25.36, 12.05, 12.3), ("LEDNet", 25.74, 7.4, 38.65), ("RetinexFormer", 26.02, 1.61, 15.57),
    ("Restormer", 26.72, 26.13, 144.25), (f"{NAME}-m", 27.00, 3.31, 7.25), (f"{NAME}-l", 27.30, 12.96, 27.19),
]
fig, ax = plt.subplots(figsize=(8, 4.6))
for name, p, params, macs in table1:
    ours = name.startswith(NAME)
    ax.scatter(macs, p, s=40 + params * 22, color=ORANGE if ours else BLUE, alpha=0.85 if ours else 0.55,
               edgecolors="white", linewidths=1.5, zorder=3)
    ax.annotate(f"{name}\n{params} M", (macs, p), xytext=(9, -4), textcoords="offset points", fontsize=10,
                color=INK, fontweight="bold" if ours else "normal", va="center")
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 50, 100, 200]); ax.set_xticklabels(["5", "10", "20", "50", "100", "200"])
ax.set_xlim(5, 300); ax.set_ylim(20.8, 28)
ax.set_xlabel("GMACs (log scale, lower = cheaper)"); ax.set_ylabel("PSNR on LOLBlur (dB)")
ax.set_title("Quality vs. compute on LOLBlur (bubble size = parameters)", loc="left", fontweight="bold", fontsize=13)
ax.scatter([], [], color=ORANGE, label=NAME); ax.scatter([], [], color=BLUE, alpha=0.55, label="Other methods")
ax.legend(frameon=False, loc="lower right", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "efficiency.png"); plt.close(fig)

# ------------------------------------------------ 2. cascade vs end-to-end: Table 2 (LOLBlur, zero-shot 2-step pipelines)
table2 = [
    ("NAFNet → RetinexFormer", 14.66), ("Chen → Zero-DCE", 17.02), ("RetinexFormer → NAFNet", 17.16),
    ("MIMO → Zero-DCE", 17.52), ("Zero-DCE → MIMO", 17.68), ("RUAS → MIMO", 17.81),
    ("DeblurGAN-v2 → Zero-DCE", 18.33), (f"{NAME}-m (end-to-end)", 27.00),
]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
names = [t[0] for t in table2]; vals = [t[1] for t in table2]
bars = ax.barh(names, vals, height=0.62, color=[ORANGE if n.startswith(NAME) else BLUE for n in names])
for b, v in zip(bars, vals):
    ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.2f}", va="center", fontsize=10, color=INK)
ax.set_xlim(0, 31); ax.set_xlabel("PSNR on LOLBlur (dB)")
ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0, labelsize=10)
ax.set_title("Two separate models vs. one joint model", loc="left", fontweight="bold", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "cascade.png"); plt.close(fig)

# ------------------------------------------------ 3. ablation: Tables 6, 7 (vs 27.00) and A (losses), ΔPSNR vs full model
arch = [  # Table 6, full DarkIR-m = 26.90 at 256px crops; Table 7 row uses its own baseline 27.00
    ("All blocks = EBlock", 26.17 - 26.90), ("Decoder = NAFBlock", 26.23 - 26.90),
    ("All blocks = NAFBlock", 26.24 - 26.90), ("FreMLP also edits phase", 26.30 - 26.90),
    ("Di-SpAM → LKA (Table 7)", 26.45 - 27.00), ("Encoder = NAFBlock", 26.51 - 26.90),
    ("No extra depthwise in DBlock", 26.63 - 26.90), ("All blocks = DBlock", 26.68 - 26.90),
]
loss = [  # Table A, full loss = 26.9
    ("L1 + L_lol", 26.19 - 26.9), ("L1 only", 26.34 - 26.9),
    ("L1 + Percep + Edge (no L_lol)", 26.61 - 26.9), ("L1 + L_lol + Edge", 26.717 - 26.9),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})
for ax, rows, title in ((axes[0], arch, "Architecture change (Tables 6–7)"), (axes[1], loss, "Loss terms used (Table A)")):
    names = [r[0] for r in rows][::-1]; vals = [r[1] for r in rows][::-1]
    bars = ax.barh(names, vals, height=0.6, color=BLUE)
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f"{v:+.2f} ", va="center", ha="right", fontsize=10, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlim(-0.95, 0.05); ax.set_xlabel(f"ΔPSNR vs. full {NAME}-m (dB)")
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0, labelsize=10)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=13)
fig.tight_layout(w_pad=2); fig.savefig(OUT / "ablation.png"); plt.close(fig)

# ------------------------------------------------ 4. error maps on the paper's released LOLv2-real outputs (image 00733)
A = ROOT / "assets/lolv2real"
gt = load(A / "normal00733.png")
cols = [("Low-light input", A / "low00733_low.png"), ("SNR-Net", A / "00733_snr.png"),
        ("RetinexFormer", A / "low00733_retinexformer.png"), (NAME, A / "low00733_darkir.png")]
imgs = [(n, load(p)) for n, p in cols]
errs = [np.abs(im - gt).mean(axis=2) for _, im in imgs]
vmax = float(np.percentile(np.concatenate([e.ravel() for e in errs[1:]]), 99))
fig, axes = plt.subplots(2, 5, figsize=(15, 4.6))
for c, ((name, im), err) in enumerate(zip(imgs, errs)):
    p = psnr(im, gt); computed[f"psnr_00733_{name}"] = round(p, 2)
    axes[0, c].imshow(im); bare(axes[0, c])
    axes[0, c].set_title(f"{name}\nPSNR {p:.2f} dB", fontsize=11, fontweight="bold" if name == NAME else "normal")
    m = axes[1, c].imshow(err, cmap="Blues", vmin=0, vmax=vmax); bare(axes[1, c])
    axes[1, c].set_xlabel(f"mean |error| {err.mean():.3f}", fontsize=10, color=MUTED)
axes[0, 4].imshow(gt); bare(axes[0, 4]); axes[0, 4].set_title("Ground truth", fontsize=11)
bare(axes[1, 4])
cax = axes[1, 4].inset_axes([0.05, 0.05, 0.07, 0.9])
cb = fig.colorbar(m, cax=cax)
cb.set_label("|output − GT|\n(darker = larger error)", fontsize=10)
fig.tight_layout(h_pad=1.2); fig.savefig(OUT / "error_maps.png"); plt.close(fig)

# ------------------------------------------------ 5. Fourier amplitude swap on LOL-v2 real test pairs
def swap(phase_src, amp_src):
    """Per channel: keep phase of phase_src, take amplitude of amp_src."""
    out = np.empty_like(phase_src)
    for ch in range(3):
        Fp, Fa = np.fft.fft2(phase_src[..., ch]), np.fft.fft2(amp_src[..., ch])
        out[..., ch] = np.real(np.fft.ifft2(np.abs(Fa) * np.exp(1j * np.angle(Fp))))
    return np.clip(out, 0, 1)


D = ROOT / "data/datasets/LOL-v2/Real_captured/Test"
pairs = sorted((p, D / "Normal" / p.name.replace("low", "normal")) for p in (D / "Low").glob("*.png"))
p_low, p_swap = [], []
for lp, np_ in pairs:
    lo, no = load(lp), load(np_)
    p_low.append(psnr(lo, no)); p_swap.append(psnr(swap(lo, no), no))
computed["amp_swap_n_pairs"] = len(pairs)
computed["amp_swap_psnr_low_vs_gt_mean"] = round(float(np.mean(p_low)), 2)
computed["amp_swap_psnr_lowphase_gtamp_vs_gt_mean"] = round(float(np.mean(p_swap)), 2)

lo, no = load(D / "Low/low00733.png"), load(D / "Normal/normal00733.png")
panels = [("Low-light input", lo), ("Low phase + GT amplitude", swap(lo, no)),
          ("GT phase + low amplitude", swap(no, lo)), ("Ground truth", no)]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, (name, im) in zip(axes, panels):
    ax.imshow(im); bare(ax)
    ax.set_title(f"{name}\nmean Y = {luma(im).mean():.2f}, PSNR {psnr(im, no):.1f} dB" if name != "Ground truth"
                 else f"{name}\nmean Y = {luma(im).mean():.2f}", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "amplitude_swap.png"); plt.close(fig)

print(json.dumps(computed, indent=1))
