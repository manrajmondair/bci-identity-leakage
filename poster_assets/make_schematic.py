"""Threat-model / pipeline schematic for the poster methods column."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CARD = "#8C1515"      # Stanford cardinal
GREEN = "#2E7D32"
GREENF = "#E3F2E4"
REDF = "#FBE3E3"
BLUE = "#1565C0"
BLUEF = "#E3ECF7"
GREY = "#444444"

fig, ax = plt.subplots(figsize=(13.6, 5.8), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

def box(x, y, w, h, text, ec, fc, fs=13, bold=True, tc="black", rad=0.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={rad}",
                       linewidth=2.2, edgecolor=ec, facecolor=fc, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color=tc, zorder=5)

def arrow(x1, y1, x2, y2, color, lw=2.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=22, linewidth=lw, color=color, linestyle=ls, zorder=4))

# ---- Top row: the LEGITIMATE motor-imagery pipeline (green) ----
ytop = 31; h1 = 9
box(1,  ytop, 19, h1, "EEG signal\n(3 public corpora)", GREEN, GREENF, 13)
box(28, ytop, 24, h1, "BCI decoder\ntrained for MOTOR IMAGERY\nFBCSP · Riemann · EEGNet", GREEN, GREENF, 12.5)
box(60, ytop, 22, h1, "Intended output:\nmotor command\n(left / right / feet)", GREEN, GREENF, 12.5)
arrow(20, ytop+h1/2, 28, ytop+h1/2, GREEN)
arrow(52, ytop+h1/2, 60, ytop+h1/2, GREEN)
ax.text(50, 42.5, "Intended use  —  decode mental commands", ha="center",
        fontsize=12.5, style="italic", color=GREEN, fontweight="bold")

# ---- Defense layer sits ON the decoder's representation (blue) ----
box(28, 20.5, 24, 6.5, "DEFENSES on the representation\nD1 ad-hoc · D2 DANN · D3 DP-SGD · D4 federated",
    BLUE, BLUEF, 10.5)
arrow(40, ytop, 40, 27, BLUE, lw=2.2, style="-|>", ls=(0,(4,3)))

# ---- Adversary path (red): identity leaks as a side-channel ----
box(28, 8, 24, 8.5, "ADVERSARY\nreads model outputs /\npenultimate embeddings", CARD, REDF, 12)
arrow(40, 20.5, 40, 16.5, CARD, lw=2.8)
ax.text(57.5, 18.4, "side-channel\nleakage", ha="center", fontsize=10.5,
        color=CARD, style="italic", fontweight="bold")
box(60, 8, 30, 8.5, "SUBJECT  RE-IDENTIFIED\nWho produced this EEG?", CARD, CARD, 13.5, tc="white")
arrow(52, 12.25, 60, 12.25, CARD, lw=2.8)

# ---- Attack legend strip ----
box(1, 0.2, 89, 5.4,
    "Attacks  A1 closed-set re-ID · A2 cross-task · A3 cross-session · A4 open-set verification (unseen subjects) · A5 membership inference",
    GREY, "white", 10.5, bold=False, tc=GREY, rad=0.4)

plt.tight_layout(pad=0.3)
out = "/Users/manrajmondair/bci-review/poster_assets/fig_schematic.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", out)
from PIL import Image
print("size", Image.open(out).size)
