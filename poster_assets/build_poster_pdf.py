"""Print-ready 36x24in poster, Stanford CS template style. Verifiable (renders to PDF+PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import textwrap, os

CARD = "#8C1515"; CARDD = "#660C0C"; INK = "#1a1a1a"; GREY = "#555555"
A = "/Users/manrajmondair/bci-review/poster_assets"

fig = plt.figure(figsize=(36, 24), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 36); ax.set_ylim(0, 24)
ax.invert_yaxis(); ax.axis("off")

def rrect(x, y, w, h, fc, ec="none", lw=0, rad=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1, zorder=2))

def header(x, y, w, text, h=0.75, fs=30):
    rrect(x, y, w, h, CARD, rad=0.1)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold", zorder=3)

def subhdr(x, y, w, text, fs=30):
    ax.text(x, y, text, ha="left", va="top", color=CARD, fontsize=fs, fontweight="bold", zorder=3)
    ax.plot([x, x+w], [y+0.62, y+0.62], color=CARD, lw=2.5, zorder=3)
    return y + 0.95

def bullets(x, y, items, wrap, fs=27, lead=1.32, gap=0.16, bullet="•", color=INK):
    """items: list of (text, bold_prefix_len_or_0). Returns ending y."""
    lh = fs * lead / 72.0
    for it in items:
        txt = it; bold0 = 0
        if isinstance(it, tuple): txt, bold0 = it
        lines = textwrap.wrap(txt, wrap)
        for i, ln in enumerate(lines):
            bx = x if i == 0 else x + 0.34
            if i == 0:
                ax.text(x, y, bullet, ha="left", va="top", fontsize=fs, color=CARD, fontweight="bold")
                ax.text(x + 0.34, y, ln, ha="left", va="top", fontsize=fs, color=color)
            else:
                ax.text(x + 0.34, y, ln, ha="left", va="top", fontsize=fs, color=color)
            y += lh
        y += gap
    return y

def para(x, y, txt, wrap, fs=27, lead=1.32, color=INK, bold=False, italic=False):
    lh = fs * lead / 72.0
    for ln in textwrap.wrap(txt, wrap):
        ax.text(x, y, ln, ha="left", va="top", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", style="italic" if italic else "normal")
        y += lh
    return y

def image(path, x, y, w, valign_top=True):
    im = mpimg.imread(path); ih, iw = im.shape[0], im.shape[1]
    h = w * ih / iw
    ax.imshow(im, extent=(x, x+w, y+h, y), zorder=2, aspect="auto")
    return y + h

def caption(x, y, lead_bold, rest, w, fs=20):
    # bold lead-in then rest, wrapped together
    full = lead_bold + " " + rest
    lh = fs * 1.28 / 72.0
    lines = textwrap.wrap(full, int(w / (fs*0.0078)))
    for i, ln in enumerate(lines):
        ax.text(x, y, ln, ha="left", va="top", fontsize=fs, color=GREY)
        y += lh
    return y + 0.05

def callout(x, y, w, big, sub, fs_big=36, fs_sub=18, h=1.62):
    rrect(x, y, w, h, "#F7ECEC", ec=CARD, lw=2, rad=0.1)
    ax.text(x+0.35, y+0.16, big, ha="left", va="top", fontsize=fs_big, color=CARD, fontweight="bold")
    yy = y + 0.16 + fs_big*1.1/72.0
    for ln in textwrap.wrap(sub, int(w/(fs_sub*0.0072))):
        ax.text(x+0.35, yy, ln, ha="left", va="top", fontsize=fs_sub, color=INK); yy += fs_sub*1.22/72.0
    return y + h + 0.18

# ===================== TITLE BAND =====================
rrect(0, 0, 36, 3.15, CARD, rad=0.0)
ax.text(18.2, 0.9, "Subject Re-Identification Leakage in Motor-Imagery BCI Models",
        ha="center", va="center", color="white", fontsize=50, fontweight="bold")
ax.text(18.2, 1.85, "Manraj Singh Mondair", ha="center", va="center", color="white",
        fontsize=30, style="italic", fontweight="bold")
ax.text(18.2, 2.45, "Department of Computer Science, Stanford University  ·  CS 281: Ethics of AI",
        ha="center", va="center", color="#f0dada", fontsize=22, style="italic")
# Real Stanford seal (top-left) and CS logo (top-right)
seal = mpimg.imread(f"{A}/seal.png")
ax.imshow(seal, extent=(0.35, 2.95, 2.95, 0.35), zorder=4, aspect="auto")
rrect(31.55, 2.5, 4.15, 1.0, "white", ec="#d9c89a", lw=2)
clogo = mpimg.imread(f"{A}/cslogo.png")
ax.imshow(clogo, extent=(31.85, 35.4, 3.4, 2.6), zorder=4, aspect="auto")

# ===================== LEFT COLUMN =====================
LX, LW = 0.43, 7.95
header(LX, 4.03, LW, "Project Overview")
y = 5.0
y = bullets(LX+0.05, y, [
 "Brain–computer interfaces (BCIs) decode EEG into commands for assistive devices and consumer neurotech. We ask the privacy question task benchmarks ignore: does the signature that enables decoding also reveal WHO the user is?",
 "We show motor-imagery BCI models leak stable subject identity as a side-channel — recoverable by an attacker holding only model outputs or embeddings.",
 "Headline: 100% subject re-identification across 104 people (chance 0.96%) — while the same decoder reaches only ~35% task accuracy. Privacy leakage is decoupled from task utility.",
 "Ethics: a model trained to read commands can fingerprint its cohort. EEG should be treated as biometric data under GDPR Article 9 and emerging neurorights.",
 "Contribution: an audit-clean benchmark — 3 corpora × 5 attacks × 4 defenses × adaptive attackers, every number with bootstrap CIs + a 240-invariant audit.",
], wrap=40, fs=24.5)

header(LX, 15.14, LW, "Datasets & Metrics")
y = 16.1
# mini table
tx = LX+0.05; tw = LW-0.1
rows = [("Corpus", "Subj.", "Role", True),
        ("PhysioNet EEG-MMIDB", "104", "primary", False),
        ("BCI IV-2a", "9×2", "cross-session", False),
        ("Lee 2019 OpenBMI", "54×2", "replication", False)]
cw = [4.5, 1.2, 2.2]; rh = 0.62
for r,(c0,c1,c2,hd) in enumerate(rows):
    yy = y + r*rh
    if hd: rrect(tx, yy, tw, rh, CARD, rad=0.05)
    elif r%2==0: rrect(tx, yy, tw, rh, "#f2e9e9", rad=0.0)
    tcol = "white" if hd else INK
    ax.text(tx+0.12, yy+rh/2, c0, ha="left", va="center", fontsize=18, color=tcol, fontweight="bold" if hd else "normal")
    ax.text(tx+cw[0]+0.5, yy+rh/2, c1, ha="center", va="center", fontsize=18, color=tcol, fontweight="bold" if hd else "normal")
    ax.text(tx+cw[0]+cw[1]+1.2, yy+rh/2, c2, ha="left", va="center", fontsize=17, color=tcol, fontweight="bold" if hd else "normal")
y = y + 4*rh + 0.35
y = bullets(tx, y, [
 "Closed-set top-1 — name the subject among N enrolled (chance 1/N).",
 "Open-set AUC / EER — same-vs-different verification on subjects NEVER seen in training (chance AUC 0.5).",
 "Membership-inference AUC + advantage — was this person in the training set?",
 "Utility cost — motor-imagery accuracy (pp) a defense sacrifices.",
], wrap=42, fs=21)

# ===================== MIDDLE COLUMN =====================
MX, MW = 9.02, 14.49
header(MX, 4.02, MW, "Methods & Experiments")
y = 5.0
y = para(MX+0.05, y, "Three victim decoders, attacked five ways (A1–A5); four defenses (D1–D4) stress-tested against generic AND adaptive attackers.", wrap=92, fs=23, italic=True, color=CARD)
y += 0.1
yimg = image(f"{A}/fig_schematic.png", MX+0.05, y, MW-0.1)
y = yimg + 0.25
y = bullets(MX+0.05, y, [
 "Victims: FBCSP+LDA and Riemannian tangent-space+LR (classical); EEGNet (deep CNN); contrastive EEGNet embedder for open-set verification.",
 "Adaptive threat model — the real test: the attacker knows the defense and re-trains the encoder end-to-end on identity labels (“encoder fine-tune”).",
 "Rigor: 1000-resample trial-grouped bootstrap CIs; shuffled-label negative control; run provenance + a 240-invariant audit on every commit.",
], wrap=88, fs=22)

header(MX, 16.02, MW, "Discussions & Future Research")
y = 17.0
ax.text(MX+0.05, y, "Discussions:", ha="left", va="top", fontsize=22, color=CARD, fontweight="bold"); y += 0.5
y = bullets(MX+0.05, y, [
 "Identity leaks through every decoder family; survives task change, session change, and even resting state — and verifies subjects the model never saw (AUC 0.92 on two independent corpora).",
 "Ad-hoc transforms and DANN cut leakage against a generic probe but COLLAPSE under an adaptive attacker (re-ID restored to 0.80) — security by obscurity fails.",
 "Only DP-SGD is adaptively robust: ε≤1 blocks both re-ID and DP-aware membership inference at ~6pp task cost; ε=3 stops re-ID but NOT membership inference — the defense story splits by attack type.",
], wrap=90, fs=21)
ax.text(MX+0.05, y, "Future Research:", ha="left", va="top", fontsize=22, color=CARD, fontweight="bold"); y += 0.5
y = bullets(MX+0.05, y, [
 "Population-scale cohorts (TUH-EEG ~10⁴), other paradigms (sleep, ERP), tighter federated budgets, per-subject ancestry metadata for fairness audits.",
], wrap=90, fs=21)
y += 0.05
ax.text(MX+0.05, y, "References:", ha="left", va="top", fontsize=17, color=CARD, fontweight="bold"); y += 0.34
refs = ["[1] Schalk et al. BCI2000/PhysioNet, IEEE TBME 2004.  [2] Lee et al. OpenBMI, GigaScience 2019.  [3] Lawhern et al. EEGNet, J. Neural Eng. 2018.",
        "[4] Abadi et al. DP-SGD, ACM CCS 2016.  [5] Shokri et al. Membership Inference, IEEE S&P 2017.  [6] Yeom et al. Privacy & Overfitting, IEEE CSF 2018."]
for r in refs:
    ax.text(MX+0.05, y, r, ha="left", va="top", fontsize=15, color=GREY); y += 0.3

# ===================== RIGHT COLUMN (RESULTS) =====================
header(24.21, 4.02, 11.36, "Results")
RX, RW = 24.3, 11.2
def cimg(path, y, w):   # centered image in results column
    return image(path, RX + (RW - w)/2.0, y, w)

y = subhdr(RX, 5.0, RW, "Attacks — identity leaks at chance task accuracy", fs=26)
y = para(RX, y, "Classical decoders re-identify all 104 subjects perfectly; the signal survives task and session change and generalizes to unseen people.", wrap=74, fs=19, color=INK)
y += 0.05
y = cimg(f"{A}/fig_closed_set.png", y, 10.6)
y = caption(RX, y+0.05, "100% re-ID (Riemann) at 0.96% chance:", "identity is trivially recoverable from a model trained only to decode commands.", RW)
y = callout(RX, y+0.05, RW, "AUC 0.925 / 0.920",
            "Open-set verification on subjects NEVER seen in training — PhysioNet 0.925, replicates 0.920 on Lee 2019. EEG behaves like a biometric template.")

y2 = subhdr(RX, y+0.12, RW, "Defenses — only DP-SGD survives adaptive attack", fs=26)
y2 = cimg(f"{A}/fig_dp_sweep.png", y2, 9.0)
y2 = caption(RX, y2+0.04, "DP-SGD privacy–utility frontier:", "ε≤1 holds the adaptive encoder-fine-tune attacker near chance at ~6pp task cost.", RW)
y2 = cimg(f"{A}/fig_dpmia.png", y2+0.05, 7.3)
y2 = caption(RX, y2+0.04, "DP-aware membership inference:", "ε=3 fails (AUC 0.89 ≈ undefended); only ε≤1 reaches chance, tracking the Yeom (2018) bound.", RW)

out_pdf = "/Users/manrajmondair/Downloads/BCI_Poster_Mondair.pdf"
out_png = "/Users/manrajmondair/bci-review/poster_assets/poster_preview.png"
fig.savefig(out_pdf, dpi=150); fig.savefig(out_png, dpi=72)
print("wrote", out_pdf, "and", out_png)
print("bottom-of-column y left/mid/right-top/right-bot:", round(y,2), round(y2,2))
