import re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
    "axes.labelsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42, "axes.titlesize": 9, "axes.titleweight": "normal"})
C = dict(frozen="#7f7f7f", N="#8ab6d6", S="#D55E00", P="#0072B2", DS="#009E73", DP="#8fd3b6")
KEYS = ["llava", "qwen", "smol", "q7"]; NAMES = ["LLaVA-1.5-7B", "Qwen2.5-VL-3B", "SmolVLM-2.3B", "Qwen2.5-VL-7B"]; SH = ["LLaVA\n1.5-7B", "Qwen\n2.5-VL-3B", "SmolVLM\n2.3B", "Qwen\n2.5-VL-7B"]
W2, W1 = 7.16, 3.5          # IEEE double / single column width (inches)
def save(fig, n): fig.savefig(f"{n}.pdf", bbox_inches="tight", pad_inches=.03); fig.savefig(f"{n}.png", dpi=300, bbox_inches="tight", pad_inches=.03); plt.close(fig)
def panel(a, s): a.text(-0.02, 1.08, s, transform=a.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")

# ---- parse the final hierarchical run (B=5000, all four models)
cell, seeds = {}, {}
mod = None; part = "cells"
for ln in open("n10_full_run.txt"):
    m = re.match(r"==== (\w+): ", ln)
    if m: mod = m.group(1); part = "cells"; continue
    if "INDIVIDUAL SEEDS" in ln: part = "seeds"; continue
    if "PAIRED CONTRASTS" in ln: part = "x"; continue
    if not mod: continue
    n = r"([+-]?\d\.\d+)"
    if part == "cells":
        m = re.match(rf"\s+(crude|medium|clean)/(\w+)\s+(\d+)\s+{n} \[{n},{n}\]\s+{n} \[{n},{n}\]\s+(\d\.\d+) \[(\d\.\d+),(\d\.\d+)\]\s+{n} \[{n},{n}\]", ln)
        if m:
            g = m.groups(); cell[(mod, g[0], g[1])] = dict(k=int(g[2]), sh=float(g[3]), lo=float(g[4]), hi=float(g[5]), rem=float(g[9]), rlo=float(g[10]), rhi=float(g[11]))
    elif part == "seeds":
        m = re.match(rf"\s+(crude|medium|clean)/(\w+)\s+(\d+)\s+{n} \[{n},{n}\]\s+(\d\.\d+) \[", ln)
        if m: seeds.setdefault((mod, m.group(1), m.group(2)), []).append(float(m.group(4)))
assert len(cell) == 39 and ("q7","crude","S") in cell and ("llava","crude","DP") in cell, len(cell)

# =============== Fig 3: shortcut (hierarchical CIs + individual seeds) ===============
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.55), sharey=True)
for a, lv, t, ps in zip(ax, ["crude", "clean"], ["Crude edits, tested on crude placebo edits", "Clean edits, tested on Stable Diffusion placebo edits"], "ab"):
    x = np.arange(4); w = .34
    for k, c in enumerate(["S", "P"]):
        pts = np.array([cell[(m, lv, c)]["sh"] for m in KEYS]) * 100; lo = np.array([cell[(m, lv, c)]["lo"] for m in KEYS]) * 100; hi = np.array([cell[(m, lv, c)]["hi"] for m in KEYS]) * 100
        xs = x + (k - .5) * w
        a.bar(xs, pts, w * .92, color=C[c], zorder=2, label={"S": "S: standard edit training", "P": "P: placebo-aware training"}[c])
        a.errorbar(xs, pts, yerr=[pts - lo, hi - pts], fmt="none", ecolor="k", elinewidth=.9, capsize=2.2, capthick=.9, zorder=4)
        for i, m in enumerate(KEYS):
            sv = np.array(seeds.get((m, lv, c), [])) * 100
            if len(sv) > 1: a.scatter(np.full(len(sv), xs[i]) + np.linspace(-.05, .05, len(sv)), sv, s=7, facecolor="white", edgecolor="k", lw=.6, zorder=5)
    a.axhline(0, color="k", lw=.7, zorder=3); a.set_xticks(x); a.set_xticklabels(SH); a.set_title(t, pad=8); panel(a, f"({ps})")
    a.grid(axis="y", color="#dddddd", lw=.5, zorder=0); a.set_axisbelow(True)
ax[0].set_ylabel("Shortcut (percentage points)\nfrozen $-$ trained placebo Yes-rate")
ax[0].set_ylim(-5, 24)
h, l = ax[0].get_legend_handles_labels(); h.append(Line2D([], [], marker="o", ls="", mfc="white", mec="k", ms=3.5, mew=.6)); l.append("single training seed")
ax[0].legend(h, l, frameon=False, loc="upper right", handlelength=1.2)
fig.tight_layout(w_pad=1.2); save(fig, "fig_shortcut")

# =============== Fig 4: removal detection ===============
FZ = {"crude": [.685, .894, .766, .906], "clean": [.868, .979, .923, .974]}
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.55), sharey=True)
for a, lv, t, ps in zip(ax, ["crude", "clean"], ["Crude removal edits", "Clean removal edits (Stable Diffusion test)"], "ab"):
    x = np.arange(4); w = .26
    for k, (c, lab) in enumerate([("frozen", "Frozen model"), ("N", "No-edit control (N)"), ("S", "Edit training (S)")]):
        xs = x + (k - 1) * w
        if c == "frozen": pts = np.array(FZ[lv]); a.bar(xs, pts, w * .92, color=C[c], label=lab, zorder=2)
        else:
            pts = np.array([cell[(m, lv, c)]["rem"] for m in KEYS]); lo = np.array([cell[(m, lv, c)]["rlo"] for m in KEYS]); hi = np.array([cell[(m, lv, c)]["rhi"] for m in KEYS])
            a.bar(xs, pts, w * .92, color=C[c], label=lab, zorder=2); a.errorbar(xs, pts, yerr=[pts - lo, hi - pts], fmt="none", ecolor="k", elinewidth=.9, capsize=2, capthick=.9, zorder=4)
    a.set_xticks(x); a.set_xticklabels(SH); a.set_ylim(.5, 1.03); a.set_title(t, pad=8); panel(a, f"({ps})"); a.grid(axis="y", color="#dddddd", lw=.5, zorder=0); a.set_axisbelow(True)
ax[0].set_ylabel("Removed images answered “No”")
hh, ll = ax[0].get_legend_handles_labels(); fig.legend(hh, ll, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08), handlelength=1.2)
fig.tight_layout(w_pad=1.2); save(fig, "fig_removal")

# =============== Fig 5: trade-off curves (from N7 output) ===============
T = {}; mo = lv = cur = None
for line in open("cal_15.txt"):
    m = re.match(r"==== (\w+) / (\w+)", line)
    if m: mo, lv, cur = m.group(1).lower(), m.group(2), None; continue
    h = re.match(r"\s+(frozen|S|P|no-edit control)\s+POPE acc\s+(.*)", line)
    if h and mo: cur = h.group(1); T[(mo, lv, cur)] = {"pope": [float(v) for v in h.group(2).split()]}; continue
    r = re.match(r"\s+removal\s+(.*)", line)
    if r and cur and mo and "rem" not in T[(mo, lv, cur)]: T[(mo, lv, cur)]["rem"] = [float(v) for v in r.group(1).split()]
fig, ax = plt.subplots(1, 4, figsize=(W2, 2.35), sharey=True)
sty = [("frozen", C["frozen"], "Frozen model", "-", "s"), ("no-edit control", C["N"], "No-edit control (N)", "-", "^"), ("S", C["S"], "Edit training (S)", "-", "o"), ("P", C["P"], "Placebo-aware (P)", (0, (3, 1.5)), "o")]
for a, (m, nm), ps in zip(ax, [("llava", "LLaVA-1.5-7B"), ("qwen", "Qwen2.5-VL-3B"), ("smol", "SmolVLM-2.3B"), ("q7", "Qwen2.5-VL-7B")], "abcd"):
    for cond, col, lab, ls, mk in sty:
        d = T[(m, "crude", cond)]; a.plot(d["pope"], d["rem"], ls=ls, color=col, lw=1.3, marker=mk, ms=2.8, label=lab, zorder=3 if cond in ("S", "P") else 2)
    a.set_title(nm, fontsize=8.5, pad=6); a.set_xlabel("POPE accuracy"); a.grid(color="#e3e3e3", lw=.5); a.set_axisbelow(True); panel(a, f"({ps})")
    allx = np.concatenate([T[(m, "crude", c)]["pope"] for c in ["frozen", "no-edit control", "S", "P"]]); lo, hi = allx.min(), allx.max(); pad = .04 * (hi - lo)
    a.set_xlim(lo - pad, hi + pad); a.set_ylim(-.04, 1.04)
    a.xaxis.set_major_locator(plt.MaxNLocator(4))
ax[0].set_ylabel("Removal detection")
hh, ll = ax[0].get_legend_handles_labels()
fig.legend(hh, ll, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.09), handlelength=2.2)
fig.tight_layout(w_pad=.6); save(fig, "fig_tradeoff")

# =============== Fig 6: POPE accuracy before / after label-free repair (dot plot) ===============
pope = {"Frozen model": [.851, .867, .831, .849], "S, threshold 0": [.835, .851, .774, .847], "S, after repair": [.861, .862, .832, .847], "N, threshold 0": [.859, .887, .845, .876], "N, after repair": [.853, .863, .835, .852]}
fig, ax = plt.subplots(figsize=(W1, 2.9)); x = np.arange(4)
fz = np.array(pope["Frozen model"])
ax.scatter(x, fz, marker="D", s=34, color=C["frozen"], zorder=4, label="Frozen model", edgecolor="k", lw=.4)
for key, mk, col, ec, off in [("S, threshold 0", "o", "white", C["S"], -.12), ("S, after repair", "o", C["S"], C["S"], -.12), ("N, threshold 0", "^", "white", "#4a86b5", .12), ("N, after repair", "^", C["N"], "#4a86b5", .12)]:
    ax.scatter(x + off, pope[key], marker=mk, s=34, facecolor=col, edgecolor=ec, lw=1.2, zorder=5, label=key.replace("S,", "Edit training (S),").replace("N,", "No-edit control (N),"))
for i in range(4):
    for a_, b_, off, col in [("S, threshold 0", "S, after repair", -.12, C["S"]), ("N, threshold 0", "N, after repair", .12, "#4a86b5")]:
        ax.annotate("", xy=(i + off, pope[b_][i]), xytext=(i + off, pope[a_][i]), arrowprops=dict(arrowstyle="->", color=col, lw=.8, shrinkA=3.5, shrinkB=3.5), zorder=3)
ax.set_xticks(x); ax.set_xticklabels(SH); ax.set_ylim(.76, .90); ax.set_ylabel("POPE accuracy"); ax.grid(axis="y", color="#e3e3e3", lw=.5); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.5), fontsize=7, handletextpad=.2, columnspacing=.8)
fig.tight_layout(); save(fig, "fig_repair")

# =============== Fig 7: DPO-style vs cross-entropy (hierarchical CIs) ===============
fig, a = plt.subplots(figsize=(W1, 2.9)); mods = ["llava", "qwen", "smol"]; x = np.arange(3); w = .19
conds = [("S", "crude", "S", "Cross-entropy (S)"), ("P", "crude", "P", "Cross-entropy + placebo (P)"), ("DS", "crude", "DS", "DPO-style"), ("DP", "crude", "DP", "DPO-style + placebo")]
for k, (c, lv, key, lab) in enumerate(conds):
    first = True
    for i, m in enumerate(mods):
        d = cell.get((m, lv, key))
        if d is None: continue
        xs = i + (k - 1.5) * w; p = d["sh"] * 100
        a.bar(xs, p, w * .92, color=C[c], label=lab if first else None, zorder=2); first = False
        a.errorbar(xs, p, yerr=[[p - d["lo"] * 100], [d["hi"] * 100 - p]], fmt="none", ecolor="k", elinewidth=.9, capsize=2, zorder=4)
a.axhline(0, color="k", lw=.7, zorder=3); a.set_xticks(x); a.set_xticklabels(SH[:3]); a.set_ylabel("Shortcut, crude edits\n(percentage points)"); a.grid(axis="y", color="#e3e3e3", lw=.5); a.set_axisbelow(True)
a.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28), columnspacing=.8, handletextpad=.3); fig.tight_layout(); save(fig, "fig_dpo")
print("ok", len(cell), sorted({k[2] for k in cell}))

# =============== Fig 2: edit examples (re-laid out from the saved mosaic) ===============
from PIL import Image
im = np.array(Image.open("edits_src.png").convert("RGB")); H, Wd = im.shape[:2]; im = im[16:]
ih = W2 * im.shape[0] / Wd; hh = .42
fig = plt.figure(figsize=(W2, ih + hh)); ax = fig.add_axes([0, 0, 1, ih / (ih + hh)]); ax.imshow(im, aspect="auto"); ax.axis("off")
heads = ["(a) Original", "(b) Crude:\nflat fill, hard edge", "(c) Flat fill,\nfeathered edge", "(d) Flat fill,\nnoise texture", "(e) Medium:\nsmooth Telea fill", "(f) Clean:\nLaMa inpainting"]
for i, h in enumerate(heads): fig.text((i + .5) / 6, 1 - .03 / (ih + hh), h, ha="center", va="top", fontsize=8, linespacing=1.15)
save(fig, "fig_edits")
