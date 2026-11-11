# >>> N10: HIERARCHICAL BOOTSTRAP (training seeds x source images). CPU only, ~1-3 min. No training, no overwriting: reads preds, writes NEW files only.
# Reads the saved per-seed prediction files {item_id: margin}. Needs only numpy (no pest_common / GPU).
#   * Every training seed stays its own run. Seeds are NEVER averaged before resampling.
#   * Each replicate: (1) draw training seeds with replacement inside every trained cell, (2) draw source images with replacement
#     (all test groups of a drawn image come along). The statistic is computed on that resampled (seed x group) table.
#   * Paired comparisons: ONE image draw per model is shared by every condition (frozen, S, P, N, DS, DP) in every replicate.
#     Seeds are drawn independently per cell (seed 0 of S and seed 0 of P are different training runs, so they are not paired).
#   * The frozen model is one deterministic run: image-level variation only.
#   * Individual seeds are also reported (each alone, image-cluster CI).
# Metrics per test group (matching edit: crude->crude, medium->med, clean->Stable-Diffusion test editor):
#   keep = answered Yes on the matching PLACEBO image;  shortcut = keep(frozen) - keep(trained)   (> 0 = lost Yes-answers)
#   rem  = answered No on the matching REMOVAL image;   gain = rem(trained) - rem(frozen)
import os, re, sys, glob, json, csv, collections
import numpy as np

PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/pest_v4/preds"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/working/n10_out"
B = int(os.environ.get("N10_B", 2000))
os.makedirs(OUT_DIR, exist_ok=True)
MATCH = {"crude": ("crude_placebo", "crude_remove"), "medium": ("med_placebo", "med_remove"), "clean": ("sd_placebo", "sd_remove")}
rx = re.compile(r"^(?P<m>[a-z0-9]+)_(?P<lv>none|crude|medium|clean)_(?P<c>S|P|N|DS|DP)_s(?P<s>\d+)\.json$")

cells = collections.defaultdict(dict); frozen = {}
for f in sorted(glob.glob(os.path.join(PRED_DIR, "*.json"))):
    b = os.path.basename(f); mm = rx.match(b)
    if mm: cells[(mm["m"], mm["lv"], mm["c"])][int(mm["s"])] = json.load(open(f))
    elif re.match(r"^[a-z0-9]+_A\.json$", b): frozen[b.split("_")[0]] = json.load(open(f))

rng = np.random.default_rng(0)
gid_of = lambda k: k.split("|")[0]
img_of = lambda g: g.rsplit("_", 1)[1]                      # gid = val_<category>_<COCO image id>
def groups_of(p): return sorted({gid_of(k) for k in p if k.endswith("|crude_remove") and gid_of(k).startswith("val_") and k.count("val_") == 1})

def table(runs, kind, gl, yes):
    out = []
    for s in sorted(runs):
        v = np.array([runs[s].get(f"{g}|{kind}", np.nan) for g in gl], float)
        if np.isnan(v).any(): raise SystemExit(f"incomplete prediction file: missing {kind} items")
        out.append((v > 0) if yes else (v <= 0))
    return np.array(out, float)

lines = []; rows = []
def P(s=""): print(s); lines.append(s)
ci = lambda x: np.percentile(x, [2.5, 97.5])
def fmt(pt, x): lo, hi = ci(x); return f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}]"
def fmtp(pt, x): lo, hi = ci(x); return f"{pt:.3f} [{lo:.3f},{hi:.3f}]"

def stat(T, cid, nc, W, seed_level):
    """T (S,G) 0/1. Weighted by the image draw W (B,nc); seeds redrawn per replicate when seed_level (never pre-averaged)."""
    S = T.shape[0]
    sums = np.stack([np.bincount(cid, weights=T[s], minlength=nc) for s in range(S)])
    cnt = np.bincount(cid, minlength=nc).astype(float)
    per_seed = (sums @ W.T) / (W @ cnt)                    # (S,B): each seed's accuracy-like value in each image draw
    if not seed_level or S == 1: return per_seed.mean(0)
    pick = rng.integers(0, S, (B, S))
    return np.take_along_axis(per_seed.T, pick, axis=1).mean(1)

P("=" * 118); P(f"N10  HIERARCHICAL BOOTSTRAP: training seeds x source images   (B={B}; seeds never averaged beforehand; one shared image draw => paired)"); P("=" * 118)
P("cells: " + ", ".join(f"{m}/{lv}/{c}[{len(v)}]" for (m, lv, c), v in sorted(cells.items())))
P("frozen files: " + (", ".join(sorted(frozen)) or "none") + "\n")
for _m in sorted({k[0] for k in cells} - set(frozen)): P(f"WARNING: model {_m} has trained runs but no {_m}_A.json (frozen) -> skipped")

for m in sorted(frozen):
    mine = {k: v for k, v in cells.items() if k[0] == m}
    gl = groups_of(frozen[m]); ims = sorted({img_of(g) for g in gl}); ix = {i: j for j, i in enumerate(ims)}
    cid = np.array([ix[img_of(g)] for g in gl]); nc = len(ims)
    W = np.stack([np.bincount(rng.integers(0, nc, nc), minlength=nc) for _ in range(B)])
    P(f"==== {m}: {len(gl)} test groups from {nc} source images ====")
    FA = {lv: (table({0: frozen[m]}, MATCH[lv][0], gl, True), table({0: frozen[m]}, MATCH[lv][1], gl, False)) for lv in MATCH}
    store = {}
    for (mm_, lv, c), R in mine.items():
        levels = ["crude", "medium", "clean"] if c == "N" else [lv]       # N has no edit level: score it on every test edit
        for l2 in levels:
            kp, kr = MATCH[l2]
            try: store[(l2, c)] = (sorted(R), table(R, kp, gl, True), table(R, kr, gl, False))
            except SystemExit: pass
    BS = {}
    P(f"  {'cell':<13}{'seeds':>5}  {'shortcut [hier. 95% CI]':<27}{'  images-only CI (seeds fixed)':<31}{'removal rate [hier.]':<24}{'gain vs frozen [hier.]'}")
    for (lv, c), (seeds, kT, rT) in sorted(store.items()):
        kA, rA = FA[lv]
        bk = stat(kT, cid, nc, W, True); br = stat(rT, cid, nc, W, True)
        bkA = stat(kA, cid, nc, W, False); brA = stat(rA, cid, nc, W, False)
        bk_fix = stat(kT, cid, nc, W, False)
        sh, gn = bkA - bk, br - brA
        BS[(lv, c)] = dict(short=sh, rem=br)
        ptS, ptG = kA.mean() - kT.mean(), rT.mean() - rA.mean()
        P(f"  {lv+'/'+c:<13}{len(seeds):>5}  {fmt(ptS, sh):<27}  {fmt(ptS, bkA-bk_fix):<29}{fmtp(rT.mean(), br):<24}{fmt(ptG, gn)}")
        rows.append(dict(model=m, level=lv, cond=c, n_seeds=len(seeds), shortcut=round(ptS, 4), short_lo=round(ci(sh)[0], 4), short_hi=round(ci(sh)[1], 4),
                         short_images_only_lo=round(ci(bkA-bk_fix)[0], 4), short_images_only_hi=round(ci(bkA-bk_fix)[1], 4),
                         removal=round(rT.mean(), 4), rem_lo=round(ci(br)[0], 4), rem_hi=round(ci(br)[1], 4), gain=round(ptG, 4), gain_lo=round(ci(gn)[0], 4), gain_hi=round(ci(gn)[1], 4)))
    P("\n  INDIVIDUAL SEEDS (each run alone; CI over source images only):")
    P(f"  {'cell':<13}{'seed':>5}  {'shortcut [95% CI]':<27}{'removal rate [95% CI]'}")
    for (lv, c), (seeds, kT, rT) in sorted(store.items()):
        kA, rA = FA[lv]; bkA = stat(kA, cid, nc, W, False); sc = []
        for i, s in enumerate(seeds):
            bk = stat(kT[i:i+1], cid, nc, W, False); br = stat(rT[i:i+1], cid, nc, W, False); sh = bkA - bk; sc.append(kA.mean() - kT[i].mean())
            P(f"  {lv+'/'+c:<13}{s:>5}  {fmt(sc[-1], sh):<27}{fmtp(rT[i].mean(), br)}")
            rows.append(dict(model=m, level=lv, cond=c, seed=s, shortcut=round(sc[-1], 4), short_lo=round(ci(sh)[0], 4), short_hi=round(ci(sh)[1], 4),
                             removal=round(rT[i].mean(), 4), rem_lo=round(ci(br)[0], 4), rem_hi=round(ci(br)[1], 4)))
        if len(seeds) > 1: P(f"  {lv+'/'+c:<13}  between-seed shortcut: min {min(sc):+.3f}  max {max(sc):+.3f}  SD {np.std(sc, ddof=1):.3f}")
    P("\n  PAIRED CONTRASTS (shared image draw; seeds redrawn independently; * = 95% CI excludes 0):")
    def con(name, a, b, key):
        if a in BS and b in BS:
            d = BS[a][key] - BS[b][key]; lo, hi = ci(d)
            P(f"    {name:<52}{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {'*' if lo > 0 or hi < 0 else ''}")
            rows.append(dict(model=m, contrast=name, diff=round(d.mean(), 4), lo=round(lo, 4), hi=round(hi, 4)))
    for lv in MATCH:
        con(f"shortcut {lv}: S minus P", (lv, "S"), (lv, "P"), "short")
        con(f"shortcut {lv}: S minus N (no-edit control)", (lv, "S"), (lv, "N"), "short")
        con(f"removal  {lv}: S minus N", (lv, "S"), (lv, "N"), "rem")
        con(f"removal  {lv}: P minus N", (lv, "P"), (lv, "N"), "rem")
        con(f"shortcut {lv}: DS minus S (DPO-style vs CE)", (lv, "DS"), (lv, "S"), "short")
        con(f"shortcut {lv}: DS minus DP", (lv, "DS"), (lv, "DP"), "short")
    P("")

P("NOTES")
P(" * With 2-3 seeds the seed level has only a handful of distinct resamples: it widens the interval but cannot estimate between-seed variance well. Read it together with the individual-seed lines and the between-seed range.")
P(" * N (no-edit control) has no edit level; it is scored on every test edit and compared with the trained cell that uses the same edit. Its 'shortcut' is the effect of fine-tuning alone.")
P(" * POPE, MME and natural-image items are not tied to the 235 source-image groups and are not in this script.")
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open(os.path.join(OUT_DIR, "n10_hier_boot.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
open(os.path.join(OUT_DIR, "n10_hier_boot.txt"), "w").write("\n".join(lines))
print("wrote", OUT_DIR)
