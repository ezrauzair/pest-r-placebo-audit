# >>> N11: HUMAN EDIT-VALIDITY AUDIT PACKAGE (CPU only, ~2-5 min). Builds the package only. NO audit has been done by this code.
# Reuses the edit images that already exist from the test set (no new editing, no model, no GPU). Writes only into a NEW folder.
# Run on Kaggle after restoring pest_v4 (same as the other analysis cells). Set EDITOR to "sd" (main audit: the held-out test editor the paper
# reports as 'clean'), then optionally run again with "crude" (and "lama"/"medium" if wanted). Output is one folder per editor.
#
# Design (independent of model tuning):
#   * Sample N_GROUPS groups at random from the 235 matched TEST groups with a fixed seed. Predictions, margins and results are never read.
#     At most one group per COCO source image. Sampling is not stratified by anything the models did.
#   * Each item shows: ORIGINAL | EDIT 1 | EDIT 2. Which edit is the removal and which is the placebo is randomised per item and HIDDEN from reviewers
#     (the key is a separate file). Reviewers therefore answer the same questions about both edits, so (a) and (b) are judged blind to intent.
#   * Two separate workbooks (reviewer_A.xlsx, reviewer_B.xlsx) with different row orders and coded item ids: reviewers work independently.
#   * Questions per edit: is the target object still visible?  are there visible editing artifacts / an obvious failure?  Plus: is the object in the original?
#   * Unblinding (done only after both files are returned) maps the answers to the three audit questions:
#       (a) target removed       = removal edit judged "Not visible"
#       (b) target preserved     = placebo edit judged "Clearly visible"
#       (c) artifacts / failure  = "Minor" / "Obvious failure" on either edit
import os, sys, json, csv, random, shutil, math, collections
sys.path.insert(0, "/kaggle/working")
from PIL import Image, ImageDraw, ImageFont

EDITOR = os.environ.get("AUDIT_EDITOR", "sd")             # sd | lama | crude | medium
N_GROUPS = int(os.environ.get("AUDIT_N", 100))
SAMPLE_SEED = 20261106                                    # fixed, written to the Instructions sheet
OUT = os.environ.get("AUDIT_OUT", f"/kaggle/working/pest_audit_{EDITOR}")
PANEL_W = 420                                             # montage panel width (full-resolution PNGs are copied separately)

OPT_VIS = ["Clearly visible", "Partly visible", "Not visible", "Cannot tell"]
OPT_ART = ["None", "Minor", "Obvious failure"]
OPT_YN = ["Yes", "No", "Cannot tell"]

def edit_paths(g, editor, P=None, cpath=None, mpath=None):
    """(removal_path, placebo_path) of the EXISTING edit files for this editor."""
    if editor in ("sd", "lama"): return g["files"][f"{editor}_remove"], g["files"][f"{editor}_placebo"]
    if editor == "crude": return cpath(g["gid"], "remove"), cpath(g["gid"], "placebo")
    if editor == "medium": return mpath(g["gid"], "remove"), mpath(g["gid"], "placebo")
    raise SystemExit("unknown editor")

def sample_groups(M, n, seed):
    rnd = random.Random(seed); order = sorted(M, key=lambda g: g["gid"]); rnd.shuffle(order)
    seen, out = set(), []
    for g in order:
        if g["img_id"] in seen: continue
        seen.add(g["img_id"]); out.append(g)
        if len(out) == n: break
    return out

def font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()

def montage(paths, labels, out_path):
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB"); h = int(im.height * PANEL_W / im.width); ims.append(im.resize((PANEL_W, h), Image.LANCZOS))
    H = max(i.height for i in ims) + 30; W = PANEL_W * len(ims) + 8 * (len(ims) - 1)
    M = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(M); f = font(18)
    for k, (im, lab) in enumerate(zip(ims, labels)):
        x = k * (PANEL_W + 8); M.paste(im, (x, 30)); d.text((x + 6, 5), lab, fill=(0, 0, 0), font=f)
    M.save(out_path, "JPEG", quality=88); return M.size

def build(M, editor, out, n, seed, P=None, cpath=None, mpath=None, qtext=None):
    S = sample_groups(M, n, seed)
    if len(S) < n: print(f"WARNING: only {len(S)} usable groups (asked {n})")
    for d in ("images", "montages"): os.makedirs(os.path.join(out, d), exist_ok=True)
    rnd = random.Random(seed + 1); codes = [f"A{i+1:03d}" for i in range(len(S))]; rnd.shuffle(codes)   # codes carry no information about the group
    key, items = [], []
    for g, code in zip(S, codes):
        rem, pla = edit_paths(g, editor, P, cpath, mpath)
        for p in (g["files"]["original"], rem, pla):
            if not os.path.exists(p): raise SystemExit(f"missing existing image file: {p}")
        removal_is_1 = rnd.random() < 0.5
        e1, e2 = (rem, pla) if removal_is_1 else (pla, rem)
        names = {"original": g["files"]["original"], "edit1": e1, "edit2": e2}
        for k, p in names.items(): shutil.copyfile(p, os.path.join(out, "images", f"{code}_{k}.png"))      # lossless copies, no re-encoding
        montage([names["original"], e1, e2], ["ORIGINAL", "EDIT 1", "EDIT 2"], os.path.join(out, "montages", f"{code}.jpg"))
        obj = g["cat"].replace("_", " ")
        items.append(dict(code=code, obj=obj, question=(qtext(g["cat"]) if qtext else f"Is there a {obj} in the image?")))
        key.append(dict(code=code, gid=g["gid"], img_id=g["img_id"], category=g["cat"], editor=editor, edit1_is="removal" if removal_is_1 else "placebo",
                        edit2_is="placebo" if removal_is_1 else "removal", original=g["files"]["original"], removal=rem, placebo=pla,
                        placebo_area=g.get("placebo_area"), remove_area=g.get("remove_area")))
    with open(os.path.join(out, "audit_KEY_do_not_share_with_reviewers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    return items, key

INSTR = [
 ("PEST-R edit-validity audit  -  reviewer instructions", ""),
 ("Status", "THIS IS A BLANK AUDIT FORM. No human audit has been completed by the person or system that built it. Results exist only after two reviewers return filled forms."),
 ("Purpose", "A research paper edits COCO photographs with an image editor to (i) REMOVE a named object and (ii) make a control edit of the same size and shape somewhere ELSE that should leave the named object visible. You check whether the edits worked. Please do not guess what the research hypothesis is."),
 ("Data disclosure", "All images are taken from the held-out TEST set of the study (not from training data), and all edits are the pre-existing test edits (editor: %s). The groups were sampled uniformly at random with fixed seed %d, at most one per source photo, without looking at any model output. Reviewers must not use any model to help judge."),
 ("What you see", "For every row: the ORIGINAL photo and two edited versions, EDIT 1 and EDIT 2 (shown in one strip in the 'Image' column; full-resolution PNG files are in the images/ folder under the names given in the 'Full-size files' column). One of the two edits is a removal and the other is a control edit - you are NOT told which, and their order is random. Judge each edit on its own."),
 ("Q1  Target visible in the ORIGINAL?", "Yes / No / Cannot tell. Is the named object (column 'Target object') present in the original photo? (A sanity check on the data.)"),
 ("Q2  Target visible in EDIT 1 / EDIT 2", "Clearly visible = you can recognise the named object at a glance.  Partly visible = only a part, or a remnant/ghost that a person could still recognise.  Not visible = no recognisable trace of the object remains.  Cannot tell = the image is too unclear to decide. Use only what you can see; do not infer from the edit's shape."),
 ("Q3  Artifacts in EDIT 1 / EDIT 2", "None = the edit looks natural.  Minor = a visible smear, texture, colour or edge mismatch, but the image is otherwise sensible.  Obvious failure = a flat patch, large distortion, an object that was added, a duplicated object, or anything that makes the edit clearly unusable."),
 ("Notes", "Free text. Write anything unusual (e.g. 'another object of the same kind is still visible', 'wrong object removed', 'image broken'). Leave blank if nothing."),
 ("Independence", "Work alone. Do not discuss rows with the other reviewer until both files are submitted. Fill ONLY the yellow columns. Do not reorder, delete or rename rows. Save as .xlsx and return the file."),
 ("Time", "About 1 to 2 minutes per row; plan about 2-3 hours for 100 rows. It is fine to work in several sittings."),
]

def write_workbooks(items, key, out, editor, seed):
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XImg
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.styles import PatternFill, Font, Alignment
    yellow = PatternFill("solid", fgColor="FFF2A8"); head = PatternFill("solid", fgColor="D9E2F3")
    for who, shuf in (("A", seed + 11), ("B", seed + 22)):
        order = list(range(len(items))); random.Random(shuf).shuffle(order)
        wb = Workbook(); ws0 = wb.active; ws0.title = "Instructions"; ws0.column_dimensions["A"].width = 34; ws0.column_dimensions["B"].width = 130
        for r, (a, b) in enumerate(INSTR, 1):
            b = b % (editor, seed) if "%s" in b else b
            ws0.cell(r, 1, a).font = Font(bold=True); c = ws0.cell(r, 2, b); c.alignment = Alignment(wrap_text=True, vertical="top"); ws0.cell(r, 1).alignment = Alignment(wrap_text=True, vertical="top")
        ws0.cell(len(INSTR) + 2, 1, "Reviewer").font = Font(bold=True); ws0.cell(len(INSTR) + 2, 2, "Reviewer %s  -  enter your name/initials here: ______" % who)
        ws = wb.create_sheet("Audit")
        cols = ["Row", "Item code", "Target object", "Image (ORIGINAL | EDIT 1 | EDIT 2)", "Full-size files", "Q1 Target in ORIGINAL?", "Q2 Edit 1: target visible?", "Q3 Edit 1: artifacts?",
                "Q2 Edit 2: target visible?", "Q3 Edit 2: artifacts?", "Notes"]
        widths = [6, 10, 16, 95, 34, 16, 20, 18, 20, 18, 40]
        for j, (c, w) in enumerate(zip(cols, widths), 1):
            cell = ws.cell(1, j, c); cell.font = Font(bold=True); cell.fill = head; cell.alignment = Alignment(wrap_text=True, vertical="center"); ws.column_dimensions[chr(64 + j)].width = w
        dvs = {"F": OPT_YN, "G": OPT_VIS, "H": OPT_ART, "I": OPT_VIS, "J": OPT_ART}
        for col, opts in dvs.items():
            dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"{col}2:{col}{len(items) + 1}")
        for r, i in enumerate(order, 2):
            it = items[i]; ws.row_dimensions[r].height = 175
            ws.cell(r, 1, r - 1); ws.cell(r, 2, it["code"]); ws.cell(r, 3, it["obj"])
            ws.cell(r, 5, f"images/{it['code']}_original.png\nimages/{it['code']}_edit1.png\nimages/{it['code']}_edit2.png")
            for j in (3, 5): ws.cell(r, j).alignment = Alignment(wrap_text=True, vertical="top")
            img = XImg(os.path.join(out, "montages", it["code"] + ".jpg")); sc = 650 / img.width; img.width, img.height = int(img.width * sc), int(img.height * sc); ws.add_image(img, f"D{r}")
            for j in range(6, 12): ws.cell(r, j).fill = yellow; ws.cell(r, j).alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "D2"
        wb.save(os.path.join(out, f"reviewer_{who}.xlsx"))

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return p, c - h, c + h

def kappa(a, b):
    labs = sorted(set(a) | set(b)); n = len(a)
    if n == 0: return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n; pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

def analyse(out, fileA=None, fileB=None):
    """RUN ONLY AFTER both reviewers returned their filled forms. Unblinds with the key, reports rates, 95% Wilson CIs, agreement."""
    from openpyxl import load_workbook
    fileA = fileA or os.path.join(out, "reviewer_A_FILLED.xlsx"); fileB = fileB or os.path.join(out, "reviewer_B_FILLED.xlsx")
    key = {r["code"]: r for r in csv.DictReader(open(os.path.join(out, "audit_KEY_do_not_share_with_reviewers.csv")))}
    def read(f):
        ws = load_workbook(f)["Audit"]; R = {}
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[1]: R[row[1]] = dict(orig=row[5], v1=row[6], a1=row[7], v2=row[8], a2=row[9], note=row[10])
        return R
    RA, RB = read(fileA), read(fileB); lines = []
    def pr(s): print(s); lines.append(s)
    def per_item(R):
        o = {}
        for c, r in R.items():
            k = key[c]; rem1 = k["edit1_is"] == "removal"
            vr, ar = (r["v1"], r["a1"]) if rem1 else (r["v2"], r["a2"]); vp, ap = (r["v2"], r["a2"]) if rem1 else (r["v1"], r["a1"])
            nn = lambda x: x is not None and str(x).strip() != ""
            o[c] = dict(removed=(vr == "Not visible") if nn(vr) else None, preserved=(vp == "Clearly visible") if nn(vp) else None,
                        art_rem=(ar in ("Minor", "Obvious failure")) if nn(ar) else None, art_pla=(ap in ("Minor", "Obvious failure")) if nn(ap) else None,
                        fail_rem=(ar == "Obvious failure") if nn(ar) else None, fail_pla=(ap == "Obvious failure") if nn(ap) else None)
        return o
    IA, IB = per_item(RA), per_item(RB); codes = sorted(set(IA) & set(IB))
    pr(f"Editor: {next(iter(key.values()))['editor']}   items in both files: {len(codes)}   (completeness is counted separately for every metric)")
    for name, f in (("(a) target removed in the removal edit", "removed"), ("(b) target preserved in the placebo edit", "preserved"), ("(c1) any artifact in the removal edit", "art_rem"),
                    ("(c2) any artifact in the placebo edit", "art_pla"), ("(c3) obvious failure, removal edit", "fail_rem"), ("(c4) obvious failure, placebo edit", "fail_pla")):
        for who, I in (("A", IA), ("B", IB)):
            ok = [I[c][f] for c in codes if I[c][f] is not None]; p, lo, hi = wilson(sum(ok), len(ok))
            pr(f"  {name:<44} reviewer {who}: {sum(ok)}/{len(ok)} = {p:.3f} [{lo:.3f},{hi:.3f}]   (answered {len(ok)}/{len(codes)})")
        pair = [c for c in codes if IA[c][f] is not None and IB[c][f] is not None]
        both = [IA[c][f] and IB[c][f] for c in pair]; p, lo, hi = wilson(sum(both), len(pair))
        k = kappa([IA[c][f] for c in pair], [IB[c][f] for c in pair])
        pr(f"  {'':<44} BOTH yes: {sum(both)}/{len(pair)} = {p:.3f} [{lo:.3f},{hi:.3f}]   kappa = {k:.2f}   (rated by both: {len(pair)}/{len(codes)})")
    pr("Disagreements and notes to adjudicate (item code: A vs B):")
    for c in codes:
        for f in ("removed", "preserved"):
            if IA[c][f] is not None and IB[c][f] is not None and IA[c][f] != IB[c][f]: pr(f"  {c} {f}: A={IA[c][f]} B={IB[c][f]}")
    open(os.path.join(out, "audit_analysis.txt"), "w").write("\n".join(lines)); return lines

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyse":
        analyse(OUT); raise SystemExit
    from pest_common import *
    try:
        import pest_study as PS
    except Exception as e:
        PS = None; print('pest_study not importable (fine for editor sd/lama):', e)
    Gf, M, D = load_matched()
    if os.path.exists(OUT): raise SystemExit(f"{OUT} already exists - refusing to overwrite. Set AUDIT_OUT to a new folder.")
    items, key = build(M, EDITOR, OUT, N_GROUPS, SAMPLE_SEED, cpath=getattr(PS, 'cpath', None), mpath=getattr(PS, 'mpath', None), qtext=None)
    write_workbooks(items, key, OUT, EDITOR, SAMPLE_SEED)
    cnt = collections.Counter(k["category"] for k in key)
    print(f"built {len(items)} items in {OUT}"); print("category mix:", dict(cnt.most_common()))
    print("files: reviewer_A.xlsx, reviewer_B.xlsx (give one to each reviewer, with the images/ folder),\n       audit_KEY_do_not_share_with_reviewers.csv (keep yourself)")
