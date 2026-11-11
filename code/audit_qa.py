#!/usr/bin/env python3
"""QA of the audit package BEFORE it goes to reviewers. Read-only except for one new file (qa_contact_sheet.jpg).
   python audit_qa.py  PATH_TO_pest_audit_sd      (folder that holds reviewer_A.xlsx, reviewer_B.xlsx, images/, montages/, the KEY csv)
Needs: openpyxl, pillow. Prints PASS/FAIL lines; open qa_contact_sheet.jpg yourself and look at it."""
import os, sys, csv, random, re
from openpyxl import load_workbook
from PIL import Image
root = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/pest_audit_sd"
ok = True
def chk(cond, msg):
    global ok; ok &= bool(cond); print(("PASS  " if cond else "FAIL  ") + msg)
key = list(csv.DictReader(open(os.path.join(root, "audit_KEY_do_not_share_with_reviewers.csv"))))
codes = [k["code"] for k in key]; n = len(codes)
chk(n >= int(os.environ.get("AUDIT_N", 100)) * 0.9, f"key has {n} items (expected about {os.environ.get('AUDIT_N', 100)}); codes unique: {len(set(codes)) == n}")
chk(len({k['img_id'] for k in key}) == n, "one group per source photo")
chk(all(k["edit1_is"] != k["edit2_is"] for k in key), "every item has one removal and one placebo edit")
r1 = sum(k["edit1_is"] == "removal" for k in key); chk(abs(r1 - n / 2) <= 0.2 * n + 3, f"removal is Edit 1 in {r1}/{n} items (should be near half, so position gives nothing away)")
miss = [c for c in codes for s in ("original", "edit1", "edit2") if not os.path.exists(os.path.join(root, "images", f"{c}_{s}.png"))]
chk(not miss, f"all {3*n} full-size PNGs present" + ("" if not miss else f" (missing e.g. {miss[:3]})"))
chk(all(os.path.exists(os.path.join(root, "montages", c + ".jpg")) for c in codes), "all montages present")
sizes = [Image.open(os.path.join(root, "images", f"{c}_original.png")).size for c in codes[:20]]
chk(all(w >= 200 and h >= 200 for w, h in sizes), f"images open and look full size (e.g. {sizes[0]})")
secret = [k["gid"] for k in key] + [k["removal"] for k in key[:5]]
orders = []
for who in "AB":
    p = os.path.join(root, f"reviewer_{who}.xlsx"); wb = load_workbook(p)
    chk(set(wb.sheetnames) == {"Instructions", "Audit"}, f"reviewer_{who}: sheets {wb.sheetnames}")
    ws = wb["Audit"]; rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[1]]
    chk(len(rows) == n, f"reviewer_{who}: {len(rows)} item rows; embedded images: {len(ws._images)}")
    chk(len(ws._images) == n, f"reviewer_{who}: one picture per row")
    dv = [str(d.sqref) for d in ws.data_validations.dataValidation]
    chk(len(dv) == 5, f"reviewer_{who}: 5 dropdown columns {dv}")
    blank = all(all(c is None for c in r[5:11]) for r in rows); chk(blank, f"reviewer_{who}: answer columns are blank")
    text = " ".join(str(c) for r in rows for c in r[:5] if c is not None) + " ".join(str(c.value) for row in wb["Instructions"].iter_rows() for c in row if c.value)
    leak = [w for w in ("removal", "placebo") if re.search(w, " ".join(str(c) for r in rows for c in r[:5] if c is not None), re.I)]
    chk(not leak, f"reviewer_{who}: item rows do not say which edit is which (found words: {leak or 'none'})")
    chk(not any(g in text for g in secret), f"reviewer_{who}: no group ids or source paths in the workbook")
    orders.append([r[1] for r in rows])
chk(orders[0] != orders[1], "the two reviewers see different row orders")
chk("audit_KEY" not in " ".join(os.listdir(os.path.join(root, "images"))), "no key file inside images/")
print("\nFilenames inside images/ look like:", sorted(os.listdir(os.path.join(root, "images")))[:3], "(code + original/edit1/edit2 only - good)")
# contact sheet: 6 random items, enlarge to eyeball the edits
random.seed(1); pick = random.sample(codes, 6); ims = [Image.open(os.path.join(root, "montages", c + ".jpg")) for c in pick]
W = max(i.width for i in ims); H = sum(i.height for i in ims); sheet = Image.new("RGB", (W, H), "white"); y = 0
for i in ims: sheet.paste(i, (0, y)); y += i.height
sheet.save(os.path.join(root, "qa_contact_sheet.jpg"), quality=88); print("wrote", os.path.join(root, "qa_contact_sheet.jpg"), "- OPEN IT and check: object visibly gone in one edit, still visible in the other, no broken images")
print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED - do not send to reviewers yet")
