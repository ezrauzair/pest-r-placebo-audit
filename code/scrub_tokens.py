#!/usr/bin/env python3
"""Find (and optionally remove) leaked Hugging Face tokens in your old notebooks and scripts. Runs on YOUR computer, no internet.
   python scrub_tokens.py  FOLDER            -> report only (file, line, masked preview). Never prints a full token.
   python scrub_tokens.py  FOLDER --fix      -> writes a cleaned COPY next to each affected file (name_CLEAN.ext). Originals are not touched.
Then: (1) revoke the old tokens at huggingface.co/settings/tokens (this is the step that actually protects you),
      (2) replace the originals with the _CLEAN copies, (3) re-run this script until it reports nothing."""
import os, re, sys, json
PAT = re.compile(r"hf_[A-Za-z0-9]{20,}")                                              # Hugging Face token shape
ASSIGN = re.compile(r"""((?:HF_TOKEN|hf_token|tok|token|TOKEN)\s*=\s*)(\\?["'])([A-Za-z0-9_\-]{20,})\2""")   # token = "long string"
EXT = (".ipynb", ".py", ".txt", ".md", ".json", ".sh", ".yaml", ".yml", ".cfg", ".env")
def mask(t): return t[:3] + "*" * 6 + t[-2:]
def clean_text(s):
    s = PAT.sub("", s); return ASSIGN.sub(lambda m: m.group(1) + m.group(2) + m.group(2), s)
def scan(path):
    hits = []
    try: txt = open(path, encoding="utf8", errors="ignore").read()
    except Exception: return hits
    for n, line in enumerate(txt.splitlines(), 1):
        for m in PAT.finditer(line): hits.append((n, mask(m.group(0))))
        for m in ASSIGN.finditer(line): hits.append((n, mask(m.group(3))))
    return hits
def main():
    root = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "."
    fix = "--fix" in sys.argv; found = 0
    for d, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(EXT) or "_CLEAN" in f: continue
            p = os.path.join(d, f); h = scan(p)
            if not h: continue
            found += 1; print(f"LEAK  {p}: {len(h)} hit(s), lines {sorted({n for n, _ in h})[:8]}   e.g. {h[0][1]}")
            if fix:
                s = open(p, encoding="utf8", errors="ignore").read(); c = clean_text(s)
                if p.lower().endswith(".ipynb"):
                    try: json.loads(c)
                    except Exception: print("      cleaned notebook is not valid JSON - not written; clean this one by hand"); continue
                q = os.path.splitext(p)[0] + "_CLEAN" + os.path.splitext(p)[1]; open(q, "w", encoding="utf8").write(c); print("      wrote", q)
    print(f"\n{found} affected file(s)." + ("" if found else " Nothing found. Still revoke any token you ever pasted into a chat."))
if __name__ == "__main__": main()
