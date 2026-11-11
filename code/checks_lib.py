# Helper functions used by pest_checks.ipynb (Kaggle, Internet ON, CPU). Read-only: nothing is trained, nothing is overwritten.
import re, io, json, time, difflib, urllib.request, urllib.parse

UA = {"User-Agent": "pest-r-checks/1.0"}                    # generic client name, no personal contact data
def http(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()

# ---------------------------------------------------------------- 1) literature: grep the ORIGINAL PDFs, appendices included
PAPERS = {
 "ROHE / oDPO (NAACL 2025)":      "https://aclanthology.org/2025.naacl-long.349.pdf",
 "POPEv2 / Obliviate (2508.04567)": "https://arxiv.org/pdf/2508.04567",
 "HII-DPO (2602.10425)":          "https://arxiv.org/pdf/2602.10425",
 "HalluSegBench (2506.21546)":    "https://arxiv.org/pdf/2506.21546",
}
KEYS = ["placebo", "control edit", "irrelevant region", "unrelated region", "random region", "random location", "different location", "other region",
        "edit artifact", "editing artifact", "artifact", "shortcut", "spurious", "edited image", "edit cue", "inpainting trace", "same size", "same shape",
        "sanity check", "negative control", "confound", "learn to detect", "tell whether"]
def pdf_text(url):
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(http(url, 120))); return [(i + 1, (p.extract_text() or "")) for i, p in enumerate(r.pages)]
def grep_papers(papers=PAPERS, keys=KEYS, ctx=170):
    out = []
    for name, url in papers.items():
        out.append("=" * 110); out.append(name + "\n" + url)
        try: pages = pdf_text(url)
        except Exception as e: out.append(f"   COULD NOT FETCH/READ: {e}\n   -> open the PDF yourself and search for these words: {', '.join(keys[:8])}"); continue
        full = "\n".join(t for _, t in pages)
        app = [n for n, t in pages if re.search(r"^\s*(appendix|a\s+appendix|supplementary)", t, re.I | re.M)]
        out.append(f"   pages: {len(pages)}   characters: {len(full)}   pages that start an appendix/supplement: {app or 'none detected'}")
        if len(full) < 5000: out.append("   WARNING: very little text extracted (scanned or protected PDF?) - search it by hand.")
        for k in keys:
            hits = [(n, m.start()) for n, t in pages for m in re.finditer(re.escape(k), t, re.I)]
            if not hits: continue
            out.append(f"   [{k}]  {len(hits)} hit(s)")
            for n, _ in hits[:4]:
                t = dict(pages)[n]; m = re.search(re.escape(k), t, re.I)
                out.append(f"      p.{n}: ..." + re.sub(r"\s+", " ", t[max(0, m.start() - ctx): m.end() + ctx]) + "...")
    return "\n".join(out)

# ---------------------------------------------------------------- 2) references: compare every BibTeX entry with Crossref / arXiv
def _field(body, name):
    m = re.search(r"\b" + name + r"\s*=\s*\{", body, re.I)
    if not m: return ""
    i, depth = m.end(), 1
    while i < len(body) and depth:
        depth += (body[i] == "{") - (body[i] == "}"); i += 1
    return body[m.end(): i - 1]
def parse_bib(txt):
    ents = []
    for chunk in re.split(r"(?m)^(?=@\w+\{)", txt):
        m = re.match(r"@(\w+)\{([^,]+),", chunk)
        if not m: continue
        body = chunk[m.end():]; g = lambda f: re.sub(r"[{}]", "", _field(body, f)).strip()
        ents.append(dict(key=m.group(2).strip(), type=m.group(1), title=g("title"), year=g("year"), venue=g("journal") or g("booktitle"), author=g("author"), body=body))
    return ents
def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower())
def check_ref(e):
    t = e["title"]; res = {"key": e["key"], "bib_title": t, "bib_year": e["year"], "bib_venue": e["venue"]}
    am = re.search(r"arXiv:(\d{4}\.\d{4,5})", e["body"])
    try:
        if am and "arxiv" in e["venue"].lower():
            x = http("https://export.arxiv.org/api/query?id_list=" + am.group(1)).decode("utf8", "ignore")
            mt = re.search(r"<title>(.*?)</title>.*?<published>(\d{4})", x.split("<entry>", 1)[1], re.S)
            res.update(found_title=re.sub(r"\s+", " ", mt.group(1)).strip(), found_year=mt.group(2), found_venue="arXiv " + am.group(1), src="arXiv")
        else:
            j = json.loads(http("https://api.crossref.org/works?rows=2&select=title,issued,container-title,author&query.bibliographic=" + urllib.parse.quote(t + " " + e["author"].split(" and ")[0])))
            it = j["message"]["items"][0]
            res.update(found_title=(it.get("title") or [""])[0], found_year=str((it.get("issued", {}).get("date-parts") or [[""]])[0][0]),
                       found_venue=(it.get("container-title") or [""])[0], src="Crossref")
    except Exception as ex:
        res.update(found_title="", found_year="", found_venue="", src=f"lookup failed: {ex}")
    sim = difflib.SequenceMatcher(None, norm(t), norm(res["found_title"])).ratio() if res["found_title"] else 0
    ydiff = abs(int(res["found_year"]) - int(e["year"])) if res["found_year"].isdigit() and e["year"].isdigit() else None
    res["similarity"] = round(sim, 2)
    res["flag"] = "OK" if sim >= 0.9 and (ydiff is None or ydiff <= 1) else ("CHECK BY HAND (title or year differs)" if res["found_title"] else "NOT FOUND - check by hand")
    return res
def check_all(bibtxt, pause=0.4):
    out = []
    for e in parse_bib(bibtxt): out.append(check_ref(e)); time.sleep(pause)
    return out

# ---------------------------------------------------------------- 3) licences of the models actually used (from the Hugging Face model cards)
MODELS = ["llava-hf/llava-1.5-7b-hf", "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "HuggingFaceTB/SmolVLM-Instruct",
          "google/owlvit-base-patch32", "stable-diffusion-v1-5/stable-diffusion-inpainting"]
def licences(token=None):
    from huggingface_hub import model_info
    out = []
    for m in MODELS:
        try:
            i = model_info(m, token=token); cd = i.card_data.to_dict() if i.card_data else {}
            out.append((m, str(cd.get("license")), str(cd.get("license_name") or ""), str(cd.get("license_link") or ""), "gated" if i.gated else "open"))
        except Exception as e: out.append((m, "LOOKUP FAILED", str(e)[:80], "", ""))
    return out
