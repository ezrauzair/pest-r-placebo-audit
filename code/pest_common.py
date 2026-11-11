#!/usr/bin/env python
"""
PEST-R v4 pipeline: does fine-tuning on inpainting-based counterfactual edits teach a VLM an
"edit artifact => object absent" shortcut, and does a placebo edit remove it?

Design principles (so that nothing has to be redone):
  * Every stage is a separate process that reads its inputs from disk and writes its outputs to disk.
    There is no state in a notebook kernel. Restarting the kernel, or crashing, loses nothing.
  * Every stage is idempotent and resumable: finished outputs are detected and skipped, and the
    slow loops skip every item whose output file already exists.
  * Train/val/test are separated. Thresholds are frozen in the config below. The test set is never used for tuning.
  * Models stage runs on both GPUs at once, with batched evaluation and saved LoRA adapters.
  * Statistics support multiple training seeds, paired bootstrap over groups, AUROC, and a paired probe analysis.

v4 changes vs v3: dual independent detectors (OWL-ViT + Faster R-CNN) gate every edit; crop-based Stable Diffusion with
one identical prompt for relevant and placebo edits; a DEV part of the test pool for looking at images and a untouched FINAL part;
categories dropped by an edit-quality rule (never by model results); probes trained per editor; bootstrap clustered by source image.
"""
import os, sys, json, random, time, gc, re, math, shutil, hashlib, collections, glob, argparse, subprocess, traceback

# ============================== CONFIG (frozen before any test-set result is looked at) ==============================
WORK = os.environ.get("PEST_WORK", "/kaggle/working/pest_v4")
RESUME_FROM = None            # e.g. "/kaggle/input/<earlier-run-output>/pest" to reuse a finished run
SEED = 0
CATS = ["cat", "dog", "bed", "train", "bus", "tv", "laptop", "couch", "bicycle", "motorcycle", "airplane", "giraffe"]
TRAIN_CAND_PER_CAT, TEST_CAND_PER_CAT, NAT_PER_CAT = 80, 150, 20   # test: take (almost) every eligible single-instance val image
MIN_FRAC, MAX_FRAC = 0.02, 0.25          # target mask area as a fraction of the image
DILATE_PX, COMPOSITE = 15, True          # one value, used identically for relevant and placebo edits
SD_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SD_STEPS, SD_GUIDANCE, SD_STRENGTH = 30, 5.0, 0.95
DEV_FRAC = 0.15                          # share of test-pool images reserved as DEV (images may be inspected); the rest is the untouched FINAL test
# independent-detector (OWL-ViT) rules
ORIG_MIN, REMOVE_MAX_ABS, REMOVE_MAX_REL, PLACEBO_MIN_REL, NEG_MAX = 0.12, 0.10, 0.5, 0.7, 0.08
# second, independent detector (torchvision Faster R-CNN v2): BOTH detectors must agree for an edit to count as valid
ORIG2_MIN, REMOVE2_MAX, PLACEBO2_MIN_REL = 0.50, 0.20, 0.7
# a category is dropped if too few of its edits are valid (decided from detector pass rates only, never from model results)
CAT_MIN_PASS, CAT_MIN_N = 0.25, 15
AUDIT_N = 40
POPE_PER_SPLIT, POPE_EVAL_PER_SPLIT = 500, 300
VAL_FRAC = 0.15                           # share of passing train groups held out as validation (never trained on)
# training: QLoRA on LLaVA-1.5-7B
EPOCHS, LR, ACCUM, LORA_R = 2, 1e-4, 8, 16
CONDS = ["A", "B", "C", "D", "E"]
#  A frozen | B standard counterfactual | C standard, matched to D in examples | D placebo-aware | E standard + real unedited negatives
SEEDS = {"A": [0], "B": [0], "C": [0], "D": [0], "E": [0]}     # raise C and D (and B) to [0, 1, 2] once the pilot looks promising
JOB_ORDER = ["A", "B", "D", "C", "E"]
EVAL_BS = 8
BOOT = 2000
LABELS = {"A": "A frozen", "B": "B standard CF", "C": "C standard (=D size)", "D": "D placebo-aware", "E": "E std + real negs"}
# =====================================================================================================================

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def P(*a): return os.path.join(WORK, *a)


def jload(p):
    with open(p) as f: return json.load(f)


def jsave(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f: json.dump(o, f)
    os.replace(tmp, p)                      # atomic: a crash can never leave a half-written json


TAG = ""
def set_tag(t):
    global TAG; TAG = t
def log(*a): print(time.strftime("%H:%M:%S"), TAG, *a, flush=True)


def prog(i, n, t0, tag, every=50):
    if i % every == 0 or i == n:
        el = time.time() - t0; eta = el / max(i, 1) * (n - i)
        log(f"[{tag}] {i}/{n}   elapsed {el/60:.1f} min   eta {eta/60:.1f} min")


def Q(cat): return f"Is there a {cat} in the image?"


# ------------------------------------------------------------------ COCO / POPE ------------------------------------------------------------------
def find_coco_root():
    pref = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017"
    if os.path.exists(f"{pref}/annotations/instances_train2017.json"): return pref
    for k in range(1, 6):
        hits = glob.glob("/kaggle/input/" + "*/" * k + "annotations/instances_train2017.json")
        if hits: return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("COCO 2017 not found under /kaggle/input. Attach the dataset awsaf49/coco-2017-dataset.")


class Coco:
    def __init__(self):
        from pycocotools.coco import COCO
        root = find_coco_root(); log("COCO root:", root)
        self.coco = {"train": COCO(f"{root}/annotations/instances_train2017.json"),
                     "val": COCO(f"{root}/annotations/instances_val2017.json")}
        self.img_dir = {"train": f"{root}/train2017", "val": f"{root}/val2017"}
        self.name2id = {c["name"]: c["id"] for c in self.coco["val"].loadCats(self.coco["val"].getCatIds())}

    def path(self, split, img_id): return f"{self.img_dir[split]}/{self.coco[split].imgs[img_id]['file_name']}"


def get_pope_ids():
    f = P("pope_ids.json")
    if os.path.exists(f): return set(jload(f))
    from datasets import load_dataset
    srcs = load_dataset("lmms-lab/POPE")["test"].select_columns(["image_source"])["image_source"]
    ids = sorted({int(re.findall(r"\d{6,}", s)[-1]) for s in srcs})
    jsave(f, ids); return set(ids)


# ------------------------------------------------------------------ independent detector ------------------------------------------------------------------
class Det:
    """OWL-ViT used only to filter groups and verify labels; scores are cached on disk and the model loads lazily."""
    def __init__(self):
        self.f = P("det_cache.json")
        self.cache = jload(self.f) if os.path.exists(self.f) else {}
        self.pipe, self.new = None, 0

    def score(self, path, cat):
        from PIL import Image
        k = f"{path}|{cat}"
        if k not in self.cache:
            if self.pipe is None:
                from transformers import pipeline
                self.pipe = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32", device=0)
            out = self.pipe(Image.open(path).convert("RGB"), candidate_labels=[f"a photo of a {cat}"], threshold=0.01)
            self.cache[k] = max([o["score"] for o in out], default=0.0)
            self.new += 1
            if self.new % 200 == 0: self.save()
        return self.cache[k]

    def save(self): jsave(self.f, self.cache)

    def free(self):
        self.save(); self.pipe = None; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass


def ok_rule(o, rm, pl):
    """Detector sees the target in the original, no longer sees it after removal, and still sees it after the placebo edit."""
    return bool(o >= ORIG_MIN and rm <= min(REMOVE_MAX_ABS, REMOVE_MAX_REL * o) and pl >= PLACEBO_MIN_REL * o)


# ------------------------------------------------------------------ image editing helpers ------------------------------------------------------------------
def dilate(m, px):
    import cv2, numpy as np
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m.astype(np.uint8), k)


def placebo_mask(target_mask, r, tries=300):
    """Same-shape copy of the target mask, moved to a spot clear of the target (area and shape match exactly)."""
    import numpy as np
    ys, xs = np.nonzero(target_mask)
    h, w = target_mask.shape
    forbid = dilate(target_mask, 2 * DILATE_PX)
    for _ in range(tries):
        dy = r.randint(-int(ys.min()), h - 1 - int(ys.max()))
        dx = r.randint(-int(xs.min()), w - 1 - int(xs.max()))
        m = np.zeros((h, w), dtype=np.uint8)
        m[ys + dy, xs + dx] = 1
        if (m & forbid).sum() == 0: return m
    return None


def composite(orig, edited, m):
    import cv2, numpy as np
    from PIL import Image
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3)
    soft = np.maximum(soft, m.astype(np.float32))[..., None]
    o = np.asarray(orig).astype(np.float32)
    e = np.asarray(edited.resize(orig.size)).astype(np.float32)
    return Image.fromarray((o * (1 - soft) + e * soft).round().astype(np.uint8))


def mask_area(path):
    import numpy as np
    from PIL import Image
    return int((np.array(Image.open(path)) > 0).sum())


LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_CKPT = "/tmp/big-lama.pt"


class SimpleLama:
    """Self-contained LaMa (same checkpoint and padding logic as the simple-lama-inpainting package, without its numpy/pillow pins)."""
    def __init__(self):
        import torch
        if not os.path.exists(LAMA_CKPT): torch.hub.download_url_to_file(LAMA_URL, LAMA_CKPT)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(LAMA_CKPT, map_location=self.device).eval().to(self.device)

    def __call__(self, image, mask):
        import numpy as np, torch
        from PIL import Image
        img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        msk = (np.asarray(mask.convert("L")) > 0).astype(np.float32)
        H, W = msk.shape; ph, pw = (-H) % 8, (-W) % 8
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        msk = np.pad(msk, ((0, ph), (0, pw)), mode="symmetric")
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(self.device)
        m = torch.from_numpy(msk)[None, None].to(self.device)
        with torch.inference_mode():
            out = self.model(x, m)[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8)[:H, :W])


# ============================================================ v4 additions ============================================================
def part_of(img_id):
    """DEV (may be inspected, used to sanity-check the edit procedure) vs FINAL test (never looked at before the frozen run)."""
    return "dev" if int(hashlib.md5(f"dev{img_id}".encode()).hexdigest(), 16) % 100 < int(DEV_FRAC * 100) else "test"


class Det2:
    """Second, independent detector: torchvision Faster R-CNN v2 (COCO). Scores cached on disk; loads lazily."""
    def __init__(self):
        self.f = P("det2_cache.json")
        self.cache = jload(self.f) if os.path.exists(self.f) else {}
        self.model, self.new = None, 0

    def _load(self):
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.names = {n: i for i, n in enumerate(w.meta["categories"])}
        self.model = fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.05).eval().to("cuda")

    def score(self, path, cat):
        k = f"{path}|{cat}"
        if k not in self.cache:
            import torch
            from PIL import Image
            from torchvision.transforms.functional import to_tensor
            if self.model is None: self._load()
            with torch.inference_mode():
                out = self.model([to_tensor(Image.open(path).convert("RGB")).to("cuda")])[0]
            lab = self.names[cat]
            self.cache[k] = max([float(s) for s, l in zip(out["scores"], out["labels"]) if int(l) == lab], default=0.0)
            self.new += 1
            if self.new % 200 == 0: self.save()
        return self.cache[k]

    def save(self): jsave(self.f, self.cache)

    def free(self):
        self.save(); self.model = None; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass


def ok2(o, rm, pl):
    return bool(o >= ORIG2_MIN and rm <= REMOVE2_MAX and pl >= PLACEBO2_MIN_REL * o)


def ok_dual(s1, s2, a, b):
    """s1 = OWL-ViT scores dict, s2 = Faster R-CNN scores dict; a, b = keys of the removed and the placebo image."""
    return ok_rule(s1["original"], s1[a], s1[b]) and ok2(s2["original"], s2[a], s2[b])


# Fixed in advance and identical for the relevant and the placebo edit (so the two edits differ only in WHERE they are, not in HOW they are made).
BG_PROMPT = {
    "cat": "Floor, fabric or grass, matching the nearby background.",
    "dog": "Grass, ground or floor, matching the nearby background.",
    "bed": "Bedroom wall, floor and furniture, matching the nearby background.",
    "train": "Continuous railway tracks, gravel and distant trees.",
    "bus": "Empty paved road, buildings and trees in the background.",
    "tv": "Plain wall and furniture, matching the nearby background.",
    "laptop": "Plain tabletop or desk surface, matching the nearby background.",
    "couch": "Plain wall and floor of a living room, matching the nearby background.",
    "bicycle": "Pavement, street and background scenery, matching the nearby background.",
    "motorcycle": "Empty road surface and background scenery, matching the nearby background.",
    "airplane": "Open sky with soft clouds, matching the nearby background.",
    "giraffe": "Natural grass, shrubs and distant vegetation.",
}
TARGET_TERMS = {
    "cat": "cat, kitten, animal", "dog": "dog, puppy, animal", "bed": "bed, mattress, pillow, blanket",
    "train": "train, locomotive, railcar, railway vehicle", "bus": "bus, coach, vehicle", "tv": "television, tv, monitor, screen",
    "laptop": "laptop, notebook computer, computer screen", "couch": "couch, sofa, cushions", "bicycle": "bicycle, bike, wheels",
    "motorcycle": "motorcycle, motorbike, scooter", "airplane": "airplane, aeroplane, aircraft, jet, flying vehicle",
    "giraffe": "giraffe, long-necked animal, wildlife",
}
GENERIC_NEG = "text, lettering, watermark, logo, duplicated objects, new foreground objects, distorted geometry, cartoon, illustration"


def crop_box(m, w, h):
    import numpy as np
    ys, xs = np.nonzero(m)
    bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    mx, my = max(40, int(bw * 0.9)), max(40, int(bh * 0.9))
    return (max(0, int(xs.min()) - mx), max(0, int(ys.min()) - my), min(w, int(xs.max()) + mx + 1), min(h, int(ys.max()) + my + 1))


def square_inputs(image, mask, size=512):
    from PIL import Image
    w, h = image.size
    sc = min(size / w, size / h); nw, nh = max(1, round(w * sc)), max(1, round(h * sc))
    left, top = (size - nw) // 2, (size - nh) // 2
    ic = Image.new("RGB", (size, size)); ic.paste(image.resize((nw, nh), Image.LANCZOS), (left, top))
    mc = Image.new("L", (size, size), 0); mc.paste(mask.resize((nw, nh), Image.NEAREST), (left, top))
    return ic, mc, (left, top, nw, nh)


def sd_crop_edit(sd, img, m, cat, seed):
    """Inpaint the mask on a crop around it (SD works at 512 px), then paste the crop back with the same soft-edge composite as LaMa."""
    import numpy as np, torch
    from PIL import Image
    w, h = img.size
    box = crop_box(m, w, h); x1, y1, x2, y2 = box
    crop = img.crop(box); cmask = Image.fromarray((m[y1:y2, x1:x2] > 0).astype(np.uint8) * 255)
    ic, mc, (left, top, nw, nh) = square_inputs(crop, cmask)
    with torch.inference_mode():
        gen = sd(prompt=BG_PROMPT[cat], negative_prompt=GENERIC_NEG + ", " + TARGET_TERMS[cat], image=ic, mask_image=mc,
                 num_inference_steps=SD_STEPS, guidance_scale=SD_GUIDANCE, strength=SD_STRENGTH,
                 generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    gen = gen.crop((left, top, left + nw, top + nh)).resize((x2 - x1, y2 - y1), Image.LANCZOS)
    full = img.copy(); full.paste(gen, (x1, y1))
    mfull = (m > 0).astype(np.uint8)
    return composite(img, full, mfull) if COMPOSITE else full


def load_matched():
    """(all groups, FINAL matched test groups, DEV matched groups). Matched = valid for BOTH editors (dual-detector) and in a kept category."""
    Gf = jload(P("groups_final.json")); keep = set(jload(P("final_cats.json"))["keep"])
    ok = lambda g: g["ok_lama"] and g.get("ok_sd") and g["cat"] in keep
    return Gf, [g for g in Gf["test"] if ok(g) and g["part"] == "test"], [g for g in Gf["test"] if ok(g) and g["part"] == "dev"]
