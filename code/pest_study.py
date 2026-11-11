import sys; sys.path.insert(0, "/kaggle/working")
from pest_common import *
import pest_models as PM

# =====================================================================================================================================
# PEST-R FULL STUDY. Question: does fine-tuning on counterfactual edits teach an "edited => object absent" shortcut, how does its size
# depend on how DETECTABLE the edits are, and does a placebo edit (same-shape edit elsewhere, answer stays Yes) prevent it?
#   edit LEVELS : crude  (flat colour patch, hard edge)  |  medium (Telea smear, feathered edge)  |  clean (LaMa + soft blend; test also on held-out SD)
#   conditions  : S = standard (original Yes, removed No)  |  P = placebo-aware (S + placebo-edited Yes)  |  A = frozen model
#   models      : LLaVA-1.5-7B and Qwen2.5-VL-3B (both 4-bit QLoRA);   seeds: 0, 1, 2
# =====================================================================================================================================
MODELS = ["llava", "qwen", "smol", "q7"]          # smol = the third model (SmolVLM-Instruct, 2.3B); its jobs live in third_jobs()
LEVELS = ["crude", "medium", "clean"]
STUDY_SEEDS = [0, 1, 2]
POPE_N = 150                                    # POPE questions per split used for evaluation (3 splits)
QID = "Qwen/Qwen2.5-VL-3B-Instruct"
QID7 = "Qwen/Qwen2.5-VL-7B-Instruct"          # the larger model (same family as qwen, 7B instead of 3B): does the result hold at scale?
SID = "HuggingFaceTB/SmolVLM-Instruct"        # third model: SigLIP vision encoder + SmolLM2-1.7B, Apache-2.0, native transformers (no remote code)
SMOL_LONGEST = 384                            # one 384-px image, no tiling (about 64-81 visual tokens): fast on a T4
QMIN_PIX, QMAX_PIX = 128 * 28 * 28, 320 * 28 * 28   # at most ~320 visual tokens per image (keeps a T4 happy)
EST_DEFAULT = {("llava", "S"): 70, ("llava", "P"): 90, ("llava", "A"): 30, ("qwen", "S"): 40, ("qwen", "P"): 50, ("qwen", "A"): 15, ("llava", "N"): 70, ("qwen", "N"): 40,
               ("smol", "A"): 15, ("smol", "S"): 35, ("smol", "P"): 45, ("smol", "N"): 35,
               ("q7", "A"): 35, ("q7", "S"): 95, ("q7", "P"): 115, ("q7", "N"): 95}   # minutes (first guess; replaced by measured times)

# --------------------------------------------------------------------------------- edits
def cpath(gid, kind): return P("crude", gid, f"{kind}.png")
def mpath(gid, kind): return P("medium", gid, f"{kind}.png")


def xpath(code, gid, kind): return P("mech_" + code, gid, f"{kind}.png")          # mechanism-test edits: code "ff" = crude flat fill with a feathered edge, "fn" = crude flat fill with noise texture


def crude_fill(img, m):
    import numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB")).copy(); mb = m > 0
    ring = (dilate(mb.astype(np.uint8), 12) > 0) & ~mb
    col = a[ring].mean(0) if ring.any() else a.reshape(-1, 3).mean(0)
    a[mb] = np.round(col).astype(np.uint8)
    return Image.fromarray(a)


def medium_fill(img, m):
    """Classical Telea inpainting (blurry smear), blended into the image with the same soft edge as the clean LaMa edits."""
    import cv2, numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB")); mb = (m > 0).astype(np.uint8)
    out = cv2.inpaint(a, mb * 255, 3, cv2.INPAINT_TELEA)
    return composite(img, Image.fromarray(out), mb)


def flatfeather_fill(img, m):
    """Mechanism test, ONE change from the crude edit: same flat surrounding-mean colour, but the edge is feathered (alpha ramps over about 16 px instead of a hard cut)."""
    import cv2, numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB")).astype(np.float32); mb = m > 0
    ring = (dilate(mb.astype(np.uint8), 12) > 0) & ~mb
    col = a[ring].mean(0) if ring.any() else a.reshape(-1, 3).mean(0)
    alpha = cv2.GaussianBlur(mb.astype(np.float32), (0, 0), 8)[..., None]
    return Image.fromarray(np.clip(np.round(a * (1 - alpha) + col * alpha), 0, 255).astype(np.uint8))


def flatnoise_fill(img, m):
    """Mechanism test, ONE change from the crude edit: same hard edge and same mean colour, but the fill carries low-pass noise with the same spread as the surroundings (no longer a flat patch)."""
    import cv2, numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB")).astype(np.float32).copy(); mb = m > 0
    ring = (dilate(mb.astype(np.uint8), 12) > 0) & ~mb
    col = a[ring].mean(0) if ring.any() else a.reshape(-1, 3).mean(0)
    sd = a[ring].std(0) if ring.any() else a.reshape(-1, 3).std(0)
    rng = np.random.default_rng(12345 + int(mb.sum()))
    nz = cv2.GaussianBlur(rng.normal(size=a.shape).astype(np.float32), (0, 0), 2.0)
    nz = nz / (nz[mb].std(0) + 1e-6) if mb.any() else nz
    a[mb] = np.clip(col + nz[mb] * sd * 0.7, 0, 255)
    return Image.fromarray(np.round(a).astype(np.uint8))


def make_edits():
    import numpy as np
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    Gf, M, D = load_matched(); Gl, sp = jload(P("groups_lama.json")), jload(P("splits.json"))
    byg = {g["gid"]: g for g in Gl["train"]}
    todo = [byg[x] for x in sp["D"]] + M

    def one(g):
        n, img = 0, None
        for kind in ["remove", "placebo"]:
            for fn, pf in [(crude_fill, cpath), (medium_fill, mpath), (flatfeather_fill, lambda gid, k: xpath("ff", gid, k)), (flatnoise_fill, lambda gid, k: xpath("fn", gid, k))]:
                out = pf(g["gid"], kind)
                if os.path.exists(out): continue
                if img is None: img = Image.open(g["files"]["original"]).convert("RGB")
                m = np.array(Image.open(P("groups", g["gid"], f"{kind}_mask.png"))) > 0
                os.makedirs(os.path.dirname(out), exist_ok=True); fn(img, m).save(out); n += 1
        return n
    with ThreadPoolExecutor(4) as ex: n = sum(ex.map(one, todo))
    log(f"edits: {n} new images written for {len(todo)} groups ({len(sp['D'])} training + {len(M)} FINAL test)")


# --------------------------------------------------------------------------------- items and training sets
TEST_KINDS = [("crude_remove", "no", lambda g: cpath(g["gid"], "remove")), ("crude_placebo", "yes", lambda g: cpath(g["gid"], "placebo")),
              ("med_remove", "no", lambda g: mpath(g["gid"], "remove")), ("med_placebo", "yes", lambda g: mpath(g["gid"], "placebo")),
              ("lama_remove", "no", lambda g: g["files"]["lama_remove"]), ("lama_placebo", "yes", lambda g: g["files"]["lama_placebo"]),
              ("sd_remove", "no", lambda g: g["files"]["sd_remove"]), ("sd_placebo", "yes", lambda g: g["files"]["sd_placebo"]),
              ("ff_remove", "no", lambda g: xpath("ff", g["gid"], "remove")), ("ff_placebo", "yes", lambda g: xpath("ff", g["gid"], "placebo")),
              ("fn_remove", "no", lambda g: xpath("fn", g["gid"], "remove")), ("fn_placebo", "yes", lambda g: xpath("fn", g["gid"], "placebo"))]


def study_items():
    Gf, M, D = load_matched(); items = []
    for g in M:
        items.append(dict(id=f"{g['gid']}|original", path=g["files"]["original"], question=Q(g["cat"]), expected="yes", kind="original"))
        for kind, exp, pf in TEST_KINDS:
            items.append(dict(id=f"{g['gid']}|{kind}", path=pf(g), question=Q(g["cat"]), expected=exp, kind=kind))
    for n in jload(P("nat.json")):
        items.append(dict(id=n["id"], path=n["path"], question=Q(n["cat"]), expected=n["expected"], kind=n["kind"]))
    cnt = collections.Counter()
    for it in jload(P("pope_items.json")):
        if cnt[it["kind"]] < POPE_N: items.append(it); cnt[it["kind"]] += 1
    return items


def study_train(level, cond):
    cond = {"DS": "S", "DP": "P"}.get(cond, cond)               # the DPO-style baseline trains on exactly the same examples as S / P, only the loss differs
    Gf, byg, sp, M = PM.load_world(); D = [byg[x] for x in sp["D"]]
    rm = {"none": None, "flatfeather": lambda g: xpath("ff", g["gid"], "remove"), "flatnoise": lambda g: xpath("fn", g["gid"], "remove"), "crude": lambda g: cpath(g["gid"], "remove"), "medium": lambda g: mpath(g["gid"], "remove"), "clean": lambda g: g["files"]["lama_remove"]}[level]
    pl = {"none": None, "flatfeather": lambda g: xpath("ff", g["gid"], "placebo"), "flatnoise": lambda g: xpath("fn", g["gid"], "placebo"), "crude": lambda g: cpath(g["gid"], "placebo"), "medium": lambda g: mpath(g["gid"], "placebo"), "clean": lambda g: g["files"]["lama_placebo"]}[level]
    ex = lambda path, cat, ans: dict(path=path, question=Q(cat), answer=ans)
    out = []
    if cond == "N":
        for g in D: out += [ex(g["files"]["original"], g["cat"], "yes"), ex(sp["real_neg"][g["gid"]], g["cat"], "no")]
        return out
    for g in D:
        out += [ex(g["files"]["original"], g["cat"], "yes"), ex(rm(g), g["cat"], "no")]
        if cond == "P": out.append(ex(pl(g), g["cat"], "yes"))
    return out


def mech_jobs():
    """Mechanism test (LLaVA, standard training only): which property of the crude edit triggers the shortcut?
    Each new edit changes exactly ONE thing from the crude edit (flat colour + hard edge): flatfeather = softer edge, flatnoise = textured fill.
    Reference points already done: crude (flat, hard) -> shortcut; medium (smooth Telea fill, hard edge) -> no shortcut."""
    J = [dict(model="llava", level="mech", cond="A", seed=0, tag="llava_Amech")]
    for s_ in [0, 1, 2]:
        for lv in ["flatfeather", "flatnoise"]: J.append(dict(model="llava", level=lv, cond="S", seed=s_, tag=f"llava_{lv}_S_s{s_}"))
    return J


def more_jobs():
    """Remaining experiments, in priority order (a job that does not fit in the session is skipped and picked up by the next run):
    1) Qwen no-edit control, seeds 1-2 (the control currently has one seed), 2) LLaVA crude S/P seeds 3-4 (the shortcut is unstable, so 5 seeds), 3) LLaVA clean S/P seeds 3-4 (the main null result)."""
    J = []
    add = lambda model, level, cond, s: J.append(dict(model=model, level=level, cond=cond, seed=s, tag=f"{model}_{level}_{cond}_s{s}"))
    for s_ in [1, 2]: add("qwen", "none", "N", s_)
    for lv in ["crude", "clean"]:
        for s_ in [3, 4]:
            for cd in ["S", "P"]: add("llava", lv, cd, s_)
    return J


def rem_jobs(which):
    """The remaining experiments split into two parts of about equal length (one Kaggle session each).
    Part A (about 5 h on two GPUs): mechanism test (LLaVA, 2 new edit types x 3 seeds + one extra frozen evaluation) + Qwen no-edit control seeds 1-2.
    Part B (about 6.5 h): LLaVA crude and clean, S and P, seeds 3-4 (5 seeds in total)."""
    if which == "A":
        J = mech_jobs(); return J[:1] + [dict(model="qwen", level="none", cond="N", seed=k, tag=f"qwen_none_N_s{k}") for k in (1, 2)] + J[1:]
    J = []
    for lv in ["crude", "clean"]:
        for k in (3, 4):
            for cd in ["S", "P"]: J.append(dict(model="llava", level=lv, cond=cd, seed=k, tag=f"llava_{lv}_{cd}_s{k}"))
    return J


def third_jobs():
    """THE LAST EXPERIMENT: third model (smol), same recipe, same edits, same test set. Priority order (a job that does not fit in the session is skipped
    and picked up the next time the cell is run): frozen, crude+clean S/P x 3 seeds (core), no-edit control x 3 seeds, medium S/P x 3 seeds."""
    J = [dict(model="smol", level="-", cond="A", seed=0, tag="smol_A")]
    add = lambda level, cond, s: J.append(dict(model="smol", level=level, cond=cond, seed=s, tag=f"smol_{level}_{cond}_s{s}"))
    for s_ in range(3):
        for lv in ["crude", "clean"]:
            for cd in ["S", "P"]: add(lv, cd, s_)
    for s_ in range(3): add("none", "N", s_)
    for s_ in range(3):
        for cd in ["S", "P"]: add("medium", cd, s_)
    return J


# --------------------------------------------------------------------------------- second benchmark: MME (yes/no perception questions)
MME_CATS = ["existence", "count", "position", "color", "scene"]      # object-level subtasks + scene; every MME question is Yes/No; questions come in pairs per image (one Yes, one No)


def mme_items():
    """MME items from the Hugging Face dataset lmms-lab/MME (images are saved to disk once)."""
    import re
    f = P("mme_items.json")
    if os.path.exists(f):
        L = jload(f)
        if L and all(os.path.exists(it["path"]) for it in L[:25]): return L
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/MME", split="test"); cats = [str(c).lower() for c in ds["category"]]
    os.makedirs(P("mme_img"), exist_ok=True); out = []
    for i, c in enumerate(cats):
        if c not in MME_CATS: continue
        r = ds[i]; path = P("mme_img", f"{i}.jpg")
        if not os.path.exists(path): r["image"].convert("RGB").save(path, quality=95)
        q = re.sub(r"\s*Please answer yes or no\.?\s*$", "", str(r["question"]).strip(), flags=re.I)
        out.append(dict(id=f"mme|{c}|{i}", path=path, question=q, expected="yes" if str(r["answer"]).strip().lower().startswith("y") else "no",
                        kind=f"mme_{c}", qid=str(r.get("question_id", i))))
    jsave(f, out); return out


def mme_check():
    """Run once before the long job: loads MME, prints what it found, fails loudly if the data look wrong."""
    L = mme_items(); cnt = collections.Counter(it["kind"] for it in L); yes = sum(it["expected"] == "yes" for it in L)
    print("MME items:", len(L), dict(cnt), f"| share Yes {yes / max(len(L), 1):.2f} (should be about 0.50)")
    print("example:", L[0]["question"], "->", L[0]["expected"], "|", L[0]["path"])
    assert len(L) > 300 and 0.4 < yes / len(L) < 0.6 and all(os.path.exists(it["path"]) for it in L[:50]), "MME did not load as expected"
    log("MME CHECK PASSED")


def bench_jobs():
    """Larger model (q7 = Qwen2.5-VL-7B, crude + clean x S/P x 2 seeds) + second benchmark (MME) for every model. Priority order, so a partial run stays balanced:
    frozen runs and the cheap models first, then the 7B seed 0, then LLaVA's retrain, then the 7B seed 1, then the no-edit controls on MME."""
    J = [dict(model="q7", level="-", cond="A", seed=0, tag="q7_A", eval="study+mme")]
    for m in ["llava", "qwen", "smol"]: J.append(dict(model=m, level="-", cond="A", seed=0, tag=f"{m}_Amme", eval="mme", est=8))
    bench = lambda m, lv, cd: dict(model=m, level=lv, cond=cd, seed=0, tag=f"bench_{m}_{lv}_{cd}_s0", eval="bench")
    q7 = lambda lv, cd, s: dict(model="q7", level=lv, cond=cd, seed=s, tag=f"q7_{lv}_{cd}_s{s}", eval="study+mme")
    for m in ["smol", "qwen"]:
        for cd in "SP": J.append(bench(m, "clean", cd))
    for lv in ["clean", "crude"]:
        for cd in "SP": J.append(q7(lv, cd, 0))
    for cd in "SP": J.append(bench("llava", "clean", cd))
    for lv in ["clean", "crude"]:
        for cd in "SP": J.append(q7(lv, cd, 1))
    J += [bench("smol", "none", "N"), bench("qwen", "none", "N")]
    return J


def last_jobs():
    """Last experiments. (1) DPO-STYLE BASELINE: the same edit data as S / P (original=Yes, removed=No [, placebo=Yes]) but trained with a DPO preference loss
    (chosen = correct answer, rejected = the other answer, reference = the frozen model) instead of cross-entropy. Question: does a preference-optimisation recipe
    also learn the shortcut, and does placebo data also protect it? (2) The missing 7B no-edit control. Priority order, so a partial run stays balanced."""
    dpo = lambda m, lv, cd, s, est: dict(model=m, level=lv, cond=cd, seed=s, tag=f"{m}_{lv}_{cd}_s{s}", est=est)
    J = [dpo("llava", "crude", "DS", 0, 90), dpo("smol", "crude", "DS", 0, 45), dpo("qwen", "crude", "DS", 0, 55),
         dpo("llava", "crude", "DP", 0, 115), dpo("llava", "crude", "DS", 1, 90),
         dict(model="q7", level="none", cond="N", seed=0, tag="q7_none_N_s0", est=95, eval="study+mme"),
         dpo("smol", "crude", "DS", 1, 45), dpo("qwen", "crude", "DS", 1, 55), dpo("llava", "crude", "DP", 1, 115),
         dpo("llava", "crude", "DS", 2, 90), dpo("smol", "crude", "DS", 2, 45), dpo("qwen", "crude", "DS", 2, 55),
         dpo("llava", "clean", "DS", 0, 90),
         dict(model="q7", level="none", cond="N", seed=1, tag="q7_none_N_s1", est=95, eval="study+mme")]
    return J


def extra_jobs():
    """No-edit control: same questions and the same number of examples, but NO edited images (original = Yes, natural image without the object = No)."""
    J = []
    for m, seeds in [("llava", [0, 1, 2]), ("qwen", [0])]:
        for s in seeds: J.append(dict(model=m, level="none", cond="N", seed=s, tag=f"{m}_none_N_s{s}"))
    return J


# --------------------------------------------------------------------------------- jobs
def study_jobs():
    """Priority order: cheap frozen evals, then the experiments that matter most, seed by seed so partial results stay balanced."""
    J = []
    add = lambda model, level, cond, s: J.append(dict(model=model, level=level, cond=cond, seed=s, tag=f"{model}_{level}_{cond}_s{s}"))
    for m in ["llava", "qwen"]: J.append(dict(model=m, level="-", cond="A", seed=0, tag=f"{m}_A"))
    for level, m, seeds in [("medium", "llava", STUDY_SEEDS), ("crude", "qwen", STUDY_SEEDS), ("clean", "qwen", STUDY_SEEDS),
                            ("medium", "qwen", STUDY_SEEDS), ("clean", "llava", STUDY_SEEDS)]:
        for s in seeds:
            for cond in ["S", "P"]: add(m, level, cond, s)
    return J


LEGACY_TAGS = {f"llava_crude_{c}_s{k}" for c in "SP" for k in range(3)} | {"llava_clean_S_s0", "llava_clean_P_s0"}
N_PARTS = 2


def part_jobs(part):
    """Fixed split of the work into 2 parts of about equal cost (uses the default estimates, so it never changes between sessions).
    Part n returns its own jobs first, then jobs left over from earlier parts (so nothing is ever skipped). Jobs of later parts are never touched."""
    if part == "extra": return extra_jobs()
    if part == "mech": return mech_jobs()
    if part == "more": return more_jobs()
    if part == "third": return third_jobs()
    if part == "q7bench": return bench_jobs()
    if part == "last": return last_jobs()
    if part in ("remA", "remB"): return rem_jobs(part[-1])
    J = study_jobs(); w = [0 if j["tag"] in LEGACY_TAGS else EST_DEFAULT[(j["model"], j["cond"])] for j in J]
    tot = sum(w); own, acc = {}, 0.0
    for j, x in zip(J, w):
        own[j["tag"]] = min(N_PARTS, int(acc / (tot / N_PARTS)) + 1); acc += x
    mine = [j for j in J if own[j["tag"]] == part]; left = [j for j in J if own[j["tag"]] < part]
    return mine + left


def est_minutes(job):
    """Measured median duration of finished jobs of the same kind (model, condition); otherwise the first-guess default."""
    if "est" in job: return float(job["est"])
    pat = P("times", f"{job['model']}_A.json") if job["cond"] == "A" else P("times", f"{job['model']}_*_{job['cond']}_s*.json")
    ts = sorted(jload(f)["min"] for f in glob.glob(pat))
    return float(ts[len(ts) // 2]) if ts else float(EST_DEFAULT[(job["model"], job["cond"])])


# --------------------------------------------------------------------------------- backends
_ORIG = dict(build_inputs=PM.build_inputs, load_base=PM.load_base)
_QWEN = {}
_SMOL = {}


def _qwen_inputs(paths, questions):
    import torch
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths]
    texts = [_QWEN["proc"].apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{q} Answer Yes or No."}]}],
                                               tokenize=False, add_generation_prompt=True) for q in questions]
    return _QWEN["proc"](text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda:0", torch.float16)


def _qwen_load_base():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    qid = _QWEN.get("id", QID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(qid, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16)


def _smol_inputs(paths, questions):
    import torch
    from PIL import Image
    imgs = [[Image.open(p).convert("RGB")] for p in paths]
    texts = [_SMOL["proc"].apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{q} Answer Yes or No."}]}],
                                               add_generation_prompt=True) for q in questions]
    return _SMOL["proc"](text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda:0", torch.float16)


def _smol_load_base():
    import torch
    from transformers import BitsAndBytesConfig
    try: from transformers import AutoModelForImageTextToText as AM
    except ImportError: from transformers import AutoModelForVision2Seq as AM
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    return AM.from_pretrained(SID, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16)


def _set_lora_regex(regex):
    """PM.train builds its LoRA config from a fixed regex (language_model...). For SmolVLM the language layers are called text_model, so the regex is widened for that model only."""
    import peft
    if not hasattr(peft, "_LoraConfig_orig"): peft._LoraConfig_orig = peft.LoraConfig
    if regex is None: peft.LoraConfig = peft._LoraConfig_orig
    else: peft.LoraConfig = lambda **kw: peft._LoraConfig_orig(**{**kw, "target_modules": regex})


def set_backend(model):
    """Point the (tested) LLaVA training/evaluation code at the chosen model. LoRA target regex in PM.train matches both models."""
    if model != "smol": _set_lora_regex(None)
    if model == "llava":
        PM.build_inputs, PM.load_base = _ORIG["build_inputs"], _ORIG["load_base"]
        PM.init_model_env()
    elif model == "smol":
        from transformers import AutoProcessor
        if "proc" not in _SMOL:
            pr = AutoProcessor.from_pretrained(SID); pr.tokenizer.padding_side = "left"
            try: pr.image_processor.do_image_splitting = False; pr.image_processor.size = {"longest_edge": SMOL_LONGEST}
            except Exception: log("WARNING: could not set image size for the third model")
            _SMOL["proc"] = pr
        tok = _SMOL["proc"].tokenizer; ids = {}
        for k, w in [("YES", "Yes"), ("NO", "No")]:
            sp = tok.encode(" " + w, add_special_tokens=False)      # after "Assistant:" the model's answer starts with a space
            ids[k] = sp[0]; assert len(sp) == 1, f"' {w}' is not a single token for this tokenizer: {sp}"
        PM.ENV.update(proc=_SMOL["proc"], tok=tok, **ids)
        PM.build_inputs, PM.load_base = _smol_inputs, _smol_load_base
        _set_lora_regex(r".*(language_model|text_model).*\.(q_proj|k_proj|v_proj|o_proj)")
        log("third-model answer tokens:", tok.convert_ids_to_tokens([PM.ENV["YES"], PM.ENV["NO"]]))
    else:
        from transformers import AutoProcessor
        _QWEN["id"] = QID7 if model == "q7" else QID
        if "proc" not in _QWEN:
            _QWEN["proc"] = AutoProcessor.from_pretrained(QID, min_pixels=QMIN_PIX, max_pixels=QMAX_PIX)
            _QWEN["proc"].tokenizer.padding_side = "left"
        tok = _QWEN["proc"].tokenizer
        PM.ENV.update(proc=_QWEN["proc"], tok=tok, YES=tok.encode("Yes", add_special_tokens=False)[0], NO=tok.encode("No", add_special_tokens=False)[0])
        PM.build_inputs, PM.load_base = _qwen_inputs, _qwen_load_base
        log(f"{model} answer tokens:", tok.convert_ids_to_tokens([PM.ENV["YES"], PM.ENV["NO"]]))


DPO_BETA, DPO_LR = 0.1, 5e-5                                     # standard beta; learning rate lower than the cross-entropy runs (1e-4); NOT tuned


def train_dpo(model, examples, seed, tag):
    """DPO-style training for single-token Yes/No answers. For a question with correct answer c and wrong answer r the DPO loss is
    -log sigmoid( beta * [ (logp(c) - logp_ref(c)) - (logp(r) - logp_ref(r)) ] ). Both log-probs share one softmax normaliser, so this equals
    -log sigmoid( beta * sign * (margin - margin_ref) ) with margin = logit(Yes) - logit(No), sign = +1 if the correct answer is Yes.
    Reference model = the same network with the LoRA adapter switched off (the frozen model). Same LoRA setup, epochs, accumulation and seeds as PM.train."""
    import numpy as np, torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import get_cosine_schedule_with_warmup
    torch.manual_seed(seed); rng = random.Random(seed)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=PM.LORA_R, lora_alpha=2 * PM.LORA_R, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)")
    model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=DPO_LR, weight_decay=0.0)
    total = max(1, PM.EPOCHS * len(examples) // PM.ACCUM)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)
    scaler = torch.amp.GradScaler("cuda")
    model.train(); seen, run, acc, t0 = 0, [], [], time.time()
    for ep in range(PM.EPOCHS):
        order = examples[:]; rng.shuffle(order)
        for e in order:
            inp = PM.build_inputs([e["path"]], [e["question"]])
            with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.float16):
                lr_ = model(**inp).logits[0, -1].float()
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(**inp).logits[0, -1].float()
            m_ref = (lr_[PM.ENV["YES"]] - lr_[PM.ENV["NO"]]).detach(); m_pol = lg[PM.ENV["YES"]] - lg[PM.ENV["NO"]]
            sign = 1.0 if e["answer"] == "yes" else -1.0
            z = DPO_BETA * sign * (m_pol - m_ref); loss = -F.logsigmoid(z) / PM.ACCUM
            scaler.scale(loss).backward(); run.append(loss.item() * PM.ACCUM); acc.append(float(z.item() > 0)); seen += 1
            if seen % PM.ACCUM == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if seen % 100 == 0:
                log(f"[train-dpo {tag}] ep{ep} {seen}/{PM.EPOCHS*len(examples)}  loss {np.mean(run[-100:]):.3f}  pref-acc {np.mean(acc[-100:]):.2f}  {(time.time()-t0)/60:.1f} min")
    return model


def smoke(model, n_train=6, dpo=False):
    """~5 min: load the model, score a few items, take a few training steps. Run once per model before the long run."""
    import torch, math
    set_backend(model); items = [it for it in study_items() if it["kind"] in ("original", "sd_remove", "nat_neg", "crude_placebo")][:8]
    t0 = time.time(); m = PM.load_base(); log(f"[{model}] loaded in {time.time()-t0:.0f}s | GPU memory {torch.cuda.memory_allocated()/1e9:.1f} GB")
    t0 = time.time(); ms = PM.margins_batch(m, items); log(f"[{model}] margins (Yes - No) for 8 items:", [round(x, 2) for x in ms], f"| {(time.time()-t0)/8:.2f} s/item")
    assert all(math.isfinite(x) for x in ms), "non-finite margins: fp16 problem for this model on this GPU"
    with torch.no_grad():
        inp = PM.build_inputs([it["path"] for it in items], [it["question"] for it in items]); log(f"[{model}] input shape (batch, tokens):", tuple(inp["input_ids"].shape))
        top = m(**inp).logits[:, -1].argmax(-1).tolist()
    toks = PM.ENV["tok"].convert_ids_to_tokens(top); share = sum(t in (PM.ENV["YES"], PM.ENV["NO"]) for t in top) / len(top)
    log(f"[{model}] most likely next token for the 8 items:", toks, f"| {share:.0%} are the Yes/No answer tokens (should be about 100%)")
    assert share >= 0.75, "the model's first answer token is not the Yes/No token used for scoring: the prompt or answer tokens need fixing"
    ex = study_train("crude", "P")[:n_train]; t0 = time.time()
    m = (train_dpo if dpo else PM.train)(m, ex, 0, "smoke"); log(f"[{model}] trained {n_train}x{PM.EPOCHS} examples in {time.time()-t0:.0f}s ({(time.time()-t0)/(n_train*PM.EPOCHS):.2f} s/example) | peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    ms = PM.margins_batch(m, items); assert all(math.isfinite(x) for x in ms), "non-finite margins after training"
    log(f"[{model}] SMOKE TEST PASSED")


def study_worker(w, budget_min, part=None):
    set_tag(f"[w{w}]")
    import torch
    t_start = time.time(); log("GPU:", torch.cuda.get_device_name(0), f"| time budget for this worker: {budget_min:.0f} min")
    items = study_items(); cur = None; os.makedirs(P("times"), exist_ok=True); _it = {}
    def items_for(job):
        ev = job.get("eval", "study")
        if ev == "study": return items
        if ev not in _it:
            mm = mme_items()
            _it[ev] = {"mme": mm, "bench": mm + [x for x in items if x["kind"].startswith("pope")], "study+mme": items + mm}[ev]
        return _it[ev]
    for job in (study_jobs() if part is None else part_jobs(part)):
        tag = job["tag"]; out = P("preds", tag + ".json")
        if os.path.exists(out): continue
        el = (time.time() - t_start) / 60; est = est_minutes(job)
        if el + est > budget_min: continue                         # would not finish inside the session: try a shorter job
        if not PM.claim(tag): continue
        t0 = time.time(); log("=" * 10, "job", tag, f"(estimate {est:.0f} min, {el:.0f} min used)")
        try:
            if cur != job["model"]: set_backend(job["model"]); cur = job["model"]
            model = PM.load_base()
            if job["cond"] != "A":
                adir = P("adapters", tag)
                if os.path.exists(os.path.join(adir, "adapter_config.json")):
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, adir)
                else:
                    model = (train_dpo if job["cond"] in ("DS", "DP") else PM.train)(model, study_train(job["level"], job["cond"]), job["seed"], tag); model.save_pretrained(adir)
            res = PM.evaluate(model, items_for(job), tag); jsave(out, res)
            mins = (time.time() - t0) / 60; jsave(P("times", tag + ".json"), dict(min=mins)); log(f"job {tag} finished in {mins:.1f} min")
        except Exception:
            log("JOB FAILED", tag); traceback.print_exc()
            os.makedirs(P("failed"), exist_ok=True)
            with open(P("failed", tag + ".txt"), "w") as f: f.write(traceback.format_exc())
            try: os.remove(P("locks", tag + ".lock"))               # let the other GPU's worker try this job too
            except OSError: pass
        model = None; gc.collect(); torch.cuda.empty_cache()
    log("worker finished (no job left that fits the time budget)")


def import_legacy():
    """Reuse finished LLaVA runs from the earlier notebooks (same groups, same edits, same training recipe)."""
    mp = [(f"Bc_s{s}", f"llava_crude_S_s{s}") for s in range(3)] + [(f"Dc_s{s}", f"llava_crude_P_s{s}") for s in range(3)] + [("B_s0", "llava_clean_S_s0"), ("D_s0", "llava_clean_P_s0")]
    n = 0
    for a, b in mp:
        if os.path.exists(P("preds", a + ".json")) and not os.path.exists(P("preds", b + ".json")): shutil.copy(P("preds", a + ".json"), P("preds", b + ".json")); n += 1
    log(f"legacy runs imported: {n}")


def _upload_part(PART, WTOKEN, quiet=False):
    """Zip this part's results (predictions, timings) and upload them to Hugging Face. Also called every ~90 min while the workers run, so a session that dies loses at most 90 min."""
    from huggingface_hub import HfApi
    import zipfile
    root = os.path.dirname(WORK.rstrip("/")); name = f"pest_part{PART}_results.zip"; zp = os.path.join(root, name)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ["preds", "times", "study_probe.json", "edit_levels_preview.png", "mme_items.json"]:
            p = P(sub)
            for f in ([p] if os.path.isfile(p) else [os.path.join(d, x) for d, _, xs in os.walk(p) for x in xs] if os.path.isdir(p) else []):
                z.write(f, os.path.relpath(f, root))
    if not quiet: print(name, round(os.path.getsize(zp) / 1e6, 1), "MB")
    HfApi(token=WTOKEN).upload_file(path_or_fileobj=zp, path_in_repo=name, repo_id="Uzairezra/pest-backup", repo_type="dataset")
    return name


def run_part(PART, SESSION_LIMIT_H=9.0, SAFETY_H=0.4, MANUAL_USED_H=None, HF_TOKEN=""):
    """One third of the study: train + evaluate on both GPUs in the background, follow the logs, then zip and upload this part's results."""
    import subprocess, psutil
    from getpass import getpass
    WTOKEN = HF_TOKEN.strip() or getpass("Hugging Face WRITE token (used once, at the end of this part, to upload): ")
    try: used = (time.time() - psutil.Process(1).create_time()) / 3600
    except Exception: used = 1.0
    if MANUAL_USED_H is not None: used = MANUAL_USED_H
    HOURS = max(0.0, round(SESSION_LIMIT_H - used - SAFETY_H, 2))
    import_legacy()
    jobs = part_jobs(PART); todo = [j for j in jobs if not os.path.exists(P("preds", j["tag"] + ".json"))]; est = sum(est_minutes(j) for j in todo)
    print(f"PART {PART}: {len(jobs) - len(todo)}/{len(jobs)} jobs finished, {len(todo)} to go. Estimated {est/60:.1f} GPU-hours = about {est/120:.1f} h on two GPUs. Session used {used:.2f} h -> workers get {HOURS} h.")
    alive = subprocess.run(["pgrep", "-f", "study_worker"], capture_output=True, text=True).stdout.split()
    os.makedirs(P("logs"), exist_ok=True)
    if todo:
        if alive:
            print(f"{len(alive)} worker(s) already running: following their logs"); logs = sorted(glob.glob(P("logs", "s*.log")))
        else:
            for j in study_jobs() + extra_jobs() + mech_jobs() + more_jobs() + third_jobs() + bench_jobs():
                for f in glob.glob(P("locks", j["tag"] + ".lock")): os.remove(f)
            shutil.rmtree(P("failed"), ignore_errors=True)
            n = min(PM.gpu_count(), 2, len(todo)); logs = []
            print(f"starting {n} GPU worker(s)")
            for w in range(n):
                env = dict(os.environ)
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                if n > 1: env["CUDA_VISIBLE_DEVICES"] = str(w)
                lp = P("logs", f"s{w}.log"); logs.append(lp)
                subprocess.Popen([sys.executable, "-u", "-c", f"import sys; sys.path.insert(0, '/kaggle/working'); import pest_study; pest_study.study_worker({w}, {HOURS * 60}, part={PART!r})"],
                                 env=env, stdout=open(lp, "w"), stderr=subprocess.STDOUT)
                time.sleep(20)
        pos = {lp: 0 for lp in logs}
        def tail():
            for lp in logs:
                if not os.path.exists(lp): continue
                with open(lp) as f:
                    f.seek(pos[lp]); data = f.read(); pos[lp] = f.tell()
                keep = [l for l in data.splitlines() if "Loading weights" not in l and "it/s]" not in l and "B/s]" not in l]
                if keep: print("\n".join(keep), flush=True)
        last_up = time.time()
        while subprocess.run(["pgrep", "-f", "study_worker"], capture_output=True, text=True).stdout.split():
            tail(); time.sleep(20)
            if time.time() - last_up > 5400:                       # safety copy every 90 min
                try: _upload_part(PART, WTOKEN, quiet=True); print(f"{time.strftime('%H:%M:%S')}  (safety copy of the finished results uploaded)", flush=True)
                except Exception as e: print("safety upload failed (will retry later):", str(e)[:120], flush=True)
                last_up = time.time()
        tail()
        left = [j["tag"] for j in jobs if not os.path.exists(P("preds", j["tag"] + ".json"))]
        print(f"\nWORKERS FINISHED. {len(jobs) - len(left)}/{len(jobs)} jobs of this part (and the parts before it) done." + (f" Not finished (picked up automatically by the next part): {left}" if left else " Part complete."))
        if os.path.exists(P("failed")): print("FAILED jobs (send me these):", os.listdir(P("failed")))
    else:
        print("nothing left in this part")
    name = _upload_part(PART, WTOKEN)
    print(f"UPLOADED {name}. PART {PART} is saved on Hugging Face. You can stop the session now (saves quota). Delete the token when all parts are done.")
