import sys; sys.path.insert(0, "/kaggle/working")
from pest_common import *

def load_world():
    Gf, Gl, sp = jload(P("groups_final.json")), jload(P("groups_lama.json")), jload(P("splits.json"))
    M = [g for g in Gf["test"] if g["ok_lama"] and g.get("ok_sd")]
    byg = {g["gid"]: g for g in Gl["train"]}
    return Gf, byg, sp, M


def build_eval_items():
    Gf, byg, sp, M = load_world()
    items = []
    for g in M:
        for kind, exp in [("original", "yes"), ("lama_remove", "no"), ("lama_placebo", "yes"), ("sd_remove", "no"), ("sd_placebo", "yes")]:
            items.append(dict(id=f"{g['gid']}|{kind}", path=g["files"][kind], question=Q(g["cat"]), expected=exp, kind=kind))
    for gid in sp["val"]:
        g = byg[gid]
        for kind, exp in [("original", "yes"), ("lama_remove", "no"), ("lama_placebo", "yes")]:
            items.append(dict(id=f"{gid}|val_{kind}", path=g["files"][kind], question=Q(g["cat"]), expected=exp, kind=f"val_{kind}"))
    for n in jload(P("nat.json")):
        items.append(dict(id=n["id"], path=n["path"], question=Q(n["cat"]), expected=n["expected"], kind=n["kind"]))
    pope = jload(P("pope_items.json")); cnt = collections.Counter()
    for it in pope:
        if cnt[it["kind"]] < POPE_EVAL_PER_SPLIT: items.append(it); cnt[it["kind"]] += 1
    return items


def build_train_sets():
    Gf, byg, sp, M = load_world()
    def ex(path, cat, ans): return dict(path=path, question=Q(cat), answer=ans)
    D, Cg = [byg[x] for x in sp["D"]], [byg[x] for x in sp["C"]]
    f = lambda g, k: g["files"][k]
    return {
        "B": [e for g in D for e in (ex(f(g, "original"), g["cat"], "yes"), ex(f(g, "lama_remove"), g["cat"], "no"))],
        "C": [e for g in Cg for e in (ex(f(g, "original"), g["cat"], "yes"), ex(f(g, "lama_remove"), g["cat"], "no"))],
        "D": [e for g in D for e in (ex(f(g, "original"), g["cat"], "yes"), ex(f(g, "lama_remove"), g["cat"], "no"), ex(f(g, "lama_placebo"), g["cat"], "yes"))],
        "E": [e for g in D for e in (ex(f(g, "original"), g["cat"], "yes"), ex(f(g, "lama_remove"), g["cat"], "no"), ex(sp["real_neg"][g["gid"]], g["cat"], "no"))],
    }


MID = "llava-hf/llava-1.5-7b-hf"
ENV = {}


def init_model_env():
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MID)
    tok = proc.tokenizer; tok.padding_side = "left"
    ENV.update(proc=proc, tok=tok, YES=tok.encode("Yes", add_special_tokens=False)[0], NO=tok.encode("No", add_special_tokens=False)[0])
    log("answer tokens:", tok.convert_ids_to_tokens([ENV["YES"], ENV["NO"]]))


def load_base():
    import torch
    from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    return LlavaForConditionalGeneration.from_pretrained(MID, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16)


def prompt_for(q): return f"USER: <image>\n{q} Answer Yes or No. ASSISTANT:"


def build_inputs(paths, questions):
    import torch
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return ENV["proc"](images=imgs, text=[prompt_for(q) for q in questions], return_tensors="pt", padding=True).to("cuda:0", torch.float16)


def margins_batch(model, batch):
    import torch
    with torch.no_grad():
        inp = build_inputs([b["path"] for b in batch], [b["question"] for b in batch])
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**inp).logits[:, -1]
        return (logits[:, ENV["YES"]] - logits[:, ENV["NO"]]).float().cpu().tolist()


def evaluate(model, items, tag):
    import torch
    part = P("preds_partial", f"{tag}.json")
    res = jload(part) if os.path.exists(part) else {}
    todo = [it for it in items if it["id"] not in res]
    model.eval(); t0, last = time.time(), time.time()
    for i in range(0, len(todo), EVAL_BS):
        batch = todo[i:i + EVAL_BS]
        try:
            ms = margins_batch(model, batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); ms = [margins_batch(model, [b])[0] for b in batch]
        for b, m in zip(batch, ms): res[b["id"]] = m
        if time.time() - last > 120 or i + EVAL_BS >= len(todo):
            jsave(part, res); last = time.time()
            log(f"[eval {tag}] {len(res)}/{len(items)}   elapsed {(time.time()-t0)/60:.1f} min")
    return res


def train(model, examples, seed, tag):
    import numpy as np, torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import get_cosine_schedule_with_warmup
    torch.manual_seed(seed); rng = random.Random(seed)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)")
    model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    total = max(1, EPOCHS * len(examples) // ACCUM)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)
    scaler = torch.amp.GradScaler("cuda")
    model.train(); seen, run, t0 = 0, [], time.time()
    for ep in range(EPOCHS):
        order = examples[:]; rng.shuffle(order)
        for e in order:
            inp = build_inputs([e["path"]], [e["question"]])
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**inp).logits[0, -1]
            tgt = torch.tensor([ENV["YES"] if e["answer"] == "yes" else ENV["NO"]], device="cuda:0")
            loss = F.cross_entropy(logits.float().unsqueeze(0), tgt) / ACCUM
            scaler.scale(loss).backward(); run.append(loss.item() * ACCUM); seen += 1
            if seen % ACCUM == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if seen % 100 == 0:
                log(f"[train {tag}] ep{ep} {seen}/{EPOCHS*len(examples)}  loss {np.mean(run[-100:]):.3f}  {(time.time()-t0)/60:.1f} min")
    return model


def claim(tag):
    os.makedirs(P("locks"), exist_ok=True)
    try:
        fd = os.open(P("locks", tag + ".lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); return True
    except FileExistsError:
        return False


def stage_models_worker(w):
    set_tag(f"[w{w}]")
    import torch
    log("GPU:", torch.cuda.get_device_name(0))
    init_model_env()
    items, TR = build_eval_items(), build_train_sets()
    log("evaluation items per model:", len(items), "| training examples:", {k: len(v) for k, v in TR.items()})
    jobs = [(c, s) for c in JOB_ORDER for s in SEEDS[c]]
    for c, s in jobs:
        tag = f"{c}_s{s}"; out = P("preds", tag + ".json")
        if os.path.exists(out) or not claim(tag): continue
        t0 = time.time(); log("=" * 10, "job", tag)
        try:
            model = load_base()
            if c != "A":
                adir = P("adapters", tag)
                if os.path.exists(os.path.join(adir, "adapter_config.json")):
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, adir); log("loaded saved adapter", tag)
                else:
                    model = train(model, TR[c], s, tag); model.save_pretrained(adir)
            res = evaluate(model, items, tag)
            jsave(out, res)
            log(f"job {tag} finished in {(time.time()-t0)/60:.1f} min")
        except Exception:
            log("JOB FAILED", tag); traceback.print_exc()
            os.makedirs(P("failed"), exist_ok=True)
            with open(P("failed", tag + ".txt"), "w") as f: f.write(traceback.format_exc())
        model = None; gc.collect(); torch.cuda.empty_cache()


def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return max(1, len([l for l in out.splitlines() if l.startswith("GPU")]))
    except Exception: return 1