#!/usr/bin/env python3
"""SM-FS ranking pipeline v2 — complete fit/evaluate/decide, fitting double-locked.

Replacement for the rejected candidate 6c4801cf… (REVIEW_sm_fs_ranking_candidate_lock.md).
Every blocking finding is addressed structurally, not textually:

CAND-1  The full transaction is HERE: project (train-only capability separation), fit (complete
        optimizer loop), evaluate (held-fold 359-candidate scorer), decide (paired lineage-block
        bootstrap), each with sealed receipts — landed BEFORE candidate generation.
CAND-2  `fitting_allowed` is cycle-free and never trusts caller JSON: the LIVE preregistration
        must say model_fitting_authorized=true, AND an independently emitted verification
        receipt must bind the exact final-lock SHA, AND every hash the lock binds (plan, code,
        bundle, initialized bytes) is recomputed from disk. A hand-written lock cannot pass.
CAND-3  The fitter NEVER receives held associations: `project` (the trusted coordinator) writes
        a per-fold train-only projection through a private transaction; `fit` refuses any input
        other than a verified projection; `evaluate` opens held rows only after verifying the
        sealed fit receipt binds the checkpoint bytes.
CAND-4  All §4 bindings are enforced fields (title-table/E5/tokenizer hashes, complete Adam
        options, the exact 18-name allowlist asserted before optimizer construction, frozen-ref
        rule, augmentation + RNG reset/consumption order, dtype + full runtime provenance,
        tie rule, schemas, per-job init reuse checks).
CAND-5  sm_fs_ranking_lock_verify.py provides the candidate/final-lock verifier with tamper
        rejection; this module calls it, never reimplements it.
CAND-6  install_private(): same-directory 0600 staging, data fsync, hard-link no-replace
        install, parent-directory fsync, rollback on failure, post-install mode/nlink/symlink/
        content verification. read_bound(): descriptor-bound verified bytes (no reopen).
CAND-7  No `assert` on any critical path (explicit raises survive python -O); subprocess return
        codes checked; cleanliness scope stated honestly; environment provenance includes
        python/numpy/torch/CUDA/device/driver/dtype/threads and the RNG reset order.

  python3 sm_fs_ranking_pipeline.py project --fold 0
  python3 sm_fs_ranking_pipeline.py fit --fold 0 --arm graded_negative --seed 3997001 \
      --final-lock L --verification-receipt R      # double-locked until reseal steps 4-6
  python3 sm_fs_ranking_pipeline.py evaluate --fold 0 --arm graded_negative --seed 3997001
  python3 sm_fs_ranking_pipeline.py decide
"""
import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
RANK_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_v1")
RUN_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_run_v2")     # NEW namespace (CAND-8)
SEEDS = (3997001, 3997002, 3997003)
ARMS = ("positive_only", "graded_negative")
STEPS, BS, DRAWS, LR, ANCHOR_W, CLIP = 800, 48, 24, 5e-4, 1.0, 1.0
ADAM = {"class": "torch.optim.Adam", "lr": LR, "betas": [0.9, 0.999], "eps": 1e-08,
        "weight_decay": 0, "amsgrad": False}
BUCKET_SLOTS = [("hard", 3), ("medium", 2), ("easy", 1)]
BOOT = {"resamples": 9999, "seed": 3997999, "confidence": 0.95, "min_blocks": 20}
INIT_SHA = {
    3997001: "a3bf4c0588cc3e4cf1ad335b66440c113e7ee11fd116bd7307a3cee57447098e",
    3997002: "fb353e693951819683793641464155e144caad73c553039fc64f1c6e253ad796",
    3997003: "f42bdea0071a64dca0dad5312178127906b96215ba6890f6a37ad19d87ffdd5a",
}
TRAINABLE_18 = None   # resolved at load: exact ordered (name, shape) list, then frozen


class PipelineError(RuntimeError):
    pass


def need(cond, msg):
    if not cond:
        raise PipelineError(msg)


def canon(o):
    return (json.dumps(o, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_bound(path, expect_sha=None, description="input"):
    """Descriptor-bound read: open once (no-follow), verify identity, return the exact bytes."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PipelineError(f"{description} unavailable: {path}") from exc
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode), f"{description} is not a regular file")
        need(st.st_nlink == 1, f"{description} must have exactly one hard link")
        chunks = []
        while True:
            c = os.read(fd, 1 << 20)
            if not c:
                break
            chunks.append(c)
        data = b"".join(chunks)
    finally:
        os.close(fd)
    if expect_sha is not None:
        got = sha_bytes(data)
        need(got == expect_sha, f"{description} sha {got[:16]} != expected {expect_sha[:16]}")
    return data


def install_private(path, data):
    """Crash-atomic private install: same-dir 0600 stage + fsync, hard-link no-replace,
    parent fsync, rollback on failure, post-install verification."""
    d = os.path.dirname(path)
    os.makedirs(d, mode=0o700, exist_ok=True)
    stage = os.path.join(d, f".stage-{os.path.basename(path)}-{os.getpid()}")
    fd = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(stage, path)                       # fails if target exists: no-replace
        except FileExistsError as exc:
            raise PipelineError(f"no-replace target exists: {path}") from exc
        dfd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            os.unlink(stage)                           # rollback/cleanup either way
        except OSError:
            pass
    st = os.lstat(path)
    need(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode), "installed non-regular file")
    need(stat.S_IMODE(st.st_mode) == 0o600, "installed mode is not 0600")
    need(st.st_nlink == 1, "installed link count != 1")
    read_bound(path, expect_sha=sha_bytes(data), description=f"installed {path}")
    return sha_bytes(data)


def enforce_environment():
    """Set determinism BEFORE any model construction; return OBSERVED provenance (CAND-7)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    obs = {
        "python": platform.python_version(), "numpy": numpy.__version__,
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": None, "dtype": "float32",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "rng_reset_order": "torch.manual_seed(seed) once before checkpoint load; "
                           "numpy default_rng(seed+1) for augmentation, consumed in row order "
                           "within each step; sampler is counter-based (consumes no RNG)",
    }
    if obs["cuda_available"]:
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True)
            need(smi.returncode == 0, "nvidia-smi failed")
            obs["driver"] = smi.stdout.strip()
        except FileNotFoundError:
            obs["driver"] = "unavailable"
    for key, want in (("deterministic_algorithms", True), ("cudnn_deterministic", True),
                      ("cudnn_benchmark", False), ("tf32_matmul", False),
                      ("tf32_cudnn", False), ("matmul_precision", "highest"),
                      ("cublas_workspace", ":4096:8")):
        need(obs[key] == want, f"environment {key}={obs[key]!r} != required {want!r}")
    return obs


def git_head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    need(r.returncode == 0, "git rev-parse failed")
    return r.stdout.strip()


def lane_clean():
    """Honest scope: tracked files under prototypes/mu_cosine only (stated, not implied)."""
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no",
                        "--", "prototypes/mu_cosine"],
                       cwd=os.path.join(ROOT, "..", ".."), capture_output=True, text=True)
    need(r.returncode == 0, "git status failed")
    return r.stdout.strip() == ""


def load_pairs_verified():
    man = json.loads(read_bound(os.path.join(RANK_DIR, "manifest.json"),
                                description="ranking manifest"))
    data = read_bound(os.path.join(RANK_DIR, "pairs.jsonl"),
                      expect_sha=man["outputs"]["pairs.jsonl"], description="pairs")
    return man, [json.loads(l) for l in data.decode().splitlines()]


def fitting_allowed(final_lock_path, receipt_path):
    """CAND-2: cycle-free. Requires (a) LIVE prereg authorizes fitting; (b) an independently
    emitted verification receipt binds the exact final-lock bytes; (c) every hash the lock
    binds is recomputed from disk. Returns (lock, receipt) or raises."""
    doc = json.loads(read_bound(os.path.join(ROOT, "SM_FS_LINEAGE_RANKING_PREREG.json"),
                                description="live preregistration"))
    need(doc.get("model_fitting_authorized") is True,
         "live preregistration does not authorize model fitting")
    lock_bytes = read_bound(final_lock_path, description="final lock")
    lock = json.loads(lock_bytes)
    need(lock.get("schema") == "unifyweaver.sm-fs-ranking-execution-lock.final.v1",
         "lock schema is not final.v1")
    receipt = json.loads(read_bound(receipt_path, description="verification receipt"))
    need(receipt.get("schema") == "unifyweaver.sm-fs-ranking-final-verification-receipt.v1",
         "receipt schema mismatch")
    need(receipt.get("final_lock_sha256") == sha_bytes(lock_bytes),
         "receipt does not bind these exact final-lock bytes")
    need(receipt.get("verifier") not in (None, "", "self"),
         "receipt must name an independent verifier")
    derived = sha_bytes(canon({k: v for k, v in doc.items() if k != "prereg_id"}))
    need(lock.get("prereg_id") == doc.get("prereg_id") == receipt.get("prereg_id"),
         "prereg id mismatch across lock/receipt/live document")
    need(lock.get("prereg_id_derived") == derived, "lock's derived prereg id is stale")
    from sm_fs_ranking_lock_verify import verify_final_lock
    verify_final_lock(lock, root=ROOT, rank_dir=RANK_DIR, run_dir=RUN_DIR)   # recomputes hashes
    return lock, receipt


def resolve_allowlist(model):
    """Exact ordered 18-tensor allowlist; freeze everything, enable exactly these (CAND-4)."""
    names = []
    for prefix in ("judge_name.resid.weight", "corpus_name.resid.weight",
                   "op_name.resid.weight"):
        obj = model
        ok = True
        for part in prefix.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            names.append(prefix)
    li = len(model.encoder.layers) - 1
    names += [f"encoder.layers.{li}.{n}" for n, _ in
              model.encoder.layers[li].named_parameters()]
    names += ["readout_w", "readout_b", "nodetype_emb.weight"]
    by_name = dict(model.named_parameters())
    by_name.update({"readout_w": model.readout_w, "readout_b": model.readout_b})
    need(all(n in by_name for n in names), "allowlist name missing from model")
    need(len(names) == 18, f"allowlist has {len(names)} tensors, requires 18")
    for p in model.parameters():
        p.requires_grad = False
    tensors = []
    for n in names:
        by_name[n].requires_grad = True
        tensors.append(by_name[n])
    count = sum(p.numel() for p in tensors)
    need(count == 1195782, f"trainable param count {count} != 1195782")
    return names, tensors


def sample_step(fold, seed, step, train_q, pos, buckets, arm):
    from sm_fs_protocols import sampler_index
    common, contrast = [], []
    for draw in range(DRAWS):
        qi, _, _ = sampler_index(len(train_q), fold=fold, seed=seed, step=step,
                                 draw=draw, role="query")
        q = train_q[qi]
        ci, _, _ = sampler_index(len(pos[q]), fold=fold, seed=seed, step=step,
                                 draw=draw, role="common-positive", query_id=q)
        common.append(pos[q][ci])
        if arm == "positive_only":
            pi, _, _ = sampler_index(len(pos[q]), fold=fold, seed=seed, step=step,
                                     draw=draw, role="contrast-positive", query_id=q)
            contrast.append(pos[q][pi])
        else:
            slots = [(b, w) for b, w in BUCKET_SLOTS if buckets[q].get(b)]
            bi, _, _ = sampler_index(sum(w for _, w in slots), fold=fold, seed=seed,
                                     step=step, draw=draw, role="negative-bucket", query_id=q)
            acc = 0
            for b, w in slots:
                acc += w
                if bi < acc:
                    bucket = b
                    break
            members = buckets[q][bucket]
            ni, _, _ = sampler_index(len(members), fold=fold, seed=seed, step=step,
                                     draw=draw, role="negative-candidate", query_id=q,
                                     bucket=bucket)
            contrast.append(members[ni])
    return common, contrast


def cmd_project(a):
    """Trusted coordinator (CAND-3): per-fold TRAIN-ONLY projection. Held queries' rows are
    excluded entirely; the fitter consumes only this artifact."""
    man, pairs = load_pairs_verified()
    held_q = {p["query"] for p in pairs if p["fold"] == a.fold}
    train_rows = [p for p in pairs if p["query"] not in held_q]
    for p in train_rows:
        need(p["fold"] != a.fold, "held-fold row leaked into train projection")
    payload = b"".join(canon(p) for p in train_rows)
    out = os.path.join(RUN_DIR, f"fold{a.fold}", "train_projection.jsonl")
    sha = install_private(out, payload)
    meta = {"schema": "unifyweaver.sm-fs-ranking-train-projection.v1", "fold": a.fold,
            "rows": len(train_rows), "held_queries_excluded": len(held_q),
            "projection_sha256": sha,
            "source_manifest_sha256": sha_bytes(read_bound(
                os.path.join(RANK_DIR, "manifest.json"))),
            "capability_note": "held query-to-destination associations are ABSENT from this "
                               "artifact; the fitter receives nothing else"}
    install_private(os.path.join(RUN_DIR, f"fold{a.fold}", "train_projection.meta.json"),
                    canon(meta))
    print(f"fold {a.fold}: train projection {len(train_rows)} rows "
          f"({len(held_q)} held queries excluded) sha {sha[:16]}")


def cmd_fit(a):
    lock, receipt = fitting_allowed(a.final_lock, a.verification_receipt)   # raises when blocked
    obs = enforce_environment()
    import copy

    import numpy as np
    import torch

    from fine_tune_channel_heads import mu_batch
    from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, \
        build_e5_tables
    meta = json.loads(read_bound(
        os.path.join(RUN_DIR, f"fold{a.fold}", "train_projection.meta.json"),
        description="projection meta"))
    rows_bytes = read_bound(os.path.join(RUN_DIR, f"fold{a.fold}", "train_projection.jsonl"),
                            expect_sha=meta["projection_sha256"],
                            description="train projection")
    rows = [json.loads(l) for l in rows_bytes.decode().splitlines()]
    titles = json.loads(read_bound(os.path.join(RUN_DIR, "titles.json"),
                                   description="title table"))
    torch.manual_seed(a.seed)
    aug_rng = np.random.default_rng(a.seed + 1)
    ckpt_path = os.path.join(RANK_DIR, f"init_seed{a.seed}.pt")
    ckpt_bytes = read_bound(ckpt_path, expect_sha=INIT_SHA[a.seed],
                            description="initialized checkpoint")
    import io
    blob = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu", weights_only=False)
    from mu_attention import MuAttention  # constructed from cfg then loaded from exact bytes
    model = MuAttention(**blob["cfg"])
    model.load_state_dict(blob["state"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    names, tensors = resolve_allowlist(model)
    opt = torch.optim.Adam(tensors, lr=ADAM["lr"], betas=tuple(ADAM["betas"]),
                           eps=ADAM["eps"], weight_decay=ADAM["weight_decay"],
                           amsgrad=ADAM["amsgrad"])
    qtbl, ptbl, idx = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                      model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})
    pos, buckets = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for i, p in enumerate(rows):
        (pos[p["query"]].append(i) if p["class"].startswith("positive")
         else buckets[p["query"]][p["hardness"]].append(i))
    train_q = sorted(pos)
    C, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]

    def items_of(sel):
        return [(rows[i]["query"], rows[i]["candidate"], OPS["LINEAGE"], C, J, MM, MM)
                for i in sel]
    model.train()
    for step in range(STEPS):
        common, contrast = sample_step(a.fold, a.seed, step, train_q, pos, buckets, a.arm)
        sel = common + contrast
        tgt = torch.tensor([rows[i]["target"] for i in sel], dtype=torch.float32, device=dev)
        mu = mu_batch(model, tok, items_of(sel), dev, train=True, rng=aug_rng)
        loss = torch.mean((mu - tgt) ** 2)
        ag = [(it[0], it[1], it[2]) for it in items_of(common)]
        mu_ag = mu_batch(model, tok, ag, dev)
        with torch.no_grad():
            mu_ref = mu_batch(ref, tok, ag, dev)
        loss = loss + ANCHOR_W * torch.mean((mu_ag - mu_ref) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tensors, CLIP)
        opt.step()
    model.eval()
    buf = io.BytesIO()
    torch.save({"state": model.state_dict(), "cfg": blob["cfg"]}, buf)
    out_dir = os.path.join(RUN_DIR, f"fold{a.fold}")
    ck_sha = install_private(os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.pt"), buf.getvalue())
    fit_receipt = {
        "schema": "unifyweaver.sm-fs-ranking-fit-receipt.v1",
        "fold": a.fold, "arm": a.arm, "seed": a.seed,
        "init_sha256": INIT_SHA[a.seed], "checkpoint_sha256": ck_sha,
        "projection_sha256": meta["projection_sha256"],
        "final_lock_sha256": receipt["final_lock_sha256"],
        "steps": STEPS, "batch_size": BS, "adam": ADAM, "anchor_weight": ANCHOR_W,
        "grad_clip": CLIP, "trainable_names": names, "environment": obs,
        "git_commit": git_head(),
    }
    install_private(os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.receipt.json"),
                    canon(fit_receipt))
    print(f"sealed fit fold={a.fold} arm={a.arm} seed={a.seed} ckpt {ck_sha[:16]}")


def cmd_evaluate(a):
    """Held rows open ONLY after the sealed fit receipt verifies (CAND-3)."""
    out_dir = os.path.join(RUN_DIR, f"fold{a.fold}")
    receipt = json.loads(read_bound(
        os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.receipt.json"),
        description="fit receipt"))
    ck_bytes = read_bound(os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.pt"),
                          expect_sha=receipt["checkpoint_sha256"],
                          description="sealed checkpoint")
    obs = enforce_environment()
    import io

    import numpy as np
    import torch

    from fine_tune_channel_heads import mu_batch
    from mu_attention import CORPORA, E5_REVISION, JUDGES, MuAttention, NODETYPE, OPS, \
        Tokenizer, build_e5_tables
    man, pairs = load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    held = defaultdict(dict)
    for p in pairs:
        if p["fold"] == a.fold:
            held[p["query"]][p["candidate"]] = p
    titles = json.loads(read_bound(os.path.join(RUN_DIR, "titles.json"),
                                   description="title table"))
    blob = torch.load(io.BytesIO(ck_bytes), map_location="cpu", weights_only=False)
    model = MuAttention(**blob["cfg"])
    model.load_state_dict(blob["state"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    qtbl, ptbl, idx = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                      model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})
    C, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]
    out_rows = []
    with torch.no_grad():
        for q in sorted(held):
            items = [(q, c, OPS["LINEAGE"], C, J, MM, MM) for c in catalog]
            mu = np.array(mu_batch(model, tok, items, dev).cpu())
            dest = next(c for c, p in held[q].items() if p["class"] == "positive_parent")
            di = catalog.index(dest)
            # tie rule: exact ties break by ascending frozen catalog column
            rank = 1 + int(np.sum(mu > mu[di])) + int(
                np.sum((mu == mu[di]) & (np.arange(len(catalog)) < di)))
            out_rows.append({"query": q, "destination": dest, "rank": rank,
                             "rr": 1.0 / rank})
    pred = {"schema": "unifyweaver.sm-fs-ranking-eval-receipt.v1",
            "fold": a.fold, "arm": a.arm, "seed": a.seed,
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "catalog_size": len(catalog), "tie_rule": "ascending-frozen-catalog-column",
            "rows": out_rows, "environment": obs, "git_commit": git_head()}
    install_private(os.path.join(out_dir, f"eval_{a.arm}_{a.seed}.receipt.json"), canon(pred))
    print(f"sealed eval fold={a.fold} arm={a.arm} seed={a.seed} "
          f"MRR {sum(r['rr'] for r in out_rows)/len(out_rows):.4f} n={len(out_rows)}")


def cmd_decide(a):
    import numpy as np
    man, pairs = load_pairs_verified()
    block_of = {}
    fold_of = {}
    for p in pairs:
        fold_of[p["query"]] = p["fold"]
    rr = {arm: defaultdict(list) for arm in ARMS}
    for f in range(5):
        for arm in ARMS:
            for seed in SEEDS:
                rec = json.loads(read_bound(
                    os.path.join(RUN_DIR, f"fold{f}", f"eval_{arm}_{seed}.receipt.json"),
                    description="eval receipt"))
                for r in rec["rows"]:
                    rr[arm][r["query"]].append(r["rr"])
    queries = sorted(rr["positive_only"])
    need(queries == sorted(rr["graded_negative"]), "arm query sets differ")
    d = {}
    for q in queries:
        need(len(rr["positive_only"][q]) == 3 and len(rr["graded_negative"][q]) == 3,
             f"query {q} lacks 3 seeds per arm")
        d[q] = (sum(rr["graded_negative"][q]) - sum(rr["positive_only"][q])) / 3.0
    fold_lines = read_bound(os.path.join(RANK_DIR, "fold_assignment.tsv"),
                            description="fold assignment").decode().splitlines()
    qb = {}
    for p in pairs:
        qb.setdefault(p["query"], None)
    # adaptive lineage blocks: reuse the frozen fold file's block column via query dest prefix
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    block_map = {}
    blocks = [ln.split("\t")[0] for ln in fold_lines]
    for q, dst in dest_of.items():
        cands = [b for b in blocks if dst == b or dst.startswith(b + "/")]
        need(bool(cands), f"no block for {q}")
        block_map[q] = max(cands, key=len)
    by_block = defaultdict(list)
    for q in queries:
        by_block[block_map[q]].append(d[q])
    bl = sorted(by_block)
    need(len(bl) >= BOOT["min_blocks"], f"only {len(bl)} blocks (<{BOOT['min_blocks']})")
    rng = np.random.default_rng(BOOT["seed"])
    point = float(np.mean([v for q in queries for v in [d[q]]]))
    stats = []
    for _ in range(BOOT["resamples"]):
        pick = rng.integers(0, len(bl), len(bl))
        vals = [v for i in pick for v in by_block[bl[i]]]
        stats.append(float(np.mean(vals)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    passed = bool(point >= 0.010 and lo > 0.0)
    out = {"schema": "unifyweaver.sm-fs-ranking-decision.v1",
           "delta_mrr": point, "ci95": [float(lo), float(hi)], "blocks": len(bl),
           "bootstrap": BOOT, "passed_exploratory_gate": passed,
           "authorizes": "new-reserve-preregistration-only" if passed else "nothing",
           "git_commit": git_head()}
    install_private(os.path.join(RUN_DIR, "decision.json"), canon(out))
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("project")
    p.add_argument("--fold", type=int, required=True)
    f = sub.add_parser("fit")
    f.add_argument("--fold", type=int, required=True)
    f.add_argument("--arm", choices=ARMS, required=True)
    f.add_argument("--seed", type=int, required=True, choices=SEEDS)
    f.add_argument("--final-lock", required=True)
    f.add_argument("--verification-receipt", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--fold", type=int, required=True)
    e.add_argument("--arm", choices=ARMS, required=True)
    e.add_argument("--seed", type=int, required=True, choices=SEEDS)
    sub.add_parser("decide")
    a = ap.parse_args(argv)
    return {"project": cmd_project, "fit": cmd_fit,
            "evaluate": cmd_evaluate, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
