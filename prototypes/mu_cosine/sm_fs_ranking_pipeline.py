#!/usr/bin/env python3
"""SM-FS ranking pipeline v3 — CAND2-1..8 replacement; fitting double-locked.

Changes from the rejected v2 candidate (REVIEW_sm_fs_ranking_candidate_v2_lock.md):

CAND2-1  Checkpoints load through the ESTABLISHED path: verified descriptor-bound bytes are
         staged to a 0600 tempfile and loaded with fine_tune_channel_heads.load_expanded, which
         accepts the countersigned legacy cfg ({heads, layers, judge_name, ridge, ...}).
         An executable test loads a real countersigned checkpoint, constructs the optimizer,
         and runs one fit step.
CAND2-2  The sampler seam is ID-free (sm_fs_sampler.py); this pipeline imports no module that
         hard-codes a preregistration ID. `plan` emits the fresh v3 plan: the exact 30-job
         matrix, ALL FIVE per-fold projection identities, schedule populations, and its own hash.
CAND2-3  Finalization chain lives in sm_fs_ranking_lock_verify: the final lock must bind the
         reviewed candidate SHA and the independent review ID from the rigor lane's COMMITTED
         review artifact — caller-authored JSON cannot substitute.
CAND2-4  Nothing trusts adjacent self-authored metadata: `fit` authenticates its projection
         against the PLAN-bound hash; `evaluate` fully validates the fit receipt (schema, job
         identity, countersigned init hash, plan-bound projection, lock chain, trainable names,
         environment) and verifies the checkpoint bytes BEFORE any held outcome is opened;
         `decide` validates every eval receipt (schema, job identity, checkpoint chain, catalog
         hash, tie rule, finiteness, query uniqueness, exact fold membership, complete
         361-query population). Eval receipts store the FULL per-candidate score vector plus
         registered diagnostics (nDCG, AUC, MSE/MAE, per-relation and per-hardness) so no
         reporting choice survives scoring.
CAND2-6  Decision inference = sm_fs_bootstrap (frozen SHA-256 rejection sampler, 82 draws per
         replicate, query-weighted block mean, nearest-rank endpoints 249/9749). Nonfinite
         scores fail closed at scoring time — a NaN can never rank.
CAND2-7  install_private: parent-directory mode/owner/symlink checks, unlink-target rollback on
         post-verify failure, directory fsync after staging cleanup; read_bound(private=True)
         requires mode 0600.
CAND2-8  test_sm_fs_ranking_pipeline.py exercises the real loader+optimizer+one-step fit,
         scoring with NaN rejection, receipt acceptance/rejection, bootstrap known answers, and
         the decision path.

Fitting remains blocked: cmd_fit calls sm_fs_ranking_lock_verify.fitting_allowed, which requires
the live prereg to authorize fitting AND a final lock bound to the reviewed candidate SHA and
review ID. No such artifacts exist yet.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
RANK_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_v1")
RUN_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_run_v3")     # fresh namespace
SEEDS = (3997001, 3997002, 3997003)
ARMS = ("positive_only", "graded_negative")
STEPS, BS, DRAWS, LR, ANCHOR_W, CLIP = 800, 48, 24, 5e-4, 1.0, 1.0
ADAM = {"class": "torch.optim.Adam", "lr": LR, "betas": [0.9, 0.999], "eps": 1e-08,
        "weight_decay": 0, "amsgrad": False}
BUCKET_SLOTS = [("hard", 3), ("medium", 2), ("easy", 1)]
INIT_SHA = {
    3997001: "a3bf4c0588cc3e4cf1ad335b66440c113e7ee11fd116bd7307a3cee57447098e",
    3997002: "fb353e693951819683793641464155e144caad73c553039fc64f1c6e253ad796",
    3997003: "f42bdea0071a64dca0dad5312178127906b96215ba6890f6a37ad19d87ffdd5a",
}
PLAN_SCHEMA = "unifyweaver.sm-fs-ranking-training-plan.v3"
PROJ_SCHEMA = "unifyweaver.sm-fs-ranking-train-projection.v1"
FIT_SCHEMA = "unifyweaver.sm-fs-ranking-fit-receipt.v2"
EVAL_SCHEMA = "unifyweaver.sm-fs-ranking-eval-receipt.v2"
DEC_SCHEMA = "unifyweaver.sm-fs-ranking-decision.v2"


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


def read_bound(path, expect_sha=None, description="input", private=False):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PipelineError(f"{description} unavailable: {path}") from exc
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode), f"{description} is not a regular file")
        need(st.st_nlink == 1, f"{description} must have exactly one hard link")
        if private:
            need(stat.S_IMODE(st.st_mode) == 0o600, f"{description} is not mode 0600")
            need(st.st_uid == os.geteuid(), f"{description} is not owned by this user")
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


def _check_parent(d):
    st = os.lstat(d)
    need(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode),
         f"parent {d} is not a real directory")
    need(st.st_uid == os.geteuid(), f"parent {d} not owned by this user")
    need(stat.S_IMODE(st.st_mode) == 0o700, f"parent {d} is not mode 0700")


def _fsync_dir(d):
    dfd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def install_private(path, data):
    """Crash-atomic private install: parent checks, 0600 staging + fsync, hard-link no-replace,
    dir fsync, staging cleanup + dir fsync, post-install verification with TARGET ROLLBACK."""
    d = os.path.dirname(path)
    os.makedirs(d, mode=0o700, exist_ok=True)
    _check_parent(d)
    stage = os.path.join(d, f".stage-{os.path.basename(path)}-{os.getpid()}")
    fd = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    installed = False
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(stage, path)
        except FileExistsError as exc:
            raise PipelineError(f"no-replace target exists: {path}") from exc
        installed = True
        _fsync_dir(d)
        os.unlink(stage)
        _fsync_dir(d)                                  # final one-link namespace made durable
        st = os.lstat(path)
        need(stat.S_ISREG(st.st_mode), "installed non-regular file")
        need(stat.S_IMODE(st.st_mode) == 0o600, "installed mode is not 0600")
        need(st.st_nlink == 1, "installed link count != 1")
        read_bound(path, expect_sha=sha_bytes(data), private=True,
                   description=f"installed {path}")
        return sha_bytes(data)
    except BaseException:
        for p in (stage, path if installed else None):
            if p:
                try:
                    os.unlink(p)                       # rollback the partial target too
                except OSError:
                    pass
        raise


def enforce_environment():
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
    invariant = {
        "python": platform.python_version(), "numpy": numpy.__version__,
        "torch": torch.__version__, "dtype": "float32",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "threads": torch.get_num_threads(),
        "rng_reset_order": "torch.manual_seed(seed) once before checkpoint load; "
                           "numpy default_rng(seed+1) for augmentation, consumed in row order; "
                           "training sampler and bootstrap are counter-based (no RNG)",
    }
    runtime = {"cuda_available": torch.cuda.is_available(),
               "cuda_version": getattr(torch.version, "cuda", None),
               "device_name": (torch.cuda.get_device_name(0)
                               if torch.cuda.is_available() else None)}
    for key, want in (("deterministic_algorithms", True), ("cudnn_deterministic", True),
                      ("cudnn_benchmark", False), ("tf32_matmul", False),
                      ("tf32_cudnn", False), ("matmul_precision", "highest"),
                      ("cublas_workspace", ":4096:8")):
        need(invariant[key] == want, f"environment {key}={invariant[key]!r} != {want!r}")
    return {"invariant": invariant, "runtime": runtime}


def git_head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    need(r.returncode == 0, "git rev-parse failed")
    return r.stdout.strip()


def lane_clean():
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
    folds = read_bound(os.path.join(RANK_DIR, "fold_assignment.tsv"),
                       expect_sha=man["outputs"]["fold_assignment.tsv"],
                       description="fold assignment")
    return man, [json.loads(l) for l in data.decode().splitlines()], folds.decode()


def load_checkpoint_bytes(ckpt_bytes, dev):
    """CAND2-1: established loader over verified bytes (0600 tempfile → load_expanded)."""
    from fine_tune_channel_heads import load_expanded
    fd, tmp = tempfile.mkstemp(prefix=".ckpt-bound-", suffix=".pt")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(ckpt_bytes)
        model, cfg = load_expanded(tmp, dev=dev)
        return model, cfg
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def resolve_allowlist(model):
    names = []
    for prefix in ("judge_name.resid.weight", "corpus_name.resid.weight",
                   "op_name.resid.weight"):
        obj = model
        found = True
        for part in prefix.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                found = False
                break
        if found:
            names.append(prefix)
    li = len(model.encoder.layers) - 1
    names += [f"encoder.layers.{li}.{n}" for n, _ in
              model.encoder.layers[li].named_parameters()]
    names += ["readout_w", "readout_b", "nodetype_emb.weight"]
    by_name = dict(model.named_parameters())
    by_name.setdefault("readout_w", model.readout_w)
    by_name.setdefault("readout_b", model.readout_b)
    need(all(n in by_name for n in names), "allowlist name missing from model")
    need(len(names) == 18, f"allowlist has {len(names)} tensors, requires 18")
    for p in model.parameters():
        p.requires_grad = False
    tensors = []
    shapes = []
    for n in names:
        by_name[n].requires_grad = True
        tensors.append(by_name[n])
        shapes.append([n, list(by_name[n].shape)])
    count = sum(p.numel() for p in tensors)
    need(count == 1195782, f"trainable param count {count} != 1195782")
    return names, shapes, tensors


def sample_step(fold, seed, step, train_q, pos, buckets, arm):
    from sm_fs_sampler import sampler_index
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


def fit_one_step(model, ref, tensors, opt, tok, rows, sel_common, sel_contrast,
                 aug_rng, dev):
    """The exact per-step objective — factored so tests can execute it (CAND2-8)."""
    import torch

    from fine_tune_channel_heads import mu_batch
    from mu_attention import CORPORA, JUDGES, NODETYPE, OPS
    C, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]

    def items(sel):
        return [(rows[i]["query"], rows[i]["candidate"], OPS["LINEAGE"], C, J, MM, MM)
                for i in sel]
    sel = sel_common + sel_contrast
    tgt = torch.tensor([rows[i]["target"] for i in sel], dtype=torch.float32, device=dev)
    mu = mu_batch(model, tok, items(sel), dev, train=True, rng=aug_rng)
    loss = torch.mean((mu - tgt) ** 2)
    ag = [(it[0], it[1], it[2]) for it in items(sel_common)]
    mu_ag = mu_batch(model, tok, ag, dev)
    with torch.no_grad():
        mu_ref = mu_batch(ref, tok, ag, dev)
    loss = loss + ANCHOR_W * torch.mean((mu_ag - mu_ref) ** 2)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tensors, CLIP)
    opt.step()
    return float(loss.detach().cpu())


def score_catalog(mu_scores, catalog, dest_index):
    """Frozen tie rule with fail-closed finiteness (CAND2-6): any nonfinite score aborts."""
    for j, v in enumerate(mu_scores):
        need(math.isfinite(v), f"nonfinite score at catalog column {j}")
    d = mu_scores[dest_index]
    rank = 1
    for j, v in enumerate(mu_scores):
        if v > d or (v == d and j < dest_index):
            rank += 1
    return rank


def plan_projection_identity(pairs, fold):
    held_q = sorted({p["query"] for p in pairs if p["fold"] == fold})
    train_rows = [p for p in pairs if p["query"] not in set(held_q)]
    payload = b"".join(canon(p) for p in train_rows)
    return held_q, train_rows, sha_bytes(payload), payload


def cmd_plan(a):
    man, pairs, fold_txt = load_pairs_verified()
    projections = {}
    for f in range(5):
        held_q, train_rows, proj_sha, payload = plan_projection_identity(pairs, f)
        out = os.path.join(RUN_DIR, f"fold{f}", "train_projection.jsonl")
        if not os.path.exists(out):
            install_private(out, payload)
        else:
            read_bound(out, expect_sha=proj_sha, private=True,
                       description=f"existing fold{f} projection")
        pos_counts = defaultdict(int)
        bucket_counts = defaultdict(lambda: defaultdict(int))
        for p in train_rows:
            if p["class"].startswith("positive"):
                pos_counts[p["query"]] += 1
            else:
                bucket_counts[p["query"]][p["hardness"]] += 1
        projections[str(f)] = {
            "projection_sha256": proj_sha, "rows": len(train_rows),
            "held_queries": len(held_q),
            "train_query_list_sha256": sha_bytes("\n".join(sorted(pos_counts)).encode()),
            "population_sha256": sha_bytes(canon(
                {q: {"positives": pos_counts[q],
                     "buckets": dict(sorted(bucket_counts[q].items()))}
                 for q in sorted(pos_counts)})),
        }
    plan = {
        "schema": PLAN_SCHEMA,
        "bundle_manifest_sha256": sha_bytes(read_bound(
            os.path.join(RANK_DIR, "manifest.json"))),
        "jobs": [{"fold": f, "arm": arm, "seed": s}
                 for f in range(5) for s in SEEDS for arm in ARMS],
        "job_count": 30, "steps": STEPS, "batch_size": BS, "query_draws_per_step": DRAWS,
        "adam": ADAM, "anchor_weight": ANCHOR_W, "grad_clip": CLIP, "early_stopping": False,
        "projections": projections,
        "sampler_module_sha256": sha_bytes(read_bound(
            os.path.join(ROOT, "sm_fs_sampler.py"))),
        "bootstrap_module_sha256": sha_bytes(read_bound(
            os.path.join(ROOT, "sm_fs_bootstrap.py"))),
        "initialized_checkpoints": {str(s): INIT_SHA[s] for s in SEEDS},
        "receipt_schemas": [PROJ_SCHEMA, FIT_SCHEMA, EVAL_SCHEMA, DEC_SCHEMA],
        "fitting_authorized": False,
    }
    data = canon(plan)
    out = os.path.join(RUN_DIR, "training_plan.json")
    if os.path.exists(out):
        need(read_bound(out, private=True) == data,
             "training_plan.json exists and differs — no-replace contract")
        print(f"plan unchanged -> {out}")
    else:
        install_private(out, data)
        print(f"plan v3 -> {out} (sha {sha_bytes(data)[:16]}, 30 jobs, all 5 projections)")


def _load_plan():
    return json.loads(read_bound(os.path.join(RUN_DIR, "training_plan.json"),
                                 private=True, description="training plan"))


def cmd_fit(a):
    from sm_fs_ranking_lock_verify import fitting_allowed
    lock, receipt = fitting_allowed(a.final_lock, a.verification_receipt)   # raises today
    env = enforce_environment()
    import copy

    import numpy as np
    import torch

    from mu_attention import E5_REVISION, Tokenizer, build_e5_tables
    plan = _load_plan()
    proj = plan["projections"][str(a.fold)]
    rows_bytes = read_bound(os.path.join(RUN_DIR, f"fold{a.fold}", "train_projection.jsonl"),
                            expect_sha=proj["projection_sha256"], private=True,
                            description="plan-bound train projection")
    rows = [json.loads(l) for l in rows_bytes.decode().splitlines()]
    for p in rows:
        need(p["fold"] != a.fold, "held-fold row inside plan-bound projection")
    titles = json.loads(read_bound(os.path.join(RUN_DIR, "titles.json"), private=True,
                                   description="title table"))
    torch.manual_seed(a.seed)
    aug_rng = np.random.default_rng(a.seed + 1)
    ckpt_bytes = read_bound(os.path.join(RANK_DIR, f"init_seed{a.seed}.pt"),
                            expect_sha=INIT_SHA[a.seed], private=True,
                            description="initialized checkpoint")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint_bytes(ckpt_bytes, dev)
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    names, shapes, tensors = resolve_allowlist(model)
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
    model.train()
    losses = []
    for step in range(STEPS):
        c, k = sample_step(a.fold, a.seed, step, train_q, pos, buckets, a.arm)
        losses.append(fit_one_step(model, ref, tensors, opt, tok, rows, c, k, aug_rng, dev))
    model.eval()
    import io
    buf = io.BytesIO()
    torch.save({"state": model.state_dict(), "cfg": cfg}, buf)
    out_dir = os.path.join(RUN_DIR, f"fold{a.fold}")
    ck_sha = install_private(os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.pt"), buf.getvalue())
    fit_receipt = {
        "schema": FIT_SCHEMA, "fold": a.fold, "arm": a.arm, "seed": a.seed,
        "init_sha256": INIT_SHA[a.seed], "checkpoint_sha256": ck_sha,
        "projection_sha256": proj["projection_sha256"],
        "plan_sha256": sha_bytes(read_bound(os.path.join(RUN_DIR, "training_plan.json"),
                                            private=True)),
        "final_lock_sha256": receipt["final_lock_sha256"],
        "steps": STEPS, "batch_size": BS, "adam": ADAM, "anchor_weight": ANCHOR_W,
        "grad_clip": CLIP, "trainable_names_shapes": shapes,
        "loss_first_last": [losses[0], losses[-1]],
        "environment": env, "git_commit": git_head(),
    }
    install_private(os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.receipt.json"),
                    canon(fit_receipt))
    print(f"sealed fit fold={a.fold} arm={a.arm} seed={a.seed} ckpt {ck_sha[:16]}")


def validate_fit_receipt(rec, fold, arm, seed, plan):
    need(rec.get("schema") == FIT_SCHEMA, "fit receipt schema mismatch")
    need((rec.get("fold"), rec.get("arm"), rec.get("seed")) == (fold, arm, seed),
         "fit receipt job identity mismatch")
    need(rec.get("init_sha256") == INIT_SHA[seed],
         "fit receipt init hash is not the countersigned checkpoint")
    need(rec.get("projection_sha256")
         == plan["projections"][str(fold)]["projection_sha256"],
         "fit receipt projection is not the plan-bound projection")
    need(isinstance(rec.get("trainable_names_shapes"), list)
         and len(rec["trainable_names_shapes"]) == 18, "fit receipt trainable inventory")
    need(isinstance(rec.get("environment"), dict)
         and rec["environment"].get("invariant", {}).get("dtype") == "float32",
         "fit receipt environment missing")
    need(isinstance(rec.get("checkpoint_sha256"), str)
         and len(rec["checkpoint_sha256"]) == 64, "fit receipt checkpoint hash malformed")


def cmd_evaluate(a):
    plan = _load_plan()
    out_dir = os.path.join(RUN_DIR, f"fold{a.fold}")
    rec = json.loads(read_bound(
        os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.receipt.json"), private=True,
        description="fit receipt"))
    validate_fit_receipt(rec, a.fold, a.arm, a.seed, plan)          # BEFORE held bytes
    ck_bytes = read_bound(os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.pt"),
                          expect_sha=rec["checkpoint_sha256"], private=True,
                          description="sealed checkpoint")
    env = enforce_environment()
    import numpy as np
    import torch

    from fine_tune_channel_heads import mu_batch
    from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, \
        build_e5_tables
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint_bytes(ck_bytes, dev)                 # checkpoint verified first
    model.eval()
    man, pairs, _ = load_pairs_verified()                           # held bytes open ONLY now
    catalog = sorted({p["candidate"] for p in pairs})
    catalog_sha = sha_bytes("\n".join(catalog).encode())
    held = defaultdict(dict)
    for p in pairs:
        if p["fold"] == a.fold:
            held[p["query"]][p["candidate"]] = p
    titles = json.loads(read_bound(os.path.join(RUN_DIR, "titles.json"), private=True))
    qtbl, ptbl, idx = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                      model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})
    C, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]
    out_rows = []
    with torch.no_grad():
        for q in sorted(held):
            items = [(q, c, OPS["LINEAGE"], C, J, MM, MM) for c in catalog]
            mu = [float(v) for v in mu_batch(model, tok, items, dev).cpu()]
            dest = next(c for c, p in held[q].items() if p["class"] == "positive_parent")
            di = catalog.index(dest)
            rank = score_catalog(mu, catalog, di)                   # fail-closed nonfinite
            errs = [(mu[catalog.index(c)], p["target"], p.get("relation"),
                     p.get("hardness")) for c, p in held[q].items()]
            mse = sum((m - t) ** 2 for m, t, _, _ in errs) / len(errs)
            mae = sum(abs(m - t) for m, t, _, _ in errs) / len(errs)
            pos_set = {c for c, p in held[q].items() if p["class"].startswith("positive")}
            pos_scores = [mu[catalog.index(c)] for c in pos_set]
            neg_scores = [mu[catalog.index(c)] for c in held[q] if c not in pos_set]
            wins = sum(1 for ps in pos_scores for ns in neg_scores if ps > ns)
            ties = sum(1 for ps in pos_scores for ns in neg_scores if ps == ns)
            auc = ((wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores))
                   if pos_scores and neg_scores else None)
            order = sorted(range(len(catalog)), key=lambda j: (-mu[j], j))
            dcg = sum(held[q].get(catalog[j], {}).get("target", 0.0)
                      / math.log2(r + 2) for r, j in enumerate(order[:10]))
            ideal = sorted((p["target"] for p in held[q].values()), reverse=True)[:10]
            idcg = sum(t / math.log2(r + 2) for r, t in enumerate(ideal))
            out_rows.append({"query": q, "destination": dest, "rank": rank,
                             "rr": 1.0 / rank, "scores": mu, "mse": mse, "mae": mae,
                             "auc": auc, "ndcg10": (dcg / idcg if idcg else None)})
    pred = {"schema": EVAL_SCHEMA, "fold": a.fold, "arm": a.arm, "seed": a.seed,
            "checkpoint_sha256": rec["checkpoint_sha256"],
            "fit_receipt_sha256": sha_bytes(canon(rec)),
            "catalog_sha256": catalog_sha, "catalog_size": len(catalog),
            "tie_rule": "ascending-frozen-catalog-column", "rows": out_rows,
            "held_query_count": len(out_rows), "environment": env,
            "git_commit": git_head()}
    install_private(os.path.join(out_dir, f"eval_{a.arm}_{a.seed}.receipt.json"), canon(pred))
    mrr = sum(r["rr"] for r in out_rows) / len(out_rows)
    print(f"sealed eval fold={a.fold} arm={a.arm} seed={a.seed} MRR {mrr:.4f} n={len(out_rows)}")


def cmd_decide(a):
    from sm_fs_bootstrap import decide as boot_decide
    plan = _load_plan()
    man, pairs, fold_txt = load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    catalog_sha = sha_bytes("\n".join(catalog).encode())
    fold_of = {p["query"]: p["fold"] for p in pairs}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    rr = {arm: defaultdict(dict) for arm in ARMS}
    per_seed_mrr = defaultdict(dict)
    for f in range(5):
        for arm in ARMS:
            for seed in SEEDS:
                rec = json.loads(read_bound(
                    os.path.join(RUN_DIR, f"fold{f}", f"eval_{arm}_{seed}.receipt.json"),
                    private=True, description="eval receipt"))
                need(rec.get("schema") == EVAL_SCHEMA, "eval receipt schema")
                need((rec["fold"], rec["arm"], rec["seed"]) == (f, arm, seed),
                     "eval receipt job identity mismatch")
                need(rec.get("catalog_sha256") == catalog_sha, "eval catalog mismatch")
                need(rec.get("tie_rule") == "ascending-frozen-catalog-column", "tie rule")
                seen = set()
                for row in rec["rows"]:
                    q = row["query"]
                    need(q not in seen, f"duplicate query {q} in receipt")
                    seen.add(q)
                    need(fold_of.get(q) == f, f"query {q} not in fold {f}")
                    need(math.isfinite(row["rr"]) and 0 < row["rr"] <= 1, "rr out of range")
                    rr[arm].setdefault(q, {})[seed] = row["rr"]
                per_seed_mrr[arm][(f, seed)] = (
                    sum(r["rr"] for r in rec["rows"]) / len(rec["rows"]))
    all_held = {q for q, f in fold_of.items()}
    for arm in ARMS:
        need(sorted(rr[arm]) == sorted(all_held),
             f"{arm} evaluated {len(rr[arm])} queries; population requires {len(all_held)}")
        for q, by_seed in rr[arm].items():
            need(sorted(by_seed) == sorted(SEEDS), f"query {q} lacks all 3 seeds ({arm})")
    d = {q: (sum(rr["graded_negative"][q].values()) / 3.0
             - sum(rr["positive_only"][q].values()) / 3.0) for q in sorted(all_held)}
    blocks = [ln.split("\t")[0] for ln in fold_txt.splitlines()]
    block_values = {b: [] for b in blocks}
    for q, val in d.items():
        cands = [b for b in blocks if dest_of[q] == b or dest_of[q].startswith(b + "/")]
        need(bool(cands), f"no lineage block for {q}")
        block_values[max(cands, key=len)].append(val)
    result = boot_decide(block_values)
    result.update({
        "schema": DEC_SCHEMA, "catalog_sha256": catalog_sha,
        "unique_destinations": len(set(dest_of[q] for q in all_held)),
        "per_seed_mrr": {arm: {f"fold{f}/seed{s}": per_seed_mrr[arm][(f, s)]
                               for f in range(5) for s in SEEDS} for arm in ARMS},
        "plan_sha256": sha_bytes(canon(plan)),
        "authorizes": ("new-reserve-preregistration-only"
                       if result["passed_exploratory_gate"] else "nothing"),
        "git_commit": git_head(),
    })
    install_private(os.path.join(RUN_DIR, "decision.json"), canon(result))
    print(json.dumps({k: result[k] for k in
                      ("delta_mrr", "ci95", "passed_exploratory_gate", "authorizes")}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
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
    return {"plan": cmd_plan, "fit": cmd_fit,
            "evaluate": cmd_evaluate, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
