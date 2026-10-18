#!/usr/bin/env python3
"""SM-FS ranking pipeline v4 — CAND3-1..5 replacement; fitting chain-locked.

Changes from rejected candidate v3 (REVIEW_sm_fs_ranking_candidate_v3_lock.md):

CAND3-1  Finalization lives in sm_fs_ranking_chain: the accepted review must be GIT-TRACKED with
         bytes equal to the blob at HEAD, schema-registered, ID-rederived, and explicitly accept
         the exact candidate SHA; the final lock's execution bindings are compared field-by-field
         with the reviewed candidate (drift outside the explicit amendment whitelist rejected);
         both preregistrations are bound and the retention cascade checked; the chain context
         hands VERIFIED BYTES (plan, titles) to every capability.
CAND3-2  Fit receipts bind the full chain (candidate SHA, review ID, final-lock SHA, plan SHA,
         execution commit, exact inventory vs the candidate's bound inventory, environment,
         optimizer/budget) and validate_fit_receipt checks every one against the live chain.
         `decide` RECOMPUTES destination, rank, and reciprocal rank from each receipt's bound
         finite score vector — a caller-supplied rr is never trusted — and validates the complete
         30-job/query/fold population plus checkpoint and fit-receipt chains.
CAND3-3  Weighted MSE/MAE (pair objective weights), relation & hardness slices, R@1/5/10,
         nDCG, AUC, best-leaf-title sensitivity, per-candidate scores in every eval receipt;
         decision reports three whole-population seed-specific results per arm, bootstrap
         mean/attempts, held/blocks/unique-destination counts. Runtime provenance includes the
         driver (required when CUDA is present). Control arms (unchanged warm start, frozen e5
         title-cosine) are registered evaluate-time descriptive arms under the same chain gate.
CAND3-4  install_private: rollback failures are RAISED (never suppressed into success), the
         rollback unlink is made durable with a directory fsync, and errors compose; the fitter
         consumes chain-verified bytes instead of reopening unauthenticated paths.
CAND3-5  test_sm_fs_ranking_pipeline.py adds adversarial end-to-end coverage: forged/untracked
         reviews, post-review drift, forged fit and eval receipts, decision recomputation on a
         full synthetic chain in a throwaway git repo, and rollback-failure injection.

Fitting remains blocked: no accepted review, amended preregistration, or final lock exists.
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
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))
RANK_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_v1")
RUN_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_run_v4")     # fresh namespace
SEEDS = (3997001, 3997002, 3997003)
ARMS = ("positive_only", "graded_negative")
CONTROL_ARMS = ("control_warm_start", "control_e5_cosine")
STEPS, BS, DRAWS, LR, ANCHOR_W, CLIP = 800, 48, 24, 5e-4, 1.0, 1.0
ADAM = {"class": "torch.optim.Adam", "lr": LR, "betas": [0.9, 0.999], "eps": 1e-08,
        "weight_decay": 0, "amsgrad": False}
BUCKET_SLOTS = [("hard", 3), ("medium", 2), ("easy", 1)]
INIT_SHA = {
    3997001: "a3bf4c0588cc3e4cf1ad335b66440c113e7ee11fd116bd7307a3cee57447098e",
    3997002: "fb353e693951819683793641464155e144caad73c553039fc64f1c6e253ad796",
    3997003: "f42bdea0071a64dca0dad5312178127906b96215ba6890f6a37ad19d87ffdd5a",
}
PLAN_SCHEMA = "unifyweaver.sm-fs-ranking-training-plan.v4"
FIT_SCHEMA = "unifyweaver.sm-fs-ranking-fit-receipt.v3"
EVAL_SCHEMA = "unifyweaver.sm-fs-ranking-eval-receipt.v3"
DEC_SCHEMA = "unifyweaver.sm-fs-ranking-decision.v3"


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
    """Crash-atomic private install. Rollback failures are RAISED, never suppressed into
    success, and the rollback unlink is itself made durable (CAND3-4)."""
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
        _fsync_dir(d)
        st = os.lstat(path)
        need(stat.S_ISREG(st.st_mode), "installed non-regular file")
        need(stat.S_IMODE(st.st_mode) == 0o600, "installed mode is not 0600")
        need(st.st_nlink == 1, "installed link count != 1")
        read_bound(path, expect_sha=sha_bytes(data), private=True,
                   description=f"installed {path}")
        return sha_bytes(data)
    except BaseException as primary:
        rollback_errors = []
        for p in (stage, path if installed else None):
            if not p:
                continue
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
            except OSError as exc:
                rollback_errors.append(f"{p}: {exc}")
        try:
            _fsync_dir(d)                              # make the rollback durable too
        except OSError as exc:
            rollback_errors.append(f"fsync {d}: {exc}")
        if rollback_errors:
            raise PipelineError(
                f"install failed AND rollback incomplete ({'; '.join(rollback_errors)}); "
                f"original error: {primary}") from primary
        raise


def _nvidia_driver():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    except FileNotFoundError:
        return None


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
    cuda = torch.cuda.is_available()
    runtime = {"cuda_available": cuda,
               "cuda_version": getattr(torch.version, "cuda", None),
               "device_name": torch.cuda.get_device_name(0) if cuda else None,
               "driver": _nvidia_driver() if cuda else None}
    for key, want in (("deterministic_algorithms", True), ("cudnn_deterministic", True),
                      ("cudnn_benchmark", False), ("tf32_matmul", False),
                      ("tf32_cudnn", False), ("matmul_precision", "highest"),
                      ("cublas_workspace", ":4096:8")):
        need(invariant[key] == want, f"environment {key}={invariant[key]!r} != {want!r}")
    if cuda:
        need(runtime["driver"], "CUDA present but driver provenance unavailable (required)")
    return {"invariant": invariant, "runtime": runtime}


def git_head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    need(r.returncode == 0, "git rev-parse failed")
    return r.stdout.strip()


def lane_clean():
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no",
                        "--", "prototypes/mu_cosine"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
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
    from fine_tune_channel_heads import load_expanded
    fd, tmp = tempfile.mkstemp(prefix=".ckpt-bound-", suffix=".pt")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(ckpt_bytes)
        return load_expanded(tmp, dev=dev)
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
    tensors, shapes = [], []
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


def recompute_rank(scores, catalog, dest_index):
    """Rank from a bound finite score vector (CAND3-2: never trust a stored rank/rr)."""
    need(len(scores) == len(catalog), "score vector length != catalog size")
    for j, v in enumerate(scores):
        need(isinstance(v, (int, float)) and not isinstance(v, bool)
             and math.isfinite(v), f"nonfinite score at catalog column {j}")
    d = scores[dest_index]
    rank = 1
    for j, v in enumerate(scores):
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
        pos_counts, bucket_counts = defaultdict(int), defaultdict(lambda: defaultdict(int))
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
        "job_count": 30,
        "control_arms": list(CONTROL_ARMS),
        "steps": STEPS, "batch_size": BS, "query_draws_per_step": DRAWS,
        "adam": ADAM, "anchor_weight": ANCHOR_W, "grad_clip": CLIP, "early_stopping": False,
        "projections": projections,
        "sampler_module_sha256": sha_bytes(read_bound(
            os.path.join(ROOT, "sm_fs_sampler.py"))),
        "bootstrap_module_sha256": sha_bytes(read_bound(
            os.path.join(ROOT, "sm_fs_bootstrap.py"))),
        "chain_module_sha256": sha_bytes(read_bound(
            os.path.join(ROOT, "sm_fs_ranking_chain.py"))),
        "initialized_checkpoints": {str(s): INIT_SHA[s] for s in SEEDS},
        "receipt_schemas": [FIT_SCHEMA, EVAL_SCHEMA, DEC_SCHEMA],
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
        print(f"plan v4 -> {out} (sha {sha_bytes(data)[:16]}, 30 jobs + controls)")


def chain_context(final_lock_path, receipt_path):
    """Full chain verification; the context every capability requires (CAND3-1/2)."""
    import sm_fs_ranking_chain as chain
    from sm_fs_ranking_lock_verify import BOUND_KEYS, _bindings
    lock_bytes = read_bound(final_lock_path, description="final lock")
    receipt_bytes = read_bound(receipt_path, description="verification receipt")
    lock, candidate, review = chain.verify_final_state(
        lock_bytes, RUN_DIR, REPO_ROOT, _bindings, BOUND_KEYS)
    receipt = chain.verify_receipt(receipt_bytes, lock_bytes)
    plan_bytes = read_bound(os.path.join(RUN_DIR, "training_plan.json"),
                            expect_sha=candidate["training_plan_sha256"], private=True,
                            description="chain-bound training plan")
    titles_bytes = read_bound(os.path.join(RUN_DIR, "titles.json"),
                              expect_sha=candidate["title_table_sha256"], private=True,
                              description="chain-bound title table")
    return {
        "lock": lock, "candidate": candidate, "review": review, "receipt": receipt,
        "final_lock_sha256": sha_bytes(lock_bytes),
        "candidate_sha256": lock["candidate_sha256"], "review_id": lock["review_id"],
        "plan": json.loads(plan_bytes), "plan_sha256": sha_bytes(plan_bytes),
        "titles": json.loads(titles_bytes),
        "execution_commit": lock["execution_commit"],
    }


def cmd_fit(a):
    ctx = chain_context(a.final_lock, a.verification_receipt)       # raises today
    env = enforce_environment()
    import copy
    import io

    import numpy as np
    import torch

    from mu_attention import E5_REVISION, Tokenizer, build_e5_tables
    plan = ctx["plan"]
    proj = plan["projections"][str(a.fold)]
    rows_bytes = read_bound(os.path.join(RUN_DIR, f"fold{a.fold}", "train_projection.jsonl"),
                            expect_sha=proj["projection_sha256"], private=True,
                            description="plan-bound train projection")
    rows = [json.loads(l) for l in rows_bytes.decode().splitlines()]
    for p in rows:
        need(p["fold"] != a.fold, "held-fold row inside plan-bound projection")
    titles = ctx["titles"]                                          # chain-verified bytes
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
    need(shapes == ctx["candidate"]["trainable_names_shapes"],
         "resolved inventory differs from the reviewed candidate's bound inventory")
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
    buf = io.BytesIO()
    torch.save({"state": model.state_dict(), "cfg": cfg}, buf)
    out_dir = os.path.join(RUN_DIR, f"fold{a.fold}")
    ck_sha = install_private(os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.pt"), buf.getvalue())
    fit_receipt = {
        "schema": FIT_SCHEMA, "fold": a.fold, "arm": a.arm, "seed": a.seed,
        "init_sha256": INIT_SHA[a.seed], "checkpoint_sha256": ck_sha,
        "projection_sha256": proj["projection_sha256"],
        "plan_sha256": ctx["plan_sha256"], "final_lock_sha256": ctx["final_lock_sha256"],
        "candidate_sha256": ctx["candidate_sha256"], "review_id": ctx["review_id"],
        "execution_commit": ctx["execution_commit"],
        "steps": STEPS, "batch_size": BS, "adam": ADAM, "anchor_weight": ANCHOR_W,
        "grad_clip": CLIP, "trainable_names_shapes": shapes,
        "loss_first_last": [losses[0], losses[-1]],
        "environment": env, "git_commit": git_head(),
    }
    install_private(os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.receipt.json"),
                    canon(fit_receipt))
    print(f"sealed fit fold={a.fold} arm={a.arm} seed={a.seed} ckpt {ck_sha[:16]}")


def validate_fit_receipt(rec, fold, arm, seed, ctx):
    """Exact, chain-bound validation (CAND3-2): every field against the live chain."""
    need(rec.get("schema") == FIT_SCHEMA, "fit receipt schema mismatch")
    need((rec.get("fold"), rec.get("arm"), rec.get("seed")) == (fold, arm, seed),
         "fit receipt job identity mismatch")
    need(rec.get("init_sha256") == INIT_SHA[seed],
         "fit receipt init hash is not the countersigned checkpoint")
    need(rec.get("projection_sha256")
         == ctx["plan"]["projections"][str(fold)]["projection_sha256"],
         "fit receipt projection is not the plan-bound projection")
    need(rec.get("plan_sha256") == ctx["plan_sha256"], "fit receipt plan chain broken")
    need(rec.get("final_lock_sha256") == ctx["final_lock_sha256"],
         "fit receipt final-lock chain broken")
    need(rec.get("candidate_sha256") == ctx["candidate_sha256"],
         "fit receipt candidate chain broken")
    need(rec.get("review_id") == ctx["review_id"], "fit receipt review chain broken")
    need(rec.get("execution_commit") == ctx["execution_commit"],
         "fit receipt commit differs from the bound execution commit")
    need(rec.get("trainable_names_shapes") == ctx["candidate"]["trainable_names_shapes"],
         "fit receipt inventory differs from the reviewed candidate's bound inventory")
    need(rec.get("adam") == ADAM and rec.get("steps") == STEPS
         and rec.get("batch_size") == BS and rec.get("anchor_weight") == ANCHOR_W
         and rec.get("grad_clip") == CLIP, "fit receipt optimizer/budget mismatch")
    env = rec.get("environment")
    need(isinstance(env, dict)
         and env.get("invariant") == ctx["candidate"]["environment_invariant"],
         "fit receipt environment invariant differs from the reviewed candidate")
    rt = env.get("runtime", {}) if isinstance(env, dict) else {}
    need("cuda_available" in rt, "fit receipt runtime missing")
    if rt.get("cuda_available"):
        need(rt.get("driver"), "fit receipt lacks required driver provenance")
    need(isinstance(rec.get("checkpoint_sha256"), str)
         and len(rec["checkpoint_sha256"]) == 64, "fit receipt checkpoint hash malformed")


def _leaf(path):
    return path.rsplit("/", 1)[-1]


def cmd_evaluate(a):
    ctx = chain_context(a.final_lock, a.verification_receipt)
    out_dir = os.path.join(RUN_DIR, f"fold{a.fold}")
    is_control = a.arm in CONTROL_ARMS
    rec = None
    ck_bytes = None
    if not is_control:
        rec = json.loads(read_bound(
            os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.receipt.json"), private=True,
            description="fit receipt"))
        validate_fit_receipt(rec, a.fold, a.arm, a.seed, ctx)       # BEFORE held bytes
        ck_bytes = read_bound(os.path.join(out_dir, f"fit_{a.arm}_{a.seed}.pt"),
                              expect_sha=rec["checkpoint_sha256"], private=True,
                              description="sealed checkpoint")
    elif a.arm == "control_warm_start":
        ck_bytes = read_bound(os.path.join(RANK_DIR, f"init_seed{a.seed}.pt"),
                              expect_sha=INIT_SHA[a.seed], private=True,
                              description="warm-start control checkpoint")
    env = enforce_environment()
    import torch

    from fine_tune_channel_heads import mu_batch
    from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, \
        build_e5_tables
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    titles = ctx["titles"]
    qtbl, ptbl, idx = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                      model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})
    model = None
    if a.arm != "control_e5_cosine":
        model, _ = load_checkpoint_bytes(ck_bytes, dev)
        model.eval()
    man, pairs, _ = load_pairs_verified()                           # held bytes open here
    catalog = sorted({p["candidate"] for p in pairs})
    catalog_sha = sha_bytes("\n".join(catalog).encode())
    held = defaultdict(dict)
    for p in pairs:
        if p["fold"] == a.fold:
            held[p["query"]][p["candidate"]] = p
    C, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]
    Pn = ptbl.numpy()
    Qn = qtbl.numpy()
    out_rows = []
    with torch.no_grad():
        for q in sorted(held):
            if a.arm == "control_e5_cosine":
                qv = Qn[idx[q]]
                mu = [float(qv @ Pn[idx[c]]) for c in catalog]
            else:
                items = [(q, c, OPS["LINEAGE"], C, J, MM, MM) for c in catalog]
                mu = [float(v) for v in mu_batch(model, tok, items, dev).cpu()]
            dest = next(c for c, p in held[q].items() if p["class"] == "positive_parent")
            di = catalog.index(dest)
            rank = recompute_rank(mu, catalog, di)
            wsum = werr = wabs = 0.0
            slices = defaultdict(lambda: [0.0, 0])
            for c, p in held[q].items():
                m = mu[catalog.index(c)]
                w = float(p.get("weight", 1.0))
                wsum += w
                werr += w * (m - p["target"]) ** 2
                wabs += w * abs(m - p["target"])
                for key in (p.get("relation") or p["class"],
                            ("hardness:" + p["hardness"]) if "hardness" in p else None):
                    if key:
                        slices[key][0] += (m - p["target"]) ** 2
                        slices[key][1] += 1
            pos_set = {c for c, p in held[q].items() if p["class"].startswith("positive")}
            ps = [mu[catalog.index(c)] for c in pos_set]
            ns = [mu[catalog.index(c)] for c in held[q] if c not in pos_set]
            wins = sum(1 for x in ps for y in ns if x > y)
            ties = sum(1 for x in ps for y in ns if x == y)
            auc = (wins + 0.5 * ties) / (len(ps) * len(ns)) if ps and ns else None
            order = sorted(range(len(catalog)), key=lambda j: (-mu[j], j))
            dcg = sum(held[q].get(catalog[j], {}).get("target", 0.0)
                      / math.log2(r + 2) for r, j in enumerate(order[:10]))
            ideal = sorted((p["target"] for p in held[q].values()), reverse=True)[:10]
            idcg = sum(t / math.log2(r + 2) for r, t in enumerate(ideal))
            leaf = _leaf(dest)
            title_rank = min(recompute_rank(mu, catalog, j)
                             for j, c in enumerate(catalog) if _leaf(c) == leaf)
            out_rows.append({
                "query": q, "destination": dest, "rank": rank, "rr": 1.0 / rank,
                "scores": mu, "weighted_mse": werr / wsum, "weighted_mae": wabs / wsum,
                "slice_mse": {k: v[0] / v[1] for k, v in sorted(slices.items())},
                "auc": auc, "ndcg10": (dcg / idcg if idcg else None),
                "title_equivalent_rank": title_rank,
            })
    pred = {"schema": EVAL_SCHEMA, "fold": a.fold, "arm": a.arm, "seed": a.seed,
            "checkpoint_sha256": (rec["checkpoint_sha256"] if rec
                                  else (INIT_SHA[a.seed]
                                        if a.arm == "control_warm_start" else None)),
            "fit_receipt_sha256": (sha_bytes(canon(rec)) if rec else None),
            "final_lock_sha256": ctx["final_lock_sha256"],
            "candidate_sha256": ctx["candidate_sha256"], "review_id": ctx["review_id"],
            "catalog_sha256": catalog_sha, "catalog_size": len(catalog),
            "tie_rule": "ascending-frozen-catalog-column", "rows": out_rows,
            "held_query_count": len(out_rows),
            "r_at": {str(k): sum(1 for r in out_rows if r["rank"] <= k) / len(out_rows)
                     for k in (1, 5, 10)},
            "environment": env, "git_commit": git_head()}
    install_private(os.path.join(out_dir, f"eval_{a.arm}_{a.seed}.receipt.json"), canon(pred))
    mrr = sum(r["rr"] for r in out_rows) / len(out_rows)
    print(f"sealed eval fold={a.fold} arm={a.arm} seed={a.seed} MRR {mrr:.4f} n={len(out_rows)}")


def cmd_decide(a):
    ctx = chain_context(a.final_lock, a.verification_receipt)
    from sm_fs_bootstrap import decide as boot_decide
    man, pairs, fold_txt = load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    catalog_sha = sha_bytes("\n".join(catalog).encode())
    cat_index = {c: j for j, c in enumerate(catalog)}
    fold_of = {p["query"]: p["fold"] for p in pairs}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    rr = {arm: defaultdict(dict) for arm in ARMS}
    for f in range(5):
        for arm in ARMS:
            for seed in SEEDS:
                fit_rec = json.loads(read_bound(
                    os.path.join(RUN_DIR, f"fold{f}", f"fit_{arm}_{seed}.receipt.json"),
                    private=True, description="fit receipt"))
                validate_fit_receipt(fit_rec, f, arm, seed, ctx)
                rec = json.loads(read_bound(
                    os.path.join(RUN_DIR, f"fold{f}", f"eval_{arm}_{seed}.receipt.json"),
                    private=True, description="eval receipt"))
                need(rec.get("schema") == EVAL_SCHEMA, "eval receipt schema")
                need((rec["fold"], rec["arm"], rec["seed"]) == (f, arm, seed),
                     "eval receipt job identity mismatch")
                need(rec.get("checkpoint_sha256") == fit_rec["checkpoint_sha256"],
                     "eval receipt checkpoint chain broken")
                need(rec.get("fit_receipt_sha256") == sha_bytes(canon(fit_rec)),
                     "eval receipt does not bind the validated fit receipt")
                need(rec.get("final_lock_sha256") == ctx["final_lock_sha256"]
                     and rec.get("candidate_sha256") == ctx["candidate_sha256"]
                     and rec.get("review_id") == ctx["review_id"],
                     "eval receipt chain fields broken")
                need(rec.get("catalog_sha256") == catalog_sha
                     and rec.get("catalog_size") == len(catalog), "eval catalog mismatch")
                need(rec.get("tie_rule") == "ascending-frozen-catalog-column", "tie rule")
                seen = set()
                for row in rec["rows"]:
                    q = row["query"]
                    need(q not in seen, f"duplicate query {q} in receipt")
                    seen.add(q)
                    need(fold_of.get(q) == f, f"query {q} not in fold {f}")
                    need(dest_of.get(q) == row.get("destination"),
                         f"receipt destination for {q} differs from the frozen bundle")
                    rank = recompute_rank(row.get("scores", []), catalog,
                                          cat_index[dest_of[q]])
                    need(rank == row.get("rank"), f"stored rank for {q} != recomputed")
                    rr[arm].setdefault(q, {})[seed] = 1.0 / rank    # recomputed, not trusted
    all_held = set(fold_of)
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
        "unique_destinations": len({dest_of[q] for q in all_held}),
        "seed_specific_mrr": {arm: {str(s): (sum(rr[arm][q][s] for q in all_held)
                                             / len(all_held)) for s in SEEDS}
                              for arm in ARMS},
        "plan_sha256": ctx["plan_sha256"], "final_lock_sha256": ctx["final_lock_sha256"],
        "candidate_sha256": ctx["candidate_sha256"], "review_id": ctx["review_id"],
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
    for name in ("fit", "evaluate", "decide"):
        p = sub.add_parser(name)
        p.add_argument("--final-lock", required=True)
        p.add_argument("--verification-receipt", required=True)
        if name != "decide":
            p.add_argument("--fold", type=int, required=True)
            p.add_argument("--arm", required=True,
                           choices=ARMS + (CONTROL_ARMS if name == "evaluate" else ()))
            p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "fit": cmd_fit,
            "evaluate": cmd_evaluate, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
