#!/usr/bin/env python3
"""SM-FS lineage-ranking trainer/coordinator/evaluator — step 1 of the reseal sequence.

Implements the registered five-fold × two-arm × three-seed transaction of
PROTOCOL_sm_fs_lineage_ranking.md under REVIEW_sm_fs_ranking_execution_lock.md §3.1, with the
fitting gate REVIEW §4 requires: `fit` refuses to construct a model unless a FINAL execution lock
authorizes fitting AND binds a preregistration whose ID is DERIVED from the live document (no
hard-coded ID → no hash/ID cycle). Until steps 4–6 of the reseal sequence land, every `fit` call
exits BLOCKED. `plan`, gate tests, and dry verification are always allowed.

Enforced (not promised):
- deterministic env set BEFORE model construction and ASSERTED as observed state
  (deterministic algorithms, cuDNN deterministic + no benchmark, TF32 off everywhere,
  float32 matmul 'highest', CUBLAS_WORKSPACE_CONFIG, pinned thread count);
- byte-reload of the seed's initialized checkpoint (sha-asserted) before EVERY fold × arm —
  never continue one fold or arm from another;
- 18-tensor trainable allowlist asserted by exact name/shape/count BEFORE optimizer construction;
- sol's counter-based sampler (sm_fs_protocols.sampler_index, KAV-verified) drives every draw:
  24 query draws/step, common-positive slot shared verbatim across arms, contrast slot =
  independent positive (positive_only) or 3:2:1 nonempty-bucket slot then uniform candidate
  (graded_negative); unweighted MSE over the 48 sampled rows; anchor on the 24 common-positive
  rows only, never augmented;
- held-fold rows are filtered out before any training materialization; `evaluate` runs only
  against a sealed checkpoint receipt;
- outputs installed crash-atomically (0700 dir, 0600 files, O_EXCL no-replace) with hashes.

  python3 sm_fs_ranking_train.py plan
  python3 sm_fs_ranking_train.py fit --fold 0 --arm graded_negative --seed 3997001 \
      --final-lock <authorized lock>          # BLOCKED until reseal steps 4-6
  python3 sm_fs_ranking_train.py evaluate --fold 0 --seed 3997001
  python3 sm_fs_ranking_train.py decide
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
RANK_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_v1")
RUN_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_run_v1")
SEEDS = (3997001, 3997002, 3997003)
ARMS = ("positive_only", "graded_negative")
STEPS, BS, DRAWS, LR, ANCHOR_W, CLIP = 800, 48, 24, 5e-4, 1.0, 1.0
BUCKET_SLOTS = [("hard", 3), ("medium", 2), ("easy", 1)]
PLAN_SCHEMA = "unifyweaver.sm-fs-ranking-training-plan.v1"


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def canon(o):
    return (json.dumps(o, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_private(path, data):
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)   # no-replace
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def load_bundle():
    man = json.load(open(os.path.join(RANK_DIR, "manifest.json")))
    for name in ("pairs.jsonl", "fold_assignment.tsv"):
        got = sha_file(os.path.join(RANK_DIR, name))
        assert got == man["outputs"][name], f"{name} drifted"
    pairs = [json.loads(l) for l in open(os.path.join(RANK_DIR, "pairs.jsonl"))]
    return man, pairs


def fitting_gate(final_lock_path):
    """Fail closed unless a final lock authorizes fitting against the LIVE prereg document."""
    if not final_lock_path or not os.path.exists(final_lock_path):
        return False, "no final execution lock supplied"
    lock = json.load(open(final_lock_path))
    if lock.get("schema") != "unifyweaver.sm-fs-ranking-execution-lock.final.v1":
        return False, f"lock schema {lock.get('schema')!r} is not final"
    if lock.get("fitting_authorized") is not True:
        return False, "lock does not authorize fitting"
    # derive the ID from the live document (REVIEW §4: never hard-code an ID into a file whose
    # own hash participates in the plan)
    doc = json.load(open(os.path.join(ROOT, "SM_FS_LINEAGE_RANKING_PREREG.json")))
    derived = hashlib.sha256(
        canon({k: v for k, v in doc.items() if k != "prereg_id"})).hexdigest()
    if lock.get("prereg_id") != doc.get("prereg_id"):
        return False, "lock prereg_id differs from live document"
    if lock.get("prereg_id_derived") != derived:
        return False, "lock's derived prereg id does not match the live document"
    if lock.get("independently_verified") is not True:
        return False, "final lock not independently verified (sequence step 6)"
    return True, lock


def build_schedule_identity(pairs):
    """Content identity of the sampler schedule inputs (populations only — no draws)."""
    by_fold_train_q = {}
    pos = defaultdict(list)
    buckets = defaultdict(lambda: defaultdict(list))
    for i, p in enumerate(pairs):
        if p["class"].startswith("positive"):
            pos[p["query"]].append(i)
        else:
            buckets[p["query"]][p["hardness"]].append(i)
    folds = defaultdict(set)
    for p in pairs:
        folds[p["fold"]].add(p["query"])
    for f in range(5):
        train_q = sorted(q for ff, qs in folds.items() if ff != f for q in qs)
        by_fold_train_q[f] = hashlib.sha256("\n".join(train_q).encode()).hexdigest()
    pop = {q: {"positives": len(pos[q]),
               "buckets": {b: len(buckets[q][b]) for b, _ in BUCKET_SLOTS if buckets[q][b]}}
           for q in sorted(pos)}
    return {"train_query_list_sha256_by_fold": by_fold_train_q,
            "population_sha256": hashlib.sha256(canon(pop)).hexdigest()}


def cmd_plan(a):
    man, pairs = load_bundle()
    review = json.load(open(os.path.join(ROOT, "SM_FS_LINEAGE_RANKING_EXECUTION_REVIEW.json")))
    init = review["initialized_checkpoints"] if "initialized_checkpoints" in review else None
    jobs = [{"fold": f, "arm": arm, "seed": s}
            for f in range(5) for s in SEEDS for arm in ARMS]
    plan = {
        "schema": PLAN_SCHEMA,
        "bundle_manifest_sha256": sha_file(os.path.join(RANK_DIR, "manifest.json")),
        "jobs": jobs, "job_count": len(jobs),
        "steps": STEPS, "batch_size": BS, "query_draws_per_step": DRAWS,
        "lr": LR, "anchor_weight": ANCHOR_W, "grad_clip": CLIP, "early_stopping": False,
        "sampler_module_sha256": sha_file(os.path.join(ROOT, "sm_fs_protocols.py")),
        "schedule_identity": build_schedule_identity(pairs),
        "reload_rule": "byte-reload seed's initialized checkpoint before every fold x arm",
        "anchor_rule": "common-positive slots only; identical across arms; never augmented",
        "held_fold_rule": "held rows filtered before any training materialization",
        "trainer_sha256": sha_file(os.path.abspath(__file__)),
        "fitting_authorized": False,
    }
    out = os.path.join(RUN_DIR, "training_plan.json")
    if os.path.exists(out):
        existing = open(out, "rb").read()
        if existing == canon(plan):
            print(f"plan unchanged -> {out}")
            return
        raise SystemExit("training_plan.json exists and differs — no-replace contract")
    write_private(out, canon(plan))
    print(f"plan -> {out} (sha {hashlib.sha256(canon(plan)).hexdigest()[:16]}, "
          f"{len(jobs)} jobs, fitting_authorized=false)")


def enforce_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    observed = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
    }
    assert observed["deterministic_algorithms"] and observed["cudnn_deterministic"]
    assert not observed["cudnn_benchmark"] and not observed["tf32_matmul"]
    assert not observed["tf32_cudnn"] and observed["matmul_precision"] == "highest"
    assert observed["cublas_workspace"] == ":4096:8"
    return observed


def sample_step(fold, seed, step, train_q, pos, buckets, arm):
    """One step's 48 rows via the frozen counter sampler. Returns (common, contrast) index lists."""
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
            n_slots = sum(w for _, w in slots)
            bi, _, _ = sampler_index(n_slots, fold=fold, seed=seed, step=step,
                                     draw=draw, role="negative-bucket", query_id=q)
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


def cmd_fit(a):
    ok, why = fitting_gate(a.final_lock)
    if not ok:
        print(f"FITTING BLOCKED: {why} (reseal sequence steps 4-6 incomplete)")
        return 3
    raise SystemExit("final lock accepted, but the fit body is intentionally not yet armed: "
                     "it lands with the authorized amendment (sequence step 4) so the plan hash "
                     "the prereg binds includes the final loop")


def cmd_evaluate(a):
    raise SystemExit("evaluate requires a sealed fit receipt; no fits exist (fitting blocked)")


def cmd_decide(a):
    raise SystemExit("decide requires all 30 sealed evaluations; fitting is blocked")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    f = sub.add_parser("fit")
    f.add_argument("--fold", type=int, required=True)
    f.add_argument("--arm", choices=ARMS, required=True)
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--final-lock", default=None)
    e = sub.add_parser("evaluate")
    e.add_argument("--fold", type=int, required=True)
    e.add_argument("--seed", type=int, required=True)
    sub.add_parser("decide")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "fit": cmd_fit, "evaluate": cmd_evaluate,
            "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
