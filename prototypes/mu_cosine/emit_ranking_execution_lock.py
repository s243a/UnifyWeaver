#!/usr/bin/env python3
"""Emit the content-bound execution lock for the SM-FS lineage-ranking paired runs.

PROTOCOL_sm_fs_lineage_ranking.md §4 blocks model fitting until the preregistration is amended
with the exact initialized checkpoint + post-growth hashes, loading/growth code hash,
tokenizer/text-table hash, optimizer class/options, trainable parameter names/counts,
frozen-reference construction, augmentation algorithm, numeric precision, deterministic
environment, and the training-plan hash. This script PRODUCES those bindings; it performs NO
optimizer step. Its output is the amendment proposal for the rigor lane to reseal.

Proposed warm start (engineering lane, for rigor-lane countersign): the migrated pre-Pearltrees
base `model_prod_namecond_full.pt` (the retention protocol's operative Track-T source) with one
LINEAGE growth per seed from those exact bytes — both ranking arms stay Pearltrees-clean, and the
later retention/transfer arms share the same base lineage.

Embedding text per §4: certified MAP TITLE and candidate LEAF TITLE only — exact paths stay
opaque identities; no path components, ancestors, fold labels, or reserve text enter the encoder.

  python3 emit_ranking_execution_lock.py        # -> ~/mu_data/sm_fs_ranking_v1/execution_lock.json
"""
import hashlib
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
RANK_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_v1")
BASE = os.path.join(ROOT, "model_prod_namecond_full.pt")
BASE_SHA_FROZEN = "9bb60915e9bcf8d4a89293c538284fd5ac631d9729f355917e128db7353bc8ef"
SEEDS = (3997001, 3997002, 3997003)


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    import numpy as np
    import torch

    from fine_tune_pearltrees_filing import load_with_lineage_ops
    from mu_attention import E5_REVISION, build_e5_tables

    base_sha = sha_file(BASE)
    assert base_sha == BASE_SHA_FROZEN, f"base checkpoint drifted: {base_sha[:16]}"

    man = json.load(open(os.path.join(RANK_DIR, "manifest.json")))
    assert man["schema"] == "unifyweaver.sm-fs-lineage-ranking-bundle.v1"
    pairs_sha = sha_file(os.path.join(RANK_DIR, "pairs.jsonl"))
    assert pairs_sha == man["outputs"]["pairs.jsonl"], "pairs drifted from manifest"

    # per-seed LINEAGE growth from the exact migrated bytes; save + hash initialized checkpoints
    init = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model, cfg = load_with_lineage_ops(BASE, dev="cpu")
        out = os.path.join(RANK_DIR, f"init_seed{seed}.pt")
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            torch.save({"state": model.state_dict(), "cfg": cfg}, f)
        # trainable inventory (frozen recipe) — names + counts, identical across seeds
        trainable = []
        for name in ("judge_name.resid.weight", "corpus_name.resid.weight",
                     "op_name.resid.weight"):
            mod = model
            try:
                for part in name.split("."):
                    mod = getattr(mod, part)
                trainable.append((name, tuple(mod.shape)))
            except AttributeError:
                pass
        for n, p in model.encoder.layers[-1].named_parameters():
            trainable.append((f"encoder.layers.{len(model.encoder.layers)-1}.{n}",
                              tuple(p.shape)))
        trainable += [("readout_w", tuple(model.readout_w.shape)),
                      ("readout_b", tuple(model.readout_b.shape)),
                      ("nodetype_emb.weight", tuple(model.nodetype_emb.weight.shape))]
        init[str(seed)] = {"path": out, "sha256": sha_file(out)}
    n_params = sum(int(np.prod(s)) for _, s in trainable)

    # tokenizer/text table: titles only (map title + candidate LEAF title), paths = opaque ids
    pairs_titles = {}
    for ln in open(os.path.expanduser("~/mu_data/sm_fs_v3/lineage_fs_targets.tsv"),
                   encoding="utf-8"):
        if ln.startswith("#"):
            if ln[1:].strip().startswith("map_path"):
                cols = ln[1:].strip().split("\t")
            continue
        r = dict(zip(cols, ln.rstrip("\n").split("\t")))
        pairs_titles[r["map_path"]] = r["map_title"]
        pairs_titles[r["ancestor_path"]] = r["ancestor_title"]   # leaf title of that path
    names = sorted(pairs_titles)
    e5_cache = os.path.join(RANK_DIR, "ranking_text_e5.pt")
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=e5_cache, batch_size=128,
                                      texts=pairs_titles, model_revision=E5_REVISION)
    text_table = {"names_sha256": sha_bytes("\n".join(names).encode()),
                  "texts_sha256": sha_bytes("\n".join(
                      f"{n}\t{pairs_titles[n]}" for n in names).encode()),
                  "e5_cache_sha256": sha_file(e5_cache), "e5_revision": E5_REVISION,
                  "embedding_text_rule": "map-title-and-candidate-leaf-title-only"}

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    lock = {
        "schema": "unifyweaver.sm-fs-ranking-execution-lock.v1",
        "status": "PROPOSED — awaiting rigor-lane prereg amendment/reseal; no fitting performed",
        "prereg_id": "0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2",
        "ranking_bundle_manifest_sha256": sha_file(os.path.join(RANK_DIR, "manifest.json")),
        "warm_start": {"proposal": "pre-pearltrees-migrated-base",
                       "path": BASE, "sha256": base_sha,
                       "growth": "load_with_lineage_ops once per seed from exact bytes"},
        "initialized_checkpoints": init,
        "growth_code": {"fine_tune_pearltrees_filing.py":
                        sha_file(os.path.join(ROOT, "fine_tune_pearltrees_filing.py")),
                        "mu_attention.py": sha_file(os.path.join(ROOT, "mu_attention.py"))},
        "tokenizer_text_table": text_table,
        "optimizer": {"class": "torch.optim.Adam", "lr": 0.0005, "betas": [0.9, 0.999],
                      "eps": 1e-08, "weight_decay": 0},
        "trainable_parameters": {"names_shapes": [[n, list(s)] for n, s in trainable],
                                 "tensor_count": len(trainable), "param_count": n_params},
        "frozen_reference": "copy.deepcopy of the exact initialized model, eval, requires_grad "
                            "False, before optimizer creation (per-seed)",
        "augmentation": {"algorithm": "mu_batch(train=True, rng=default_rng(seed+1)) — the "
                         "existing tokenizer train-time augmentation", "anchor_rows_augmented": False},
        "numeric_environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn_deterministic_required": True, "tf32_disabled_required": True,
            "float32_matmul_precision": "highest", "dtype": "float32",
        },
        "sampler": {"module": "sm_fs_protocols.sampler_index",
                    "sm_fs_protocols.py": sha_file(os.path.join(ROOT, "sm_fs_protocols.py"))},
        "training_plan_code": {"emit_ranking_execution_lock.py": sha_file(os.path.abspath(__file__))},
        "git_commit": commit,
    }
    out = os.path.join(RANK_DIR, "execution_lock.json")
    data = (json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode()
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    print(f"execution lock -> {out} (sha {sha_bytes(data)[:16]})")
    for s, rec in init.items():
        print(f"  init seed {s}: {rec['sha256'][:16]}")
    print(f"  trainable: {len(trainable)} tensors / {n_params} params; "
          f"text table {text_table['texts_sha256'][:16]}; commit {commit[:12]}")
    print("STATUS: fitting remains BLOCKED until the rigor lane amends and reseals the prereg "
          "with these bindings (+ the final training-plan file hash).")


if __name__ == "__main__":
    main()
