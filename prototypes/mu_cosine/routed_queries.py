#!/usr/bin/env python3
"""Public-only routed filing tasks with content-bound provenance.

The historical routed-query passes established useful exploratory engineering
evidence, but their task and pick files were not cryptographically bound and
the catalog did not enforce privacy.  This v2 interface is the prospective
contract for any further hosted judging:

  ceilings
      Recompute descriptive perfect-judge ceilings on the certified-public
      population.

  emit --margin 0.02 --menu 10 --lineage --tier-id low
      Emit an outcome-blind v2 task.  Catalog, population, ranking, privacy
      index, selection band, menus, and lineage mode are all content-bound.

  seal-picks --task TASK --raw-picks RAW ...
      Convert strict raw ``{qid,pick}`` rows into a v2 pick artifact that binds
      the exact task bytes and declared judge provenance.  This does not
      authenticate the provider; it prevents later provenance drift.

  score --task TASK --picks PICKS
      Rebuild and byte-compare the task, require an exact QID join, then report
      a paired bookmark/folder node-block interval.

  score-policy --policy POLICY --tier ID TASK PICKS [...]
      Reproduce a complete frozen multi-tier policy.  Each population row must
      match exactly one tier and every judge tier must have one bound task/pick
      pair.

Legacy unbound artifacts are intentionally rejected.  They remain valid only
as descriptive historical evidence and cannot be upgraded retroactively.
This version binds one response artifact per tier; provenance-safe provider-call
chunking and repeated-draw aggregation require a future parent-task/chunk
manifest and are not silently emulated by concatenating responses.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_filing import load_filing
from filing_privacy import PUBLIC_CATALOG_POLICY_ID
from filing_assistant import (
    TREES,
    catalog_tables,
    encode_queries,
    ranks_np,
    stable_score_order,
)
from mu_attention import E5_MODEL, E5_REVISION
from routed_policy import (
    RoutedPolicyError,
    band_qids,
    build_task_envelope,
    canonical_json_bytes,
    file_content_record,
    make_band,
    paired_node_block_bootstrap,
    policy_tier_for_margin,
    read_pick_file,
    read_policy_file,
    read_task_file,
    seal_pick_file,
    sha256_bytes,
    write_task_file,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY = ROOT / "ROUTED_POLICY_three_tier_v1.json"
EVIDENCE_STATUS = "exploratory_transductive"
JUDGE_CONTRACT_FIELDS = {
    "provider",
    "family",
    "model",
    "model_revision",
    "interface",
    "prompt_sha256",
    "temperature",
}
PRIVACY_NOTE = (
    "Only bookmarks retained inside certified-public folders, certified-public "
    "candidate folders, and certified-public lineage nodes may enter hosted "
    "tasks; private markers/restrictions are removed and unknown node visibility "
    "is quarantined."
)


def _frozen_policy():
    envelope = read_policy_file(DEFAULT_POLICY)
    prompt = envelope["policy_core"]["judge_prompt"]
    prompt_path = ROOT / prompt["path"]
    if sha256_bytes(prompt_path.read_bytes()) != prompt["sha256"]:
        raise RoutedPolicyError("frozen judge prompt does not match the policy hash")
    return envelope


def _validate_selection_against_tier(policy_envelope, selection):
    """Require a task selection to be one exact judge tier of the policy."""
    if not isinstance(selection, Mapping):
        raise RoutedPolicyError("task selection is missing")
    core = policy_envelope["policy_core"]
    matching = [
        tier
        for tier in core["tiers"]
        if tier["tier_id"] == selection.get("tier_id")
    ]
    if len(matching) != 1 or matching[0]["action"] != "judge":
        raise RoutedPolicyError("task selection is not a frozen judge tier")
    tier = matching[0]
    for field in ("band", "menu_size", "lineage", "lineage_depth"):
        if selection.get(field) != tier[field]:
            raise RoutedPolicyError(
                f"task selection {field} differs from frozen tier {tier['tier_id']}"
            )
    judge = selection.get("required_judge")
    if not isinstance(judge, Mapping) or set(judge) != JUDGE_CONTRACT_FIELDS:
        raise RoutedPolicyError("task must bind the complete declared judge contract")
    for field in ("provider", "family", "model", "model_revision", "interface"):
        if not isinstance(judge.get(field), str) or not judge[field]:
            raise RoutedPolicyError(f"required judge {field} must be a nonempty string")
    if judge["family"] != tier["required_judge_family"]:
        raise RoutedPolicyError("task judge family differs from frozen tier")
    if judge["prompt_sha256"] != core["judge_prompt"]["sha256"]:
        raise RoutedPolicyError("task prompt differs from frozen policy prompt")
    temperature = judge["temperature"]
    if (
        temperature is not None
        and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not np.isfinite(temperature)
        )
    ):
        raise RoutedPolicyError("required judge temperature must be finite or null")
    return tier


@dataclass
class BuildState:
    queries: list[tuple[str, object]]
    q_titles: list[str]
    f_ids: list[object]
    f_titles: list[str]
    truepos: list[list[int]]
    alias_truepos: list[list[int]]
    cos: np.ndarray
    ranks: np.ndarray
    alias_ranks: np.ndarray
    margin: np.ndarray
    order: np.ndarray
    legacy_manifest: str
    privacy: object
    receipt: dict


def _hash_rows(rows):
    return sha256_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def _array_record(array):
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": sha256_bytes(contiguous.tobytes(order="C")),
    }


def _implementation_record():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    files = [
        "routed_queries.py",
        "routed_policy.py",
        "filing_privacy.py",
        "eval_filing.py",
        "eval_pearltrees_filing.py",
        "filing_assistant.py",
        "mu_attention.py",
    ]
    rows = []
    for name in files:
        data = (ROOT / name).read_bytes()
        rows.append({"name": name, "sha256": sha256_bytes(data)})
    return {
        "git_commit": commit,
        "files_sha256": _hash_rows(rows),
    }


def build(a):
    queries, f_ids, f_titles, cand_vec, privacy = catalog_tables(
        a.min_bm, f"eval{a.min_bm}", return_privacy=True
    )
    by_title = {}
    for column, title in enumerate(f_titles):
        by_title.setdefault(title, []).append(column)
    queries = sorted(queries)
    if len(queries) > a.max_queries:
        queries = random.Random(a.seed).sample(queries, a.max_queries)
    q_titles = [title for title, _ in queries]
    cand_by_id = dict(zip(f_ids, f_titles))
    position_by_id = {folder_id: column for column, folder_id in enumerate(f_ids)}
    truepos = [[position_by_id[folder_id]] for _, folder_id in queries]
    alias_truepos = [
        sorted(by_title[cand_by_id[folder_id]]) for _, folder_id in queries
    ]
    legacy_manifest = hashlib.sha256(
        "\n".join(f"{title}\t{folder_id}" for title, folder_id in queries).encode()
    ).hexdigest()
    qv = encode_queries(q_titles)
    cos = qv @ cand_vec.T
    ranks = ranks_np(cos, truepos)
    alias_ranks = ranks_np(cos, alias_truepos)
    if cos.shape[1] < 2:
        raise RoutedPolicyError("at least two public candidate folders are required")
    sorted_scores = np.sort(cos, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    order = stable_score_order(cos)

    catalog_rows = [
        {"folder_id": str(folder_id), "title": title}
        for folder_id, title in zip(f_ids, f_titles)
    ]
    population_rows = [
        {
            "qid": qid,
            "bookmark": title,
            "true_folder_id": str(folder_id),
        }
        for qid, (title, folder_id) in enumerate(queries)
    ]
    ranking_rows = [
        {
            "qid": qid,
            "margin_float64_hex": float(margin[qid]).hex(),
            "top_k_folder_ids": [
                str(f_ids[int(column)]) for column in order[qid, : a.top_k]
            ],
        }
        for qid in range(len(queries))
    ]
    privacy_receipt = privacy.receipt()
    receipt = {
        "source": dict(privacy.source_snapshot),
        "privacy": {
            "schema": privacy_receipt["schema"],
            "policy_id": privacy_receipt["policy_id"],
            "manifest_sha256": privacy.manifest_sha256,
            "counts": privacy_receipt["counts"],
        },
        "catalog": {
            "policy_id": PUBLIC_CATALOG_POLICY_ID,
            "min_bm": a.min_bm,
            "folder_count": len(f_ids),
            "sha256": _hash_rows(catalog_rows),
        },
        "population": {
            "count": len(queries),
            "seed": a.seed,
            "max_queries": a.max_queries,
            "sha256": _hash_rows(population_rows),
            "legacy_sha256": legacy_manifest,
            "primary_grading": "exact_destination_id",
            "sensitivity_grading": "best_title_equivalent_destination",
        },
        "ranker": {
            "name": "e5-cos",
            "model_id": E5_MODEL,
            "model_revision": E5_REVISION,
            "model_revision_status": "immutable-huggingface-commit",
            "top_k": a.top_k,
            "tie_break": "candidate_catalog_column_ascending",
            "candidate_embeddings": _array_record(cand_vec),
            "query_embeddings": _array_record(qv),
            "ranking_sha256": _hash_rows(ranking_rows),
            "implementation": _implementation_record(),
        },
    }
    print(
        f"population: {len(queries)} public-only queries "
        f"(manifest {legacy_manifest[:16]}, privacy {privacy.manifest_sha256[:16]}), "
        f"{len(f_ids)} folders, K={a.top_k}"
    )
    return BuildState(
        queries=queries,
        q_titles=q_titles,
        f_ids=f_ids,
        f_titles=f_titles,
        truepos=truepos,
        alias_truepos=alias_truepos,
        cos=cos,
        ranks=ranks,
        alias_ranks=alias_ranks,
        margin=margin,
        order=order,
        legacy_manifest=legacy_manifest,
        privacy=privacy,
        receipt=receipt,
    )


def menu_hit(state, qid, menu_size):
    truth = set(state.truepos[qid])
    return any(int(column) in truth for column in state.order[qid, :menu_size])


def run_ceilings(a):
    state = build(a)
    baseline = float(np.mean(state.ranks <= 1))
    print(f"\nno-routing baseline R@1: {baseline:.3f}")
    print(
        f"{'t':>7s} {'routed':>7s} {'kept R@1':>9s} "
        + " ".join(f"ceil@N={n:<3d}" for n in a.menus)
    )
    for threshold in a.grid:
        routed = state.margin < threshold
        kept = ~routed
        kept_r1 = float(np.mean(state.ranks[kept] <= 1)) if kept.any() else float("nan")
        cells = []
        for menu_size in a.menus:
            rescued = sum(
                menu_hit(state, int(qid), menu_size) for qid in np.where(routed)[0]
            )
            policy = (int((state.ranks[kept] <= 1).sum()) + rescued) / len(state.q_titles)
            cells.append(f"{policy:9.3f}")
        print(
            f"{threshold:7.3f} {routed.mean():7.2f} {kept_r1:9.3f} "
            + " ".join(cells)
        )
    print(
        "\nDESCRIPTIVE public-only ceilings only. Threshold/menu selection here "
        "cannot be called confirmatory on this population."
    )


def _selection(
    *,
    tier_id,
    margin_low,
    margin_high,
    menu_size,
    lineage,
    lineage_depth,
    required_judge,
):
    if not isinstance(tier_id, str) or not tier_id:
        raise RoutedPolicyError("tier_id is required")
    if menu_size <= 0:
        raise RoutedPolicyError("menu size must be positive")
    return {
        "tier_id": tier_id,
        "band": make_band(margin_low, margin_high),
        "menu_size": int(menu_size),
        "lineage": bool(lineage),
        "lineage_depth": int(lineage_depth),
        "required_judge": dict(required_judge),
    }


def _task_core(state, selection):
    policy = _frozen_policy()
    return {
        **state.receipt,
        "selection": dict(selection),
        "policy_provenance": {
            "policy_name": "three-tier-margin-v1",
            "policy_id": policy["policy_id"],
            "evidence_status": EVIDENCE_STATUS,
            "selection_note": (
                "Thresholds, menu sizes, judge family, and lineage context were "
                "selected on the historical 1,200-row transductive benchmark."
            ),
        },
        "judge_contract": {
            "pick_semantics": "zero-based-menu-position-or-null",
            "required_context": "title-plus-lineage"
            if selection["lineage"]
            else "title-only",
            "privacy": PRIVACY_NOTE,
            "required_judge": dict(selection["required_judge"]),
        },
        "execution_contract": {
            "response_artifacts_per_tier": 1,
            "chunked_provider_calls": "unsupported-requires-bound-chunk-manifest",
            "repeated_draw_aggregation": "unsupported-requires-bound-draw-manifest",
        },
    }


def _task_rows(state, selection):
    qids = band_qids(state.margin, selection["band"])
    maximum_menu = min(len(state.f_ids), state.receipt["ranker"]["top_k"])
    if selection["menu_size"] > maximum_menu:
        raise RoutedPolicyError(
            f"menu_size {selection['menu_size']} exceeds ranked public pool {maximum_menu}"
        )
    lineage = {}
    if selection["lineage"]:
        from eval_pearltrees_filing import folder_lineage

        _, candidates, privacy = load_filing(TREES, state.receipt["catalog"]["min_bm"],
                                             return_privacy=True)
        if privacy.manifest_sha256 != state.privacy.manifest_sha256:
            raise RoutedPolicyError("privacy index drifted while constructing lineage")
        _, _, lineage_by_id = folder_lineage(
            candidates,
            depth=selection["lineage_depth"],
            public_tree_titles=privacy.public_title_by_id,
            return_id_chains=True,
        )
        lineage = lineage_by_id
    rows = []
    for qid in qids:
        menu = []
        for position, column in enumerate(
            state.order[qid, : selection["menu_size"]]
        ):
            column = int(column)
            title = state.f_titles[column]
            item = {
                "pos": position,
                "folder_id": str(state.f_ids[column]),
                "title": title,
            }
            if selection["lineage"]:
                item["path"] = " > ".join(
                    reversed(lineage.get(str(state.f_ids[column]), []))
                )
            menu.append(item)
        rows.append(
            {
                "record_type": "task",
                "qid": int(qid),
                "bookmark": state.q_titles[qid],
                "menu": menu,
            }
        )
    return rows


def _default_task_path(a):
    low = "min" if a.margin_low is None else str(a.margin_low)
    high = "max" if a.margin is None else str(a.margin)
    suffix = "_lin" if a.lineage else ""
    return os.path.expanduser(
        f"~/mu_data/routed_tasks_{a.tier_id}_{low}_{high}_n{a.menu}{suffix}.jsonl"
    )


def run_emit(a):
    state = build(a)
    policy = _frozen_policy()
    prompt = policy["policy_core"]["judge_prompt"]
    required_judge = {
        "provider": a.required_judge_provider,
        "family": a.required_judge_family,
        "model": a.required_judge_model,
        "model_revision": a.required_model_revision,
        "interface": a.required_interface,
        "prompt_sha256": prompt["sha256"],
        "temperature": a.required_temperature,
    }
    selection = _selection(
        tier_id=a.tier_id,
        margin_low=a.margin_low,
        margin_high=a.margin,
        menu_size=a.menu,
        lineage=a.lineage,
        lineage_depth=a.lineage_depth,
        required_judge=required_judge,
    )
    _validate_selection_against_tier(policy, selection)
    rows = _task_rows(state, selection)
    out = a.out or _default_task_path(a)
    header, record = write_task_file(out, _task_core(state, selection), rows)
    print(
        f"emitted {len(rows)} certified-public tasks -> {out}\n"
        f"task_id {header['task_id']}  file_sha256 {record['sha256']}"
    )
    print(
        f"Raw judge output must begin with a header binding task_id "
        f"{header['task_id']}, then contain exactly one {{qid,pick}} row per task; "
        "run seal-picks before score."
    )
    print(
        "Current v2 sealing accepts one complete response artifact for this tier; "
        "do not concatenate unbound provider-call chunks or repeated draws."
    )


def _verify_task_rebuild(state, task_path):
    header, rows, file_record = read_task_file(task_path)
    core = header["task_core"]
    selection = core.get("selection")
    if not isinstance(selection, dict):
        raise RoutedPolicyError("task selection is missing")
    expected_rows = _task_rows(state, selection)
    expected = build_task_envelope(_task_core(state, selection), expected_rows)
    if header != expected or rows != expected_rows:
        raise RoutedPolicyError(
            "task does not reproduce from the current source/privacy/catalog/ranking state"
        )
    return header, rows, file_record, selection


def run_seal_picks(a):
    task_header, _, _ = read_task_file(a.task)
    required = task_header["task_core"]["selection"]["required_judge"]
    prompt = Path(a.prompt_file).read_bytes()
    provenance = {
        "provider": a.judge_provider,
        "family": a.judge_family,
        "model": a.judge_model,
        "model_revision": a.model_revision,
        "interface": a.interface,
        "prompt_sha256": sha256_bytes(prompt),
        "temperature": a.temperature,
        "run_id": a.run_id,
        "provenance_status": "declared",
    }
    for field, expected in required.items():
        if provenance.get(field) != expected:
            raise RoutedPolicyError(
                f"declared judge {field}={provenance.get(field)!r} "
                f"!= task requirement {expected!r}"
            )
    header, record = seal_pick_file(
        a.task, a.raw_picks, a.out, provenance
    )
    print(
        f"sealed picks -> {a.out}\n"
        f"pick_id {header['pick_id']}  file_sha256 {record['sha256']}"
    )
    print("Judge identity is declared and content-bound; it is not provider-authenticated.")


def _policy_correct(state, task_rows, picks, *, title_equivalence=False):
    ranks = state.alias_ranks if title_equivalence else state.ranks
    true_positions = state.alias_truepos if title_equivalence else state.truepos
    correct = np.array(ranks <= 1, dtype=bool)
    for task in task_rows:
        qid = task["qid"]
        correct[qid] = False
        pick = picks[qid]
        if pick is not None:
            column = int(state.order[qid, pick])
            if column in set(true_positions[qid]):
                correct[qid] = True
    return correct


def _result_receipt(
    state,
    correct,
    baseline,
    interval,
    artifacts,
    policy_id,
    *,
    evaluation_scope,
    frozen_policy_id=None,
    task_id=None,
    alias_correct=None,
):
    receipt = {
        "schema": "unifyweaver.routed-result.v1",
        "evidence_status": EVIDENCE_STATUS,
        "policy_id": policy_id,
        "evaluation_scope": evaluation_scope,
        "primary_grading": "exact_destination_id",
        "population": state.receipt["population"],
        "privacy": state.receipt["privacy"],
        "ranker": state.receipt["ranker"],
        "artifacts": artifacts,
        "baseline_correct": int(baseline.sum()),
        "policy_correct": int(correct.sum()),
        "query_count": len(correct),
        "baseline_r1": float(baseline.mean()),
        "policy_r1": float(correct.mean()),
        "delta": float(correct.mean() - baseline.mean()),
        "paired_node_block_interval": interval,
    }
    if frozen_policy_id is not None:
        receipt["frozen_policy_id"] = frozen_policy_id
    if task_id is not None:
        receipt["task_id"] = task_id
    if alias_correct is not None:
        receipt["title_equivalence_sensitivity"] = {
            "policy_correct": int(alias_correct.sum()),
            "baseline_correct": int((state.alias_ranks <= 1).sum()),
            "policy_r1": float(alias_correct.mean()),
            "baseline_r1": float(np.mean(state.alias_ranks <= 1)),
        }
    core = dict(receipt)
    receipt["result_id"] = sha256_bytes(canonical_json_bytes(core))
    return receipt


def _write_result(path, receipt):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(canonical_json_bytes(receipt))
        print(f"result receipt -> {path}")


def _print_result(receipt):
    ci = receipt["paired_node_block_interval"]
    print(
        f"\npolicy R@1 {receipt['policy_r1']:.3f} vs baseline "
        f"{receipt['baseline_r1']:.3f} (delta {receipt['delta']:+.3f})"
    )
    print(
        "paired bookmark/folder node-block 95% interval: "
        f"[{ci['lower_0.025']:+.3f}, {ci['upper_0.975']:+.3f}] "
        f"({ci['block_count']} blocks, {ci['replicates']} draws)"
    )
    print("EVIDENCE STATUS: exploratory/transductive; not a confirmatory deployment claim.")


def run_score(a):
    state = build(a)
    task_header, task_rows, task_record, selection = _verify_task_rebuild(
        state, a.task
    )
    frozen = _frozen_policy()
    _validate_selection_against_tier(frozen, selection)
    pick_header, _, picks = read_pick_file(a.picks, a.task)
    judge = pick_header["pick_core"]["judge"]
    for field, expected in selection["required_judge"].items():
        if judge.get(field) != expected:
            raise RoutedPolicyError(f"sealed judge {field} differs from task requirement")
    correct = _policy_correct(state, task_rows, picks)
    alias_correct = _policy_correct(
        state, task_rows, picks, title_equivalence=True
    )
    baseline = np.array(state.ranks <= 1, dtype=bool)
    interval = paired_node_block_bootstrap(
        correct,
        baseline,
        state.q_titles,
        [str(folder_id) for _, folder_id in state.queries],
        replicates=a.bootstrap_replicates,
        seed=a.bootstrap_seed,
    )
    partial_policy_id = sha256_bytes(
        canonical_json_bytes(
            {
                "scope": "single-tier-plus-auto-baseline",
                "frozen_policy_id": frozen["policy_id"],
                "task_id": task_header["task_id"],
            }
        )
    )
    receipt = _result_receipt(
        state,
        correct,
        baseline,
        interval,
        {
            "task": {**task_record, "task_id": task_header["task_id"]},
            "picks": {
                **file_content_record(a.picks),
                "pick_id": pick_header["pick_id"],
            },
        },
        partial_policy_id,
        evaluation_scope="single-tier-plus-auto-baseline",
        frozen_policy_id=frozen["policy_id"],
        task_id=task_header["task_id"],
        alias_correct=alias_correct,
    )
    _print_result(receipt)
    _write_result(a.out, receipt)


def _parse_tier_args(values):
    out = {}
    for tier_id, task, picks in values:
        if tier_id in out:
            raise RoutedPolicyError(f"duplicate --tier binding: {tier_id}")
        out[tier_id] = (task, picks)
    return out


def run_score_policy(a):
    state = build(a)
    envelope = read_policy_file(a.policy)
    frozen = _frozen_policy()
    if envelope["policy_id"] != frozen["policy_id"]:
        raise RoutedPolicyError("score-policy accepts only the frozen policy id")
    core = envelope["policy_core"]
    supplied = _parse_tier_args(a.tier)
    expected_judge_ids = {
        tier["tier_id"] for tier in core["tiers"] if tier["action"] == "judge"
    }
    if set(supplied) != expected_judge_ids:
        raise RoutedPolicyError(
            f"tier artifacts do not match policy; expected={sorted(expected_judge_ids)}, "
            f"supplied={sorted(supplied)}"
        )

    baseline = np.array(state.ranks <= 1, dtype=bool)
    correct = baseline.copy()
    alias_correct = np.array(state.alias_ranks <= 1, dtype=bool)
    artifacts = {"policy": {**file_content_record(a.policy), "policy_id": envelope["policy_id"]}}
    observed_judge_qids = set()
    for tier in core["tiers"]:
        tier_qids = set(band_qids(state.margin, tier["band"]))
        if tier["action"] == "auto_top1":
            continue
        task_path, picks_path = supplied[tier["tier_id"]]
        task_header, task_rows, task_record, selection = _verify_task_rebuild(
            state, task_path
        )
        if (
            task_header["task_core"]["policy_provenance"]["policy_id"]
            != envelope["policy_id"]
        ):
            raise RoutedPolicyError(f"task is bound to a different policy: {tier['tier_id']}")
        try:
            validated_tier = _validate_selection_against_tier(envelope, selection)
        except RoutedPolicyError as exc:
            raise RoutedPolicyError(
                f"task selection does not match tier {tier['tier_id']}: {exc}"
            ) from exc
        if validated_tier["tier_id"] != tier["tier_id"]:
            raise RoutedPolicyError(f"task selection does not match tier {tier['tier_id']}")
        task_qids = {row["qid"] for row in task_rows}
        if task_qids != tier_qids or observed_judge_qids & task_qids:
            raise RoutedPolicyError(f"task qids do not exactly partition tier {tier['tier_id']}")
        observed_judge_qids.update(task_qids)
        pick_header, _, picks = read_pick_file(picks_path, task_path)
        judge = pick_header["pick_core"]["judge"]
        for field, expected in selection["required_judge"].items():
            if judge.get(field) != expected:
                raise RoutedPolicyError(
                    f"wrong judge {field} for tier {tier['tier_id']}"
                )
        tier_correct = _policy_correct(state, task_rows, picks)
        tier_alias_correct = _policy_correct(
            state, task_rows, picks, title_equivalence=True
        )
        for qid in task_qids:
            correct[qid] = tier_correct[qid]
            alias_correct[qid] = tier_alias_correct[qid]
        artifacts[tier["tier_id"]] = {
            "task": {**task_record, "task_id": task_header["task_id"]},
            "picks": {
                **file_content_record(picks_path),
                "pick_id": pick_header["pick_id"],
            },
        }

    for qid, margin in enumerate(state.margin):
        tier = policy_tier_for_margin(core, float(margin))
        if tier["action"] == "judge" and qid not in observed_judge_qids:
            raise RoutedPolicyError(f"judge tier qid {qid} lacks a bound artifact")
        if tier["action"] == "auto_top1" and qid in observed_judge_qids:
            raise RoutedPolicyError(f"auto tier qid {qid} appears in judge artifacts")

    interval = paired_node_block_bootstrap(
        correct,
        baseline,
        state.q_titles,
        [str(folder_id) for _, folder_id in state.queries],
        replicates=core["bootstrap"]["replicates"],
        seed=core["bootstrap"]["seed"],
    )
    receipt = _result_receipt(
        state,
        correct,
        baseline,
        interval,
        artifacts,
        envelope["policy_id"],
        evaluation_scope="complete-frozen-policy",
        alias_correct=alias_correct,
    )
    _print_result(receipt)
    _write_result(a.out, receipt)


def _add_population_args(parser):
    parser.add_argument("--min-bm", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=100)


def _add_bootstrap_args(parser):
    parser.add_argument("--bootstrap-replicates", type=int, default=9999)
    parser.add_argument("--bootstrap-seed", type=int, default=3867001)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    ceilings = sub.add_parser("ceilings")
    _add_population_args(ceilings)
    ceilings.add_argument(
        "--grid", type=float, nargs="*", default=(0.005, 0.01, 0.02, 0.03, 0.05, 0.08)
    )
    ceilings.add_argument("--menus", type=int, nargs="*", default=(5, 10, 20, 50))

    emit = sub.add_parser("emit")
    _add_population_args(emit)
    emit.add_argument("--tier-id", required=True)
    emit.add_argument("--margin-low", type=float, default=None)
    emit.add_argument("--margin", type=float, default=0.02, help="exclusive upper band boundary")
    emit.add_argument("--menu", type=int, default=10)
    emit.add_argument("--lineage", action="store_true")
    emit.add_argument("--lineage-depth", type=int, default=3)
    emit.add_argument("--required-judge-family", default="sonnet")
    emit.add_argument("--required-judge-provider", required=True)
    emit.add_argument("--required-judge-model", required=True)
    emit.add_argument("--required-model-revision", required=True)
    emit.add_argument("--required-interface", required=True)
    emit.add_argument("--required-temperature", type=float, default=None)
    emit.add_argument("--out")

    seal = sub.add_parser("seal-picks")
    seal.add_argument("--task", required=True)
    seal.add_argument("--raw-picks", required=True)
    seal.add_argument("--out", required=True)
    seal.add_argument("--judge-provider", required=True)
    seal.add_argument("--judge-family", required=True)
    seal.add_argument("--judge-model", required=True)
    seal.add_argument("--model-revision", required=True)
    seal.add_argument("--interface", required=True)
    seal.add_argument("--prompt-file", required=True)
    seal.add_argument("--temperature", type=float, default=None)
    seal.add_argument("--run-id", required=True)

    score = sub.add_parser("score")
    _add_population_args(score)
    _add_bootstrap_args(score)
    score.add_argument("--task", required=True)
    score.add_argument("--picks", required=True)
    score.add_argument("--out")

    score_policy = sub.add_parser("score-policy")
    _add_population_args(score_policy)
    score_policy.add_argument("--policy", default=str(DEFAULT_POLICY))
    score_policy.add_argument(
        "--tier",
        nargs=3,
        action="append",
        metavar=("TIER_ID", "TASK", "PICKS"),
        default=[],
    )
    score_policy.add_argument("--out")

    args = parser.parse_args(argv)
    actions = {
        "ceilings": run_ceilings,
        "emit": run_emit,
        "seal-picks": run_seal_picks,
        "score": run_score,
        "score-policy": run_score_policy,
    }
    try:
        actions[args.mode](args)
    except (OSError, RoutedPolicyError, ValueError) as exc:
        parser.error(f"FAIL-CLOSED: {exc}")


if __name__ == "__main__":
    main()
