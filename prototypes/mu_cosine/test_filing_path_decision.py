#!/usr/bin/env python3
"""Acceptance tests for filing path decoder Stage A.

These follow the numbered list in ``DESIGN_filing_path_decoder_handoff.md`` §9,
restricted to the Stage A / schema surface that §11 authorizes for the first
engineering PR.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from filing_path_decision import (
    ABSTAIN,
    DECISION_SCHEMA,
    GRAPH_SNAPSHOT_SCHEMA,
    PATH_RECORDS_SCHEMA,
    POLICY_SCHEMA,
    PROPOSE_NEW,
    REQUEST_SCHEMA,
    SELECT_EXISTING,
    FilingPathError,
    assert_not_circular_grading,
    build_request,
    check_external_naming,
    confirm_decision,
    decide_stage_a,
    derive_privacy_class,
    make_typed_id,
    parse_typed_id,
    policy_id_for,
    read_decision,
    read_principal_path_records,
    require_proposal_authorization,
    validate_decision,
    write_decision,
)
from routed_policy import (
    canonical_json_bytes,
    float64_hex,
    make_band,
    sha256_bytes,
    write_task_file,
)

CORPUS = "fixture-account"
PRIVACY_MANIFEST = "b" * 64
CATALOG_SHA = "c" * 64
POPULATION_SHA = "d" * 64


def _typed(stable_value: str) -> dict[str, str]:
    return {
        "node_type": "folder",
        "corpus_or_account": CORPUS,
        "stable_value": stable_value,
    }


def _hash_rows(rows) -> str:
    return sha256_bytes(b"".join(canonical_json_bytes(dict(row)) for row in rows))


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _menu(folder_ids=("10", "11")):
    # "first" is duplicated on purpose: duplicate titles must stay distinct.
    titles = {"10": "first", "11": "first", "12": "second", "13": "third"}
    return [
        {"pos": pos, "folder_id": fid, "title": titles.get(fid, f"title {fid}")}
        for pos, fid in enumerate(folder_ids)
    ]


def _task_rows(folder_ids=("10", "11"), qids=(0, 1)):
    return [
        {
            "record_type": "task",
            "qid": qid,
            "bookmark": f"bookmark {qid}",
            "menu": _menu(folder_ids),
        }
        for qid in qids
    ]


def _ranking_rows(qids=(0, 1), margins=(0.25, 0.25), folder_ids=("10", "11")):
    return [
        {
            "qid": qid,
            "margin_float64_hex": float(margin).hex(),
            "top_k_folder_ids": list(folder_ids),
        }
        for qid, margin in zip(qids, margins)
    ]


def _task_core(ranking_sha, *, public=True, menu_size=2):
    return {
        "source": {"members_sha256": "a" * 64},
        "privacy": {
            "policy_id": "pearltrees-public-only-v1" if public else "unverified",
            "manifest_sha256": PRIVACY_MANIFEST,
        },
        "catalog": {
            "policy_id": "pearltrees-public-alphanumeric-title-v1",
            "sha256": CATALOG_SHA,
        },
        "population": {"sha256": POPULATION_SHA},
        "ranker": {"ranking_sha256": ranking_sha},
        "selection": {
            "tier_id": "low",
            "band": make_band(None, 0.02),
            "menu_size": menu_size,
            "lineage": True,
            "lineage_depth": 3,
        },
        "policy_provenance": {"evidence_status": "exploratory_transductive"},
    }


def _write_graph(path, edges):
    rows = [
        {"record_type": "edge", "parent_id": _typed(parent), "child_id": _typed(child)}
        for parent, child in edges
    ]
    header = {
        "schema": GRAPH_SNAPSHOT_SCHEMA,
        "record_type": "graph_snapshot_header",
        "corpus_or_account": CORPUS,
        "edge_table_sha256": _hash_rows(rows),
    }
    path.write_bytes(
        canonical_json_bytes(header) + b"".join(canonical_json_bytes(r) for r in rows)
    )
    return path


def _write_paths(path, paths_by_folder, titles_by_folder=None):
    titles_by_folder = titles_by_folder or {}
    rows = []
    for folder, chain in paths_by_folder.items():
        rows.append(
            {
                "record_type": "principal_path",
                "folder_id": _typed(folder),
                "path_ids": [_typed(step) for step in chain],
                "titles": titles_by_folder.get(
                    folder, [f"n{step}" for step in chain]
                ),
            }
        )
    header = {
        "schema": PATH_RECORDS_SCHEMA,
        "record_type": "principal_path_header",
        "corpus_or_account": CORPUS,
        "principal_path_policy_id": "record-majority-principal-parent-v1",
        "principal_path_records_sha256": _hash_rows(rows),
    }
    path.write_bytes(
        canonical_json_bytes(header) + b"".join(canonical_json_bytes(r) for r in rows)
    )
    return path


def _write_policy(
    path,
    *,
    margin_threshold=None,
    partition="train_only",
    propose_new=False,
    maximum_nodes=64,
    maximum_steps=64,
    external_naming=False,
):
    document = {
        "schema": POLICY_SCHEMA,
        "abstain": {
            "margin_threshold_float64_hex": float64_hex(margin_threshold),
            "selection_partition": partition,
        },
        "propose_new_enabled": propose_new,
        "search_limits": {
            "maximum_search_nodes": maximum_nodes,
            "maximum_search_steps": maximum_steps,
        },
        "external_naming": {"enabled": external_naming},
    }
    document["policy_id"] = policy_id_for(document)
    path.write_bytes(canonical_json_bytes(document))
    return path


def _write_ranking(path, rows):
    path.write_bytes(b"".join(canonical_json_bytes(r) for r in rows))
    return path


@pytest.fixture
def world(tmp_path):
    """A two-candidate catalog under one allowed root, with a multi-parent leaf."""

    ranking_rows = _ranking_rows()
    core = _task_core(_hash_rows(ranking_rows))
    task_path = tmp_path / "task.jsonl"
    write_task_file(task_path, core, _task_rows())
    edges = [
        ("root", "10"),
        ("root", "20"),
        ("20", "11"),
        ("root", "11"),  # 11 is genuinely multi-parent
    ]
    return {
        "tmp": tmp_path,
        "task": task_path,
        "graph": _write_graph(tmp_path / "graph.jsonl", edges),
        "paths": _write_paths(
            tmp_path / "paths.jsonl",
            {"10": ["root", "10"], "11": ["root", "20", "11"]},
        ),
        "policy": _write_policy(tmp_path / "policy.json"),
        "ranking": _write_ranking(tmp_path / "ranking.jsonl", ranking_rows),
        "roots": [_typed("root")],
    }


def _build(world, **overrides):
    kwargs = dict(
        request_id="req-1",
        task_path=world["task"],
        qid=0,
        graph_path=world["graph"],
        principal_paths_path=world["paths"],
        policy_path=world["policy"],
        allowed_roots=world["roots"],
    )
    kwargs.update(overrides)
    return build_request(**kwargs)


# --------------------------------------------------------------------------
# 1. duplicate titles remain separate and actions use typed IDs
# --------------------------------------------------------------------------


def test_duplicate_titles_stay_distinct_and_actions_use_typed_ids(world):
    decision = decide_stage_a(_build(world))
    assert decision["decision"] == SELECT_EXISTING
    folder_id = decision["payload"]["folder_id"]
    # Both candidates carry the title "first"; only the typed ID disambiguates.
    assert folder_id == _typed("10")
    assert parse_typed_id(folder_id).stable_value == "10"
    for step in decision["payload"]["authoritative_path_ids"]:
        parse_typed_id(step)


def test_title_only_identity_is_rejected():
    with pytest.raises(FilingPathError, match="typed identity object"):
        parse_typed_id("first")
    with pytest.raises(FilingPathError, match="untyped or malformed"):
        parse_typed_id({"title": "first"})


# --------------------------------------------------------------------------
# 2. a multi-parent folder returns the frozen principal breadcrumb
# --------------------------------------------------------------------------


def test_multi_parent_folder_keeps_frozen_principal_path(world):
    # Candidate 11 is reachable directly from root, but its recorded principal
    # path goes through 20.  Search reachability is not breadcrumb authority.
    ranking_rows = _ranking_rows(folder_ids=("11", "10"))
    core = _task_core(_hash_rows(ranking_rows))
    task_path = world["tmp"] / "task_11.jsonl"
    write_task_file(task_path, core, _task_rows(folder_ids=("11", "10")))
    decision = decide_stage_a(_build(world, task_path=task_path))
    assert decision["payload"]["folder_id"] == _typed("11")
    assert decision["payload"]["authoritative_path_ids"] == [
        _typed("root"),
        _typed("20"),
        _typed("11"),
    ]
    assert decision["payload"]["path_source"] == "frozen_principal_path_records"


# --------------------------------------------------------------------------
# 3. dynamic depth: 1, current maximum, and greater than 26
# --------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 5, 27, 40])
def test_paths_round_trip_at_any_depth_without_fixed_slots(tmp_path, depth):
    chain = [f"n{index}" for index in range(depth)]
    path_file = _write_paths(tmp_path / f"paths_{depth}.jsonl", {chain[-1]: chain})
    records = read_principal_path_records(path_file)
    folder = make_typed_id(chain[-1], node_type="folder", corpus_or_account=CORPUS)
    assert [step.stable_value for step in records.path_for(folder)] == chain


def test_depth_greater_than_twenty_six_decides_end_to_end(tmp_path):
    chain = [f"n{index}" for index in range(30)]
    leaf = chain[-1]
    ranking_rows = _ranking_rows(folder_ids=(leaf, "10"))
    core = _task_core(_hash_rows(ranking_rows))
    task_path = tmp_path / "task.jsonl"
    write_task_file(task_path, core, _task_rows(folder_ids=(leaf, "10")))
    verified = build_request(
        request_id="deep",
        task_path=task_path,
        qid=0,
        graph_path=_write_graph(
            tmp_path / "graph.jsonl",
            list(zip(chain, chain[1:])) + [(chain[0], "10")],
        ),
        principal_paths_path=_write_paths(tmp_path / "paths.jsonl", {leaf: chain}),
        policy_path=_write_policy(tmp_path / "policy.json"),
        allowed_roots=[_typed(chain[0])],
    )
    decision = decide_stage_a(verified)
    assert decision["decision"] == SELECT_EXISTING
    assert len(decision["payload"]["authoritative_path_ids"]) == 30


# --------------------------------------------------------------------------
# 4. the three actions are mutually exclusive and schema valid
# --------------------------------------------------------------------------


def test_actions_are_mutually_exclusive(world):
    decision = decide_stage_a(_build(world))
    validate_decision(decision)
    assert decision["schema"] == DECISION_SCHEMA
    assert decision["requires_user_confirmation"] is True

    mixed = json.loads(json.dumps(decision))
    mixed["payload"]["reason_code"] = "ambiguous_existing"
    mixed.pop("decision_sha256")
    mixed["decision_sha256"] = sha256_bytes(canonical_json_bytes(mixed))
    with pytest.raises(FilingPathError, match="ABSTAIN fields"):
        validate_decision(mixed)


def test_propose_new_payloads_are_refused(world):
    decision = json.loads(json.dumps(decide_stage_a(_build(world))))
    decision["decision"] = PROPOSE_NEW
    decision["payload"] = {
        "anchor_parent_id": _typed("root"),
        "proposed_segments": [{"local_proposal_id": "p1", "suggested_title": "new"}],
    }
    decision.pop("decision_sha256")
    decision["decision_sha256"] = sha256_bytes(canonical_json_bytes(decision))
    with pytest.raises(FilingPathError, match="not implemented"):
        validate_decision(decision)


# --------------------------------------------------------------------------
# 5. no eligible candidate leaves no stale selected/proposed target
# --------------------------------------------------------------------------


def test_absent_eligible_candidate_abstains_without_a_target(world):
    empty_paths = _write_paths(world["tmp"] / "empty_paths.jsonl", {"99": ["root", "99"]})
    decision = decide_stage_a(_build(world, principal_paths_path=empty_paths))
    assert decision["decision"] == ABSTAIN
    assert decision["payload"]["reason_code"] == "no_eligible_candidate"
    assert "folder_id" not in decision["payload"]
    assert "authoritative_path_ids" not in decision["payload"]
    assert decision["payload"]["candidate_ids_considered"] == [_typed("10"), _typed("11")]
    validate_decision(decision)


def test_candidate_outside_allowed_roots_is_ineligible_not_a_provenance_fault(world):
    """Request scope narrows eligibility; it abstains rather than blocking."""

    other_root_paths = _write_paths(
        world["tmp"] / "other_root.jsonl",
        {"10": ["elsewhere", "10"], "11": ["elsewhere", "11"]},
    )
    graph = _write_graph(
        world["tmp"] / "other_graph.jsonl",
        [("elsewhere", "10"), ("elsewhere", "11")],
    )
    decision = decide_stage_a(
        _build(world, principal_paths_path=other_root_paths, graph_path=graph)
    )
    assert decision["decision"] == ABSTAIN
    assert decision["payload"]["reason_code"] == "no_eligible_candidate"
    assert "folder_id" not in decision["payload"]
    reasons = {
        item["reason"] for item in decision["search_receipt"]["ineligible_candidates"]
    }
    assert reasons == {"outside_allowed_roots"}


# --------------------------------------------------------------------------
# 6. a low margin cannot authorize PROPOSE_NEW without matching calibration
# --------------------------------------------------------------------------


def test_low_margin_abstains_and_never_proposes(world):
    ranking_rows = _ranking_rows(margins=(0.001, 0.5))
    core = _task_core(_hash_rows(ranking_rows))
    task_path = world["tmp"] / "task_narrow.jsonl"
    write_task_file(task_path, core, _task_rows())
    verified = _build(
        world,
        task_path=task_path,
        policy_path=_write_policy(world["tmp"] / "margin.json", margin_threshold=0.05),
        ranking_detail_path=_write_ranking(world["tmp"] / "rank2.jsonl", ranking_rows),
    )
    decision = decide_stage_a(verified)
    assert decision["decision"] == ABSTAIN
    assert decision["payload"]["reason_code"] == "ambiguous_existing"
    # The abstention records ambiguity; it emits no novelty score of any kind.
    assert "not evidence that a folder is missing" in decision["evidence_summary"]["note"]
    assert "calibrated_open_set_score" not in json.dumps(decision)
    assert decision["calibration_receipt_sha256"] is None


def test_proposal_requires_population_matched_calibration(world):
    disabled = _write_policy(world["tmp"] / "p_off.json", propose_new=False)
    enabled = _write_policy(world["tmp"] / "p_on.json", propose_new=True)
    from filing_path_decision import read_decision_policy

    with pytest.raises(FilingPathError, match="disabled by policy"):
        require_proposal_authorization(
            read_decision_policy(disabled),
            population_id=POPULATION_SHA,
            calibration_receipt={"population_id": POPULATION_SHA, "regime": "x"},
        )
    policy = read_decision_policy(enabled)
    with pytest.raises(FilingPathError, match="not a novelty probability"):
        require_proposal_authorization(
            policy, population_id=POPULATION_SHA, calibration_receipt=None
        )
    with pytest.raises(FilingPathError, match="different population"):
        require_proposal_authorization(
            policy,
            population_id=POPULATION_SHA,
            calibration_receipt={
                "population_id": "0" * 64,
                "regime": "prospective_naturally_absent",
            },
        )
    with pytest.raises(FilingPathError, match="research pilot"):
        require_proposal_authorization(
            policy,
            population_id=POPULATION_SHA,
            calibration_receipt={
                "population_id": POPULATION_SHA,
                "regime": "simulated_absence",
            },
        )
    # A prospective cohort is the only regime that licenses a live proposal.
    require_proposal_authorization(
        policy,
        population_id=POPULATION_SHA,
        calibration_receipt={
            "population_id": POPULATION_SHA,
            "regime": "prospective_naturally_absent",
        },
    )


def test_stage_a_has_no_reachable_proposal_path(world):
    enabled = _write_policy(world["tmp"] / "p_on2.json", propose_new=True)
    decision = decide_stage_a(_build(world, policy_path=enabled))
    assert decision["decision"] in (SELECT_EXISTING, ABSTAIN)


# --------------------------------------------------------------------------
# 7. threshold fitting cannot read outer labels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("partition", ["outer", "test", "simulated_absence_outer", None])
def test_threshold_selection_partition_must_be_inner(world, partition):
    from filing_path_decision import read_decision_policy

    path = _write_policy(world["tmp"] / f"policy_{partition}.json", partition=partition)
    with pytest.raises(FilingPathError, match="inner train/calibration data only"):
        read_decision_policy(path)


@pytest.mark.parametrize("partition", ["train_only", "train_calibration_only"])
def test_inner_partitions_are_accepted(world, partition):
    from filing_path_decision import read_decision_policy

    path = _write_policy(world["tmp"] / f"ok_{partition}.json", partition=partition)
    assert read_decision_policy(path).selection_partition == partition


def test_margin_rule_requires_the_bound_inherited_ranking(world):
    with pytest.raises(FilingPathError, match="ranking detail"):
        _build(
            world,
            policy_path=_write_policy(
                world["tmp"] / "needs_rank.json", margin_threshold=0.05
            ),
        )


def test_ranking_detail_must_match_the_inherited_hash(world):
    forged = _write_ranking(
        world["tmp"] / "forged.jsonl", _ranking_rows(margins=(0.9, 0.9))
    )
    with pytest.raises(FilingPathError, match="inherited ranking_sha256"):
        _build(
            world,
            policy_path=_write_policy(world["tmp"] / "m.json", margin_threshold=0.05),
            ranking_detail_path=forged,
        )


def test_ranking_detail_order_must_match_the_parent_menu(world):
    rows = _ranking_rows(folder_ids=("11", "10"))
    core = _task_core(_hash_rows(rows))
    task_path = world["tmp"] / "task_mismatch.jsonl"
    # menu says 10, 11 but the bound ranking says 11, 10
    write_task_file(task_path, core, _task_rows(folder_ids=("10", "11")))
    with pytest.raises(FilingPathError, match="disagrees with the parent task menu"):
        _build(
            world,
            task_path=task_path,
            policy_path=_write_policy(world["tmp"] / "m2.json", margin_threshold=0.05),
            ranking_detail_path=_write_ranking(world["tmp"] / "r3.jsonl", rows),
        )


# --------------------------------------------------------------------------
# 9. controls run on byte-identical manifests
# --------------------------------------------------------------------------


def test_always_existing_control_shares_byte_identical_inputs(world):
    """The margin control and the always-existing control differ only in policy."""

    always = _build(world, policy_path=_write_policy(world["tmp"] / "always.json"))
    ranking_rows = _ranking_rows(margins=(0.001, 0.5))
    core = _task_core(_hash_rows(ranking_rows))
    task_path = world["tmp"] / "task_ctrl.jsonl"
    write_task_file(task_path, core, _task_rows())
    margin = build_request(
        request_id="req-1",
        task_path=task_path,
        qid=0,
        graph_path=world["graph"],
        principal_paths_path=world["paths"],
        policy_path=_write_policy(world["tmp"] / "ctrl.json", margin_threshold=0.05),
        allowed_roots=world["roots"],
        ranking_detail_path=_write_ranking(world["tmp"] / "r4.jsonl", ranking_rows),
    )
    always_supplement = always.request["path_supplement"]
    margin_supplement = margin.request["path_supplement"]
    assert always_supplement == margin_supplement
    assert decide_stage_a(always)["decision"] == SELECT_EXISTING
    assert decide_stage_a(margin)["decision"] == ABSTAIN


# --------------------------------------------------------------------------
# 10. resource exhaustion returns best_so_far, not the last iterate
# --------------------------------------------------------------------------


def test_exhausted_scan_returns_best_so_far(world):
    """With 10 ineligible, a one-node budget censors before reaching 11."""

    only_11 = _write_paths(
        world["tmp"] / "only11.jsonl", {"11": ["root", "20", "11"]}
    )
    censored = _write_policy(world["tmp"] / "tight.json", maximum_nodes=1, maximum_steps=1)
    decision = decide_stage_a(
        _build(world, principal_paths_path=only_11, policy_path=censored)
    )
    assert decision["decision"] == ABSTAIN
    assert decision["payload"]["reason_code"] == "resource_censored"
    assert decision["search_receipt"]["resource_censored"] is True
    assert decision["search_receipt"]["best_so_far"] is None

    # A two-node budget reaches 11 and returns it as the scored best_so_far.
    roomier = _write_policy(world["tmp"] / "roomy.json", maximum_nodes=2, maximum_steps=2)
    decision = decide_stage_a(
        _build(world, principal_paths_path=only_11, policy_path=roomier)
    )
    assert decision["decision"] == SELECT_EXISTING
    assert decision["payload"]["folder_id"] == _typed("11")
    assert decision["search_receipt"]["best_so_far"]["folder_id"] == _typed("11")
    assert decision["search_receipt"]["ineligible_candidates"][0]["reason"] == (
        "no_authoritative_path"
    )


# --------------------------------------------------------------------------
# 11. the request rederives its exact parent row
# --------------------------------------------------------------------------


def test_request_rederives_the_parent_and_derives_the_row_digest(world):
    verified = _build(world)
    assert verified.request["schema"] == REQUEST_SCHEMA
    parent = verified.request["parent_task"]
    assert parent["qid"] == 0
    expected_row = sha256_bytes(canonical_json_bytes(verified.task_row))
    assert parent["row_sha256"] == expected_row


def test_tampered_parent_task_blocks(world):
    raw = Path(world["task"]).read_bytes().replace(b"bookmark 0", b"bookmark X")
    tampered = world["tmp"] / "tampered.jsonl"
    tampered.write_bytes(raw)
    with pytest.raises(FilingPathError, match="failed re-derivation"):
        _build(world, task_path=tampered)


def test_missing_qid_blocks(world):
    with pytest.raises(FilingPathError, match="exactly one row for qid"):
        _build(world, qid=99)


def test_request_cannot_restate_inherited_authority(world):
    with pytest.raises(FilingPathError, match="must not restate inherited"):
        _build(world, extra={"ranker": {"ranking_sha256": "0" * 64}})


# --------------------------------------------------------------------------
# 12. private/unknown inputs and external naming
# --------------------------------------------------------------------------


def _naming_receipt(request_sha, item_sha, privacy_sha, *, hours=1):
    return {
        "request_sha256": request_sha,
        "item_content_sha256": item_sha,
        "inherited_privacy_receipt_sha256": privacy_sha,
        "provider": "example",
        "model": "namer",
        "model_revision": "r1",
        "permitted_fields": ["bookmark_title"],
        "purpose": "folder_name_suggestion",
        "expires_at": (
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)
        ).isoformat(),
    }


def test_unknown_privacy_can_never_reach_an_external_namer():
    with pytest.raises(FilingPathError, match="local-only"):
        check_external_naming(
            privacy_class="unknown",
            request_sha256="a" * 64,
            item_content_sha256="b" * 64,
            inherited_privacy_receipt_sha256="c" * 64,
            receipt=_naming_receipt("a" * 64, "b" * 64, "c" * 64),
        )


def test_private_naming_requires_an_exact_content_bound_receipt():
    args = dict(
        privacy_class="private",
        request_sha256="a" * 64,
        item_content_sha256="b" * 64,
        inherited_privacy_receipt_sha256="c" * 64,
    )
    assert check_external_naming(receipt=None, **args) is None
    with pytest.raises(FilingPathError, match="different request"):
        check_external_naming(
            receipt=_naming_receipt("9" * 64, "b" * 64, "c" * 64), **args
        )
    with pytest.raises(FilingPathError, match="different item content"):
        check_external_naming(
            receipt=_naming_receipt("a" * 64, "9" * 64, "c" * 64), **args
        )
    with pytest.raises(FilingPathError, match="privacy derivation"):
        check_external_naming(
            receipt=_naming_receipt("a" * 64, "b" * 64, "9" * 64), **args
        )
    with pytest.raises(FilingPathError, match="expired"):
        check_external_naming(
            receipt=_naming_receipt("a" * 64, "b" * 64, "c" * 64, hours=-1), **args
        )
    digest = check_external_naming(
        receipt=_naming_receipt("a" * 64, "b" * 64, "c" * 64), **args
    )
    assert len(digest) == 64


def test_privacy_class_is_derived_not_accepted(world):
    ranking_rows = _ranking_rows()
    core = _task_core(_hash_rows(ranking_rows), public=False)
    assert derive_privacy_class(core) == "unknown"
    assert derive_privacy_class(_task_core(_hash_rows(ranking_rows))) == "public"
    with pytest.raises(FilingPathError, match="must not restate inherited"):
        _build(world, extra={"privacy_class": "public"})


def test_external_naming_receipt_requires_an_enabling_policy(world):
    verified = _build(world)
    receipt = _naming_receipt(
        verified.request_sha256,
        verified.request["parent_task"]["row_sha256"],
        verified.inherited_privacy_receipt_sha256,
    )
    with pytest.raises(FilingPathError, match="policy disables external naming"):
        _build(world, external_naming_authorization_receipt=receipt)


def test_request_sha256_is_stable_across_the_naming_receipt_field(world):
    """The receipt commits to request_sha256, so that hash excludes the receipt."""

    enabling = _write_policy(world["tmp"] / "naming_on.json", external_naming=True)
    unbound = _build(world, policy_path=enabling)
    assert unbound.external_naming_authorization_receipt_sha256 is None

    receipt = _naming_receipt(
        unbound.request_sha256,
        unbound.request["parent_task"]["row_sha256"],
        unbound.inherited_privacy_receipt_sha256,
    )
    authorized = _build(
        world, policy_path=enabling, external_naming_authorization_receipt=receipt
    )
    # Attaching the receipt must not move the hash the receipt commits to.
    assert authorized.request_sha256 == unbound.request_sha256
    assert authorized.external_naming_authorization_receipt_sha256 == sha256_bytes(
        canonical_json_bytes(receipt)
    )
    rebuilt = dict(authorized.request)
    rebuilt.pop("external_naming_authorization_receipt_sha256")
    assert sha256_bytes(canonical_json_bytes(rebuilt)) == authorized.request_sha256


# --------------------------------------------------------------------------
# 13. stale hashes, malformed paths, illegal transitions, nonfinite values
# --------------------------------------------------------------------------


def test_illegal_recorded_transition_blocks(world):
    detached = _write_paths(
        world["tmp"] / "detached.jsonl", {"10": ["root", "20", "10"]}
    )
    with pytest.raises(FilingPathError, match="absent from the frozen graph"):
        decide_stage_a(_build(world, principal_paths_path=detached))


def test_path_must_terminate_at_its_own_folder(world):
    bad = world["tmp"] / "bad_terminal.jsonl"
    rows = [
        {
            "record_type": "principal_path",
            "folder_id": _typed("10"),
            "path_ids": [_typed("root"), _typed("20")],
            "titles": ["root", "twenty"],
        }
    ]
    header = {
        "schema": PATH_RECORDS_SCHEMA,
        "record_type": "principal_path_header",
        "corpus_or_account": CORPUS,
        "principal_path_policy_id": "p",
    }
    bad.write_bytes(
        canonical_json_bytes(header) + b"".join(canonical_json_bytes(r) for r in rows)
    )
    with pytest.raises(FilingPathError, match="terminate at its own folder_id"):
        read_principal_path_records(bad)


def test_stale_path_records_hash_blocks(world):
    stale = world["tmp"] / "stale.jsonl"
    raw = Path(world["paths"]).read_bytes().splitlines(keepends=True)
    header = json.loads(raw[0])
    header["principal_path_records_sha256"] = "0" * 64
    stale.write_bytes(canonical_json_bytes(header) + b"".join(raw[1:]))
    with pytest.raises(FilingPathError, match="does not match its rows"):
        read_principal_path_records(stale)


def test_nonfinite_margin_blocks(world):
    rows = [
        {
            "qid": 0,
            "margin_float64_hex": "nan",
            "top_k_folder_ids": ["10", "11"],
        },
        _ranking_rows()[1],
    ]
    core = _task_core(_hash_rows(rows))
    task_path = world["tmp"] / "task_nan.jsonl"
    write_task_file(task_path, core, _task_rows())
    with pytest.raises(FilingPathError, match="ranking detail margin"):
        _build(
            world,
            task_path=task_path,
            policy_path=_write_policy(world["tmp"] / "m3.json", margin_threshold=0.05),
            ranking_detail_path=_write_ranking(world["tmp"] / "r5.jsonl", rows),
        )


def test_policy_id_must_bind_its_own_content(world):
    path = world["tmp"] / "forged_policy.json"
    document = json.loads(Path(world["policy"]).read_text())
    document["propose_new_enabled"] = True  # policy_id no longer matches
    path.write_bytes(canonical_json_bytes(document))
    from filing_path_decision import read_decision_policy

    with pytest.raises(FilingPathError, match="does not bind its own content"):
        read_decision_policy(path)


def test_duplicate_principal_path_record_blocks(world):
    dup = world["tmp"] / "dup.jsonl"
    row = {
        "record_type": "principal_path",
        "folder_id": _typed("10"),
        "path_ids": [_typed("root"), _typed("10")],
        "titles": ["root", "ten"],
    }
    header = {
        "schema": PATH_RECORDS_SCHEMA,
        "record_type": "principal_path_header",
        "corpus_or_account": CORPUS,
        "principal_path_policy_id": "p",
    }
    dup.write_bytes(
        canonical_json_bytes(header)
        + canonical_json_bytes(row)
        + canonical_json_bytes(row)
    )
    with pytest.raises(FilingPathError, match="duplicate principal path record"):
        read_principal_path_records(dup)


# --------------------------------------------------------------------------
# 14. recommendation cannot mutate the source graph or call a filing API
# --------------------------------------------------------------------------


def test_decision_leaves_every_input_artifact_byte_identical(world):
    before = {
        name: Path(world[name]).read_bytes()
        for name in ("task", "graph", "paths", "policy", "ranking")
    }
    decide_stage_a(_build(world))
    after = {
        name: Path(world[name]).read_bytes()
        for name in ("task", "graph", "paths", "policy", "ranking")
    }
    assert before == after


def test_module_exposes_no_mutation_surface():
    import filing_path_decision as module

    banned = ("create_folder", "move_bookmark", "apply_decision", "write_graph")
    for name in banned:
        assert not hasattr(module, name)
    source = Path(module.__file__).read_text()
    for token in ("requests.post", "urlopen", "http.client", "os.remove", "shutil.rmtree"):
        assert token not in source


def test_decision_receipt_is_no_replace(world):
    decision = decide_stage_a(_build(world))
    out = world["tmp"] / "decision.jsonl"
    write_decision(out, decision)
    assert read_decision(out)["decision_sha256"] == decision["decision_sha256"]
    with pytest.raises(FilingPathError, match="refusing to overwrite"):
        write_decision(out, decision)


def test_decision_digest_detects_tampering(world):
    decision = decide_stage_a(_build(world))
    tampered = json.loads(json.dumps(decision))
    tampered["payload"]["catalog_rank"] = 99
    with pytest.raises(FilingPathError, match="does not bind its own content"):
        validate_decision(tampered)


# --------------------------------------------------------------------------
# 15. confirmation revalidates the catalog snapshot
# --------------------------------------------------------------------------


def test_confirmation_requires_user_and_an_unchanged_catalog(world):
    verified = _build(world)
    decision = decide_stage_a(verified)
    with pytest.raises(FilingPathError, match="without user confirmation"):
        confirm_decision(
            decision,
            current_catalog_sha256=CATALOG_SHA,
            current_principal_path_records_sha256=verified.paths.records_sha256,
            user_confirmed=False,
        )
    with pytest.raises(FilingPathError, match="catalog changed"):
        confirm_decision(
            decision,
            current_catalog_sha256="0" * 64,
            current_principal_path_records_sha256=verified.paths.records_sha256,
            user_confirmed=True,
        )
    with pytest.raises(FilingPathError, match="principal path records changed"):
        confirm_decision(
            decision,
            current_catalog_sha256=CATALOG_SHA,
            current_principal_path_records_sha256="0" * 64,
            user_confirmed=True,
        )
    record = confirm_decision(
        decision,
        current_catalog_sha256=CATALOG_SHA,
        current_principal_path_records_sha256=verified.paths.records_sha256,
        user_confirmed=True,
    )
    assert record["authorizes_mutation"] is False
    assert len(record["confirmation_sha256"]) == 64


def test_select_existing_is_never_reported_as_censored(world):
    """A successful scan breaks before the budget check can fire."""

    from filing_path_decision import STAGE_A_ABSTAIN_REASONS, _build_decision

    decision = decide_stage_a(_build(world))
    assert decision["decision"] == SELECT_EXISTING
    assert decision["search_receipt"]["resource_censored"] is False

    verified = _build(world)
    with pytest.raises(FilingPathError, match="censored search cannot also report"):
        _build_decision(
            verified,
            decision=SELECT_EXISTING,
            payload=decision["payload"],
            search_receipt={**decision["search_receipt"], "resource_censored": True},
            evidence_summary={},
        )
    # Stage A declines to mint a reason code belonging to a gated stage.
    assert "proposal_need_uncertain" not in STAGE_A_ABSTAIN_REASONS
    with pytest.raises(FilingPathError, match="Stage A cannot emit abstain reason"):
        _build_decision(
            verified,
            decision=ABSTAIN,
            payload={"reason_code": "proposal_need_uncertain",
                     "candidate_ids_considered": []},
            search_receipt=decision["search_receipt"],
            evidence_summary={},
        )


def test_public_data_receipt_is_still_held_to_exact_binding(world):
    """A supplied receipt is never waved through just because data is public."""

    verified = _build(world)
    bad = _naming_receipt("9" * 64, "b" * 64, "c" * 64)
    with pytest.raises(FilingPathError, match="different request"):
        check_external_naming(
            privacy_class="public",
            request_sha256=verified.request_sha256,
            item_content_sha256="b" * 64,
            inherited_privacy_receipt_sha256="c" * 64,
            receipt=bad,
        )
    # ...and an unrecognised class is rejected rather than falling through.
    with pytest.raises(FilingPathError, match="unsupported privacy class"):
        check_external_naming(
            privacy_class="confidential",
            request_sha256="a" * 64,
            item_content_sha256="b" * 64,
            inherited_privacy_receipt_sha256="c" * 64,
            receipt=_naming_receipt("a" * 64, "b" * 64, "c" * 64),
        )


def test_search_receipt_digest_binds_the_recorded_search(world):
    decision = json.loads(json.dumps(decide_stage_a(_build(world))))
    decision["search_receipt"]["node_expansions"] = 99
    decision.pop("decision_sha256")
    decision["decision_sha256"] = sha256_bytes(canonical_json_bytes(decision))
    with pytest.raises(FilingPathError, match="does not bind the recorded search"):
        validate_decision(decision)


def test_cli_decide_emits_a_no_replace_receipt(world, capsys):
    from filing_path_decision import main

    out = world["tmp"] / "cli_decision.jsonl"
    code = main(
        [
            "decide",
            "--task", str(world["task"]),
            "--qid", "0",
            "--graph", str(world["graph"]),
            "--paths", str(world["paths"]),
            "--policy", str(world["policy"]),
            "--allowed-root", f"folder:{CORPUS}:root",
            "--request-id", "cli-1",
            "--out", str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert SELECT_EXISTING in printed
    assert "no graph mutation performed" in printed
    assert read_decision(out)["payload"]["folder_id"] == _typed("10")


def test_cli_blocks_instead_of_abstaining_on_a_provenance_failure(world, capsys):
    from filing_path_decision import main

    code = main(
        [
            "decide",
            "--task", str(world["task"]),
            "--qid", "42",
            "--graph", str(world["graph"]),
            "--paths", str(world["paths"]),
            "--policy", str(world["policy"]),
            "--allowed-root", f"folder:{CORPUS}:root",
            "--request-id", "cli-2",
        ]
    )
    assert code == 2
    assert "blocked:" in capsys.readouterr().err


# --------------------------------------------------------------------------
# 16. the evaluator rejects same-e5-only grading
# --------------------------------------------------------------------------


def test_same_embedding_grading_is_rejected_unless_labeled_circular():
    circular = {
        "construction_model_id": "intfloat/e5-small-v2",
        "construction_model_revision": "abc",
        "evaluator_model_id": "intfloat/e5-small-v2",
        "evaluator_model_revision": "abc",
    }
    with pytest.raises(FilingPathError, match="self-consistency, not recovery"):
        assert_not_circular_grading(circular)
    assert_not_circular_grading({**circular, "labeled_circular_diagnostic": True})
    assert_not_circular_grading(
        {**circular, "evaluator_model_id": "other/model", "evaluator_model_revision": "z"}
    )


# --------------------------------------------------------------------------
# transaction: prepare -> verify -> recommend reproduces at identical hashes
# --------------------------------------------------------------------------


def test_decision_is_reproducible_at_identical_hashes(world):
    first = decide_stage_a(_build(world))
    second = decide_stage_a(_build(world))
    assert first == second


def test_graph_snapshot_rejects_duplicate_and_self_edges(world):
    from filing_path_decision import read_graph_snapshot

    dup = _write_graph(world["tmp"] / "dup_edge.jsonl", [("root", "10"), ("root", "10")])
    with pytest.raises(FilingPathError, match="duplicate edge"):
        read_graph_snapshot(dup)
    loop = _write_graph(world["tmp"] / "self_edge.jsonl", [("root", "root")])
    with pytest.raises(FilingPathError, match="self edge"):
        read_graph_snapshot(loop)


def test_graph_and_paths_must_describe_one_corpus(world, tmp_path):
    other = tmp_path / "other_paths.jsonl"
    rows = [
        {
            "record_type": "principal_path",
            "folder_id": {
                "node_type": "folder",
                "corpus_or_account": "other",
                "stable_value": "10",
            },
            "path_ids": [
                {
                    "node_type": "folder",
                    "corpus_or_account": "other",
                    "stable_value": "10",
                }
            ],
            "titles": ["ten"],
        }
    ]
    header = {
        "schema": PATH_RECORDS_SCHEMA,
        "record_type": "principal_path_header",
        "corpus_or_account": "other",
        "principal_path_policy_id": "p",
    }
    other.write_bytes(
        canonical_json_bytes(header) + b"".join(canonical_json_bytes(r) for r in rows)
    )
    with pytest.raises(FilingPathError, match="different corpora"):
        _build(world, principal_paths_path=other)
