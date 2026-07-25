#!/usr/bin/env python3
"""Stage A of the filing path decoder: additive, no-mutation path decisions.

This implements the first engineering step named in
``DESIGN_filing_path_decoder_handoff.md`` §11: the common request/decision
schemas plus the frozen-e5 existing-folder baseline.  Everything here is
*additive* over ``unifyweaver.routed-task.v2``.  The parent task remains
authoritative for source bytes, privacy certification, catalog, population,
ordered candidates, the frozen e5 ranker, policy tier, prompt, and judge
contract.  This layer contributes only:

  * a graph / principal-path supplement,
  * bounded search limits and a deterministic ``best_so_far`` scan,
  * an advisory decision, and
  * a content-bound decision receipt.

Three things this module deliberately cannot do:

  * mutate a graph, create a folder, or call a filing API — no such code path
    exists here, and every artifact it opens is opened read-only;
  * emit ``PROPOSE_NEW`` — ordinary proposals stay disabled until a
    prospective naturally absent cohort produces a matching calibration
    artifact (§2 and §10 of the handoff).  A low top-two margin is ambiguity,
    never a novelty probability; and
  * generate, repair, shorten, or reorder a breadcrumb — an existing folder's
    path is copied verbatim from the frozen principal-path records.

An ordinary hard case produces ``ABSTAIN``.  A provenance failure raises
``FilingPathError`` and produces no valid decision; it is never disguised as an
abstention.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as _datetime
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filing_privacy import (
    POLICY_ID as PRIVACY_POLICY_ID,
    PUBLIC_CATALOG_POLICY_ID,
)
from routed_policy import (
    RoutedPolicyError,
    _atomic_write_jsonl_no_clobber,
    canonical_json_bytes,
    file_content_record,
    float64_hex,
    parse_float64_hex,
    read_task_file,
    sha256_bytes,
    strict_json_loads,
)


REQUEST_SCHEMA = "unifyweaver.filing-path-request.v1"
DECISION_SCHEMA = "unifyweaver.filing-path-decision.v1"
POLICY_SCHEMA = "unifyweaver.filing-path-policy.v1"
PATH_RECORDS_SCHEMA = "unifyweaver.filing-principal-paths.v1"
GRAPH_SNAPSHOT_SCHEMA = "unifyweaver.filing-graph-snapshot.v1"
SEARCH_RECEIPT_SCHEMA = "unifyweaver.filing-path-search.v1"
CONFIRMATION_SCHEMA = "unifyweaver.filing-path-confirmation.v1"

SELECT_EXISTING = "SELECT_EXISTING"
PROPOSE_NEW = "PROPOSE_NEW"
ABSTAIN = "ABSTAIN"
DECISIONS = (SELECT_EXISTING, PROPOSE_NEW, ABSTAIN)

#: §1.3 reason codes.  These are operational outputs, not scientific findings.
ABSTAIN_REASONS = (
    "ambiguous_existing",
    "proposal_need_uncertain",
    "outside_calibration_support",
    "resource_censored",
    "privacy_restricted_naming",
    "no_eligible_candidate",
)

#: The parent task is the only source of these.  A request that restates one is
#: rejected rather than silently ignored (§0, §3).
FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "bookmark",
        "candidates",
        "catalog",
        "e5_score",
        "judge_contract",
        "menu",
        "policy_provenance",
        "population",
        "privacy",
        "privacy_class",
        "ranker",
        "ranking",
        "selection",
        "source",
        "top_k_folder_ids",
    }
)

STAGE_A_OBJECTIVE = "first_eligible_in_frozen_parent_order"


class FilingPathError(ValueError):
    """A filing path request, supplement, policy, or decision failed closed."""


# --------------------------------------------------------------------------
# typed identity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedId:
    """A stable typed folder identity.  Titles are attributes, not join keys."""

    node_type: str
    corpus_or_account: str
    stable_value: str

    def as_record(self) -> dict[str, str]:
        return {
            "node_type": self.node_type,
            "corpus_or_account": self.corpus_or_account,
            "stable_value": self.stable_value,
        }


def parse_typed_id(value: Any, label: str = "identity") -> TypedId:
    if not isinstance(value, Mapping):
        raise FilingPathError(f"{label} must be a typed identity object, not a title")
    expected = {"node_type", "corpus_or_account", "stable_value"}
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise FilingPathError(
            f"{label} has untyped or malformed identity fields "
            f"(missing={missing}, unexpected={extra})"
        )
    for key in sorted(expected):
        field = value[key]
        if not isinstance(field, str) or not field:
            raise FilingPathError(f"{label}.{key} must be a nonempty string")
    return TypedId(
        node_type=value["node_type"],
        corpus_or_account=value["corpus_or_account"],
        stable_value=value["stable_value"],
    )


def make_typed_id(stable_value: str, *, node_type: str, corpus_or_account: str) -> TypedId:
    return parse_typed_id(
        {
            "node_type": node_type,
            "corpus_or_account": corpus_or_account,
            "stable_value": stable_value,
        }
    )


def _hash_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(b"".join(canonical_json_bytes(dict(row)) for row in rows))


def _read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    records = []
    raw = Path(path).read_bytes()
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        records.append(strict_json_loads(line, f"{path}:{line_no}"))
    if not records:
        raise FilingPathError(f"empty JSONL artifact: {path}")
    return records


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FilingPathError(f"{label} must be an integer")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    number = _require_int(value, label)
    if number <= 0:
        raise FilingPathError(f"{label} must be positive")
    return number


def _parse_float64_hex(value: Any, label: str) -> float | None:
    """``parse_float64_hex`` with this module's error type.

    Nonfinite and noncanonical encodings are rejected by the shared primitive;
    re-raising as ``FilingPathError`` keeps every artifact failure one type.
    """

    try:
        return parse_float64_hex(value)
    except RoutedPolicyError as exc:
        raise FilingPathError(f"{label}: {exc}") from exc


# --------------------------------------------------------------------------
# graph snapshot and principal path records (the path supplement)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSnapshot:
    """Frozen topology.  Used for legality checks and, later, Stage C search."""

    corpus_or_account: str
    edges: frozenset[tuple[TypedId, TypedId]]
    snapshot_sha256: str
    edge_table_sha256: str

    def has_edge(self, parent: TypedId, child: TypedId) -> bool:
        return (parent, child) in self.edges


def read_graph_snapshot(path: str | os.PathLike[str]) -> GraphSnapshot:
    records = _read_jsonl(path)
    header, rows = records[0], records[1:]
    if (
        not isinstance(header, Mapping)
        or header.get("schema") != GRAPH_SNAPSHOT_SCHEMA
        or header.get("record_type") != "graph_snapshot_header"
    ):
        raise FilingPathError("malformed graph snapshot header")
    corpus = header.get("corpus_or_account")
    if not isinstance(corpus, str) or not corpus:
        raise FilingPathError("graph snapshot corpus_or_account is required")
    edges: set[tuple[TypedId, TypedId]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("record_type") != "edge":
            raise FilingPathError("graph snapshot row has wrong record_type")
        parent = parse_typed_id(row.get("parent_id"), "edge parent_id")
        child = parse_typed_id(row.get("child_id"), "edge child_id")
        if parent == child:
            raise FilingPathError("graph snapshot contains a self edge")
        if (parent, child) in edges:
            raise FilingPathError("graph snapshot contains a duplicate edge")
        edges.add((parent, child))
    edge_table = _hash_rows(rows)
    declared = header.get("edge_table_sha256")
    if declared is not None and declared != edge_table:
        raise FilingPathError("graph snapshot edge_table_sha256 does not match its rows")
    return GraphSnapshot(
        corpus_or_account=corpus,
        edges=frozenset(edges),
        snapshot_sha256=file_content_record(path)["sha256"],
        edge_table_sha256=edge_table,
    )


@dataclass(frozen=True)
class PrincipalPathRecords:
    """The catalog's recorded canonical path per folder — display authority."""

    policy_id: str
    corpus_or_account: str
    by_folder: Mapping[TypedId, tuple[TypedId, ...]]
    titles_by_folder: Mapping[TypedId, tuple[str, ...]]
    records_sha256: str

    def path_for(self, folder_id: TypedId) -> tuple[TypedId, ...] | None:
        return self.by_folder.get(folder_id)


def read_principal_path_records(path: str | os.PathLike[str]) -> PrincipalPathRecords:
    records = _read_jsonl(path)
    header, rows = records[0], records[1:]
    if (
        not isinstance(header, Mapping)
        or header.get("schema") != PATH_RECORDS_SCHEMA
        or header.get("record_type") != "principal_path_header"
    ):
        raise FilingPathError("malformed principal path header")
    policy_id = header.get("principal_path_policy_id")
    corpus = header.get("corpus_or_account")
    if not isinstance(policy_id, str) or not policy_id:
        raise FilingPathError("principal_path_policy_id is required")
    if not isinstance(corpus, str) or not corpus:
        raise FilingPathError("principal path corpus_or_account is required")
    by_folder: dict[TypedId, tuple[TypedId, ...]] = {}
    titles: dict[TypedId, tuple[str, ...]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("record_type") != "principal_path":
            raise FilingPathError("principal path row has wrong record_type")
        folder_id = parse_typed_id(row.get("folder_id"), "principal path folder_id")
        raw_path = row.get("path_ids")
        if not isinstance(raw_path, list) or not raw_path:
            raise FilingPathError("path_ids must be a nonempty list")
        # No fixed-slot assumption anywhere: depth is whatever the record says.
        path_ids = tuple(
            parse_typed_id(item, f"path_ids[{index}]") for index, item in enumerate(raw_path)
        )
        if len(set(path_ids)) != len(path_ids):
            raise FilingPathError("principal path repeats a node")
        if path_ids[-1] != folder_id:
            raise FilingPathError("principal path must terminate at its own folder_id")
        row_titles = row.get("titles")
        if not isinstance(row_titles, list) or len(row_titles) != len(path_ids):
            raise FilingPathError("titles must align one-to-one with path_ids")
        for title in row_titles:
            if not isinstance(title, str) or not title:
                raise FilingPathError("principal path title must be a nonempty string")
        if folder_id in by_folder:
            raise FilingPathError("duplicate principal path record for one folder_id")
        by_folder[folder_id] = path_ids
        titles[folder_id] = tuple(row_titles)
    declared = header.get("principal_path_records_sha256")
    computed = _hash_rows(rows)
    if declared is not None and declared != computed:
        raise FilingPathError("principal_path_records_sha256 does not match its rows")
    return PrincipalPathRecords(
        policy_id=policy_id,
        corpus_or_account=corpus,
        by_folder=by_folder,
        titles_by_folder=titles,
        records_sha256=computed,
    )


def path_is_in_scope(path_ids: Sequence[TypedId], allowed_roots: Iterable[TypedId]) -> bool:
    """Request scope, not provenance: an out-of-scope folder is ineligible."""

    roots = set(allowed_roots)
    return not roots or path_ids[0] in roots


def validate_path_transitions(path_ids: Sequence[TypedId], graph: GraphSnapshot):
    """A recorded step that is not an edge is a provenance failure, not a miss."""

    for parent, child in zip(path_ids, path_ids[1:]):
        if not graph.has_edge(parent, child):
            raise FilingPathError(
                "principal path uses a transition absent from the frozen graph"
            )


def validate_path_against_graph(
    path_ids: Sequence[TypedId], graph: GraphSnapshot, allowed_roots: Iterable[TypedId]
):
    """Every recorded step must be a legal edge below an allowed root."""

    if not path_is_in_scope(path_ids, allowed_roots):
        raise FilingPathError("principal path root is outside allowed_roots")
    validate_path_transitions(path_ids, graph)


# --------------------------------------------------------------------------
# decision policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionPolicy:
    policy_id: str
    margin_threshold: float | None
    selection_partition: str
    propose_new_enabled: bool
    maximum_search_nodes: int
    maximum_search_steps: int
    external_naming_enabled: bool
    document: Mapping[str, Any]


#: Threshold fitting may read inner training/calibration data only (§7.1).
INNER_PARTITIONS = frozenset({"train_only", "train_calibration_only"})


def policy_id_for(document: Mapping[str, Any]) -> str:
    core = {key: value for key, value in document.items() if key != "policy_id"}
    return sha256_bytes(canonical_json_bytes(core))


def read_decision_policy(path: str | os.PathLike[str]) -> DecisionPolicy:
    document = strict_json_loads(Path(path).read_bytes(), source=str(path))
    if not isinstance(document, Mapping) or document.get("schema") != POLICY_SCHEMA:
        raise FilingPathError("unsupported filing path policy schema")
    declared_id = document.get("policy_id")
    computed_id = policy_id_for(document)
    if declared_id != computed_id:
        raise FilingPathError("filing path policy_id does not bind its own content")

    abstain = document.get("abstain")
    if not isinstance(abstain, Mapping):
        raise FilingPathError("policy abstain block is required")
    partition = abstain.get("selection_partition")
    if partition not in INNER_PARTITIONS:
        raise FilingPathError(
            "abstention thresholds may be selected on inner train/calibration data only; "
            f"got selection_partition={partition!r}"
        )
    threshold = _parse_float64_hex(
        abstain.get("margin_threshold_float64_hex"), "margin threshold"
    )

    limits = document.get("search_limits")
    if not isinstance(limits, Mapping):
        raise FilingPathError("policy search_limits are required")
    maximum_nodes = _require_positive_int(
        limits.get("maximum_search_nodes"), "maximum_search_nodes"
    )
    maximum_steps = _require_positive_int(
        limits.get("maximum_search_steps"), "maximum_search_steps"
    )

    propose = document.get("propose_new_enabled")
    if not isinstance(propose, bool):
        raise FilingPathError("propose_new_enabled must be a boolean")
    naming = document.get("external_naming", {})
    if not isinstance(naming, Mapping):
        raise FilingPathError("external_naming must be an object")
    naming_enabled = naming.get("enabled", False)
    if not isinstance(naming_enabled, bool):
        raise FilingPathError("external_naming.enabled must be a boolean")

    return DecisionPolicy(
        policy_id=computed_id,
        margin_threshold=threshold,
        selection_partition=partition,
        propose_new_enabled=propose,
        maximum_search_nodes=maximum_nodes,
        maximum_search_steps=maximum_steps,
        external_naming_enabled=naming_enabled,
        document=dict(document),
    )


def require_proposal_authorization(
    policy: DecisionPolicy,
    *,
    population_id: str,
    calibration_receipt: Mapping[str, Any] | None,
):
    """§2/§10: a margin never authorizes ``PROPOSE_NEW`` on its own.

    A proposal needs a calibration artifact fitted on *this* population.  A
    simulated-absence artifact can license only a labeled, owner-reviewed
    research pilot; a live proposal needs a prospective naturally absent
    cohort.  Stage A therefore has no reachable proposal path at all.
    """

    if not policy.propose_new_enabled:
        raise FilingPathError(
            "PROPOSE_NEW is disabled by policy; ordinary proposals remain blocked "
            "until a prospective naturally absent cohort supplies a calibration artifact"
        )
    if not isinstance(calibration_receipt, Mapping):
        raise FilingPathError(
            "PROPOSE_NEW requires a population-specific calibration artifact; "
            "a low top-two margin is ambiguity, not a novelty probability"
        )
    if calibration_receipt.get("population_id") != population_id:
        raise FilingPathError(
            "calibration artifact was fitted on a different population; "
            "its score is outside the calibration support claimed for it"
        )
    regime = calibration_receipt.get("regime")
    if regime == "prospective_naturally_absent":
        return
    if regime == "simulated_absence":
        if not calibration_receipt.get("research_pilot_label"):
            raise FilingPathError(
                "simulated catalog absence may support only a clearly labeled "
                "research pilot, never a live novelty claim"
            )
        review = calibration_receipt.get("owner_review_receipt_sha256")
        if not isinstance(review, str) or len(review) != 64:
            raise FilingPathError(
                "a simulated-absence research pilot requires an owner review receipt"
            )
        return
    raise FilingPathError(f"unsupported calibration regime: {regime!r}")


# --------------------------------------------------------------------------
# privacy derivation and external naming authorization
# --------------------------------------------------------------------------


def derive_privacy_class(task_core: Mapping[str, Any]) -> str:
    """Derive privacy from the reverified parent, never from a request field."""

    privacy = task_core.get("privacy")
    catalog = task_core.get("catalog")
    if not isinstance(privacy, Mapping) or not isinstance(catalog, Mapping):
        raise FilingPathError("parent task lacks a privacy or catalog receipt")
    manifest = privacy.get("manifest_sha256")
    if not isinstance(manifest, str) or len(manifest) != 64:
        raise FilingPathError("parent privacy manifest digest is missing or truncated")
    if (
        privacy.get("policy_id") == PRIVACY_POLICY_ID
        and catalog.get("policy_id") == PUBLIC_CATALOG_POLICY_ID
    ):
        return "public"
    return "unknown"


def _parse_expiry(value: Any) -> _datetime.datetime:
    if not isinstance(value, str) or not value:
        raise FilingPathError("authorization receipt expiry is required")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = _datetime.datetime.fromisoformat(text)
    except ValueError as exc:
        raise FilingPathError(f"malformed authorization expiry: {value!r}") from exc
    if parsed.tzinfo is None:
        raise FilingPathError("authorization expiry must carry a UTC offset")
    return parsed


def check_external_naming(
    *,
    privacy_class: str,
    request_sha256: str,
    item_content_sha256: str,
    inherited_privacy_receipt_sha256: str,
    receipt: Mapping[str, Any] | None,
    now: _datetime.datetime | None = None,
) -> str | None:
    """§6: private naming needs an exact content-bound receipt; unknown never.

    Returns the receipt digest when an external call is permitted, or ``None``
    when naming must stay local-only.  Raises when a caller tries to send
    private or unknown content outward without exact authorization.
    """

    if receipt is None:
        return None
    if privacy_class == "unknown":
        raise FilingPathError(
            "unknown-privacy data is local-only; owner authorization cannot "
            "substitute for a missing privacy classification"
        )
    if privacy_class == "public":
        # Certified-public data does not need a private-content receipt, but a
        # supplied receipt must still bind this exact request.
        pass
    elif privacy_class != "private":
        raise FilingPathError(f"unsupported privacy class: {privacy_class!r}")

    for field in (
        "request_sha256",
        "item_content_sha256",
        "inherited_privacy_receipt_sha256",
        "provider",
        "model",
        "model_revision",
        "permitted_fields",
        "purpose",
        "expires_at",
    ):
        if field not in receipt:
            raise FilingPathError(f"authorization receipt is missing {field}")
    if receipt["request_sha256"] != request_sha256:
        raise FilingPathError("authorization receipt is bound to a different request")
    if receipt["item_content_sha256"] != item_content_sha256:
        raise FilingPathError("authorization receipt is bound to different item content")
    if receipt["inherited_privacy_receipt_sha256"] != inherited_privacy_receipt_sha256:
        raise FilingPathError(
            "authorization receipt does not bind this content-bound privacy derivation"
        )
    permitted = receipt["permitted_fields"]
    if not isinstance(permitted, list) or not permitted:
        raise FilingPathError("authorization receipt must enumerate permitted fields")
    expiry = _parse_expiry(receipt["expires_at"])
    moment = now or _datetime.datetime.now(_datetime.timezone.utc)
    if moment >= expiry:
        raise FilingPathError("authorization receipt has expired")
    return sha256_bytes(canonical_json_bytes(dict(receipt)))


def assert_not_circular_grading(grading_spec: Mapping[str, Any]):
    """§7.4: do not grade an e5-built proposal by the same e5 alone."""

    if not isinstance(grading_spec, Mapping):
        raise FilingPathError("grading spec must be an object")
    constructor = (
        grading_spec.get("construction_model_id"),
        grading_spec.get("construction_model_revision"),
    )
    evaluator = (
        grading_spec.get("evaluator_model_id"),
        grading_spec.get("evaluator_model_revision"),
    )
    if any(part is None for part in constructor + evaluator):
        raise FilingPathError("grading spec must name construction and evaluator models")
    if constructor == evaluator and not grading_spec.get("labeled_circular_diagnostic"):
        raise FilingPathError(
            "same-embedding grading of an embedding-constructed proposal establishes "
            "self-consistency, not recovery; label it a circular diagnostic or use an "
            "independent evaluator"
        )


# --------------------------------------------------------------------------
# request construction and verification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedRequest:
    request: Mapping[str, Any]
    request_sha256: str
    task_core: Mapping[str, Any]
    task_row: Mapping[str, Any]
    menu_ids: tuple[TypedId, ...]
    graph: GraphSnapshot
    paths: PrincipalPathRecords
    policy: DecisionPolicy
    allowed_roots: tuple[TypedId, ...]
    privacy_class: str
    inherited_privacy_receipt_sha256: str
    population_id: str
    top_two_margin: float | None
    calibration_receipt_sha256: str | None
    external_naming_authorization_receipt_sha256: str | None


def _menu_typed_ids(
    row: Mapping[str, Any], *, node_type: str, corpus_or_account: str
) -> tuple[TypedId, ...]:
    menu = row.get("menu")
    if not isinstance(menu, list) or not menu:
        raise FilingPathError("parent task row has no candidate menu")
    ids = []
    seen: set[TypedId] = set()
    for position, item in enumerate(menu):
        if not isinstance(item, Mapping):
            raise FilingPathError("menu item must be an object")
        if item.get("pos") != position:
            raise FilingPathError("menu positions must be contiguous and zero based")
        raw = item.get("folder_id")
        if not isinstance(raw, str) or not raw:
            raise FilingPathError("menu folder_id must be a nonempty string")
        typed = make_typed_id(
            raw, node_type=node_type, corpus_or_account=corpus_or_account
        )
        if typed in seen:
            raise FilingPathError("parent menu repeats a folder_id")
        seen.add(typed)
        ids.append(typed)
    return tuple(ids)


def _verify_ranking_detail(
    path: str | os.PathLike[str],
    task_core: Mapping[str, Any],
    qid: int,
    menu_ids: Sequence[TypedId],
) -> float:
    """Bind the inherited ranking rows and read this row's exact margin."""

    rows = _read_jsonl(path)
    ranker = task_core.get("ranker")
    if not isinstance(ranker, Mapping):
        raise FilingPathError("parent task lacks a ranker receipt")
    if _hash_rows(rows) != ranker.get("ranking_sha256"):
        raise FilingPathError("ranking detail does not match the inherited ranking_sha256")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("qid") == qid]
    if len(matches) != 1:
        raise FilingPathError(f"ranking detail must contain exactly one row for qid {qid}")
    row = matches[0]
    margin = _parse_float64_hex(row.get("margin_float64_hex"), "ranking detail margin")
    if margin is None:
        raise FilingPathError("ranking detail row has no margin")
    ordered = row.get("top_k_folder_ids")
    if not isinstance(ordered, list):
        raise FilingPathError("ranking detail row has no top_k_folder_ids")
    expected = [typed.stable_value for typed in menu_ids]
    if ordered[: len(expected)] != expected:
        raise FilingPathError(
            "ranking detail order disagrees with the parent task menu; "
            "the frozen candidate order is inherited, not rebuilt"
        )
    return margin


def build_request(
    *,
    request_id: str,
    task_path: str | os.PathLike[str],
    qid: int,
    graph_path: str | os.PathLike[str],
    principal_paths_path: str | os.PathLike[str],
    policy_path: str | os.PathLike[str],
    allowed_roots: Sequence[Mapping[str, Any] | TypedId],
    node_type: str = "folder",
    ranking_detail_path: str | os.PathLike[str] | None = None,
    calibration_receipt: Mapping[str, Any] | None = None,
    external_naming_authorization_receipt: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> VerifiedRequest:
    """Re-derive the parent, bind the supplement, and freeze the request."""

    if not isinstance(request_id, str) or not request_id:
        raise FilingPathError("request_id is required")
    _require_int(qid, "qid")
    if extra:
        collisions = sorted(set(extra) & FORBIDDEN_REQUEST_KEYS)
        if collisions:
            raise FilingPathError(
                "the request must not restate inherited parent authority: "
                f"{collisions}"
            )

    # 1. Re-derivation through the landed v2 code path, then byte comparison.
    try:
        header, rows, task_record = read_task_file(task_path)
    except RoutedPolicyError as exc:
        raise FilingPathError(f"parent routed-task.v2 failed re-derivation: {exc}") from exc
    task_core = header["task_core"]

    # 2. Exact QID join — a copied task ID is not a join.
    matches = [row for row in rows if row.get("qid") == qid]
    if len(matches) != 1:
        raise FilingPathError(
            f"parent task must contain exactly one row for qid {qid}, found {len(matches)}"
        )
    task_row = matches[0]
    # 3. Derived from the canonical parent row.  Callers do not assert this.
    row_sha256 = sha256_bytes(canonical_json_bytes(dict(task_row)))

    catalog = task_core.get("catalog")
    if not isinstance(catalog, Mapping) or not isinstance(catalog.get("sha256"), str):
        raise FilingPathError("parent task lacks a catalog receipt")
    population = task_core.get("population")
    if not isinstance(population, Mapping) or not isinstance(population.get("sha256"), str):
        raise FilingPathError("parent task lacks a population receipt")
    population_id = population["sha256"]
    ranker = task_core.get("ranker")
    if not isinstance(ranker, Mapping) or not isinstance(ranker.get("ranking_sha256"), str):
        raise FilingPathError("parent task lacks a ranker receipt")

    graph = read_graph_snapshot(graph_path)
    paths = read_principal_path_records(principal_paths_path)
    if graph.corpus_or_account != paths.corpus_or_account:
        raise FilingPathError("graph and principal path records describe different corpora")
    corpus_or_account = graph.corpus_or_account

    roots = tuple(
        item if isinstance(item, TypedId) else parse_typed_id(item, "allowed_root")
        for item in allowed_roots
    )
    if not roots:
        raise FilingPathError("allowed_roots must be nonempty")
    for root in roots:
        if root.corpus_or_account != corpus_or_account:
            raise FilingPathError("allowed_root belongs to a different corpus")

    policy = read_decision_policy(policy_path)
    menu_ids = _menu_typed_ids(
        task_row, node_type=node_type, corpus_or_account=corpus_or_account
    )

    margin = None
    if ranking_detail_path is not None:
        margin = _verify_ranking_detail(ranking_detail_path, task_core, qid, menu_ids)
    elif policy.margin_threshold is not None:
        raise FilingPathError(
            "the policy sets a margin threshold, so the inherited ranking detail "
            "must be bound; the rule cannot be evaluated from the task file alone"
        )

    privacy_class = derive_privacy_class(task_core)
    inherited_privacy_receipt_sha256 = sha256_bytes(
        canonical_json_bytes(dict(task_core["privacy"]))
    )

    calibration_receipt_sha256 = (
        sha256_bytes(canonical_json_bytes(dict(calibration_receipt)))
        if calibration_receipt is not None
        else None
    )

    request = {
        "schema": REQUEST_SCHEMA,
        "record_type": "filing_path_request",
        "request_id": request_id,
        "parent_task": {
            "task_id": header["task_id"],
            "task_file_content_record": dict(task_record),
            "qid": qid,
            "row_sha256": row_sha256,
        },
        "path_supplement": {
            "graph_snapshot_sha256": graph.snapshot_sha256,
            "edge_table_sha256": graph.edge_table_sha256,
            "principal_path_records_sha256": paths.records_sha256,
            "principal_path_policy_id": paths.policy_id,
            "allowed_roots": [root.as_record() for root in roots],
        },
        "decision_policy_sha256": policy.policy_id,
        "calibration_receipt_sha256": calibration_receipt_sha256,
        "search_limits": {
            "maximum_search_nodes": policy.maximum_search_nodes,
            "maximum_search_steps": policy.maximum_search_steps,
        },
    }
    if extra:
        request["supplement_notes"] = dict(extra)

    # ``request_sha256`` is defined over the request *without* the naming
    # receipt digest.  An authorization receipt commits to this hash, so the
    # hash cannot in turn commit to the receipt without becoming circular.
    request_sha256 = sha256_bytes(canonical_json_bytes(request))
    naming_receipt_sha256 = check_external_naming(
        privacy_class=privacy_class,
        request_sha256=request_sha256,
        item_content_sha256=row_sha256,
        inherited_privacy_receipt_sha256=inherited_privacy_receipt_sha256,
        receipt=external_naming_authorization_receipt,
    )
    if naming_receipt_sha256 is not None and not policy.external_naming_enabled:
        raise FilingPathError("policy disables external naming for this request")
    request["external_naming_authorization_receipt_sha256"] = naming_receipt_sha256

    return VerifiedRequest(
        request=request,
        request_sha256=request_sha256,
        task_core=task_core,
        task_row=task_row,
        menu_ids=menu_ids,
        graph=graph,
        paths=paths,
        policy=policy,
        allowed_roots=roots,
        privacy_class=privacy_class,
        inherited_privacy_receipt_sha256=inherited_privacy_receipt_sha256,
        population_id=population_id,
        top_two_margin=margin,
        calibration_receipt_sha256=calibration_receipt_sha256,
        external_naming_authorization_receipt_sha256=naming_receipt_sha256,
    )


# --------------------------------------------------------------------------
# Stage A decision
# --------------------------------------------------------------------------


def _search_receipt(
    *,
    limits: Mapping[str, int],
    node_expansions: int,
    steps: int,
    best_so_far: Mapping[str, Any] | None,
    resource_censored: bool,
    skipped: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SEARCH_RECEIPT_SCHEMA,
        "stage": "A",
        "objective": STAGE_A_OBJECTIVE,
        "tie_rule": "parent_frozen_candidate_order_ascending",
        "limits": dict(limits),
        "node_expansions": node_expansions,
        "steps": steps,
        "resource_censored": resource_censored,
        "best_so_far": dict(best_so_far) if best_so_far else None,
        "ineligible_candidates": [dict(item) for item in skipped],
    }


def decide_stage_a(verified: VerifiedRequest) -> dict[str, Any]:
    """Frozen-e5 existing-folder baseline: ``SELECT_EXISTING`` or ``ABSTAIN``.

    The parent's candidate order is consumed as given.  Ineligible candidates —
    those without an authoritative recorded path, or rooted outside
    ``allowed_roots`` — are skipped and recorded, never re-scored.  The scan is
    bounded; on exhaustion it returns the deterministic ``best_so_far`` rather
    than the last iterate.
    """

    policy = verified.policy
    limits = {
        "maximum_search_nodes": policy.maximum_search_nodes,
        "maximum_search_steps": policy.maximum_search_steps,
    }

    # §2: ambiguity may abstain; it may never become a proposal.
    if policy.margin_threshold is not None:
        margin = verified.top_two_margin
        if margin is None:
            raise FilingPathError("margin rule requires a bound ranking detail")
        if margin < policy.margin_threshold:
            return _build_decision(
                verified,
                decision=ABSTAIN,
                payload={
                    "reason_code": "ambiguous_existing",
                    "candidate_ids_considered": [
                        typed.as_record() for typed in verified.menu_ids
                    ],
                },
                search_receipt=_search_receipt(
                    limits=limits,
                    node_expansions=0,
                    steps=0,
                    best_so_far=None,
                    resource_censored=False,
                    skipped=[],
                ),
                evidence_summary={
                    "rule": "top_two_margin_below_inner_selected_threshold",
                    "top_two_margin_float64_hex": float64_hex(margin),
                    "threshold_float64_hex": float64_hex(policy.margin_threshold),
                    "note": "a low margin is ambiguity, not evidence that a folder is missing",
                },
            )

    best: dict[str, Any] | None = None
    skipped: list[dict[str, Any]] = []
    node_expansions = 0
    steps = 0
    censored = False

    for rank, folder_id in enumerate(verified.menu_ids, start=1):
        if node_expansions >= policy.maximum_search_nodes or steps >= policy.maximum_search_steps:
            censored = True
            break
        node_expansions += 1
        steps += 1
        path_ids = verified.paths.path_for(folder_id)
        if path_ids is None:
            skipped.append(
                {"folder_id": folder_id.as_record(), "reason": "no_authoritative_path"}
            )
            continue
        if not path_is_in_scope(path_ids, verified.allowed_roots):
            # Request scope narrows eligibility; it is not a provenance fault.
            skipped.append(
                {"folder_id": folder_id.as_record(), "reason": "outside_allowed_roots"}
            )
            continue
        try:
            # An illegal transition in a *recorded* path is a provenance
            # failure, not an ordinary ineligible candidate.
            validate_path_transitions(path_ids, verified.graph)
        except FilingPathError as exc:
            raise FilingPathError(
                f"principal path for {folder_id.stable_value} is not usable: {exc}"
            ) from exc
        best = {
            "folder_id": folder_id.as_record(),
            "catalog_rank": rank,
            "authoritative_path_ids": [step.as_record() for step in path_ids],
            "path_titles": list(verified.paths.titles_by_folder[folder_id]),
        }
        break

    if best is None:
        reason = "resource_censored" if censored else "no_eligible_candidate"
        return _build_decision(
            verified,
            decision=ABSTAIN,
            payload={
                "reason_code": reason,
                "candidate_ids_considered": [
                    typed.as_record() for typed in verified.menu_ids
                ],
            },
            search_receipt=_search_receipt(
                limits=limits,
                node_expansions=node_expansions,
                steps=steps,
                best_so_far=None,
                resource_censored=censored,
                skipped=skipped,
            ),
            evidence_summary={
                "rule": STAGE_A_OBJECTIVE,
                "note": "no candidate carried an authoritative in-catalog path",
            },
        )

    payload = {
        "folder_id": best["folder_id"],
        "authoritative_path_ids": best["authoritative_path_ids"],
        "path_source": "frozen_principal_path_records",
        "path_display_titles": best["path_titles"],
        "catalog_rank": best["catalog_rank"],
        "e5_score": None,
        "top_two_margin": float64_hex(verified.top_two_margin),
    }
    return _build_decision(
        verified,
        decision=SELECT_EXISTING,
        payload=payload,
        search_receipt=_search_receipt(
            limits=limits,
            node_expansions=node_expansions,
            steps=steps,
            best_so_far=dict(best),
            resource_censored=censored,
            skipped=skipped,
        ),
        evidence_summary={
            "rule": STAGE_A_OBJECTIVE,
            "inherited_ranker": "frozen_e5_parent_candidate_order",
            "note": "breadcrumb copied verbatim from the frozen principal path records",
        },
    )


def _build_decision(
    verified: VerifiedRequest,
    *,
    decision: str,
    payload: Mapping[str, Any],
    search_receipt: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if decision not in (SELECT_EXISTING, ABSTAIN):
        # Stage A has no reachable proposal path; §10 keeps it gated.
        raise FilingPathError(f"Stage A cannot emit {decision}")
    if decision == ABSTAIN and payload.get("reason_code") not in ABSTAIN_REASONS:
        raise FilingPathError(f"unknown abstain reason: {payload.get('reason_code')!r}")
    record = {
        "schema": DECISION_SCHEMA,
        "record_type": "filing_path_decision",
        "request_id": verified.request["request_id"],
        "request_sha256": verified.request_sha256,
        "parent_task_id": verified.request["parent_task"]["task_id"],
        "parent_task_file_sha256": verified.request["parent_task"][
            "task_file_content_record"
        ]["sha256"],
        "parent_task_row_sha256": verified.request["parent_task"]["row_sha256"],
        "inherited_privacy_receipt_sha256": verified.inherited_privacy_receipt_sha256,
        "derived_privacy_class": verified.privacy_class,
        "decision": decision,
        "catalog_snapshot_sha256": verified.task_core["catalog"]["sha256"],
        "principal_path_records_sha256": verified.paths.records_sha256,
        "graph_snapshot_sha256": verified.graph.snapshot_sha256,
        "ranking_receipt_sha256": verified.task_core["ranker"]["ranking_sha256"],
        "calibration_receipt_sha256": verified.calibration_receipt_sha256,
        "search_receipt_sha256": sha256_bytes(canonical_json_bytes(dict(search_receipt))),
        "search_receipt": dict(search_receipt),
        "external_naming_authorization_receipt_sha256": (
            verified.external_naming_authorization_receipt_sha256
        ),
        "requires_user_confirmation": True,
        "evidence_summary": dict(evidence_summary),
        "payload": dict(payload),
    }
    record["decision_sha256"] = sha256_bytes(canonical_json_bytes(record))
    return record


def validate_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Schema-validate a decision and re-derive its self-binding digest."""

    if not isinstance(decision, Mapping) or decision.get("schema") != DECISION_SCHEMA:
        raise FilingPathError("unsupported filing path decision schema")
    action = decision.get("decision")
    if action not in DECISIONS:
        raise FilingPathError(f"unknown decision action: {action!r}")
    if decision.get("requires_user_confirmation") is not True:
        raise FilingPathError("every decision is advisory and requires confirmation")
    for field in (
        "request_sha256",
        "parent_task_id",
        "parent_task_file_sha256",
        "parent_task_row_sha256",
        "inherited_privacy_receipt_sha256",
        "catalog_snapshot_sha256",
        "principal_path_records_sha256",
        "graph_snapshot_sha256",
        "ranking_receipt_sha256",
        "search_receipt_sha256",
    ):
        if not isinstance(decision.get(field), str) or len(decision[field]) != 64:
            raise FilingPathError(f"decision {field} must be a 64-hex digest")
    receipt = decision.get("search_receipt")
    if not isinstance(receipt, Mapping):
        raise FilingPathError("decision search_receipt must be an object")
    if sha256_bytes(canonical_json_bytes(dict(receipt))) != decision["search_receipt_sha256"]:
        raise FilingPathError("search_receipt_sha256 does not bind the recorded search")
    payload = decision.get("payload")
    if not isinstance(payload, Mapping):
        raise FilingPathError("decision payload must be an object")

    # The three actions are mutually exclusive: each payload shape belongs to
    # exactly one action, and no payload may carry another action's fields.
    shapes = {
        SELECT_EXISTING: {"folder_id", "authoritative_path_ids", "catalog_rank"},
        PROPOSE_NEW: {"anchor_parent_id", "proposed_segments"},
        ABSTAIN: {"reason_code", "candidate_ids_considered"},
    }
    for other, keys in shapes.items():
        if other == action:
            continue
        overlap = sorted(keys & set(payload))
        if overlap:
            raise FilingPathError(
                f"{action} payload carries {other} fields: {overlap}"
            )
    missing = sorted(shapes[action] - set(payload))
    if missing:
        raise FilingPathError(f"{action} payload is missing {missing}")

    if action == SELECT_EXISTING:
        folder_id = parse_typed_id(payload["folder_id"], "payload.folder_id")
        path_ids = payload["authoritative_path_ids"]
        if not isinstance(path_ids, list) or not path_ids:
            raise FilingPathError("authoritative_path_ids must be a nonempty list")
        parsed = [
            parse_typed_id(step, f"authoritative_path_ids[{index}]")
            for index, step in enumerate(path_ids)
        ]
        if parsed[-1] != folder_id:
            raise FilingPathError("authoritative path must terminate at the selected folder")
        if payload.get("path_source") != "frozen_principal_path_records":
            raise FilingPathError("an existing folder's path must come from the catalog")
        _require_positive_int(payload.get("catalog_rank"), "catalog_rank")
    elif action == PROPOSE_NEW:
        raise FilingPathError(
            "PROPOSE_NEW is not implemented; it stays gated behind a prospective "
            "naturally absent calibration cohort"
        )
    else:
        if payload["reason_code"] not in ABSTAIN_REASONS:
            raise FilingPathError(f"unknown abstain reason: {payload['reason_code']!r}")
        for key in ("folder_id", "authoritative_path_ids", "anchor_parent_id"):
            if key in payload:
                raise FilingPathError(
                    f"an abstention must leave no stale target ({key} present)"
                )

    expected = dict(decision)
    declared = expected.pop("decision_sha256", None)
    if sha256_bytes(canonical_json_bytes(expected)) != declared:
        raise FilingPathError("decision_sha256 does not bind its own content")
    return dict(decision)


def write_decision(path: str | os.PathLike[str], decision: Mapping[str, Any]):
    """No-replace write.  A decision receipt is never silently overwritten."""

    validate_decision(decision)
    header = {
        "schema": DECISION_SCHEMA,
        "record_type": "filing_path_decision_envelope",
        "decision_sha256": decision["decision_sha256"],
    }
    try:
        return _atomic_write_jsonl_no_clobber(path, header, [decision])
    except RoutedPolicyError as exc:
        raise FilingPathError(str(exc)) from exc


def read_decision(path: str | os.PathLike[str]) -> dict[str, Any]:
    records = _read_jsonl(path)
    if len(records) != 2:
        raise FilingPathError("a decision artifact holds one envelope and one decision")
    header, decision = records
    if (
        not isinstance(header, Mapping)
        or header.get("schema") != DECISION_SCHEMA
        or header.get("record_type") != "filing_path_decision_envelope"
    ):
        raise FilingPathError("malformed decision envelope")
    validated = validate_decision(decision)
    if header.get("decision_sha256") != validated["decision_sha256"]:
        raise FilingPathError("decision envelope digest mismatch")
    return validated


# --------------------------------------------------------------------------
# confirmation
# --------------------------------------------------------------------------


def confirm_decision(
    decision: Mapping[str, Any],
    *,
    current_catalog_sha256: str,
    current_principal_path_records_sha256: str,
    user_confirmed: bool,
) -> dict[str, Any]:
    """Revalidate the catalog immediately before any separately authorized act.

    This function records confirmation.  It performs no action: applying a
    decision is outside this architecture and requires separate authorization.
    """

    validated = validate_decision(decision)
    if not user_confirmed:
        raise FilingPathError("a decision cannot be applied without user confirmation")
    if validated["catalog_snapshot_sha256"] != current_catalog_sha256:
        raise FilingPathError(
            "the catalog changed between recommendation and confirmation"
        )
    if validated["principal_path_records_sha256"] != current_principal_path_records_sha256:
        raise FilingPathError("principal path records changed before confirmation")
    record = {
        "schema": CONFIRMATION_SCHEMA,
        "record_type": "filing_path_confirmation",
        "decision_sha256": validated["decision_sha256"],
        "catalog_snapshot_sha256": current_catalog_sha256,
        "principal_path_records_sha256": current_principal_path_records_sha256,
        "user_confirmed": True,
        "authorizes_mutation": False,
    }
    record["confirmation_sha256"] = sha256_bytes(canonical_json_bytes(record))
    return record


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _typed_root(value: str) -> dict[str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "allowed root must be NODE_TYPE:CORPUS:STABLE_VALUE"
        )
    return {
        "node_type": parts[0],
        "corpus_or_account": parts[1],
        "stable_value": parts[2],
    }


def run_decide(a) -> int:
    verified = build_request(
        request_id=a.request_id,
        task_path=a.task,
        qid=a.qid,
        graph_path=a.graph,
        principal_paths_path=a.paths,
        policy_path=a.policy,
        allowed_roots=a.allowed_root,
        node_type=a.node_type,
        ranking_detail_path=a.ranking_detail,
    )
    decision = decide_stage_a(verified)
    if a.out:
        record = write_decision(a.out, decision)
        print(f"decision written: {a.out} ({record['sha256'][:16]})")
    action = decision["decision"]
    if action == SELECT_EXISTING:
        payload = decision["payload"]
        breadcrumb = " > ".join(payload["path_display_titles"])
        print(
            f"{action} {payload['folder_id']['stable_value']} "
            f"(rank {payload['catalog_rank']}): {breadcrumb}"
        )
    else:
        print(f"{action} ({decision['payload']['reason_code']})")
    print("advisory only; user confirmation required, no graph mutation performed")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    decide = sub.add_parser("decide", help="Stage A existing-folder decision")
    decide.add_argument("--task", required=True, help="routed-task.v2 JSONL parent")
    decide.add_argument("--qid", type=int, required=True)
    decide.add_argument("--graph", required=True, help="frozen graph snapshot JSONL")
    decide.add_argument("--paths", required=True, help="principal path records JSONL")
    decide.add_argument("--policy", required=True, help="filing-path-policy.v1 JSON")
    decide.add_argument("--ranking-detail", default=None)
    decide.add_argument("--allowed-root", action="append", type=_typed_root, required=True)
    decide.add_argument("--node-type", default="folder")
    decide.add_argument("--request-id", required=True)
    decide.add_argument("--out", default=None, help="no-replace decision receipt path")
    decide.set_defaults(func=run_decide)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FilingPathError, RoutedPolicyError) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
