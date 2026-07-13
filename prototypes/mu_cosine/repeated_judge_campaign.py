#!/usr/bin/env python3
"""Outcome-blind selection and scoring schedules for a repeated-judge campaign.

The selection/scheduling path deliberately stops at immutable score-input
files.  It never loads a model or calls a judge.  The response helper validates
only transport/schema integrity; it does not analyze scores.  A separate
candidate builder must freeze the graph/Nomic candidate pool before this
selector is run.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import unicodedata


SCHEMA_VERSION = 2
ALGORITHM = "repeated-judge-campaign-selector-v2"
ENDPOINT_ROLES = ("descendant", "anchor", "adjacent", "distant")
ROW_ROLES = ("anchor", "adjacent", "distant")
DEFAULT_CORPORA = ("exploratory", "fresh")
DEFAULT_HOP_TRANSITIONS = ("h1-h2", "h2-h3", "h3-h4", "h4-h5")
DEFAULT_DEGREE_QUARTILES = ("q1", "q2", "q3", "q4")
DEFAULT_AGREEMENT_CLASSES = ("agreement", "disagreement")
DEFAULT_JUDGES = ("gpt-5.5-low", "gpt-5.6-luna")
DEFAULT_COMPONENTS_PER_CORPUS = 320
FROZEN_COMPONENT_REPEAT_GRID = frozenset({
    (160, 3), (320, 3), (512, 3), (800, 3), (320, 4), (800, 4),
})
FROZEN_NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
FROZEN_NOMIC_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
# Exact bytes prepended by the revision-pinned Nomic cache builder.  The
# trailing space is semantic input, not presentation whitespace.
FROZEN_NOMIC_PREFIX = "clustering: "
FROZEN_WALK_WEIGHTS = (1.0, 0.5, 0.25, 0.125)

CANDIDATE_REQUIRED_COLUMNS = (
    "candidate_id",
    "corpus",
    "source_component",
    "hop_transition",
    "degree_quartile",
    "agreement_class",
    "anchor_hop",
    "adjacent_hop",
    "distant_hop",
    "anchor_campaign_tag",
    "adjacent_campaign_tag",
    "distant_campaign_tag",
    "anchor_degree_quartile",
    "adjacent_degree_quartile",
    "distant_degree_quartile",
    "anchor_adjacent_direct_edge",
    "anchor_distant_distance",
    "anchor_distant_disconnected",
    "cumulative_anchor_adjacent_similarity",
    "cumulative_anchor_distant_similarity",
    "nomic_anchor_adjacent_similarity",
    "nomic_anchor_distant_similarity",
    "descendant_id",
    "descendant_title",
    "anchor_id",
    "anchor_title",
    "adjacent_id",
    "adjacent_title",
    "distant_id",
    "distant_title",
)
HISTORICAL_REQUIRED_COLUMNS = ("corpus", "endpoint_id", "endpoint_title")
NESTED_FOLD_COLUMNS = (
    "component_id", "corpus", "outer_fold", "partition", "inner_fold",
)
RESPONSE_REQUIRED_COLUMNS = (
    "request_id", "row_id", "attempt", "provider_request_id", "provider_response_id",
    "started_at_utc", "completed_at_utc", "status", "judge", "model_id",
    "model_revision", "prompt_sha256", "settings_sha256", "raw_response_sha256",
    "raw_response", "score_d", "score_s", "error_type", "parse_status",
    "parse_error_type",
)


class CampaignInputError(ValueError):
    """Raised when an outcome-blind campaign input violates its contract."""


def normalize_title(value: str) -> str:
    """Canonical identity used to catch title aliases across graph snapshots."""
    value = unicodedata.normalize("NFKC", str(value)).replace("_", " ")
    return " ".join(value.split()).casefold()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_record(value: bytes) -> dict:
    return {"size_bytes": len(value), "sha256": sha256_bytes(value)}


def file_record(path) -> dict:
    return content_record(Path(path).read_bytes())


def _stable_digest(*values) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def _read_tsv(path, required_columns, *, label):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CampaignInputError(f"{label} must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames is None:
        raise CampaignInputError(f"{label} has no header")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise CampaignInputError(f"{label} has duplicate header columns")
    missing = [name for name in required_columns if name not in reader.fieldnames]
    if missing:
        raise CampaignInputError(f"{label} misses required columns: {', '.join(missing)}")
    rows = []
    for line_number, row in enumerate(reader, 2):
        if None in row:
            raise CampaignInputError(f"{label} row {line_number} has too many fields")
        if any(value is None for value in row.values()):
            raise CampaignInputError(f"{label} row {line_number} has too few fields")
        if not any(value != "" for value in row.values()):
            raise CampaignInputError(f"{label} row {line_number} is blank")
        rows.append({name: row[name] for name in reader.fieldnames})
    if not rows:
        raise CampaignInputError(f"{label} contains no records")
    return tuple(reader.fieldnames), rows, content_record(raw)


def _load_json_object(path, *, label):
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignInputError(f"{label} must be a UTF-8 JSON document") from exc
    if not isinstance(value, dict):
        raise CampaignInputError(f"{label} must contain a JSON object")
    return value, content_record(raw)


def _require_sha256(value, *, label):
    if not isinstance(value, str) or len(value) != 64:
        raise CampaignInputError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CampaignInputError(f"{label} must be a SHA-256 hex digest") from exc
    if value != value.lower():
        raise CampaignInputError(f"{label} must use canonical lowercase hex")
    return value


def _canonical_token(value, *, label):
    if not isinstance(value, str) or not value or value != value.strip():
        raise CampaignInputError(f"{label} must be a non-empty canonical token without edge whitespace")
    return value


def _canonical_int(value, *, label, minimum=None):
    _canonical_token(value, label=label)
    if not value.isascii() or not value.isdigit() or (len(value) > 1 and value.startswith("0")):
        raise CampaignInputError(f"{label} must be a canonical non-negative integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise CampaignInputError(f"{label} must be at least {minimum}")
    return result


def _canonical_float(value, *, label):
    _canonical_token(value, label=label)
    try:
        result = float(value)
    except ValueError as exc:
        raise CampaignInputError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise CampaignInputError(f"{label} must be finite")
    return result


def _canonical_bool(value, *, label):
    _canonical_token(value, label=label)
    if value not in {"true", "false"}:
        raise CampaignInputError(f"{label} must be canonical JSON-style true or false")
    return value == "true"


def load_candidate_builder_manifest(path, candidate_artifact):
    """Validate a caller-supplied, content-addressed provenance declaration.

    This checks internal consistency and frozen geometry identifiers.  It does
    not attest that the declared builder, graph, or embedding artifacts were
    produced by repository code; that builder remains a future prerequisite.
    """
    value, artifact = _load_json_object(path, label="candidate-builder manifest")
    try:
        pool = value["candidate_pool"]
        graph = value["graph"]
        nomic = value["nomic"]
        thresholds = value["agreement_thresholds"]
    except KeyError as exc:
        raise CampaignInputError(
            f"candidate-builder manifest misses required field {exc.args[0]!r}"
        ) from exc
    if value.get("schema_version") != 1:
        raise CampaignInputError("candidate-builder manifest schema_version must equal 1")
    _canonical_token(value.get("algorithm"), label="candidate-builder algorithm")
    _require_sha256(value.get("implementation_sha256"), label="candidate-builder implementation_sha256")
    if pool != candidate_artifact:
        raise CampaignInputError(
            "candidate-builder manifest candidate_pool record does not match the supplied TSV bytes"
        )
    _require_sha256(graph.get("artifact_sha256"), label="graph artifact_sha256")
    weights = graph.get("cumulative_walk_weights")
    if not isinstance(weights, list) or tuple(float(item) for item in weights) != FROZEN_WALK_WEIGHTS:
        raise CampaignInputError(
            f"cumulative_walk_weights must equal {list(FROZEN_WALK_WEIGHTS)!r}"
        )
    if nomic.get("model_id") != FROZEN_NOMIC_MODEL:
        raise CampaignInputError(f"Nomic model_id must equal {FROZEN_NOMIC_MODEL!r}")
    if nomic.get("revision") != FROZEN_NOMIC_REVISION:
        raise CampaignInputError("Nomic revision does not match the frozen commit")
    if nomic.get("task_prefix") != FROZEN_NOMIC_PREFIX:
        raise CampaignInputError(f"Nomic task_prefix must equal {FROZEN_NOMIC_PREFIX!r}")
    _require_sha256(
        nomic.get("embedding_manifest_sha256"), label="Nomic embedding_manifest_sha256"
    )
    parsed_thresholds = {}
    for name in ("graph_delta_min", "nomic_delta_min"):
        raw_threshold = thresholds.get(name)
        if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)):
            raise CampaignInputError(f"agreement_thresholds.{name} must be numeric")
        threshold = float(raw_threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise CampaignInputError(f"agreement_thresholds.{name} must be finite and non-negative")
        parsed_thresholds[name] = threshold
    return value, artifact, parsed_thresholds


def load_request_contract(path, judges):
    """Load exact immutable model, prompt, and request settings for each judge alias."""
    value, artifact = _load_json_object(path, label="request contract")
    if value.get("schema_version") != 1:
        raise CampaignInputError("request contract schema_version must equal 1")
    specs = value.get("judges")
    if not isinstance(specs, dict) or set(specs) != set(judges):
        raise CampaignInputError("request contract judge aliases must exactly match --judges")
    output = {}
    for alias in judges:
        spec = specs[alias]
        if not isinstance(spec, dict):
            raise CampaignInputError(f"request contract judge {alias!r} must be an object")
        for field in ("model_id", "model_revision", "prompt_id", "reasoning_effort"):
            _canonical_token(spec.get(field), label=f"request contract {alias}.{field}")
        prompt_sha256 = _require_sha256(
            spec.get("prompt_sha256"), label=f"request contract {alias}.prompt_sha256"
        )
        settings = spec.get("settings")
        if not isinstance(settings, dict):
            raise CampaignInputError(f"request contract {alias}.settings must be an object")
        settings_json = json.dumps(
            settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if spec.get("stateless") is not True:
            raise CampaignInputError(f"request contract {alias}.stateless must be true")
        call_seed = spec.get("call_seed")
        if call_seed is not None and (isinstance(call_seed, bool) or not isinstance(call_seed, int)):
            raise CampaignInputError(f"request contract {alias}.call_seed must be null or an integer")
        output[alias] = {
            "request_schema_version": 1,
            "model_id": spec["model_id"],
            "model_revision": spec["model_revision"],
            "prompt_id": spec["prompt_id"],
            "prompt_sha256": prompt_sha256,
            "reasoning_effort": spec["reasoning_effort"],
            "settings_json": settings_json,
            "settings_sha256": sha256_bytes(settings_json.encode("utf-8")),
            "call_seed_base": call_seed,
            "shared_session_id": "",
            "stateless": True,
        }
    return output, value, artifact


@dataclass(frozen=True)
class Endpoint:
    graph_id: str
    title: str
    normalized_title: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    corpus: str
    source_component: str
    hop_transition: str
    degree_quartile: str
    agreement_class: str
    endpoints: tuple[Endpoint, Endpoint, Endpoint, Endpoint]
    raw: dict
    component_id: str | None = None
    fold: int | None = None
    inner_fold: int | None = None

    @property
    def cell(self):
        return (
            self.corpus,
            self.hop_transition,
            self.degree_quartile,
            self.agreement_class,
        )

    @property
    def scoped_ids(self):
        return frozenset((self.corpus, endpoint.graph_id) for endpoint in self.endpoints)

    @property
    def normalized_titles(self):
        return frozenset(endpoint.normalized_title for endpoint in self.endpoints)


@dataclass(frozen=True)
class HistoricalEndpoints:
    scoped_ids: frozenset
    normalized_titles: frozenset
    records: tuple[dict, ...]


def load_candidates(path) -> tuple[tuple[str, ...], tuple[Candidate, ...], dict]:
    """Load and strictly validate the frozen rich candidate-pool TSV."""
    columns, records, artifact = _read_tsv(
        path, CANDIDATE_REQUIRED_COLUMNS, label="candidate pool"
    )
    candidates = []
    candidate_ids = set()
    id_to_title = {}
    for row_number, record in enumerate(records, 2):
        for name in CANDIDATE_REQUIRED_COLUMNS:
            if not record[name].strip():
                raise CampaignInputError(
                    f"candidate pool row {row_number} has empty required value {name!r}"
                )
        for name in (
            "candidate_id", "corpus", "source_component", "hop_transition",
            "degree_quartile", "agreement_class", "anchor_campaign_tag",
            "adjacent_campaign_tag", "distant_campaign_tag", "anchor_degree_quartile",
            "adjacent_degree_quartile", "distant_degree_quartile",
        ):
            _canonical_token(
                record[name], label=f"candidate pool row {row_number} field {name}"
            )
        candidate_id = record["candidate_id"]
        if candidate_id in candidate_ids:
            raise CampaignInputError(f"duplicate candidate_id {candidate_id!r}")
        candidate_ids.add(candidate_id)

        anchor_hop = _canonical_int(
            record["anchor_hop"], label=f"candidate {candidate_id} anchor_hop", minimum=1
        )
        adjacent_hop = _canonical_int(
            record["adjacent_hop"], label=f"candidate {candidate_id} adjacent_hop", minimum=1
        )
        distant_hop = _canonical_int(
            record["distant_hop"], label=f"candidate {candidate_id} distant_hop", minimum=1
        )
        if adjacent_hop != distant_hop:
            raise CampaignInputError(
                f"candidate {candidate_id!r} does not match adjacent and distant hop strata"
            )
        expected_transition = f"h{min(anchor_hop, adjacent_hop)}-h{max(anchor_hop, adjacent_hop)}"
        if record["hop_transition"] != expected_transition:
            raise CampaignInputError(
                f"candidate {candidate_id!r} hop_transition does not match its structural hops"
            )
        if record["adjacent_campaign_tag"] != record["distant_campaign_tag"]:
            raise CampaignInputError(
                f"candidate {candidate_id!r} does not match comparator campaign tags"
            )
        if record["adjacent_degree_quartile"] != record["distant_degree_quartile"]:
            raise CampaignInputError(
                f"candidate {candidate_id!r} does not match comparator degree quartiles"
            )
        if record["anchor_degree_quartile"] != record["degree_quartile"]:
            raise CampaignInputError(
                f"candidate {candidate_id!r} anchor degree quartile disagrees with its balance cell"
            )
        if not _canonical_bool(
            record["anchor_adjacent_direct_edge"],
            label=f"candidate {candidate_id} anchor_adjacent_direct_edge",
        ):
            raise CampaignInputError(
                f"candidate {candidate_id!r} adjacent comparator is not a direct graph neighbor"
            )
        disconnected = _canonical_bool(
            record["anchor_distant_disconnected"],
            label=f"candidate {candidate_id} anchor_distant_disconnected",
        )
        distance = record["anchor_distant_distance"]
        if disconnected:
            if distance != "inf":
                raise CampaignInputError(
                    f"candidate {candidate_id!r} disconnected distant comparator must use distance 'inf'"
                )
        else:
            _canonical_int(
                distance, label=f"candidate {candidate_id} anchor_distant_distance", minimum=3
            )
        for name in (
            "cumulative_anchor_adjacent_similarity", "cumulative_anchor_distant_similarity",
            "nomic_anchor_adjacent_similarity", "nomic_anchor_distant_similarity",
        ):
            similarity = _canonical_float(
                record[name], label=f"candidate {candidate_id} {name}"
            )
            if not 0.0 <= similarity <= 1.0:
                raise CampaignInputError(
                    f"candidate {candidate_id!r} {name} must lie in [0,1]"
                )
        endpoints = []
        for role in ENDPOINT_ROLES:
            graph_id = record[f"{role}_id"]
            title = record[f"{role}_title"]
            _canonical_token(graph_id, label=f"candidate {candidate_id} {role}_id")
            normalized = normalize_title(title)
            if not normalized:
                raise CampaignInputError(
                    f"candidate {candidate_id!r} has empty normalized {role} title"
                )
            scoped = (record["corpus"], graph_id)
            previous = id_to_title.setdefault(scoped, normalized)
            if previous != normalized:
                raise CampaignInputError(
                    f"endpoint id {scoped!r} maps to inconsistent normalized titles"
                )
            endpoints.append(Endpoint(graph_id, title, normalized))
        if len({endpoint.graph_id for endpoint in endpoints}) != len(endpoints):
            raise CampaignInputError(
                f"candidate {candidate_id!r} repeats an endpoint id within the triple"
            )
        if len({endpoint.normalized_title for endpoint in endpoints}) != len(endpoints):
            raise CampaignInputError(
                f"candidate {candidate_id!r} repeats a normalized endpoint title within the triple"
            )
        candidates.append(Candidate(
            candidate_id=candidate_id,
            corpus=record["corpus"],
            source_component=record["source_component"],
            hop_transition=record["hop_transition"],
            degree_quartile=record["degree_quartile"],
            agreement_class=record["agreement_class"],
            endpoints=tuple(endpoints),
            raw=dict(record),
        ))
    return tuple(columns), tuple(candidates), artifact


def validate_candidate_geometry(candidates, thresholds):
    """Verify frozen agreement labels from the recorded continuous kernels."""
    for candidate in candidates:
        record = candidate.raw
        graph_delta = (
            float(record["cumulative_anchor_adjacent_similarity"])
            - float(record["cumulative_anchor_distant_similarity"])
        )
        nomic_delta = (
            float(record["nomic_anchor_adjacent_similarity"])
            - float(record["nomic_anchor_distant_similarity"])
        )
        graph_prefers_adjacent = graph_delta >= thresholds["graph_delta_min"]
        nomic_prefers_adjacent = nomic_delta >= thresholds["nomic_delta_min"]
        expected = "agreement" if graph_prefers_adjacent == nomic_prefers_adjacent else "disagreement"
        if candidate.agreement_class != expected:
            raise CampaignInputError(
                f"candidate {candidate.candidate_id!r} agreement_class={candidate.agreement_class!r} "
                f"but frozen deltas imply {expected!r}"
            )


def load_historical_endpoints(paths) -> tuple[HistoricalEndpoints, tuple[dict, ...]]:
    """Load one or more explicit historical endpoint inventories."""
    scoped_ids = set()
    normalized_titles = set()
    records = []
    artifacts = []
    for path in paths:
        _columns, rows, artifact = _read_tsv(
            path, HISTORICAL_REQUIRED_COLUMNS, label="historical endpoint inventory"
        )
        artifacts.append(artifact)
        for row_number, record in enumerate(rows, 2):
            corpus = _canonical_token(
                record["corpus"], label=f"historical endpoint row {row_number} corpus"
            )
            endpoint_id = _canonical_token(
                record["endpoint_id"], label=f"historical endpoint row {row_number} endpoint_id"
            )
            title = record["endpoint_title"]
            normalized = normalize_title(title)
            if not corpus or not endpoint_id or not normalized:
                raise CampaignInputError(
                    f"historical endpoint row {row_number} has an empty required identity"
                )
            scoped_ids.add((corpus, endpoint_id))
            normalized_titles.add(normalized)
            records.append({
                "corpus": corpus,
                "endpoint_id": endpoint_id,
                "normalized_title": normalized,
            })
    if not paths:
        raise CampaignInputError("at least one historical endpoint inventory is required")
    artifacts.sort(key=lambda value: (value["sha256"], value["size_bytes"]))
    records.sort(key=lambda value: (
        value["corpus"], value["endpoint_id"], value["normalized_title"]
    ))
    return HistoricalEndpoints(
        frozenset(scoped_ids), frozenset(normalized_titles), tuple(records)
    ), tuple(artifacts)


def expected_quotas(
    *,
    per_cell=10,
    corpora=DEFAULT_CORPORA,
    hop_transitions=DEFAULT_HOP_TRANSITIONS,
    degree_quartiles=DEFAULT_DEGREE_QUARTILES,
    agreement_classes=DEFAULT_AGREEMENT_CLASSES,
):
    if int(per_cell) != per_cell or per_cell < 1:
        raise CampaignInputError("per_cell must be a positive integer")
    return {
        (corpus, hop, degree, agreement): int(per_cell)
        for corpus in corpora
        for hop in hop_transitions
        for degree in degree_quartiles
        for agreement in agreement_classes
    }


def component_quotas(
    components_per_corpus,
    *,
    corpora=DEFAULT_CORPORA,
    hop_transitions=DEFAULT_HOP_TRANSITIONS,
    degree_quartiles=DEFAULT_DEGREE_QUARTILES,
    agreement_classes=DEFAULT_AGREEMENT_CLASSES,
):
    """Derive exact equal cell quotas from a frozen total G for each corpus."""
    if isinstance(components_per_corpus, bool) or not isinstance(components_per_corpus, int):
        raise CampaignInputError("components_per_corpus must be an integer")
    cell_count = len(hop_transitions) * len(degree_quartiles) * len(agreement_classes)
    if components_per_corpus < cell_count or components_per_corpus % cell_count:
        raise CampaignInputError(
            f"components_per_corpus must be a positive multiple of {cell_count} "
            "so every required cell has an exact equal quota"
        )
    return expected_quotas(
        per_cell=components_per_corpus // cell_count,
        corpora=corpora,
        hop_transitions=hop_transitions,
        degree_quartiles=degree_quartiles,
        agreement_classes=agreement_classes,
    )


def filter_candidate_pool(candidates, historical, quotas):
    """Reject historical endpoints and candidates outside the frozen cells."""
    kept = []
    reasons = Counter()
    for candidate in candidates:
        if candidate.cell not in quotas:
            reasons["unknown_cell"] += 1
            continue
        if candidate.scoped_ids & historical.scoped_ids:
            reasons["historical_endpoint_id"] += 1
            continue
        if candidate.normalized_titles & historical.normalized_titles:
            reasons["historical_normalized_title"] += 1
            continue
        kept.append(candidate)
    unknown = sorted({candidate.cell for candidate in candidates if candidate.cell not in quotas})
    if unknown:
        raise CampaignInputError(f"candidate pool contains unknown frozen cells; first={unknown[0]!r}")
    by_cell = Counter(candidate.cell for candidate in kept)
    shortages = {
        cell: {"available": by_cell[cell], "required": required}
        for cell, required in quotas.items()
        if by_cell[cell] < required
    }
    if shortages:
        cell = sorted(shortages)[0]
        value = shortages[cell]
        raise CampaignInputError(
            f"cell {cell!r} has {value['available']} eligible candidates; "
            f"requires {value['required']}"
        )
    return tuple(kept), dict(sorted(reasons.items()))


def _eligible(candidate, used_ids, used_titles, source_counts, source_cap):
    return (
        not (candidate.scoped_ids & used_ids)
        and not (candidate.normalized_titles & used_titles)
        and source_counts[(candidate.corpus, candidate.source_component)] < source_cap[candidate.corpus]
    )


def select_candidates(
    candidates,
    quotas,
    *,
    source_cap_fraction=0.10,
    seed=0,
    max_attempts=512,
):
    """Deterministic scarcity-first packing under endpoint and source caps."""
    candidates = tuple(candidates)
    if not 0.0 < source_cap_fraction <= 1.0:
        raise CampaignInputError("source_cap_fraction must lie in (0,1]")
    if max_attempts < 1:
        raise CampaignInputError("max_attempts must be positive")
    totals = Counter()
    for cell, count in quotas.items():
        totals[cell[0]] += count
    source_cap = {
        corpus: max(1, int(math.floor(total * source_cap_fraction + 1e-12)))
        for corpus, total in totals.items()
    }
    by_cell = defaultdict(list)
    for candidate in candidates:
        if candidate.cell in quotas:
            by_cell[candidate.cell].append(candidate)
    last_failure = None
    for attempt in range(max_attempts):
        selected = []
        selected_ids = set()
        used_ids = set()
        used_titles = set()
        source_counts = Counter()
        deficits = dict(quotas)
        while any(deficits.values()):
            choices = []
            eligible_by_cell = {}
            for cell in sorted(deficits):
                deficit = deficits[cell]
                if not deficit:
                    continue
                eligible = [
                    candidate for candidate in by_cell[cell]
                    if candidate.candidate_id not in selected_ids
                    and _eligible(candidate, used_ids, used_titles, source_counts, source_cap)
                ]
                eligible.sort(key=lambda candidate: (
                    _stable_digest(seed, attempt, candidate.candidate_id),
                    candidate.candidate_id,
                ))
                eligible_by_cell[cell] = eligible
                slack = len(eligible) - deficit
                choices.append((slack, len(eligible), cell))
            slack, available, cell = min(choices)
            if slack < 0:
                last_failure = (attempt, cell, available, deficits[cell])
                break
            candidate = eligible_by_cell[cell][0]
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
            used_ids.update(candidate.scoped_ids)
            used_titles.update(candidate.normalized_titles)
            source_counts[(candidate.corpus, candidate.source_component)] += 1
            deficits[cell] -= 1
        else:
            by_id = Counter(endpoint for candidate in selected for endpoint in candidate.scoped_ids)
            by_title = Counter(title for candidate in selected for title in candidate.normalized_titles)
            if max(by_id.values(), default=0) != 1 or max(by_title.values(), default=0) != 1:
                raise AssertionError("selected campaign is not endpoint-disjoint")
            if Counter(candidate.cell for candidate in selected) != Counter(quotas):
                raise AssertionError("selected campaign does not match frozen quotas")
            return tuple(selected), {
                "attempt": attempt,
                "source_cap_fraction": float(source_cap_fraction),
                "source_component_caps": dict(sorted(source_cap.items())),
                "source_component_counts": {
                    f"{corpus}|{component}": count
                    for (corpus, component), count in sorted(source_counts.items())
                },
            }
    raise CampaignInputError(
        "could not pack endpoint-disjoint quotas under the source-component cap "
        f"after {max_attempts} deterministic attempts; last_failure={last_failure!r}"
    )


def component_identity(candidate):
    fields = {
        "candidate_id": candidate.candidate_id,
        "cell": candidate.cell,
        "source_component": candidate.source_component,
        "endpoints": [
            [role, endpoint.graph_id, endpoint.normalized_title]
            for role, endpoint in zip(ENDPOINT_ROLES, candidate.endpoints)
        ],
    }
    return "component-" + _stable_digest(fields)[:24]


def _atomic_balanced_assignment(
    items,
    *,
    group_key,
    dimensions,
    bins,
    seed,
    label,
    max_attempts=512,
):
    """Assign dependency groups atomically under floor/ceiling balance bounds."""
    if bins < 2:
        raise CampaignInputError(f"{label} bins must be at least two")
    grouped = defaultdict(list)
    dimension_totals = Counter()
    item_dimensions = {}
    for item in items:
        key = group_key(item)
        values = tuple(dimensions(item))
        if not values or len(values) != len(set(values)):
            raise AssertionError(f"{label} dimensions must be non-empty and unique per item")
        grouped[key].append(item)
        item_dimensions[item.component_id] = values
        dimension_totals.update(values)
    lower = {dimension: total // bins for dimension, total in dimension_totals.items()}
    upper = {
        dimension: (total + bins - 1) // bins
        for dimension, total in dimension_totals.items()
    }
    vectors = {}
    for key, values in grouped.items():
        vector = Counter(
            dimension
            for item in values
            for dimension in item_dimensions[item.component_id]
        )
        impossible = [
            dimension for dimension, count in vector.items()
            if count > upper[dimension]
        ]
        if impossible:
            dimension = sorted(impossible, key=repr)[0]
            raise CampaignInputError(
                f"atomic source group {key!r} contributes {vector[dimension]} rows to "
                f"{label} stratum {dimension!r}, exceeding the per-bin ceiling "
                f"{upper[dimension]}"
            )
        vectors[key] = vector

    last_failure = None
    for attempt in range(max_attempts):
        counts = Counter()
        assignment = {}
        remaining = Counter(dimension_totals)
        keys = sorted(grouped, key=lambda key: (
            -sum(vectors[key].values()),
            -len(vectors[key]),
            _stable_digest(seed, label, "group-order", attempt, key),
            repr(key),
        ))
        failed = False
        for key in keys:
            vector = vectors[key]
            remaining.subtract(vector)
            choices = []
            for bin_index in range(bins):
                if any(
                    counts[(bin_index, dimension)] + value > upper[dimension]
                    for dimension, value in vector.items()
                ):
                    continue
                feasible = True
                for dimension in dimension_totals:
                    for other_bin in range(bins):
                        proposed = counts[(other_bin, dimension)]
                        if other_bin == bin_index:
                            proposed += vector[dimension]
                        if proposed + remaining[dimension] < lower[dimension]:
                            feasible = False
                            break
                    if not feasible:
                        break
                if not feasible:
                    continue
                deficit_filled = sum(
                    min(value, max(0, lower[dimension] - counts[(bin_index, dimension)]))
                    for dimension, value in vector.items()
                )
                peak_fraction = max(
                    (
                        (counts[(bin_index, dimension)] + value) / upper[dimension]
                        if upper[dimension] else 0.0
                    )
                    for dimension, value in vector.items()
                )
                choices.append((
                    -deficit_filled,
                    peak_fraction,
                    sum(counts[(bin_index, dimension)] for dimension in vector),
                    _stable_digest(seed, label, "bin-tie", attempt, key, bin_index),
                    bin_index,
                ))
            if not choices:
                last_failure = {"attempt": attempt, "source_group": repr(key)}
                failed = True
                break
            bin_index = min(choices)[-1]
            assignment[key] = bin_index
            for dimension, value in vector.items():
                counts[(bin_index, dimension)] += value
        if failed:
            continue
        violations = []
        for dimension in dimension_totals:
            values = [counts[(bin_index, dimension)] for bin_index in range(bins)]
            if min(values) < lower[dimension] or max(values) > upper[dimension]:
                violations.append((dimension, values))
        if violations:
            last_failure = {"attempt": attempt, "violation": repr(violations[0])}
            continue
        return assignment, counts, {
            "attempt": attempt,
            "source_group_count": len(grouped),
            "dimension_bounds": {
                repr(dimension): {"floor": lower[dimension], "ceiling": upper[dimension]}
                for dimension in sorted(dimension_totals, key=repr)
            },
        }
    raise CampaignInputError(
        f"could not assign atomic source groups to near-balanced {label} bins after "
        f"{max_attempts} deterministic attempts; last_failure={last_failure!r}"
    )


def assign_component_folds(selected, quotas, *, folds=5, seed=0):
    """Assign source-component dependency groups atomically to outer folds."""
    if folds < 2:
        raise CampaignInputError("folds must be at least two")
    enriched = []
    component_ids = set()
    for candidate in selected:
        component_id = component_identity(candidate)
        if component_id in component_ids:
            raise CampaignInputError(f"component hash collision: {component_id}")
        component_ids.add(component_id)
        enriched.append(replace(candidate, component_id=component_id))

    assignment, _counts, atomic_diagnostics = _atomic_balanced_assignment(
        enriched,
        group_key=lambda candidate: (candidate.corpus, candidate.source_component),
        dimensions=lambda candidate: (
            ("cell", *candidate.cell),
            ("corpus-total", candidate.corpus),
            ("campaign-total",),
        ),
        bins=folds,
        seed=seed,
        label="outer-fold",
    )
    output = [
        replace(
            candidate,
            fold=assignment[(candidate.corpus, candidate.source_component)],
        )
        for candidate in enriched
    ]
    output.sort(key=lambda candidate: candidate.component_id)
    fold_totals = Counter(candidate.fold for candidate in output)
    fold_source_counts = Counter(
        (candidate.fold, candidate.corpus, candidate.source_component)
        for candidate in output
    )
    diagnostics = {
        "component_counts": {str(fold): fold_totals[fold] for fold in range(folds)},
        "cell_counts": {
            str(fold): {
                "|".join(cell): sum(
                    candidate.fold == fold and candidate.cell == cell for candidate in output
                )
                for cell in sorted(quotas)
            }
            for fold in range(folds)
        },
        "source_component_counts": {
            str(fold): {
                f"{corpus}|{source}": count
                for (at, corpus, source), count in sorted(fold_source_counts.items())
                if at == fold
            }
            for fold in range(folds)
        },
        "atomic_assignment": atomic_diagnostics,
    }
    if len({candidate.component_id for candidate in output}) != len(output):
        raise AssertionError("component ids must remain unique after fold assignment")
    for cell in quotas:
        counts = [
            sum(candidate.fold == fold and candidate.cell == cell for candidate in output)
            for fold in range(folds)
        ]
        if max(counts) - min(counts) > 1:
            raise AssertionError("outer per-cell fold counts must differ by at most one")
    totals = [fold_totals[fold] for fold in range(folds)]
    if max(totals) - min(totals) > 1:
        raise AssertionError("outer fold totals must differ by at most one")
    source_folds = defaultdict(set)
    for candidate in output:
        source_folds[(candidate.corpus, candidate.source_component)].add(candidate.fold)
    if any(len(values) != 1 for values in source_folds.values()):
        raise AssertionError("a source-component dependency group crossed outer folds")
    return tuple(output), diagnostics


def assign_nested_inner_folds(components, *, outer_folds=5, inner_folds=3, seed=0):
    """Freeze one global inner label, then materialize every nested outer split."""
    if outer_folds < 2 or inner_folds < 2:
        raise CampaignInputError("outer_folds and inner_folds must both be at least two")
    components = tuple(components)
    if any(component.fold not in range(outer_folds) for component in components):
        raise CampaignInputError("every component must have a valid outer fold")
    source_outer_folds = defaultdict(set)
    for component in components:
        source_outer_folds[(component.corpus, component.source_component)].add(component.fold)
    if any(len(values) != 1 for values in source_outer_folds.values()):
        raise CampaignInputError(
            "source-component dependency groups must be atomic before inner-fold assignment"
        )
    by_cell = defaultdict(list)
    for component in components:
        by_cell[component.cell].append(component)
    assignment, _counts, atomic_diagnostics = _atomic_balanced_assignment(
        components,
        group_key=lambda component: (component.corpus, component.source_component),
        dimensions=lambda component: (
            ("cell", *component.cell),
            ("corpus-total", component.corpus),
            ("campaign-total",),
        ),
        bins=inner_folds,
        seed=seed,
        label="global-inner-fold",
    )

    enriched = tuple(sorted(
        (
            replace(
                component,
                inner_fold=assignment[(component.corpus, component.source_component)],
            )
            for component in components
        ),
        key=lambda component: component.component_id,
    ))
    global_inner_totals = Counter(component.inner_fold for component in enriched)
    global_cell_counts = Counter(
        (component.cell, component.inner_fold) for component in enriched
    )
    records = []
    diagnostics = {
        "global_inner_component_counts": {
            str(fold): global_inner_totals[fold] for fold in range(inner_folds)
        },
        "global_cell_inner_counts": {
            "|".join((*cell, str(fold))): global_cell_counts[(cell, fold)]
            for cell in sorted(by_cell)
            for fold in range(inner_folds)
        },
        "atomic_assignment": atomic_diagnostics,
        "outer_splits": {},
    }
    for outer_fold in range(outer_folds):
        held = [component for component in enriched if component.fold == outer_fold]
        training = [component for component in enriched if component.fold != outer_fold]
        for component in enriched:
            is_held = component.fold == outer_fold
            records.append({
                "component_id": component.component_id,
                "corpus": component.corpus,
                "outer_fold": outer_fold,
                "partition": "held" if is_held else "train",
                "inner_fold": "" if is_held else component.inner_fold,
            })
        diagnostics["outer_splits"][str(outer_fold)] = {
            "held_components": len(held),
            "training_components": len(training),
            "training_inner_component_counts": {
                str(fold): sum(component.inner_fold == fold for component in training)
                for fold in range(inner_folds)
            },
        }
    records.sort(key=lambda row: (
        row["outer_fold"], row["partition"] != "held", row["component_id"]
    ))
    expected = len(components) * outer_folds
    if len(records) != expected:
        raise AssertionError("nested fold table must contain every component in every outer split")
    source_inner_folds = defaultdict(set)
    for component in enriched:
        source_inner_folds[(component.corpus, component.source_component)].add(
            component.inner_fold
        )
    if any(len(values) != 1 for values in source_inner_folds.values()):
        raise AssertionError("a source-component dependency group crossed global inner folds")
    return enriched, tuple(records), diagnostics


def campaign_rows(components):
    """Expand each selected component into anchor/adjacent/distant score rows."""
    rows = []
    role_endpoint = {"anchor": 1, "adjacent": 2, "distant": 3}
    for component in components:
        descendant = component.endpoints[0]
        for role in ROW_ROLES:
            root = component.endpoints[role_endpoint[role]]
            rows.append({
                "row_id": f"{component.component_id}:{role}",
                "component_id": component.component_id,
                "fold": component.fold,
                "inner_fold": component.inner_fold,
                "candidate_id": component.candidate_id,
                "corpus": component.corpus,
                "source_component": component.source_component,
                "hop_transition": component.hop_transition,
                "degree_quartile": component.degree_quartile,
                "agreement_class": component.agreement_class,
                "role": role,
                "node_id": descendant.graph_id,
                "node_title": descendant.title,
                "node_normalized_title": descendant.normalized_title,
                "root_id": root.graph_id,
                "root_title": root.title,
                "root_normalized_title": root.normalized_title,
                "cur_relation": "subcategory",
                "conf": "1.0",
                "neighborhood": (
                    f"repeatcov_{component.corpus}_{component.hop_transition}_"
                    f"{component.degree_quartile}_{component.agreement_class}_{role}"
                ),
                "node_type": "category",
                "root_type": "category",
                "raw": "",
            })
    rows.sort(key=lambda row: (row["component_id"], ROW_ROLES.index(row["role"])))
    return tuple(rows)


def build_scoring_schedule(
    rows,
    *,
    judges=DEFAULT_JUDGES,
    repeats=3,
    batch_size=10,
    seed=0,
    judge_specs=None,
):
    """Build stable prompt blocks contained within one nested split signature."""
    rows = tuple(rows)
    if repeats < 3:
        raise CampaignInputError("repeats must be at least three for repeated-judge covariance")
    if not 1 <= batch_size <= 10:
        raise CampaignInputError("batch_size must be between one and ten rows")
    if len(set(judges)) != len(tuple(judges)) or not judges:
        raise CampaignInputError("judges must be non-empty and unique")
    if not isinstance(judge_specs, dict) or set(judge_specs) != set(judges):
        raise CampaignInputError("judge_specs must exactly cover the scheduled judges")
    by_component = defaultdict(dict)
    for row in rows:
        role = row["role"]
        component = row["component_id"]
        if role in by_component[component]:
            raise CampaignInputError(f"component {component!r} repeats role {role!r}")
        by_component[component][role] = row
    if any(set(values) != set(ROW_ROLES) for values in by_component.values()):
        raise CampaignInputError("every component must have exactly the three frozen row roles")

    signatures = {}
    for component, values in by_component.items():
        signature_values = {
            (row["corpus"], int(row["fold"]), int(row["inner_fold"]))
            for row in values.values()
        }
        if len(signature_values) != 1:
            raise CampaignInputError(
                f"component {component!r} has inconsistent nested split metadata"
            )
        signatures[component] = next(iter(signature_values))

    components_by_signature = defaultdict(list)
    for component, signature in signatures.items():
        components_by_signature[signature].append(component)
    prompt_blocks = []
    for signature in sorted(components_by_signature):
        components = sorted(components_by_signature[signature], key=lambda component: (
            _stable_digest(seed, "prompt-block-order", signature, component), component,
        ))
        for start in range(0, len(components), batch_size):
            members = tuple(components[start:start + batch_size])
            prompt_block_id = "prompt-block-" + _stable_digest(signature, members)[:24]
            prompt_blocks.append((prompt_block_id, signature, members))
    prompt_blocks.sort(key=lambda item: item[0])

    records = []
    wave_orders = {}
    for judge in judges:
        spec = judge_specs[judge]
        for repeat in range(repeats):
            wave_id = f"{judge}:repeat-{repeat}"
            requests = []
            for prompt_block_id, signature, members in prompt_blocks:
                for role in ROW_ROLES:
                    batch = [by_component[component][role] for component in members]
                    requests.append((prompt_block_id, signature, role, batch))
            requests.sort(key=lambda item: (
                _stable_digest(seed, wave_id, "request-order", item[0], item[2]),
                item[0], item[2],
            ))
            ordered_rows = []
            for batch_index, (prompt_block_id, signature, role, batch) in enumerate(requests):
                # Membership is stable, while each role x judge gets an
                # independent deterministic base order.  Evenly spaced repeat
                # rotations spread each measurement across list positions.
                batch = sorted(batch, key=lambda row: (
                    _stable_digest(
                        seed, prompt_block_id, "position-base", judge, role,
                        row["component_id"],
                    ),
                    row["component_id"],
                ))
                rotation = (repeat * len(batch)) // repeats
                batch = batch[rotation:] + batch[:rotation]
                if len({row["component_id"] for row in batch}) != len(batch):
                    raise AssertionError("a request repeats a component")
                if {
                    (row["corpus"], int(row["fold"]), int(row["inner_fold"]))
                    for row in batch
                } != {signature}:
                    raise AssertionError("a request crosses its nested split signature")
                immutable_contract = {
                    "request_schema_version": spec["request_schema_version"],
                    "model_id": spec["model_id"],
                    "model_revision": spec["model_revision"],
                    "prompt_id": spec["prompt_id"],
                    "prompt_sha256": spec["prompt_sha256"],
                    "reasoning_effort": spec["reasoning_effort"],
                    "settings_sha256": spec["settings_sha256"],
                    "call_seed_base": spec["call_seed_base"],
                    "shared_session_id": spec["shared_session_id"],
                }
                request_input_sha256 = sha256_bytes(score_input_bytes(batch))
                request_preimage = (
                    wave_id,
                    role,
                    prompt_block_id,
                    [row["row_id"] for row in batch],
                    request_input_sha256,
                    immutable_contract,
                )
                call_seed = ""
                if spec["call_seed_base"] is not None:
                    call_seed = int(
                        _stable_digest(spec["call_seed_base"], request_preimage)[:8], 16
                    ) % (2 ** 31)
                request_id = "request-" + _stable_digest(
                    request_preimage, {"call_seed": call_seed}
                )[:24]
                for batch_row, row in enumerate(batch):
                    position = len(ordered_rows)
                    ordered_rows.append(row)
                    records.append({
                        "request_schema_version": spec["request_schema_version"],
                        "model_id": spec["model_id"],
                        "model_revision": spec["model_revision"],
                        "prompt_id": spec["prompt_id"],
                        "prompt_sha256": spec["prompt_sha256"],
                        "reasoning_effort": spec["reasoning_effort"],
                        "settings_json": spec["settings_json"],
                        "settings_sha256": spec["settings_sha256"],
                        "call_seed": call_seed,
                        "shared_session_id": spec["shared_session_id"],
                        "wave_id": wave_id,
                        "judge": judge,
                        "repeat": repeat,
                        "batch_size": len(batch),
                        "batch_index": batch_index,
                        "batch_row": batch_row,
                        "global_position": position,
                        "request_id": request_id,
                        "request_input_sha256": request_input_sha256,
                        "prompt_block_id": prompt_block_id,
                        "inference_cluster_id": prompt_block_id,
                        "corpus": signature[0],
                        "outer_fold": signature[1],
                        "global_inner_fold": signature[2],
                        "row_id": row["row_id"],
                        "component_id": row["component_id"],
                        "role": role,
                    })
            if {row["row_id"] for row in ordered_rows} != {row["row_id"] for row in rows}:
                raise AssertionError("a scoring wave must contain every campaign row exactly once")
            wave_orders[wave_id] = tuple(ordered_rows)
    return tuple(records), wave_orders


def response_ingestion_schema():
    """Machine-readable contract for keyed, retry-preserving response ingestion."""
    return {
        "schema_version": 1,
        "join_key": ["request_id", "row_id"],
        "retry_key": ["request_id", "row_id", "attempt"],
        "required_columns": list(RESPONSE_REQUIRED_COLUMNS),
        "statuses": ["success", "retryable_failure", "terminal_failure"],
        "parse_statuses": ["parsed", "parse_failure", ""],
        "rules": [
            "responses join by request_id + row_id, never by row position",
            "the scorer must echo the row_id supplied in each score-input row",
            "every attempt is retained and attempts are unique non-negative integers",
            "every scheduled row has exactly one terminal outcome: success or terminal_failure",
            "attempts are contiguous from zero and every batched row records every call attempt",
            "provider-call identity, status, timestamps, and raw response are common within an attempt",
            "provider request and nonempty response identities are unique across logical call attempts",
            "immutable model, prompt, and settings identities must match the schedule",
            "successful raw response bytes are SHA-256 authenticated",
        ],
    }


def validate_response_records(schedule, responses):
    """Validate fully ingested responses without relying on file order."""
    schedule = tuple(schedule)
    responses = tuple(responses)
    scheduled = {}
    scheduled_by_request = defaultdict(set)
    for row in schedule:
        key = (row["request_id"], row["row_id"])
        if key in scheduled:
            raise CampaignInputError(f"duplicate schedule join key {key!r}")
        scheduled[key] = row
        scheduled_by_request[row["request_id"]].add(row["row_id"])
    attempts = set()
    attempts_by_key = defaultdict(list)
    successes = Counter()
    terminal_failures = Counter()
    terminal_attempt = {}
    retry_attempt_count = 0
    parsed_row_count = 0
    parse_failure_count = 0
    response_batches = defaultdict(list)
    for index, response in enumerate(responses, 1):
        missing = [column for column in RESPONSE_REQUIRED_COLUMNS if column not in response]
        if missing:
            raise CampaignInputError(
                f"response record {index} misses required columns: {', '.join(missing)}"
            )
        request_id = _canonical_token(
            response["request_id"], label=f"response record {index} request_id"
        )
        row_id = _canonical_token(response["row_id"], label=f"response record {index} row_id")
        key = (request_id, row_id)
        if key not in scheduled:
            raise CampaignInputError(f"response record {index} has unknown join key {key!r}")
        attempt = _canonical_int(
            str(response["attempt"]), label=f"response record {index} attempt", minimum=0
        )
        retry_key = (*key, attempt)
        if retry_key in attempts:
            raise CampaignInputError(f"duplicate response retry key {retry_key!r}")
        attempts.add(retry_key)
        attempts_by_key[key].append(attempt)
        response_batches[(request_id, attempt)].append(response)
        expected = scheduled[key]
        for field in (
            "judge", "model_id", "model_revision", "prompt_sha256", "settings_sha256",
        ):
            if str(response[field]) != str(expected[field]):
                raise CampaignInputError(
                    f"response record {index} immutable {field} does not match its schedule"
                )
        status = response["status"]
        if status not in {"success", "retryable_failure", "terminal_failure"}:
            raise CampaignInputError(f"response record {index} has invalid status {status!r}")
        _canonical_token(
            response["provider_request_id"],
            label=f"response record {index} provider_request_id",
        )
        _canonical_token(
            response["started_at_utc"], label=f"response record {index} started_at_utc"
        )
        _canonical_token(
            response["completed_at_utc"], label=f"response record {index} completed_at_utc"
        )
        if status == "success":
            successes[key] += 1
            terminal_attempt[key] = attempt
            _canonical_token(
                response["provider_response_id"],
                label=f"response record {index} provider_response_id",
            )
            raw = response["raw_response"]
            if not isinstance(raw, str):
                raise CampaignInputError(f"response record {index} raw_response must be text")
            digest = _require_sha256(
                response["raw_response_sha256"],
                label=f"response record {index} raw_response_sha256",
            )
            if digest != sha256_bytes(raw.encode("utf-8")):
                raise CampaignInputError(f"response record {index} raw response hash mismatch")
            if response["error_type"] not in {"", None}:
                raise CampaignInputError(
                    f"successful response record {index} must not declare error_type"
                )
            parse_status = response["parse_status"]
            if parse_status == "parsed":
                parsed_row_count += 1
                for field in ("score_d", "score_s"):
                    _canonical_float(
                        str(response[field]), label=f"response record {index} {field}"
                    )
                if response["parse_error_type"] not in {"", None}:
                    raise CampaignInputError(
                        f"parsed response record {index} must not declare parse_error_type"
                    )
            elif parse_status == "parse_failure":
                parse_failure_count += 1
                if not str(response["parse_error_type"]).strip():
                    raise CampaignInputError(
                        f"response record {index} parse_failure requires parse_error_type"
                    )
                if response["score_d"] not in {"", None} or response["score_s"] not in {"", None}:
                    raise CampaignInputError(
                        f"response record {index} parse_failure must not contain parsed scores"
                    )
            else:
                raise CampaignInputError(
                    f"successful response record {index} parse_status must be parsed or parse_failure"
                )
        else:
            if not str(response["error_type"]).strip():
                raise CampaignInputError(f"failed response record {index} requires error_type")
            if status == "terminal_failure":
                terminal_failures[key] += 1
                terminal_attempt[key] = attempt
            else:
                retry_attempt_count += 1
            if response["parse_status"] not in {"", None} or response["parse_error_type"] not in {"", None}:
                raise CampaignInputError(
                    f"provider-failure response record {index} must not declare row parse state"
                )

    common_attempt_fields = (
        "provider_request_id", "provider_response_id", "started_at_utc", "completed_at_utc",
        "status", "raw_response_sha256", "raw_response", "error_type",
    )
    provider_request_owners = {}
    provider_response_owners = {}
    for request_id, expected_rows in sorted(scheduled_by_request.items()):
        request_attempts = sorted(
            attempt for at_request, attempt in response_batches if at_request == request_id
        )
        if request_attempts != list(range(len(request_attempts))):
            raise CampaignInputError(
                f"request {request_id!r} attempts must be contiguous from zero"
            )
        for attempt in request_attempts:
            batch = response_batches[(request_id, attempt)]
            actual_rows = {record["row_id"] for record in batch}
            if actual_rows != expected_rows:
                raise CampaignInputError(
                    f"request {request_id!r} attempt {attempt} does not cover every scheduled row"
                )
            for field in common_attempt_fields:
                if len({str(record[field]) for record in batch}) != 1:
                    raise CampaignInputError(
                        f"request {request_id!r} attempt {attempt} has inconsistent provider {field}"
                    )
            owner = (request_id, attempt)
            provider_request_id = str(batch[0]["provider_request_id"])
            previous = provider_request_owners.setdefault(provider_request_id, owner)
            if previous != owner:
                raise CampaignInputError(
                    f"provider_request_id {provider_request_id!r} is reused across "
                    f"logical call attempts {previous!r} and {owner!r}"
                )
            provider_response_id = str(batch[0]["provider_response_id"])
            if provider_response_id:
                previous = provider_response_owners.setdefault(provider_response_id, owner)
                if previous != owner:
                    raise CampaignInputError(
                        f"provider_response_id {provider_response_id!r} is reused across "
                        f"logical call attempts {previous!r} and {owner!r}"
                    )
    duplicate_success = [key for key, count in successes.items() if count > 1]
    if duplicate_success:
        raise CampaignInputError(f"join key {sorted(duplicate_success)[0]!r} has multiple successes")
    duplicate_failure = [key for key, count in terminal_failures.items() if count > 1]
    if duplicate_failure:
        raise CampaignInputError(
            f"join key {sorted(duplicate_failure)[0]!r} has multiple terminal failures"
        )
    for key in sorted(scheduled):
        terminal_count = successes[key] + terminal_failures[key]
        if terminal_count != 1:
            raise CampaignInputError(
                f"scheduled join key {key!r} must have exactly one terminal outcome"
            )
        if terminal_attempt[key] != max(attempts_by_key[key]):
            raise CampaignInputError(f"join key {key!r} has attempts after its terminal outcome")
    return {
        "scheduled_rows": len(scheduled),
        "response_attempts": len(responses),
        "successful_rows": sum(successes.values()),
        "terminal_failure_rows": sum(terminal_failures.values()),
        "retry_attempts": retry_attempt_count,
        "provider_call_attempts": len(response_batches),
        "retry_call_attempts": sum(
            next(iter(records))["status"] == "retryable_failure"
            for records in response_batches.values()
        ),
        "parsed_rows": parsed_row_count,
        "parse_failure_rows": parse_failure_count,
    }


def tsv_bytes(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(columns), delimiter="\t", lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def score_input_bytes(rows):
    columns = (
        "row_id", "node_title", "root_title", "cur_relation", "conf", "neighborhood",
        "node_type", "root_type", "raw",
    )
    body = tsv_bytes(columns, rows).decode("utf-8")
    header, remainder = body.split("\n", 1)
    return ("# " + header + "\n" + remainder).encode("utf-8")


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
