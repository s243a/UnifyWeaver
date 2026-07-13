#!/usr/bin/env python3
"""Focused tests for outcome-blind repeated-campaign primitives."""

from collections import Counter, defaultdict
import csv

import pytest

from repeated_judge_campaign import (
    CANDIDATE_REQUIRED_COLUMNS,
    CampaignInputError,
    HistoricalEndpoints,
    assign_component_folds,
    assign_nested_inner_folds,
    build_scoring_schedule,
    campaign_rows,
    component_quotas,
    filter_candidate_pool,
    load_candidates,
    normalize_title,
    sha256_bytes,
    select_candidates,
    tsv_bytes,
    validate_candidate_geometry,
    validate_response_records,
)


def candidate_record(candidate_id, cell, base, *, source="source-a", shared_id=None):
    corpus, hop, degree, agreement = cell
    first_hop, second_hop = (part[1:] if part.startswith("h") else part for part in hop.split("-"))
    descendant_id = shared_id or f"id-{base}-x"
    record = {
        "candidate_id": candidate_id,
        "corpus": corpus,
        "source_component": source,
        "hop_transition": hop,
        "degree_quartile": degree,
        "agreement_class": agreement,
        "anchor_hop": first_hop,
        "adjacent_hop": second_hop,
        "distant_hop": second_hop,
        "anchor_campaign_tag": "anchor-campaign",
        "adjacent_campaign_tag": "comparator-campaign",
        "distant_campaign_tag": "comparator-campaign",
        "anchor_degree_quartile": degree,
        "adjacent_degree_quartile": "q2",
        "distant_degree_quartile": "q2",
        "anchor_adjacent_direct_edge": "true",
        "anchor_distant_distance": "3",
        "anchor_distant_disconnected": "false",
        "cumulative_anchor_adjacent_similarity": "0.9",
        "cumulative_anchor_distant_similarity": "0.1",
        "nomic_anchor_adjacent_similarity": "0.8" if agreement == "agreement" else "0.2",
        "nomic_anchor_distant_similarity": "0.2" if agreement == "agreement" else "0.8",
        "descendant_id": descendant_id,
        "descendant_title": f"Title_{descendant_id}",
        "anchor_id": f"id-{base}-a",
        "anchor_title": f"Title_{base}_a",
        "adjacent_id": f"id-{base}-b",
        "adjacent_title": f"Title_{base}_b",
        "distant_id": f"id-{base}-c",
        "distant_title": f"Title_{base}_c",
    }
    return record


def judge_specs(judges=("judge-a",)):
    return {
        judge: {
            "request_schema_version": 1,
            "model_id": f"model-{judge}",
            "model_revision": "revision-1",
            "prompt_id": "prompt-1",
            "prompt_sha256": "a" * 64,
            "reasoning_effort": "low",
            "settings_json": "{}",
            "settings_sha256": sha256_bytes(b"{}"),
            "call_seed_base": None,
            "shared_session_id": "",
            "stateless": True,
        }
        for judge in judges
    }


def write_tsv(path, columns, rows):
    path.write_bytes(tsv_bytes(columns, rows))


def test_normalization_matches_campaign_identity_contract():
    assert normalize_title("  Foo__BAR  ") == "foo bar"
    assert normalize_title("ＡＢＣ") == "abc"


def test_loader_rejects_duplicate_candidate_and_inconsistent_endpoint_identity(tmp_path):
    cell = ("exploratory", "h1-h2", "q1", "agreement")
    row = candidate_record("c0", cell, 0)
    path = tmp_path / "pool.tsv"
    write_tsv(path, CANDIDATE_REQUIRED_COLUMNS, [row, row])
    with pytest.raises(CampaignInputError, match="duplicate candidate_id"):
        load_candidates(path)

    first = candidate_record("c0", cell, 0, shared_id="same")
    second = candidate_record("c1", cell, 1, shared_id="same")
    second["descendant_title"] = "A genuinely different title"
    write_tsv(path, CANDIDATE_REQUIRED_COLUMNS, [first, second])
    with pytest.raises(CampaignInputError, match="inconsistent normalized titles"):
        load_candidates(path)


def test_historical_filter_selection_folds_and_schedule_are_deterministic(tmp_path):
    cells = (
        ("exploratory", "h1-h2", "q1", "agreement"),
        ("exploratory", "h1-h2", "q1", "disagreement"),
    )
    records = []
    for cell_index, cell in enumerate(cells):
        for value in range(6):
            base = 100 * cell_index + value
            records.append(candidate_record(
                f"candidate-{cell_index}-{value}", cell, base,
                source=f"source-{cell_index}-{value}",
            ))
    # This otherwise eligible row is removed by normalized historical title.
    records.append(candidate_record("historical-candidate", cells[0], 999))
    pool = tmp_path / "pool.tsv"
    write_tsv(pool, CANDIDATE_REQUIRED_COLUMNS, records)
    _columns, candidates, _artifact = load_candidates(pool)
    historical = HistoricalEndpoints(
        frozenset(), frozenset({normalize_title("Title_999_a")}), tuple()
    )
    quotas = {cell: 2 for cell in cells}
    eligible, exclusions = filter_candidate_pool(candidates, historical, quotas)
    assert exclusions == {"historical_normalized_title": 1}

    selected_a, audit_a = select_candidates(
        eligible, quotas, source_cap_fraction=0.50, seed=17
    )
    selected_b, audit_b = select_candidates(
        eligible, quotas, source_cap_fraction=0.50, seed=17
    )
    assert [value.candidate_id for value in selected_a] == [value.candidate_id for value in selected_b]
    assert audit_a == audit_b
    assert Counter(value.cell for value in selected_a) == Counter(quotas)
    ids = [endpoint for value in selected_a for endpoint in value.scoped_ids]
    titles = [title for value in selected_a for title in value.normalized_titles]
    assert len(ids) == len(set(ids))
    assert len(titles) == len(set(titles))
    assert max(audit_a["source_component_counts"].values()) <= 2

    folded_a, diagnostics_a = assign_component_folds(selected_a, quotas, folds=2, seed=19)
    folded_b, diagnostics_b = assign_component_folds(selected_a, quotas, folds=2, seed=19)
    assert [(value.component_id, value.fold) for value in folded_a] == [
        (value.component_id, value.fold) for value in folded_b
    ]
    assert diagnostics_a == diagnostics_b
    for fold in range(2):
        assert sum(value.fold == fold for value in folded_a) == 2
        for cell in cells:
            assert sum(value.fold == fold and value.cell == cell for value in folded_a) == 1

    folded_a, nested_a, inner_diagnostics_a = assign_nested_inner_folds(
        folded_a, outer_folds=2, inner_folds=2, seed=21
    )
    folded_b, nested_b, inner_diagnostics_b = assign_nested_inner_folds(
        folded_b, outer_folds=2, inner_folds=2, seed=21
    )
    assert nested_a == nested_b
    assert inner_diagnostics_a == inner_diagnostics_b
    assert len(nested_a) == len(folded_a) * 2
    rows = campaign_rows(folded_a)
    schedule, waves = build_scoring_schedule(
        rows,
        judges=("judge-a",),
        repeats=3,
        batch_size=2,
        seed=23,
        judge_specs=judge_specs(),
    )
    assert len(rows) == 12
    assert len(schedule) == 36
    assert len(waves) == 3
    for wave_id, order in waves.items():
        assert len(order) == len(rows)
        assert {row["row_id"] for row in order} == {row["row_id"] for row in rows}, wave_id
    by_request = Counter()
    for record in schedule:
        by_request[(record["wave_id"], record["request_id"], record["component_id"])] += 1
    assert max(by_request.values()) == 1
    request_signatures = defaultdict(set)
    request_blocks = defaultdict(set)
    for record in schedule:
        request_signatures[record["request_id"]].add(
            (record["corpus"], record["outer_fold"], record["global_inner_fold"])
        )
        request_blocks[(record["prompt_block_id"], record["role"])].add(record["component_id"])
    assert all(len(values) == 1 for values in request_signatures.values())
    assert all(record["prompt_block_id"] == record["inference_cluster_id"] for record in schedule)
    assert all(len(record["request_input_sha256"]) == 64 for record in schedule)

    positions = defaultdict(Counter)
    batch_sizes = {}
    for record in schedule:
        key = (
            record["prompt_block_id"], record["component_id"],
            record["role"], record["judge"],
        )
        positions[key][record["batch_row"]] += 1
        batch_sizes[key] = record["batch_size"]
    for key, counts in positions.items():
        values = [counts[position] for position in range(batch_sizes[key])]
        assert max(values) - min(values) <= 1

    changed_specs = judge_specs()
    changed_specs["judge-a"]["model_revision"] = "revision-2"
    changed_schedule, _ = build_scoring_schedule(
        rows,
        judges=("judge-a",),
        repeats=3,
        batch_size=2,
        seed=23,
        judge_specs=changed_specs,
    )
    assert {record["request_id"] for record in schedule}.isdisjoint(
        {record["request_id"] for record in changed_schedule}
    )
    presentation_changed = [dict(row) for row in rows]
    for row in presentation_changed:
        row["node_title"] = row["node_title"].upper()
        row["root_title"] = row["root_title"].upper()
    presentation_schedule, _ = build_scoring_schedule(
        presentation_changed,
        judges=("judge-a",),
        repeats=3,
        batch_size=2,
        seed=23,
        judge_specs=judge_specs(),
    )
    assert {record["request_id"] for record in schedule}.isdisjoint(
        {record["request_id"] for record in presentation_schedule}
    )
    seeded_specs = judge_specs()
    seeded_specs["judge-a"]["call_seed_base"] = 7
    seeded_schedule, _ = build_scoring_schedule(
        rows,
        judges=("judge-a",),
        repeats=3,
        batch_size=2,
        seed=23,
        judge_specs=seeded_specs,
    )
    seeds_by_request = defaultdict(set)
    for record in seeded_schedule:
        seeds_by_request[record["request_id"]].add(record["call_seed"])
    assert all(len(values) == 1 for values in seeds_by_request.values())
    assert len({next(iter(values)) for values in seeds_by_request.values()}) == len(
        seeds_by_request
    )


def test_source_cap_and_repeats_below_three_fail_loudly(tmp_path):
    cell = ("exploratory", "h1-h2", "q1", "agreement")
    records = [
        candidate_record(f"c-{value}", cell, value, source="only-source")
        for value in range(4)
    ]
    pool = tmp_path / "pool.tsv"
    write_tsv(pool, CANDIDATE_REQUIRED_COLUMNS, records)
    _columns, candidates, _artifact = load_candidates(pool)
    with pytest.raises(CampaignInputError, match="could not pack"):
        select_candidates(
            candidates, {cell: 4}, source_cap_fraction=0.25, max_attempts=3
        )

    unique_records = [
        candidate_record(f"unique-{value}", cell, 100 + value, source=f"source-{value}")
        for value in range(4)
    ]
    write_tsv(pool, CANDIDATE_REQUIRED_COLUMNS, unique_records)
    _columns, unique_candidates, _artifact = load_candidates(pool)
    selected, _ = select_candidates(
        unique_candidates, {cell: 4}, source_cap_fraction=1.0
    )
    folded, _ = assign_component_folds(selected, {cell: 4}, folds=2)
    folded, _nested, _diagnostics = assign_nested_inner_folds(
        folded, outer_folds=2, inner_folds=2
    )
    with pytest.raises(CampaignInputError, match="at least three"):
        build_scoring_schedule(
            campaign_rows(folded),
            judges=("judge-a",),
            repeats=2,
            batch_size=3,
            judge_specs=judge_specs(),
        )
    with pytest.raises(CampaignInputError, match="between one and ten"):
        build_scoring_schedule(
            campaign_rows(folded),
            judges=("judge-a",),
            repeats=3,
            batch_size=11,
            judge_specs=judge_specs(),
        )


def test_unknown_cells_are_rejected_not_silently_dropped(tmp_path):
    good = ("exploratory", "h1-h2", "q1", "agreement")
    bad = ("exploratory", "h9-h10", "q1", "agreement")
    pool = tmp_path / "pool.tsv"
    write_tsv(
        pool,
        CANDIDATE_REQUIRED_COLUMNS,
        [candidate_record("good", good, 1), candidate_record("bad", bad, 2)],
    )
    _columns, candidates, _artifact = load_candidates(pool)
    empty_history = HistoricalEndpoints(frozenset(), frozenset(), tuple())
    with pytest.raises(CampaignInputError, match="unknown frozen cells"):
        filter_candidate_pool(candidates, empty_history, {good: 1})


def test_candidate_structural_contract_and_canonical_identity(tmp_path):
    cell = ("exploratory", "h1-h2", "q1", "agreement")
    path = tmp_path / "pool.tsv"
    row = candidate_record(" c0", cell, 0)
    write_tsv(path, CANDIDATE_REQUIRED_COLUMNS, [row])
    with pytest.raises(CampaignInputError, match="canonical token"):
        load_candidates(path)

    row = candidate_record("c0", cell, 0)
    row["anchor_adjacent_direct_edge"] = "false"
    write_tsv(path, CANDIDATE_REQUIRED_COLUMNS, [row])
    with pytest.raises(CampaignInputError, match="not a direct graph neighbor"):
        load_candidates(path)

    row = candidate_record("c0", cell, 0)
    row["anchor_distant_distance"] = "2"
    write_tsv(path, CANDIDATE_REQUIRED_COLUMNS, [row])
    with pytest.raises(CampaignInputError, match="at least 3"):
        load_candidates(path)

    row = candidate_record("c0", cell, 0)
    row["agreement_class"] = "disagreement"
    write_tsv(path, CANDIDATE_REQUIRED_COLUMNS, [row])
    _columns, candidates, _artifact = load_candidates(path)
    with pytest.raises(CampaignInputError, match="deltas imply"):
        validate_candidate_geometry(
            candidates, {"graph_delta_min": 0.0, "nomic_delta_min": 0.0}
        )


@pytest.mark.parametrize("components_per_corpus", [160, 320, 512, 800])
def test_frozen_g_grid_has_exact_cells_and_near_balanced_outer_folds(
    tmp_path, components_per_corpus
):
    quotas = component_quotas(components_per_corpus)
    by_corpus = Counter()
    for cell, count in quotas.items():
        by_corpus[cell[0]] += count
    assert set(by_corpus.values()) == {components_per_corpus}
    assert len(set(quotas.values())) == 1

    # One synthetic cell is sufficient to pin the G=512 regression: 16 cannot
    # divide five folds exactly but must still materialize 4/3/3/3/3.
    cell = ("exploratory", "h1-h2", "q1", "agreement")
    per_cell = components_per_corpus // 32
    rows = [
        candidate_record(f"c-{value}", cell, value, source=f"source-{value}")
        for value in range(per_cell)
    ]
    path = tmp_path / "pool.tsv"
    write_tsv(path, CANDIDATE_REQUIRED_COLUMNS, rows)
    _columns, candidates, _artifact = load_candidates(path)
    selected, _ = select_candidates(
        candidates, {cell: per_cell}, source_cap_fraction=1.0
    )
    components, _ = assign_component_folds(selected, {cell: per_cell}, folds=5)
    counts = [sum(component.fold == fold for component in components) for fold in range(5)]
    assert max(counts) - min(counts) <= 1
    components, nested, _ = assign_nested_inner_folds(
        components, outer_folds=5, inner_folds=3
    )
    assert len(nested) == len(components) * 5
    for outer_fold in range(5):
        records = [row for row in nested if row["outer_fold"] == outer_fold]
        assert {row["component_id"] for row in records} == {
            component.component_id for component in components
        }
        assert all(
            (row["partition"] == "held") == (
                next(
                    component.fold for component in components
                    if component.component_id == row["component_id"]
                ) == outer_fold
            )
            for row in records
        )


def test_response_ingestion_joins_by_keys_and_preserves_retries():
    common = {
        "judge": "judge-a",
        "model_id": "model-a",
        "model_revision": "revision-1",
        "prompt_sha256": "a" * 64,
        "settings_sha256": "b" * 64,
    }
    schedule = [
        {"request_id": "request-a", "row_id": "row-a", **common},
        {"request_id": "request-b", "row_id": "row-b", **common},
    ]

    def success(request_id, row_id, score, attempt=0):
        raw = f'{{"D": {score}, "S": {score}}}'
        return {
            "request_id": request_id,
            "row_id": row_id,
            "attempt": str(attempt),
            "provider_request_id": f"provider-{request_id}-{attempt}",
            "provider_response_id": f"response-{request_id}-{attempt}",
            "started_at_utc": "2026-01-01T00:00:00Z",
            "completed_at_utc": "2026-01-01T00:00:01Z",
            "status": "success",
            **common,
            "raw_response_sha256": sha256_bytes(raw.encode()),
            "raw_response": raw,
            "score_d": str(score),
            "score_s": str(score),
            "error_type": "",
            "parse_status": "parsed",
            "parse_error_type": "",
        }

    retry = {
        "request_id": "request-a",
        "row_id": "row-a",
        "attempt": "0",
        "provider_request_id": "provider-request-a-0",
        "provider_response_id": "",
        "started_at_utc": "2026-01-01T00:00:00Z",
        "completed_at_utc": "2026-01-01T00:00:30Z",
        "status": "retryable_failure",
        **common,
        "raw_response_sha256": "",
        "raw_response": "",
        "score_d": "",
        "score_s": "",
        "error_type": "timeout",
        "parse_status": "",
        "parse_error_type": "",
    }
    responses = [
        success("request-b", "row-b", 0.4),
        retry,
        success("request-a", "row-a", 0.5, attempt=1),
    ]
    assert validate_response_records(schedule, responses) == {
        "scheduled_rows": 2,
        "response_attempts": 3,
        "successful_rows": 2,
        "terminal_failure_rows": 0,
        "retry_attempts": 1,
        "provider_call_attempts": 3,
        "retry_call_attempts": 1,
        "parsed_rows": 2,
        "parse_failure_rows": 0,
    }
    bad = [dict(record) for record in responses]
    bad[0]["model_revision"] = "wrong"
    with pytest.raises(CampaignInputError, match="immutable model_revision"):
        validate_response_records(schedule, bad)

    reused_provider_request = [dict(record) for record in responses]
    reused_provider_request[2]["provider_request_id"] = reused_provider_request[0][
        "provider_request_id"
    ]
    with pytest.raises(CampaignInputError, match="provider_request_id.*reused"):
        validate_response_records(schedule, reused_provider_request)

    reused_provider_response = [dict(record) for record in responses]
    reused_provider_response[2]["provider_response_id"] = reused_provider_response[0][
        "provider_response_id"
    ]
    with pytest.raises(CampaignInputError, match="provider_response_id.*reused"):
        validate_response_records(schedule, reused_provider_response)

    terminal = {
        "request_id": "request-b",
        "row_id": "row-b",
        "attempt": "0",
        "provider_request_id": "provider-request-b-0",
        "provider_response_id": "",
        "started_at_utc": "2026-01-01T00:00:00Z",
        "completed_at_utc": "2026-01-01T00:00:30Z",
        "status": "terminal_failure",
        **common,
        "raw_response_sha256": "",
        "raw_response": "",
        "score_d": "",
        "score_s": "",
        "error_type": "provider_rejected",
        "parse_status": "",
        "parse_error_type": "",
    }
    terminal_summary = validate_response_records(
        schedule, [retry, success("request-a", "row-a", 0.5, attempt=1), terminal]
    )
    assert terminal_summary["successful_rows"] == 1
    assert terminal_summary["terminal_failure_rows"] == 1


def test_source_dependency_groups_are_atomic_across_outer_and_inner_folds(tmp_path):
    cells = (
        ("exploratory", "h1-h2", "q1", "agreement"),
        ("exploratory", "h1-h2", "q1", "disagreement"),
    )
    records = []
    base = 0
    for cell_index, cell in enumerate(cells):
        for source in ("shared-a", "shared-b", f"unique-{cell_index}-a", f"unique-{cell_index}-b"):
            records.append(candidate_record(f"c-{base}", cell, base, source=source))
            base += 1
    path = tmp_path / "atomic.tsv"
    write_tsv(path, CANDIDATE_REQUIRED_COLUMNS, records)
    _columns, candidates, _artifact = load_candidates(path)
    quotas = {cell: 4 for cell in cells}
    selected, _ = select_candidates(candidates, quotas, source_cap_fraction=1.0)
    outer, _ = assign_component_folds(selected, quotas, folds=2, seed=51)
    outer_by_source = defaultdict(set)
    for component in outer:
        outer_by_source[(component.corpus, component.source_component)].add(component.fold)
    assert all(len(folds) == 1 for folds in outer_by_source.values())

    nested_components, _nested, _ = assign_nested_inner_folds(
        outer, outer_folds=2, inner_folds=2, seed=52
    )
    inner_by_source = defaultdict(set)
    for component in nested_components:
        inner_by_source[(component.corpus, component.source_component)].add(
            component.inner_fold
        )
    assert all(len(folds) == 1 for folds in inner_by_source.values())

    infeasible = [
        candidate_record(f"bad-{value}", cells[0], 100 + value, source=(
            "oversized" if value < 3 else "singleton"
        ))
        for value in range(4)
    ]
    write_tsv(path, CANDIDATE_REQUIRED_COLUMNS, infeasible)
    _columns, bad_candidates, _artifact = load_candidates(path)
    bad_selected, _ = select_candidates(
        bad_candidates, {cells[0]: 4}, source_cap_fraction=1.0
    )
    with pytest.raises(CampaignInputError, match="atomic source group"):
        assign_component_folds(bad_selected, {cells[0]: 4}, folds=2)


def test_batched_response_attempts_have_common_provider_identity_and_contiguous_retries():
    immutable = {
        "judge": "judge-a",
        "model_id": "model-a",
        "model_revision": "revision-1",
        "prompt_sha256": "a" * 64,
        "settings_sha256": "b" * 64,
    }
    schedule = [
        {"request_id": "request-batch", "row_id": row_id, **immutable}
        for row_id in ("row-a", "row-b")
    ]
    raw = '{"rows":[{"D":0.1,"S":0.2},{"D":0.3,"S":0.4}]}'
    responses = []
    for row_id, score in (("row-a", "0.1"), ("row-b", "0.3")):
        responses.append({
            "request_id": "request-batch",
            "row_id": row_id,
            "attempt": "0",
            "provider_request_id": "provider-call-0",
            "provider_response_id": "provider-response-0",
            "started_at_utc": "2026-01-01T00:00:00Z",
            "completed_at_utc": "2026-01-01T00:00:01Z",
            "status": "success",
            **immutable,
            "raw_response_sha256": sha256_bytes(raw.encode()),
            "raw_response": raw,
            "score_d": score,
            "score_s": score,
            "error_type": "",
            "parse_status": "parsed",
            "parse_error_type": "",
        })
    assert validate_response_records(schedule, responses)["provider_call_attempts"] == 1

    inconsistent = [dict(record) for record in responses]
    inconsistent[1]["completed_at_utc"] = "2026-01-01T00:00:02Z"
    with pytest.raises(CampaignInputError, match="inconsistent provider completed_at_utc"):
        validate_response_records(schedule, inconsistent)

    skipped_zero = [dict(record, attempt="1") for record in responses]
    with pytest.raises(CampaignInputError, match="contiguous from zero"):
        validate_response_records(schedule, skipped_zero)
