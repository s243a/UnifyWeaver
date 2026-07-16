#!/usr/bin/env python3
"""Select and schedule the frozen repeated-judge campaign without scoring it."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import tempfile

from repeated_judge_campaign import (
    ALGORITHM,
    DEFAULT_COMPONENTS_PER_CORPUS,
    DEFAULT_JUDGES,
    FROZEN_COMPONENT_REPEAT_GRID,
    NESTED_FOLD_COLUMNS,
    ROW_ROLES,
    SCHEMA_VERSION,
    CampaignInputError,
    assign_component_folds,
    assign_nested_inner_folds,
    build_scoring_schedule,
    campaign_rows,
    component_quotas,
    content_record,
    expected_quotas,
    filter_candidate_pool,
    json_bytes,
    load_candidates,
    load_candidate_builder_manifest,
    load_historical_endpoints,
    load_request_contract,
    response_ingestion_schema,
    score_input_bytes,
    select_candidates,
    tsv_bytes,
    validate_candidate_geometry,
)


ROW_COLUMNS = (
    "row_id",
    "component_id",
    "fold",
    "inner_fold",
    "candidate_id",
    "corpus",
    "source_component",
    "hop_transition",
    "degree_quartile",
    "agreement_class",
    "role",
    "node_id",
    "node_title",
    "node_normalized_title",
    "root_id",
    "root_title",
    "root_normalized_title",
    "cur_relation",
    "conf",
    "neighborhood",
    "node_type",
    "root_type",
    "raw",
)
SCHEDULE_COLUMNS = (
    "request_schema_version",
    "model_id",
    "model_revision",
    "prompt_id",
    "prompt_sha256",
    "reasoning_effort",
    "settings_json",
    "settings_sha256",
    "call_seed",
    "shared_session_id",
    "wave_id",
    "judge",
    "repeat",
    "batch_size",
    "batch_index",
    "batch_row",
    "global_position",
    "request_id",
    "request_input_sha256",
    "prompt_block_id",
    "inference_cluster_id",
    "corpus",
    "outer_fold",
    "global_inner_fold",
    "row_id",
    "component_id",
    "role",
)


def _safe_judge(value):
    output = "".join(character if character.isalnum() or character in ".-_" else "_" for character in value)
    if not output or output in {".", ".."}:
        raise CampaignInputError(f"judge name cannot form a safe artifact name: {value!r}")
    return output


def _artifact(root, relative, payload):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignInputError(f"artifact name must be relative and contained: {relative}")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"name": relative.as_posix(), **content_record(payload)}


def _component_rows(components, input_columns):
    if any(name in input_columns for name in ("component_id", "fold", "inner_fold")):
        raise CampaignInputError(
            "candidate pool may not predeclare component_id, fold, or inner_fold; "
            "they are selector outputs"
        )
    rows = []
    for component in components:
        row = {
            "component_id": component.component_id,
            "fold": component.fold,
            "inner_fold": component.inner_fold,
        }
        row.update(component.raw)
        rows.append(row)
    return ("component_id", "fold", "inner_fold", *input_columns), rows


def _cell_counts(components):
    values = Counter("|".join(component.cell) for component in components)
    return dict(sorted(values.items()))


def _configuration_classification(args, request_specs):
    deviations = []
    if args.seed != 0:
        deviations.append("selector seed differs from the frozen value 0")
    if args.per_cell is not None:
        deviations.append("--per-cell is an exploratory-only quota override")
    if (args.components_per_corpus, args.repeats) not in FROZEN_COMPONENT_REPEAT_GRID:
        deviations.append("(components_per_corpus, repeats) is outside the frozen G/R grid")
    if args.source_component_cap_fraction != 0.10:
        deviations.append("source-component cap differs from the frozen 0.10")
    if args.folds != 5 or args.inner_folds != 3:
        deviations.append("outer/inner folds differ from frozen 5/3")
    if tuple(args.judges) != DEFAULT_JUDGES:
        deviations.append("judge aliases differ from the frozen pair")
    frozen_models = {
        "gpt-5.5-low": "gpt-5.5",
        "gpt-5.6-luna": "gpt-5.6-luna",
    }
    for alias, expected_model in frozen_models.items():
        if alias in request_specs and (
            request_specs[alias]["model_id"] != expected_model
            or request_specs[alias]["reasoning_effort"] != "low"
        ):
            deviations.append(f"{alias} model identity or reasoning effort is not frozen")
    if args.batch_size > 10:
        deviations.append("prompt blocks exceed the frozen maximum of 10 rows")
    if any(not spec["stateless"] for spec in request_specs.values()):
        deviations.append("request contract is not stateless")
    # This repository does not yet contain the candidate builder or the exact
    # approved prompt/settings artifacts.  A caller-supplied content hash is
    # useful for reproducibility but is not independent verification, so even
    # an otherwise frozen-shaped materialization must not be labeled
    # confirmatory-ready.
    classification = (
        "protocol-shape-compatible-no-spend-inputs-unverified"
        if not deviations else "exploratory-smoke-only"
    )
    return classification, deviations


def build_campaign(args):
    candidate_columns, candidates, candidate_artifact = load_candidates(args.candidate_pool)
    builder_manifest, builder_artifact, thresholds = load_candidate_builder_manifest(
        args.candidate_builder_manifest, candidate_artifact
    )
    validate_candidate_geometry(candidates, thresholds)
    historical, historical_artifacts = load_historical_endpoints(args.historical_endpoints)
    if args.repeats < 3:
        raise CampaignInputError("repeats must be at least three")
    if not 1 <= args.batch_size <= 10:
        raise CampaignInputError("batch_size must be between one and ten rows")
    safe_judges = [_safe_judge(judge) for judge in args.judges]
    if len(safe_judges) != len(set(safe_judges)):
        raise CampaignInputError("judge names collide after safe artifact-name normalization")
    request_specs, request_contract, request_contract_artifact = load_request_contract(
        args.request_contract, tuple(args.judges)
    )
    if args.per_cell is None:
        quotas = component_quotas(args.components_per_corpus)
    else:
        quotas = expected_quotas(per_cell=args.per_cell)
    eligible, exclusions = filter_candidate_pool(candidates, historical, quotas)
    selected, selection = select_candidates(
        eligible,
        quotas,
        source_cap_fraction=args.source_component_cap_fraction,
        seed=args.seed,
        max_attempts=args.max_selection_attempts,
    )
    components, fold_diagnostics = assign_component_folds(
        selected, quotas, folds=args.folds, seed=args.seed
    )
    components, nested_fold_records, nested_fold_diagnostics = assign_nested_inner_folds(
        components,
        outer_folds=args.folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    rows = campaign_rows(components)
    schedule, wave_orders = build_scoring_schedule(
        rows,
        judges=tuple(args.judges),
        repeats=args.repeats,
        batch_size=args.batch_size,
        seed=args.seed,
        judge_specs=request_specs,
    )
    classification, deviations = _configuration_classification(args, request_specs)

    out = Path(args.out_dir)
    if out.exists():
        raise CampaignInputError(f"output directory already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=str(out.parent)))
    try:
        artifacts = []
        component_columns, component_records = _component_rows(components, candidate_columns)
        artifacts.append(_artifact(
            temporary, "selected_components.tsv",
            tsv_bytes(component_columns, component_records),
        ))
        artifacts.append(_artifact(
            temporary, "campaign_rows.tsv", tsv_bytes(ROW_COLUMNS, rows)
        ))
        artifacts.append(_artifact(
            temporary,
            "nested_component_folds.tsv",
            tsv_bytes(NESTED_FOLD_COLUMNS, nested_fold_records),
        ))
        artifacts.append(_artifact(
            temporary, "scoring_schedule.tsv", tsv_bytes(SCHEDULE_COLUMNS, schedule)
        ))
        artifacts.append(_artifact(
            temporary,
            "response_ingestion_schema.json",
            json_bytes(response_ingestion_schema()),
        ))

        wave_files = []
        row_by_id = {row["row_id"]: row for row in rows}
        for judge in args.judges:
            safe = _safe_judge(judge)
            for repeat in range(args.repeats):
                wave_id = f"{judge}:repeat-{repeat}"
                relative = Path("score_inputs") / f"{safe}.repeat-{repeat}.tsv"
                artifact = _artifact(
                    temporary, relative, score_input_bytes(wave_orders[wave_id])
                )
                request_artifacts = []
                wave_schedule = [row for row in schedule if row["wave_id"] == wave_id]
                request_ids = sorted({row["request_id"] for row in wave_schedule})
                for request_id in request_ids:
                    request_schedule = sorted(
                        (row for row in wave_schedule if row["request_id"] == request_id),
                        key=lambda row: row["batch_row"],
                    )
                    request_rows = [row_by_id[row["row_id"]] for row in request_schedule]
                    request_relative = (
                        Path("request_inputs") / safe / f"repeat-{repeat}" / f"{request_id}.tsv"
                    )
                    request_artifact = _artifact(
                        temporary, request_relative, score_input_bytes(request_rows)
                    )
                    if request_artifact["sha256"] != request_schedule[0]["request_input_sha256"]:
                        raise AssertionError(
                            "request ID input hash differs from materialized request bytes"
                        )
                    request_artifacts.append({
                        "request_id": request_id,
                        "prompt_block_id": request_schedule[0]["prompt_block_id"],
                        "inference_cluster_id": request_schedule[0]["inference_cluster_id"],
                        "row_count": len(request_rows),
                        "artifact": request_artifact,
                    })
                    artifacts.append(request_artifact)
                wave_files.append({
                    "wave_id": wave_id,
                    "judge": judge,
                    "repeat": repeat,
                    "batch_size": args.batch_size,
                    "request_count": len(request_ids),
                    "artifact": artifact,
                    "request_artifacts": request_artifacts,
                })
                artifacts.append(artifact)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "OUTCOME-BLIND CAMPAIGN MATERIALIZATION; NO JUDGE OR MODEL CALLS",
            "classification": classification,
            "call_authorized": False,
            "candidate_builder_verified_by_repository": False,
            "request_contract_approved_for_live_use": False,
            "configuration_deviations": deviations,
            "algorithm": ALGORITHM,
            "configuration": {
                "seed": args.seed,
                "components_per_corpus": args.components_per_corpus,
                "per_cell": args.per_cell,
                "source_component_cap_fraction": args.source_component_cap_fraction,
                "folds": args.folds,
                "inner_folds": args.inner_folds,
                "judges": list(args.judges),
                "repeats": args.repeats,
                "batch_size": args.batch_size,
                "max_selection_attempts": args.max_selection_attempts,
            },
            "inputs": {
                "candidate_pool": candidate_artifact,
                "candidate_builder_manifest": builder_artifact,
                "candidate_builder_contract": builder_manifest,
                "request_contract": request_contract_artifact,
                "request_contract_contents": request_contract,
                "historical_endpoint_inventories": list(historical_artifacts),
                "historical_unique_scoped_ids": len(historical.scoped_ids),
                "historical_unique_normalized_titles": len(historical.normalized_titles),
            },
            "selection": {
                **selection,
                "candidate_pool_rows": len(candidates),
                "eligible_candidate_rows": len(eligible),
                "exclusions": exclusions,
                "component_count": len(components),
                "row_count": len(rows),
                "cell_counts": _cell_counts(components),
                "quota_counts": {
                    "|".join(cell): count for cell, count in sorted(quotas.items())
                },
            },
            "folds": fold_diagnostics,
            "nested_folds": nested_fold_diagnostics,
            "schedule": {
                "wave_count": len(wave_orders),
                "records": len(schedule),
                "roles": list(ROW_ROLES),
                "same_component_per_request_max": 1,
                "maximum_rows_per_request": args.batch_size,
                "request_split_signature": ["corpus", "outer_fold", "global_inner_fold"],
                "prompt_block_is_inference_cluster": True,
                "prompt_block_membership_stable_across_roles_judges_repeats": True,
                "row_position_schedule": (
                    "independent deterministic base per role-by-judge; "
                    "evenly spaced rotations across repeats"
                ),
                "list_position_effect_power_sensitivity_implemented": False,
                "wave_files": wave_files,
            },
            "outputs": sorted(artifacts, key=lambda value: value["name"]),
            "non_claims": [
                "materialization does not establish covariance",
                "component folds do not consume outcomes",
                "score inputs have not been submitted to any judge",
                "candidate-builder hashes are caller-supplied provenance, not repository verification",
                "model revision, prompt hash, and settings have not been approved for live use",
                "list-position effects require a pilot and later frozen sensitivity before live use",
                "materialization never authorizes fresh calls, including protocol-shaped output",
            ],
        }
        manifest_payload = json_bytes(manifest)
        (temporary / "manifest.json").write_bytes(manifest_payload)
        os.replace(temporary, out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest, content_record(manifest_payload)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument(
        "--candidate-builder-manifest",
        required=True,
        help="content-addressed graph/Nomic builder contract for the candidate pool",
    )
    parser.add_argument(
        "--request-contract",
        required=True,
        help="immutable model revision, prompt hash, settings, and stateless-call contract",
    )
    parser.add_argument(
        "--historical-endpoints",
        action="append",
        required=True,
        help="repeatable TSV with corpus, endpoint_id, endpoint_title",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--components-per-corpus", type=int, default=DEFAULT_COMPONENTS_PER_CORPUS
    )
    parser.add_argument(
        "--per-cell", type=int, default=None,
        help="exploratory-only direct cell quota override",
    )
    parser.add_argument("--source-component-cap-fraction", type=float, default=0.10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--judges", nargs="+", default=list(DEFAULT_JUDGES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-selection-attempts", type=int, default=512)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        manifest, record = build_campaign(args)
    except CampaignInputError as exc:
        raise SystemExit(str(exc)) from exc
    print(json_bytes({
        "component_count": manifest["selection"]["component_count"],
        "row_count": manifest["selection"]["row_count"],
        "wave_count": manifest["schedule"]["wave_count"],
        "manifest_sha256": record["sha256"],
        "out_dir": os.path.abspath(args.out_dir),
    }).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
