#!/usr/bin/env python3
"""Build a Markdown report from Product-Kalman evaluation artifacts.

The report is deliberately descriptive. It records the input manifest and score
summary in a human-readable form, but it does not encode a decision rule or
claim that Product-Kalman has won.
"""

import argparse
import hashlib
import json
import sys

try:
    from .product_kalman_evaluation import (
        bootstrap_nll_improvements_from_evaluation_npz,
        evaluation_npz_score_summary,
    )
except ImportError:  # direct script execution from prototypes/mu_cosine
    from product_kalman_evaluation import (
        bootstrap_nll_improvements_from_evaluation_npz,
        evaluation_npz_score_summary,
    )


__all__ = [
    "add_artifact_bootstrap_intervals",
    "build_product_kalman_markdown_report",
    "load_json",
    "validate_artifact_score_consistency",
    "write_json",
    "write_markdown_report",
]


def load_json(path):
    """Load a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(headers, rows):
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(out)


def _score_rows(scores_json):
    scores = scores_json.get("scores", {})
    order = scores_json.get("score_order") or sorted(scores)
    rows = []
    for name in order:
        score = scores[name]
        rows.append([
            name,
            score.get("mean_nll"),
            score.get("mse"),
            score.get("n"),
            score.get("covariance_trace"),
            score.get("mean_squared_mahalanobis"),
            score.get("mahalanobis_per_dim"),
            score.get("squared_mahalanobis_q50"),
            score.get("squared_mahalanobis_q90"),
            score.get("squared_mahalanobis_q95"),
        ])
    return rows


def _improvement_rows(scores_json):
    rows = []
    for baseline_key, label in (
        ("nll_improvement_vs_prior", "prior"),
        ("nll_improvement_vs_independent_kalman", "independent_kalman"),
    ):
        for candidate, gain in sorted(scores_json.get(baseline_key, {}).items()):
            rows.append([label, candidate, gain])
    return rows


def _bootstrap_rows(scores_json):
    rows = []
    for baseline_key, label in (
        ("nll_improvement_bootstrap_vs_prior", "prior"),
        ("nll_improvement_bootstrap_vs_independent_kalman", "independent_kalman"),
    ):
        for candidate, item in sorted(scores_json.get(baseline_key, {}).items()):
            rows.append([
                label,
                candidate,
                item.get("observed_mean_gain"),
                item.get("bootstrap_mean_gain"),
                item.get("ci_low"),
                item.get("ci_high"),
                item.get("confidence"),
                item.get("n_boot"),
                item.get("seed"),
                item.get("n"),
            ])
    return rows


def _bootstrap_artifact_rows(scores_json):
    artifact = scores_json.get("bootstrap_artifact", {})
    if not artifact:
        return []
    return [
        ["evaluation_npz", artifact.get("evaluation_npz")],
        ["evaluation_npz_sha256", artifact.get("evaluation_npz_sha256")],
        ["validated_against_scores", artifact.get("validated_against_scores")],
        ["score_order", ", ".join(artifact.get("score_order", []))],
        ["n_boot", artifact.get("n_boot")],
        ["seed", artifact.get("seed")],
        ["confidence", artifact.get("confidence")],
        ["method", artifact.get("method")],
    ]


def _input_rows(scores_json, manifest):
    inputs = scores_json.get("inputs", {})
    rows = [
        ["source_table", inputs.get("source_table")],
        ["original_table", inputs.get("original_table")],
        ["split_manifest", inputs.get("split_manifest")],
        ["input_npz", inputs.get("input_npz")],
        ["input_manifest", inputs.get("input_manifest")],
        ["evaluation_npz", inputs.get("evaluation_npz")],
    ]
    if manifest:
        source = manifest.get("source_table", {})
        rows.append(["source_table_sha256", source.get("sha256")])
        rows.append(["delimiter", repr(source.get("delimiter"))])
    return rows


def _manifest_rows(manifest):
    if not manifest:
        return []
    splits = manifest.get("splits", {})
    dims = manifest.get("dimensions", {})
    ids = manifest.get("ids", {})
    H = manifest.get("H", {})
    return [
        ["calibration_rows", splits.get("calibration_rows")],
        ["evaluation_rows", splits.get("evaluation_rows")],
        ["state_dim", dims.get("state_dim")],
        ["prior_dim", dims.get("prior_dim")],
        ["observation_dim", dims.get("observation_dim")],
        ["ids_disjoint_and_unique", ids.get("disjoint_and_unique")],
        ["id_overlap_count", ids.get("overlap_count")],
        ["calibration_duplicate_count", ids.get("calibration_duplicate_count")],
        ["evaluation_duplicate_count", ids.get("evaluation_duplicate_count")],
        ["H_present", H.get("present")],
        ["H_shape", H.get("shape")],
    ]


def _column_rows(manifest):
    if not manifest:
        return []
    columns = manifest.get("columns", {})
    return [
        ["prior", ", ".join(columns.get("prior", []))],
        ["measurement", ", ".join(columns.get("measurement", []))],
        ["target_state", ", ".join(columns.get("target_state", []))],
    ]


def _calibration_rows(scores_json):
    calibration = scores_json.get("calibration", {})
    return [
        ["n_samples", calibration.get("n_samples")],
        ["state_dim", calibration.get("state_dim")],
        ["observation_dim", calibration.get("observation_dim")],
        ["ddof", calibration.get("ddof")],
        ["shrinkage", calibration.get("shrinkage")],
        ["shrinkage_target", calibration.get("shrinkage_target")],
    ]


def _grouped_covariance_rows(scores_json):
    rows = []
    for name, item in sorted(scores_json.get("grouped_covariances", {}).items()):
        counts = item.get("group_counts", {})
        rows.append([
            name,
            item.get("min_group_rows"),
            ", ".join(f"{label}:{count}" for label, count in sorted(counts.items())),
            item.get("row_covariance_shape"),
        ])
    return rows


def _artifact_path(scores_json, evaluation_npz=None):
    artifact_path = evaluation_npz or scores_json.get("inputs", {}).get("evaluation_npz")
    if not artifact_path:
        raise ValueError(
            "--evaluation-npz is required when --bootstrap-nll is set "
            "and scores JSON has no evaluation_npz input"
        )
    return artifact_path


def validate_artifact_score_consistency(scores_json, evaluation_npz=None, atol=1e-10):
    """Raise if a row-level evaluation artifact does not match the score JSON."""
    artifact_path = _artifact_path(scores_json, evaluation_npz=evaluation_npz)
    artifact = evaluation_npz_score_summary(artifact_path)
    scores = scores_json.get("scores", {})
    expected_order = list(scores_json.get("score_order") or sorted(scores))
    if artifact["score_order"] != expected_order:
        raise ValueError("evaluation artifact score_names do not match scores JSON score_order")
    for name in expected_order:
        if name not in scores:
            raise ValueError(f"scores JSON is missing score {name!r}")
        score = scores[name]
        if "mean_nll" in score:
            observed = float(score["mean_nll"])
            artifact_value = artifact["mean_nll"][name]
            if abs(observed - artifact_value) > atol:
                raise ValueError(f"evaluation artifact mean_nll for {name!r} does not match scores JSON")
        if "n" in score and int(score["n"]) != artifact["n"][name]:
            raise ValueError(f"evaluation artifact n for {name!r} does not match scores JSON")
    return artifact


def add_artifact_bootstrap_intervals(
    scores_json,
    evaluation_npz=None,
    n_boot=0,
    seed=0,
    confidence=0.95,
    overwrite=False,
    validate_artifact=True,
):
    """Return `scores_json` enriched with post-hoc bootstrap intervals from row artifacts."""
    if not n_boot:
        return scores_json
    artifact_path = _artifact_path(scores_json, evaluation_npz=evaluation_npz)
    artifact_summary = evaluation_npz_score_summary(artifact_path)
    if validate_artifact:
        artifact_summary = validate_artifact_score_consistency(scores_json, evaluation_npz=artifact_path)
    out = dict(scores_json)
    changed = False
    for key, value in bootstrap_nll_improvements_from_evaluation_npz(
        artifact_path,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
    ).items():
        if overwrite or key not in out:
            out[key] = value
            changed = True
    if changed:
        out["bootstrap_artifact"] = {
            "evaluation_npz": str(artifact_path),
            "evaluation_npz_sha256": _sha256_file(artifact_path),
            "validated_against_scores": bool(validate_artifact),
            "score_order": artifact_summary["score_order"],
            "n_boot": int(n_boot),
            "seed": int(seed),
            "confidence": float(confidence),
            "method": "paired_row_resample",
        }
    return out


def _split_materialization_rows(split_manifest):
    if not split_manifest:
        return []
    source = split_manifest.get("source_table", {})
    output = split_manifest.get("output_table", {})
    split = split_manifest.get("split", {})
    return [
        ["source_table", source.get("path")],
        ["source_table_sha256", source.get("sha256")],
        ["output_table", output.get("path")],
        ["output_table_sha256", output.get("sha256")],
        ["seed", split.get("seed")],
        ["evaluation_unit_frac", split.get("evaluation_unit_frac")],
        ["unit_columns", ", ".join(split.get("unit_columns", []))],
        ["sampled_evaluation_unit_count", split.get("sampled_evaluation_unit_count")],
        ["observed_calibration_unit_count", split.get("observed_calibration_unit_count")],
        ["observed_evaluation_unit_count", split.get("observed_evaluation_unit_count")],
        ["observed_unit_overlap_count", split.get("observed_unit_overlap_count")],
        ["disjoint_observed_units", split.get("disjoint_observed_units")],
        ["calibration_rows", split.get("calibration_rows")],
        ["evaluation_rows", split.get("evaluation_rows")],
        ["omitted_crossing_rows", split.get("omitted_crossing_rows")],
    ]


def build_product_kalman_markdown_report(
    scores_json,
    input_manifest=None,
    split_manifest=None,
    title="Product-Kalman Holdout Report",
):
    """Return a descriptive Markdown report for one Product-Kalman evaluation."""
    lines = [
        f"# {title}",
        "",
        "This report is descriptive: it records held-out scores and provenance artifacts, but it does not encode a decision rule.",
        "",
        "## Inputs",
        "",
        _markdown_table(["item", "value"], _input_rows(scores_json, input_manifest)),
        "",
    ]
    if input_manifest:
        lines.extend([
            "## Split And Schema",
            "",
            _markdown_table(["item", "value"], _manifest_rows(input_manifest)),
            "",
            "## Column Groups",
            "",
            _markdown_table(["group", "columns"], _column_rows(input_manifest)),
            "",
        ])
    if split_manifest:
        lines.extend([
            "## Split Materialization",
            "",
            _markdown_table(["item", "value"], _split_materialization_rows(split_manifest)),
            "",
        ])
    bootstrap_rows = _bootstrap_rows(scores_json)
    bootstrap_artifact_rows = _bootstrap_artifact_rows(scores_json)
    grouped_covariance_rows = _grouped_covariance_rows(scores_json)
    if grouped_covariance_rows:
        lines.extend([
            "## Grouped Covariances",
            "",
            _markdown_table(
                ["score", "min_group_rows", "group_counts", "row_covariance_shape"],
                grouped_covariance_rows,
            ),
            "",
        ])
    lines.extend([
        "## Scores",
        "",
        _markdown_table(
            [
                "model",
                "mean_nll",
                "mse",
                "n",
                "covariance_trace",
                "mean_sq_mahalanobis",
                "mahalanobis_per_dim",
                "sq_mahalanobis_q50",
                "sq_mahalanobis_q90",
                "sq_mahalanobis_q95",
            ],
            _score_rows(scores_json),
        ),
        "",
        "## NLL Improvements",
        "",
        _markdown_table(["baseline", "candidate", "mean_nll_gain"], _improvement_rows(scores_json)),
        "",
    ])
    if bootstrap_rows:
        lines.extend([
            "## NLL Improvement Bootstrap Intervals",
            "",
            _markdown_table(
                [
                    "baseline",
                    "candidate",
                    "observed_gain",
                    "bootstrap_mean",
                    "ci_low",
                    "ci_high",
                    "confidence",
                    "n_boot",
                    "seed",
                    "n",
                ],
                bootstrap_rows,
            ),
            "",
        ])
    if bootstrap_artifact_rows:
        lines.extend([
            "## Bootstrap Artifact",
            "",
            _markdown_table(["item", "value"], bootstrap_artifact_rows),
            "",
        ])
    lines.extend([
        "## Calibration Settings",
        "",
        _markdown_table(["item", "value"], _calibration_rows(scores_json)),
        "",
        "## Guardrails",
        "",
        "- Positive NLL gain means the candidate had lower held-out mean NLL than the named baseline.",
        "- For a well-scaled d-dimensional Gaussian prediction, mean squared Mahalanobis should be near d; the per-dimension value should be near 1, with tail quantiles read as empirical diagnostics rather than a decision rule.",
        "- Bootstrap intervals, when present, are paired row-resampling diagnostics for NLL gains; they are not a preregistered decision rule.",
        "- Treat this as a held-out comparison artifact, not as a training-objective decision.",
        "- Product-Kalman should be compared against the registered joint-posterior and Sigma-conditioned baselines before promotion.",
        "- Calibration rows and evaluation rows must remain disjoint; any ID overlap or duplicate count above zero should block interpretation.",
        "",
    ])
    return "\n".join(lines)


def write_json(path, data, indent=2):
    """Write a JSON artifact with stable key ordering."""
    text = json.dumps(data, indent=None if indent == 0 else indent, sort_keys=True) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_markdown_report(path, text):
    """Write a Markdown report."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Build a Markdown report from Product-Kalman JSON artifacts.")
    ap.add_argument("scores_json", help="JSON score summary from product_kalman_evaluation/table_evaluation")
    ap.add_argument("--input-manifest", help="optional input manifest JSON from product_kalman_table_to_npz")
    ap.add_argument("--split-manifest", help="optional split manifest JSON from product_kalman_split_table")
    ap.add_argument("--output-md", help="write Markdown report here instead of stdout")
    ap.add_argument("--output-json", help="write enriched score JSON here")
    ap.add_argument("--title", default="Product-Kalman Holdout Report")
    ap.add_argument("--evaluation-npz", help="optional row-level evaluation artifact NPZ for post-hoc bootstrap intervals")
    ap.add_argument("--bootstrap-nll", type=int, default=0, help="paired bootstrap replicates for NLL gains; 0 disables")
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--bootstrap-confidence", type=float, default=0.95)
    ap.add_argument("--overwrite-bootstrap", action="store_true", help="replace existing bootstrap sections in scores JSON")
    ap.add_argument("--indent", type=int, default=2, help="JSON indentation for --output-json; use 0 for compact output")
    return ap


def main(argv=None):
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    scores = load_json(args.scores_json)
    try:
        scores = add_artifact_bootstrap_intervals(
            scores,
            evaluation_npz=args.evaluation_npz,
            n_boot=args.bootstrap_nll,
            seed=args.bootstrap_seed,
            confidence=args.bootstrap_confidence,
            overwrite=args.overwrite_bootstrap,
        )
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    manifest = load_json(args.input_manifest) if args.input_manifest else None
    split_manifest = load_json(args.split_manifest) if args.split_manifest else None
    text = build_product_kalman_markdown_report(scores, manifest, split_manifest=split_manifest, title=args.title)
    if args.output_json:
        write_json(args.output_json, scores, indent=args.indent)
    if args.output_md:
        write_markdown_report(args.output_md, text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
