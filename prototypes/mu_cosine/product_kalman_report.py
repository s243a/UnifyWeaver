#!/usr/bin/env python3
"""Build a Markdown report from Product-Kalman evaluation artifacts.

The report is deliberately descriptive. It records the input manifest and score
summary in a human-readable form, but it does not encode a decision rule or
claim that Product-Kalman has won.
"""

import argparse
import json
import sys


__all__ = [
    "build_product_kalman_markdown_report",
    "load_json",
    "write_markdown_report",
]


def load_json(path):
    """Load a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
        "## Calibration Settings",
        "",
        _markdown_table(["item", "value"], _calibration_rows(scores_json)),
        "",
        "## Guardrails",
        "",
        "- Positive NLL gain means the candidate had lower held-out mean NLL than the named baseline.",
        "- For a well-scaled d-dimensional Gaussian prediction, mean squared Mahalanobis should be near d; the per-dimension value should be near 1, with tail quantiles read as empirical diagnostics rather than a decision rule.",
        "- Treat this as a held-out comparison artifact, not as a training-objective decision.",
        "- Product-Kalman should be compared against the registered joint-posterior and Sigma-conditioned baselines before promotion.",
        "- Calibration rows and evaluation rows must remain disjoint; any ID overlap or duplicate count above zero should block interpretation.",
        "",
    ])
    return "\n".join(lines)


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
    ap.add_argument("--title", default="Product-Kalman Holdout Report")
    return ap


def main(argv=None):
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    scores = load_json(args.scores_json)
    manifest = load_json(args.input_manifest) if args.input_manifest else None
    split_manifest = load_json(args.split_manifest) if args.split_manifest else None
    text = build_product_kalman_markdown_report(scores, manifest, split_manifest=split_manifest, title=args.title)
    if args.output_md:
        write_markdown_report(args.output_md, text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
