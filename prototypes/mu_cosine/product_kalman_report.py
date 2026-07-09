#!/usr/bin/env python3
"""Build a Markdown report from Product-Kalman evaluation artifacts.

The report is deliberately descriptive. It records the input manifest and score
summary in a human-readable form, but it does not encode a decision rule or
claim that Product-Kalman has won.
"""

import argparse
import hashlib
import json
import math
import sys

import numpy as np

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
    "add_artifact_validation_summary",
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


def _baseline_from_improvement_key(key):
    prefix = "nll_improvement_vs_"
    if key.startswith(prefix) and not key.startswith("nll_improvement_bootstrap_vs_"):
        return key[len(prefix):]
    return None


def _baseline_from_bootstrap_key(key):
    prefix = "nll_improvement_bootstrap_vs_"
    return key[len(prefix):] if key.startswith(prefix) else None


def _improvement_rows(scores_json):
    rows = []
    for baseline_key in sorted(scores_json):
        label = _baseline_from_improvement_key(baseline_key)
        if label is None:
            continue
        for candidate, gain in sorted(scores_json.get(baseline_key, {}).items()):
            rows.append([label, candidate, gain])
    return rows


def _bootstrap_rows(scores_json):
    rows = []
    for baseline_key in sorted(scores_json):
        label = _baseline_from_bootstrap_key(baseline_key)
        if label is None:
            continue
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
        ["pit_diagnostics_validated", artifact.get("pit_diagnostics_validated")],
        ["score_order", ", ".join(artifact.get("score_order", []))],
        ["n_boot", artifact.get("n_boot")],
        ["seed", artifact.get("seed")],
        ["confidence", artifact.get("confidence")],
        ["method", artifact.get("method")],
        ["baselines", ", ".join(artifact.get("baselines", []))],
    ]


def _artifact_validation_rows(scores_json):
    artifact = scores_json.get("evaluation_artifact_validation", {})
    if not artifact:
        return []
    return [
        ["evaluation_npz", artifact.get("evaluation_npz")],
        ["evaluation_npz_sha256", artifact.get("evaluation_npz_sha256")],
        ["validated_against_scores", artifact.get("validated_against_scores")],
        ["pit_diagnostics_validated", artifact.get("pit_diagnostics_validated")],
        ["score_order", ", ".join(artifact.get("score_order", []))],
        ["score_count", artifact.get("score_count")],
        ["score_rows", ", ".join(f"{name}:{n}" for name, n in artifact.get("score_rows", {}).items())],
        ["pit_names", ", ".join(artifact.get("pit_names", []))],
        ["pit_n", artifact.get("pit_n")],
        ["pit_dimension", artifact.get("pit_dimension")],
    ]


def _finite_float(name, value):
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _bounded_float(name, value, low=0.0, high=1.0):
    out = _finite_float(name, value)
    if out < low or out > high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return out


def _positive_int(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _sequence(name, value):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return list(value)


def _pit_diagnostic_items(scores_json):
    diagnostics = scores_json.get("pit_diagnostics", {})
    if not diagnostics:
        return []
    if not isinstance(diagnostics, dict):
        raise ValueError("pit_diagnostics must be a mapping")
    out = []
    for model, item in sorted(diagnostics.items()):
        if not isinstance(item, dict):
            raise ValueError("pit_diagnostics entries must be mappings")
        prefix = f"pit_diagnostics[{model!r}]"
        n = _positive_int(f"{prefix}.n", item.get("n"))
        dimension = _positive_int(f"{prefix}.dimension", item.get("dimension"))
        channel_names = _sequence(f"{prefix}.channel_names", item.get("channel_names", []))
        if channel_names and len(channel_names) != dimension:
            raise ValueError(f"{prefix}.channel_names length must match dimension")
        out.append((model, item, prefix, n, dimension, channel_names))
    return out


def _pit_channel_pairs(prefix, channel_ks, dimension, channel_names):
    if isinstance(channel_ks, dict):
        pairs = sorted(channel_ks.items())
        if len(pairs) != dimension:
            raise ValueError(f"{prefix}.channel_ks length must match dimension")
        return [(channel, _bounded_float(f"{prefix}.channel_ks[{channel!r}]", value)) for channel, value in pairs]
    values = _sequence(f"{prefix}.channel_ks", channel_ks)
    if len(values) != dimension:
        raise ValueError(f"{prefix}.channel_ks length must match dimension")
    pairs = []
    for idx, value in enumerate(values):
        channel = channel_names[idx] if channel_names else idx
        pairs.append((channel, _bounded_float(f"{prefix}.channel_ks[{idx}]", value)))
    return pairs


def _pit_rows(scores_json):
    rows = []
    for model, item, prefix, n, dimension, channel_names in _pit_diagnostic_items(scores_json):
        for channel, ks in _pit_channel_pairs(prefix, item.get("channel_ks", []), dimension, channel_names):
            rows.append([model, channel, ks, n, dimension, item.get("method")])
    return rows


def _pit_coverage_rows(scores_json):
    rows = []
    for model, item, prefix, n, dimension, channel_names in _pit_diagnostic_items(scores_json):
        coverage_keys = {"coverage_levels", "central_interval_coverage", "central_interval_error"}
        present = coverage_keys & set(item)
        if not present:
            continue
        if present != coverage_keys:
            missing = sorted(coverage_keys - present)
            raise ValueError(f"{prefix} missing PIT coverage fields: {missing!r}")
        levels = _sequence(f"{prefix}.coverage_levels", item["coverage_levels"])
        coverage = _sequence(f"{prefix}.central_interval_coverage", item["central_interval_coverage"])
        errors = _sequence(f"{prefix}.central_interval_error", item["central_interval_error"])
        if not levels:
            raise ValueError(f"{prefix}.coverage_levels must be nonempty")
        if len(coverage) != len(levels):
            raise ValueError(f"{prefix}.central_interval_coverage length must match coverage_levels")
        if len(errors) != len(levels):
            raise ValueError(f"{prefix}.central_interval_error length must match coverage_levels")
        seen_levels = set()
        for level_idx, raw_level in enumerate(levels):
            level = _bounded_float(f"{prefix}.coverage_levels[{level_idx}]", raw_level, low=0.0, high=1.0)
            if level <= 0.0 or level >= 1.0:
                raise ValueError(f"{prefix}.coverage_levels values must lie strictly between 0 and 1")
            if level in seen_levels:
                raise ValueError(f"{prefix}.coverage_levels values must be unique")
            seen_levels.add(level)
            row_coverage = _sequence(f"{prefix}.central_interval_coverage[{level_idx}]", coverage[level_idx])
            row_errors = _sequence(f"{prefix}.central_interval_error[{level_idx}]", errors[level_idx])
            if len(row_coverage) != dimension:
                raise ValueError(f"{prefix}.central_interval_coverage row length must match dimension")
            if len(row_errors) != dimension:
                raise ValueError(f"{prefix}.central_interval_error row length must match dimension")
            for channel_idx, raw_observed in enumerate(row_coverage):
                channel = channel_names[channel_idx] if channel_names else channel_idx
                observed = _bounded_float(
                    f"{prefix}.central_interval_coverage[{level_idx}][{channel_idx}]",
                    raw_observed,
                )
                error = _finite_float(
                    f"{prefix}.central_interval_error[{level_idx}][{channel_idx}]",
                    row_errors[channel_idx],
                )
                if abs(error - (observed - level)) > 1e-8:
                    raise ValueError(f"{prefix}.central_interval_error must equal observed minus nominal")
                rows.append([model, channel, level, observed, error, n])
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


def _decode_npz_strings(values):
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("string array must be 1-D")
    out = []
    for value in arr.tolist():
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _artifact_pit_summary(artifact_npz):
    path = str(artifact_npz)
    with np.load(path, allow_pickle=False) as data:
        if "pit_names" not in data.files:
            return {"present": False}
        required_base = {"pit_names", "pit_values", "pit_channel_ks"}
        missing_base = sorted(required_base - set(data.files))
        if missing_base:
            raise ValueError(f"evaluation artifact missing PIT arrays: {missing_base!r}")
        names = _decode_npz_strings(data["pit_names"])
        pit_values = np.asarray(data["pit_values"], dtype=float)
        channel_ks = np.asarray(data["pit_channel_ks"], dtype=float)
        if pit_values.ndim != 3 or pit_values.shape[0] != len(names):
            raise ValueError("evaluation artifact pit_values shape must match pit_names")
        if not np.isfinite(pit_values).all() or ((pit_values < 0.0) | (pit_values > 1.0)).any():
            raise ValueError("evaluation artifact pit_values must be finite and in [0, 1]")
        if channel_ks.ndim != 2 or channel_ks.shape != (len(names), pit_values.shape[2]):
            raise ValueError("evaluation artifact pit_channel_ks shape must match pit_names and pit_values")
        if not np.isfinite(channel_ks).all() or ((channel_ks < 0.0) | (channel_ks > 1.0)).any():
            raise ValueError("evaluation artifact pit_channel_ks values must be finite and in [0, 1]")
        out = {
            "present": True,
            "names": names,
            "n": int(pit_values.shape[1]),
            "dimension": int(pit_values.shape[2]),
            "channel_ks": {name: channel_ks[idx].tolist() for idx, name in enumerate(names)},
        }
        coverage_present = {
            "pit_coverage_levels",
            "pit_central_interval_coverage",
            "pit_central_interval_error",
        } & set(data.files)
        if coverage_present:
            required = {"pit_coverage_levels", "pit_central_interval_coverage", "pit_central_interval_error"}
            if coverage_present != required:
                missing = sorted(required - coverage_present)
                raise ValueError(f"evaluation artifact missing PIT coverage arrays: {missing!r}")
            levels = np.asarray(data["pit_coverage_levels"], dtype=float)
            coverage = np.asarray(data["pit_central_interval_coverage"], dtype=float)
            errors = np.asarray(data["pit_central_interval_error"], dtype=float)
            if levels.ndim != 1 or levels.size == 0:
                raise ValueError("evaluation artifact pit_coverage_levels must be a nonempty 1-D array")
            if not np.isfinite(levels).all() or ((levels <= 0.0) | (levels >= 1.0)).any():
                raise ValueError("evaluation artifact pit_coverage_levels values must lie strictly between 0 and 1")
            expected_shape = (len(names), levels.size, pit_values.shape[2])
            if coverage.shape != expected_shape:
                raise ValueError("evaluation artifact pit_central_interval_coverage shape is inconsistent")
            if errors.shape != expected_shape:
                raise ValueError("evaluation artifact pit_central_interval_error shape is inconsistent")
            if not np.isfinite(coverage).all() or ((coverage < 0.0) | (coverage > 1.0)).any():
                raise ValueError("evaluation artifact PIT coverage values must be finite and in [0, 1]")
            if not np.isfinite(errors).all():
                raise ValueError("evaluation artifact PIT coverage errors must be finite")
            out.update({
                "coverage_levels": levels.tolist(),
                "central_interval_coverage": {name: coverage[idx].tolist() for idx, name in enumerate(names)},
                "central_interval_error": {name: errors[idx].tolist() for idx, name in enumerate(names)},
            })
        return out


def _pit_channel_ks_vector(prefix, item, dimension, channel_names):
    channel_ks = item.get("channel_ks", [])
    if isinstance(channel_ks, dict):
        if not channel_names:
            raise ValueError(f"{prefix}.channel_names are required when channel_ks is a mapping")
        missing = [name for name in channel_names if name not in channel_ks]
        extra = sorted(set(channel_ks) - set(channel_names))
        if missing or extra:
            raise ValueError(f"{prefix}.channel_ks keys must match channel_names")
        return [_bounded_float(f"{prefix}.channel_ks[{name!r}]", channel_ks[name]) for name in channel_names]
    values = _sequence(f"{prefix}.channel_ks", channel_ks)
    if len(values) != dimension:
        raise ValueError(f"{prefix}.channel_ks length must match dimension")
    return [_bounded_float(f"{prefix}.channel_ks[{idx}]", value) for idx, value in enumerate(values)]


def _artifact_pit_expected_names(scores_json, diagnostics):
    ordered = [name for name in scores_json.get("score_order", []) if name in diagnostics]
    if ordered:
        missing = sorted(set(diagnostics) - set(ordered))
        if missing:
            raise ValueError(f"score_order is missing PIT diagnostic names: {missing!r}")
        return ordered
    return sorted(diagnostics)


def _validate_artifact_pit_consistency(scores_json, artifact_path, atol=1e-10):
    artifact = _artifact_pit_summary(artifact_path)
    diagnostics = scores_json.get("pit_diagnostics", {})
    if not diagnostics:
        if artifact.get("present"):
            raise ValueError("evaluation artifact has PIT diagnostics but scores JSON does not")
        return artifact
    if not isinstance(diagnostics, dict):
        raise ValueError("pit_diagnostics must be a mapping")
    if not artifact.get("present"):
        raise ValueError("scores JSON has PIT diagnostics but evaluation artifact does not")
    expected_names = _artifact_pit_expected_names(scores_json, diagnostics)
    if artifact["names"] != expected_names:
        raise ValueError("evaluation artifact pit_names do not match scores JSON PIT diagnostics")
    has_artifact_coverage = "coverage_levels" in artifact
    for model, item, prefix, n, dimension, channel_names in _pit_diagnostic_items(scores_json):
        if model not in expected_names:
            continue
        if n != artifact["n"]:
            raise ValueError(f"evaluation artifact PIT row count for {model!r} does not match scores JSON")
        if dimension != artifact["dimension"]:
            raise ValueError(f"evaluation artifact PIT dimension for {model!r} does not match scores JSON")
        json_ks = np.asarray(_pit_channel_ks_vector(prefix, item, dimension, channel_names), dtype=float)
        artifact_ks = np.asarray(artifact["channel_ks"][model], dtype=float)
        if artifact_ks.shape != json_ks.shape or not np.allclose(artifact_ks, json_ks, rtol=0.0, atol=atol):
            raise ValueError(f"evaluation artifact PIT channel_ks for {model!r} does not match scores JSON")
        coverage_keys = {"coverage_levels", "central_interval_coverage", "central_interval_error"}
        json_coverage_present = bool(coverage_keys & set(item))
        if has_artifact_coverage and not json_coverage_present:
            raise ValueError(f"evaluation artifact has PIT coverage for {model!r} but scores JSON does not")
        if json_coverage_present and not has_artifact_coverage:
            raise ValueError(f"scores JSON has PIT coverage for {model!r} but evaluation artifact does not")
        if not json_coverage_present:
            continue
        # Reuse report-table validation for per-diagnostic schema and error arithmetic.
        _pit_coverage_rows({"pit_diagnostics": {model: item}})
        json_levels = np.asarray(_sequence(f"{prefix}.coverage_levels", item["coverage_levels"]), dtype=float)
        artifact_levels = np.asarray(artifact["coverage_levels"], dtype=float)
        if json_levels.shape != artifact_levels.shape or not np.allclose(
            json_levels,
            artifact_levels,
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError(f"evaluation artifact PIT coverage levels for {model!r} do not match scores JSON")
        json_coverage = np.asarray(item["central_interval_coverage"], dtype=float)
        json_errors = np.asarray(item["central_interval_error"], dtype=float)
        artifact_coverage = np.asarray(artifact["central_interval_coverage"][model], dtype=float)
        artifact_errors = np.asarray(artifact["central_interval_error"][model], dtype=float)
        if json_coverage.shape != artifact_coverage.shape or not np.allclose(
            json_coverage,
            artifact_coverage,
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError(f"evaluation artifact PIT coverage for {model!r} does not match scores JSON")
        if json_errors.shape != artifact_errors.shape or not np.allclose(
            json_errors,
            artifact_errors,
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError(f"evaluation artifact PIT coverage error for {model!r} does not match scores JSON")
    return artifact


def validate_artifact_score_consistency(scores_json, evaluation_npz=None, atol=1e-10):
    """Raise if a row-level evaluation artifact does not match the score JSON."""
    artifact_path = _artifact_path(scores_json, evaluation_npz=evaluation_npz)
    artifact = evaluation_npz_score_summary(artifact_path)
    pit_artifact = _validate_artifact_pit_consistency(scores_json, artifact_path, atol=atol)
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
    artifact["pit_diagnostics"] = pit_artifact
    return artifact


def _artifact_validation_metadata(artifact_path, artifact_summary):
    pit = artifact_summary.get("pit_diagnostics", {})
    metadata = {
        "evaluation_npz": str(artifact_path),
        "evaluation_npz_sha256": _sha256_file(artifact_path),
        "validated_against_scores": True,
        "pit_diagnostics_validated": bool(pit.get("present")),
        "score_order": list(artifact_summary["score_order"]),
        "score_count": len(artifact_summary["score_order"]),
        "score_rows": dict(artifact_summary["n"]),
    }
    if pit.get("present"):
        metadata.update({
            "pit_names": list(pit.get("names", [])),
            "pit_n": pit.get("n"),
            "pit_dimension": pit.get("dimension"),
        })
    return metadata


def add_artifact_validation_summary(scores_json, evaluation_npz=None, overwrite=False):
    """Return `scores_json` enriched with validation metadata for a row artifact."""
    artifact_path = _artifact_path(scores_json, evaluation_npz=evaluation_npz)
    artifact_summary = validate_artifact_score_consistency(scores_json, evaluation_npz=artifact_path)
    out = dict(scores_json)
    if overwrite or "evaluation_artifact_validation" not in out:
        out["evaluation_artifact_validation"] = _artifact_validation_metadata(artifact_path, artifact_summary)
    return out


def _parse_name_list(text):
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    if not names:
        raise ValueError("name list must contain at least one name")
    if len(names) != len(set(names)):
        raise ValueError(f"name list contains duplicates: {text!r}")
    return tuple(names)


def _normalize_baselines(baselines):
    if baselines is None:
        baselines = ("prior", "independent_kalman")
    if isinstance(baselines, str):
        return _parse_name_list(baselines)
    names = tuple(str(name).strip() for name in baselines if str(name).strip())
    if not names:
        raise ValueError("baselines must contain at least one score name")
    if len(names) != len(set(names)):
        raise ValueError(f"baselines contains duplicates: {names!r}")
    return names


def add_artifact_bootstrap_intervals(
    scores_json,
    evaluation_npz=None,
    n_boot=0,
    seed=0,
    confidence=0.95,
    baselines=("prior", "independent_kalman"),
    overwrite=False,
    validate_artifact=True,
):
    """Return `scores_json` enriched with post-hoc bootstrap intervals from row artifacts."""
    if not n_boot:
        return scores_json
    baselines = _normalize_baselines(baselines)
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
        baselines=baselines,
    ).items():
        if overwrite or key not in out:
            out[key] = value
            changed = True
    if changed:
        out["bootstrap_artifact"] = {
            "evaluation_npz": str(artifact_path),
            "evaluation_npz_sha256": _sha256_file(artifact_path),
            "validated_against_scores": bool(validate_artifact),
            "pit_diagnostics_validated": bool(artifact_summary.get("pit_diagnostics", {}).get("present")),
            "score_order": artifact_summary["score_order"],
            "n_boot": int(n_boot),
            "seed": int(seed),
            "confidence": float(confidence),
            "method": "paired_row_resample",
            "baselines": list(baselines),
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
    artifact_validation_rows = _artifact_validation_rows(scores_json)
    pit_rows = _pit_rows(scores_json)
    pit_coverage_rows = _pit_coverage_rows(scores_json)
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
    if pit_rows:
        lines.extend([
            "## PIT Diagnostics",
            "",
            _markdown_table(["model", "channel", "ks_uniform", "n", "dimension", "method"], pit_rows),
            "",
        ])
    if pit_coverage_rows:
        lines.extend([
            "## PIT Central Coverage",
            "",
            _markdown_table(["model", "channel", "nominal", "observed", "error", "n"], pit_coverage_rows),
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
    if artifact_validation_rows:
        lines.extend([
            "## Evaluation Artifact Validation",
            "",
            _markdown_table(["item", "value"], artifact_validation_rows),
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
        "- PIT diagnostics, when present, summarize marginal CDF calibration; lower KS-vs-uniform and smaller absolute central-coverage error are better, but they are still exploratory shape diagnostics.",
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
    ap.add_argument(
        "--evaluation-npz",
        help="optional row-level evaluation artifact NPZ for validation and post-hoc bootstrap intervals",
    )
    ap.add_argument("--bootstrap-nll", type=int, default=0, help="paired bootstrap replicates for NLL gains; 0 disables")
    ap.add_argument(
        "--bootstrap-baselines",
        default="prior,independent_kalman",
        help="comma-separated score names used as post-hoc bootstrap baselines",
    )
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
        if args.evaluation_npz:
            scores = add_artifact_validation_summary(
                scores,
                evaluation_npz=args.evaluation_npz,
                overwrite=True,
            )
        scores = add_artifact_bootstrap_intervals(
            scores,
            evaluation_npz=args.evaluation_npz,
            n_boot=args.bootstrap_nll,
            seed=args.bootstrap_seed,
            confidence=args.bootstrap_confidence,
            baselines=_parse_name_list(args.bootstrap_baselines),
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
