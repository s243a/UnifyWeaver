#!/usr/bin/env python3
"""Holdout evaluation helpers for Product-Kalman calibration prototypes.

This module is the bridge between the covariance-fitting helpers and later real
corpus experiments. It fits Product-Kalman covariance blocks on a calibration
split, applies them to a separate evaluation split, and reports comparable
Gaussian NLL/MSE scores for the prior, a zero-cross-covariance update, and the
correlated Product-Kalman update.

The helpers are deliberately numpy-only and require caller-supplied split rows.
They do not sample data, train a model, or decide that a Product-Kalman variant
has won; they make the held-out comparison auditable.
"""

import argparse
from dataclasses import dataclass, field
import json
import math
import sys

import numpy as np

try:
    from .product_kalman import fit_residual_covariance, gaussian_nll, regularize_covariance
    from .product_kalman_calibration import (
        ProductKalmanCalibration,
        apply_product_kalman_calibration,
        assert_disjoint_ids,
        fit_product_kalman_calibration,
    )
except ImportError:  # direct script execution from prototypes/mu_cosine
    from product_kalman import fit_residual_covariance, gaussian_nll, regularize_covariance
    from product_kalman_calibration import (
        ProductKalmanCalibration,
        apply_product_kalman_calibration,
        assert_disjoint_ids,
        fit_product_kalman_calibration,
    )


DEFAULT_NLL_BASELINES = ("prior", "independent_kalman")


__all__ = [
    "GaussianScore",
    "GaussianScoreVectors",
    "GroupedGaussianScore",
    "PITDiagnostics",
    "GroupResidualCovariances",
    "ProductKalmanHoldoutEvaluation",
    "bootstrap_nll_improvements_from_evaluation_npz",
    "bootstrap_nll_improvements_from_score_rows",
    "evaluate_product_kalman_holdout",
    "evaluation_artifact_arrays",
    "fit_group_residual_covariances",
    "evaluation_npz_score_summary",
    "evaluation_to_json_dict",
    "paired_bootstrap_nll_improvement",
    "paired_bootstrap_nll_improvement_from_score_rows",
    "gaussian_marginal_pit_diagnostics",
    "gaussian_marginal_pit_values",
    "pit_diagnostics_from_values",
    "pit_uniform_ks_statistic",
    "row_covariances_from_groups",
    "run_product_kalman_holdout_npz",
    "score_gaussian_prediction_vectors_rowwise",
    "score_gaussian_prediction_vectors",
    "score_gaussian_predictions_grouped",
    "score_gaussian_predictions_rowwise",
    "score_gaussian_predictions",
    "write_evaluation_npz",
]


def _as_2d(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D row matrix; use shape (n, 1) for scalar channels")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _as_covariance(name, value, dim):
    cov = np.asarray(value, dtype=float)
    if cov.shape != (dim, dim):
        raise ValueError(f"{name} shape {cov.shape} must be ({dim}, {dim})")
    if not np.isfinite(cov).all():
        raise ValueError(f"{name} must be finite")
    return 0.5 * (cov + cov.T)


def _as_row_covariances(name, value, n, dim):
    covariances = np.asarray(value, dtype=float)
    if covariances.shape != (n, dim, dim):
        raise ValueError(f"{name} shape {covariances.shape} must be ({n}, {dim}, {dim})")
    if not np.isfinite(covariances).all():
        raise ValueError(f"{name} must be finite")
    return 0.5 * (covariances + np.swapaxes(covariances, 1, 2))


def _normalize_group_label(item):
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    elif isinstance(item, np.generic):
        item = item.item()
    if item is None:
        raise ValueError("group labels must not be None")
    if isinstance(item, float) and not np.isfinite(item):
        raise ValueError("group labels must be finite")
    try:
        hash(item)
    except TypeError as exc:
        raise ValueError("group labels must be hashable") from exc
    return item


def _as_group_labels(name, value, n=None):
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D label array")
    if n is not None and arr.size != n:
        raise ValueError(f"{name} length {arr.size} must match {n}")
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty")
    return tuple(_normalize_group_label(item) for item in arr.tolist())


def _readonly_covariance(value):
    arr = np.array(value, dtype=float, copy=True)
    arr.setflags(write=False)
    return arr


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _readonly_vector(name, value, n=None):
    arr = np.array(value, dtype=float, copy=True)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D row-score vector")
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if n is not None and arr.size != n:
        raise ValueError(f"{name} length {arr.size} must match {n}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class GroupResidualCovariances:
    """Residual covariances fitted by discrete calibration-group label.

    `fallback_covariance` is the global residual covariance. Groups with too few
    calibration rows are mapped to that fallback so later rowwise scoring does not
    silently estimate a tiny, unstable covariance block.
    """

    covariance_by_group: dict
    fallback_covariance: np.ndarray
    group_counts: dict
    min_group_rows: int

    def __post_init__(self):
        if not self.covariance_by_group:
            raise ValueError("covariance_by_group must be nonempty")
        fallback_arr = np.asarray(self.fallback_covariance, dtype=float)
        if fallback_arr.ndim != 2 or fallback_arr.shape[0] != fallback_arr.shape[1]:
            raise ValueError("fallback_covariance must be square")
        fallback = _readonly_covariance(_as_covariance("fallback_covariance", fallback_arr, fallback_arr.shape[0]))
        dim = fallback.shape[0]
        covariances = {}
        for label, cov in self.covariance_by_group.items():
            normalized = _normalize_group_label(label)
            if normalized in covariances:
                raise ValueError("covariance_by_group contains duplicate normalized labels")
            covariances[normalized] = _readonly_covariance(
                _as_covariance(f"covariance_by_group[{normalized!r}]", cov, dim)
            )
        counts = {}
        for label, count in self.group_counts.items():
            normalized = _normalize_group_label(label)
            if normalized in counts:
                raise ValueError("group_counts contains duplicate normalized labels")
            counts[normalized] = _positive_int(f"group_counts[{normalized!r}]", count)
        missing_counts = set(covariances) - set(counts)
        if missing_counts:
            raise ValueError(f"group_counts is missing labels: {sorted(str(x) for x in missing_counts)}")
        extra_counts = set(counts) - set(covariances)
        if extra_counts:
            raise ValueError(f"group_counts has extra labels: {sorted(str(x) for x in extra_counts)}")
        min_group_rows = _positive_int("min_group_rows", self.min_group_rows)
        object.__setattr__(self, "covariance_by_group", covariances)
        object.__setattr__(self, "fallback_covariance", fallback)
        object.__setattr__(self, "group_counts", counts)
        object.__setattr__(self, "min_group_rows", min_group_rows)

    def row_covariances(self, groups):
        """Return `(n, d, d)` row covariances for evaluation rows in group order."""
        return row_covariances_from_groups(groups, self.covariance_by_group, self.fallback_covariance)


def row_covariances_from_groups(groups, covariance_by_group, fallback_covariance=None):
    """Expand a group->covariance map into one covariance matrix per row."""
    if not hasattr(covariance_by_group, "items"):
        raise ValueError("covariance_by_group must be a mapping")
    normalized = {}
    dim = None
    for label, cov in covariance_by_group.items():
        key = _normalize_group_label(label)
        if key in normalized:
            raise ValueError("covariance_by_group contains duplicate normalized labels")
        arr = np.asarray(cov, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"covariance_by_group[{key!r}] must be square")
        if dim is None:
            dim = arr.shape[0]
        normalized[key] = _as_covariance(f"covariance_by_group[{key!r}]", arr, dim)
    fallback = None
    if fallback_covariance is not None:
        fallback_arr = np.asarray(fallback_covariance, dtype=float)
        if fallback_arr.ndim != 2 or fallback_arr.shape[0] != fallback_arr.shape[1]:
            raise ValueError("fallback_covariance must be square")
        if dim is None:
            dim = fallback_arr.shape[0]
        fallback = _as_covariance("fallback_covariance", fallback_arr, dim)
    if dim is None:
        raise ValueError("at least one covariance or fallback_covariance is required")
    labels = _as_group_labels("groups", groups)
    out = np.empty((len(labels), dim, dim), dtype=float)
    for i, label in enumerate(labels):
        cov = normalized.get(label, fallback)
        if cov is None:
            raise ValueError(f"no covariance for group label {label!r} and no fallback_covariance provided")
        out[i] = cov
    out.setflags(write=False)
    return out


def fit_group_residual_covariances(
    predicted,
    observed,
    groups,
    min_group_rows=None,
    shrinkage=0.0,
    jitter=1e-9,
    ddof=1,
    shrinkage_target="diagonal",
):
    """Fit residual covariances by discrete group with a global fallback.

    This is a generic bridge for hop-conditioned or regime-conditioned error
    models: fit on calibration residuals only, then call `.row_covariances()` for
    the held-out row order before using rowwise Gaussian scoring.
    """
    pred = _as_2d("predicted", predicted)
    obs = _as_2d("observed", observed)
    if obs.shape != pred.shape:
        raise ValueError(f"observed shape {obs.shape} must match predicted shape {pred.shape}")
    labels = _as_group_labels("groups", groups, n=pred.shape[0])
    if not isinstance(ddof, (int, np.integer)) or int(ddof) < 0:
        raise ValueError("ddof must be a nonnegative integer")
    ddof = int(ddof)
    if min_group_rows is None:
        min_group_rows = max(ddof + 1, pred.shape[1] + 1)
    min_group_rows = _positive_int("min_group_rows", min_group_rows)
    if min_group_rows <= ddof:
        raise ValueError("min_group_rows must exceed ddof")
    residual = obs - pred
    fallback = fit_residual_covariance(
        residual,
        shrinkage=shrinkage,
        jitter=jitter,
        ddof=ddof,
        shrinkage_target=shrinkage_target,
    )
    indices_by_group = {}
    for i, label in enumerate(labels):
        indices_by_group.setdefault(label, []).append(i)
    covariances = {}
    group_counts = {}
    for label, indices in indices_by_group.items():
        group_counts[label] = len(indices)
        if len(indices) >= min_group_rows:
            covariances[label] = fit_residual_covariance(
                residual[indices],
                shrinkage=shrinkage,
                jitter=jitter,
                ddof=ddof,
                shrinkage_target=shrinkage_target,
            )
        else:
            covariances[label] = fallback
    return GroupResidualCovariances(
        covariance_by_group=covariances,
        fallback_covariance=fallback,
        group_counts=group_counts,
        min_group_rows=min_group_rows,
    )


@dataclass(frozen=True)
class GaussianScore:
    """Scalar summary for one Gaussian prediction family on one held-out split."""

    name: str
    mean_nll: float
    mse: float
    n: int
    covariance_trace: float
    mean_squared_mahalanobis: float
    mahalanobis_per_dim: float
    squared_mahalanobis_q50: float
    squared_mahalanobis_q90: float
    squared_mahalanobis_q95: float

    def __post_init__(self):
        name = str(self.name)
        n = int(self.n)
        mean_nll = float(self.mean_nll)
        mse = float(self.mse)
        covariance_trace = float(self.covariance_trace)
        mean_squared_mahalanobis = float(self.mean_squared_mahalanobis)
        mahalanobis_per_dim = float(self.mahalanobis_per_dim)
        squared_mahalanobis_q50 = float(self.squared_mahalanobis_q50)
        squared_mahalanobis_q90 = float(self.squared_mahalanobis_q90)
        squared_mahalanobis_q95 = float(self.squared_mahalanobis_q95)
        if not name:
            raise ValueError("score name must be nonempty")
        if n <= 0:
            raise ValueError("score n must be positive")
        for field, value in (
            ("mean_nll", mean_nll),
            ("mse", mse),
            ("covariance_trace", covariance_trace),
            ("mean_squared_mahalanobis", mean_squared_mahalanobis),
            ("mahalanobis_per_dim", mahalanobis_per_dim),
            ("squared_mahalanobis_q50", squared_mahalanobis_q50),
            ("squared_mahalanobis_q90", squared_mahalanobis_q90),
            ("squared_mahalanobis_q95", squared_mahalanobis_q95),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if any(
            value < 0.0
            for value in (
                mean_squared_mahalanobis,
                mahalanobis_per_dim,
                squared_mahalanobis_q50,
                squared_mahalanobis_q90,
                squared_mahalanobis_q95,
            )
        ):
            raise ValueError("Mahalanobis diagnostics must be nonnegative")
        if not squared_mahalanobis_q50 <= squared_mahalanobis_q90 <= squared_mahalanobis_q95:
            raise ValueError("Mahalanobis quantiles must be ordered")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "mean_nll", mean_nll)
        object.__setattr__(self, "mse", mse)
        object.__setattr__(self, "covariance_trace", covariance_trace)
        object.__setattr__(self, "mean_squared_mahalanobis", mean_squared_mahalanobis)
        object.__setattr__(self, "mahalanobis_per_dim", mahalanobis_per_dim)
        object.__setattr__(self, "squared_mahalanobis_q50", squared_mahalanobis_q50)
        object.__setattr__(self, "squared_mahalanobis_q90", squared_mahalanobis_q90)
        object.__setattr__(self, "squared_mahalanobis_q95", squared_mahalanobis_q95)


@dataclass(frozen=True)
class GaussianScoreVectors:
    """Per-row Gaussian score diagnostics for one prediction family."""

    name: str
    nll: np.ndarray
    squared_error: np.ndarray
    squared_mahalanobis: np.ndarray
    dimension: int

    def __post_init__(self):
        name = str(self.name)
        if not name:
            raise ValueError("score-vector name must be nonempty")
        nll = _readonly_vector("nll", self.nll)
        squared_error = _readonly_vector("squared_error", self.squared_error, n=nll.size)
        squared_mahalanobis = _readonly_vector(
            "squared_mahalanobis",
            self.squared_mahalanobis,
            n=nll.size,
        )
        dimension = int(self.dimension)
        if dimension <= 0:
            raise ValueError("score-vector dimension must be positive")
        if (squared_error < 0.0).any() or (squared_mahalanobis < 0.0).any():
            raise ValueError("row score diagnostics must be nonnegative except nll")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "nll", nll)
        object.__setattr__(self, "squared_error", squared_error)
        object.__setattr__(self, "squared_mahalanobis", squared_mahalanobis)
        object.__setattr__(self, "dimension", dimension)

    @property
    def n(self):
        return int(self.nll.size)


@dataclass(frozen=True)
class PITDiagnostics:
    """Per-channel probability integral transform diagnostics.

    `pit` must contain CDF values in `[0, 1]` with shape `(n, d)`. A calibrated
    predictive marginal should have PIT values close to uniform; `channel_ks` is
    the one-sample Kolmogorov-Smirnov distance to `U(0, 1)` for each channel.
    """

    name: str
    pit: np.ndarray
    channel_names: tuple = ()
    channel_ks: np.ndarray = field(init=False)

    def __post_init__(self):
        name = str(self.name)
        if not name:
            raise ValueError("PIT diagnostic name must be nonempty")
        pit = _as_2d("pit", self.pit)
        if ((pit < 0.0) | (pit > 1.0)).any():
            raise ValueError("pit values must be in [0, 1]")
        pit = np.array(pit, dtype=float, copy=True)
        pit.setflags(write=False)
        channel_names = tuple(str(item) for item in self.channel_names)
        if channel_names and len(channel_names) != pit.shape[1]:
            raise ValueError("channel_names length must match pit dimension")
        channel_ks = np.array([pit_uniform_ks_statistic(pit[:, j]) for j in range(pit.shape[1])], dtype=float)
        channel_ks.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "pit", pit)
        object.__setattr__(self, "channel_names", channel_names)
        object.__setattr__(self, "channel_ks", channel_ks)

    @property
    def n(self):
        return int(self.pit.shape[0])

    @property
    def dimension(self):
        return int(self.pit.shape[1])

    def to_json_dict(self):
        return {
            "n": self.n,
            "dimension": self.dimension,
            "channel_names": list(self.channel_names),
            "channel_ks": [float(value) for value in self.channel_ks],
            "method": "marginal_pit_uniform_ks",
        }


def _normal_cdf(value):
    arr = np.asarray(value, dtype=float)
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf(arr / math.sqrt(2.0)))


def pit_uniform_ks_statistic(values):
    """Return the one-sample KS distance between PIT values and `U(0, 1)`."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("PIT values must be a nonempty 1-D array")
    if not np.isfinite(arr).all():
        raise ValueError("PIT values must be finite")
    if ((arr < 0.0) | (arr > 1.0)).any():
        raise ValueError("PIT values must be in [0, 1]")
    x = np.sort(arr)
    n = float(x.size)
    empirical_hi = np.arange(1, x.size + 1, dtype=float) / n
    empirical_lo = np.arange(0, x.size, dtype=float) / n
    return float(max(np.max(empirical_hi - x), np.max(x - empirical_lo)))


def pit_diagnostics_from_values(name, pit, channel_names=()):
    """Build JSON-ready PIT calibration diagnostics from precomputed PIT values.

    This is the generic path for mixtures: callers can compute mixture CDF values
    row by row, then use this helper for the common validation and KS summary.
    """
    return PITDiagnostics(name=name, pit=pit, channel_names=tuple(channel_names))


def _covariance_diagonal_rows(covariance, n, dim, jitter=0.0):
    jitter = float(jitter)
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be finite and nonnegative")
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim == 2:
        diag = np.diag(_as_covariance("covariance", cov, dim))[None, :]
        diag = np.repeat(diag, n, axis=0)
    elif cov.ndim == 3:
        diag = np.diagonal(_as_row_covariances("covariances", cov, n, dim), axis1=1, axis2=2)
    else:
        raise ValueError("covariance must be a covariance matrix or row-covariance array")
    diag = np.array(diag, dtype=float, copy=True)
    if jitter:
        diag += jitter
    if (diag <= 0.0).any():
        raise ValueError("marginal covariance variances must be positive")
    return diag


def gaussian_marginal_pit_values(target_state, mean, covariance, jitter=0.0):
    """Return per-row, per-channel PIT values for Gaussian predictive marginals."""
    target = _as_2d("target_state", target_state)
    pred = _as_2d("mean", mean)
    if pred.shape != target.shape:
        raise ValueError(f"mean shape {pred.shape} must match target_state shape {target.shape}")
    variances = _covariance_diagonal_rows(covariance, target.shape[0], target.shape[1], jitter=jitter)
    z = (target - pred) / np.sqrt(variances)
    pit = _normal_cdf(z)
    pit.setflags(write=False)
    return pit


def gaussian_marginal_pit_diagnostics(name, target_state, mean, covariance, jitter=0.0, channel_names=()):
    """Build PIT/KS diagnostics for Gaussian predictive marginals."""
    pit = gaussian_marginal_pit_values(target_state, mean, covariance, jitter=jitter)
    return pit_diagnostics_from_values(name, pit, channel_names=channel_names)


@dataclass(frozen=True)
class GroupedGaussianScore:
    """Gaussian score plus grouped residual-covariance provenance.

    This bundles the fitted calibration-side group covariance map with the
    evaluation row covariances actually used by rowwise scoring.
    """

    score: GaussianScore
    score_vectors: GaussianScoreVectors
    residual_covariances: GroupResidualCovariances
    row_covariances: np.ndarray

    def __post_init__(self):
        if not isinstance(self.score, GaussianScore):
            raise ValueError("score must be a GaussianScore")
        if not isinstance(self.score_vectors, GaussianScoreVectors):
            raise ValueError("score_vectors must be a GaussianScoreVectors")
        if not isinstance(self.residual_covariances, GroupResidualCovariances):
            raise ValueError("residual_covariances must be a GroupResidualCovariances")
        if self.score.name != self.score_vectors.name:
            raise ValueError("score and score_vectors names must match")
        if self.score.n != self.score_vectors.n:
            raise ValueError("score n must match score_vectors n")
        row_covariances = _as_row_covariances(
            "row_covariances",
            self.row_covariances,
            self.score_vectors.n,
            self.score_vectors.dimension,
        )
        row_covariances.setflags(write=False)
        object.__setattr__(self, "row_covariances", row_covariances)


@dataclass(frozen=True)
class ProductKalmanHoldoutEvaluation:
    """One calibration/evaluation split comparison.

    `calibration` contains the fitted correlated covariance blocks. `independent`
    is the same fit with `cross_covariance=0`, used as the explicit no-correlation
    control. The two batch updates are retained so callers can inspect row-level
    posterior means after reading the aggregate scores.
    """

    calibration: ProductKalmanCalibration
    independent_calibration: ProductKalmanCalibration
    correlated_update: object
    independent_update: object
    scores: tuple
    score_vectors: tuple = ()
    grouped_scores: tuple = ()
    pit_diagnostics: tuple = ()

    def __post_init__(self):
        if not isinstance(self.calibration, ProductKalmanCalibration):
            raise ValueError("calibration must be a ProductKalmanCalibration")
        if not isinstance(self.independent_calibration, ProductKalmanCalibration):
            raise ValueError("independent_calibration must be a ProductKalmanCalibration")
        scores = tuple(self.scores)
        for score in scores:
            if not isinstance(score, GaussianScore):
                raise ValueError("scores must contain GaussianScore objects")
        names = [s.name for s in scores]
        if len(names) != len(set(names)):
            raise ValueError("score names must be unique")
        score_vectors = tuple(self.score_vectors)
        vector_names = []
        if score_vectors:
            for vector in score_vectors:
                if not isinstance(vector, GaussianScoreVectors):
                    raise ValueError("score_vectors must contain GaussianScoreVectors objects")
            vector_names = [v.name for v in score_vectors]
            if vector_names != names:
                raise ValueError("score_vectors must appear in the same order as scores")
            for score, vector in zip(scores, score_vectors):
                if vector.n != score.n:
                    raise ValueError("score vector length must match score n")
        grouped_scores = tuple(self.grouped_scores)
        if grouped_scores:
            grouped_names = []
            score_name_set = set(names)
            vector_name_set = set(vector_names)
            for grouped in grouped_scores:
                if not isinstance(grouped, GroupedGaussianScore):
                    raise ValueError("grouped_scores must contain GroupedGaussianScore objects")
                grouped_names.append(grouped.score.name)
                if grouped.score.name not in score_name_set:
                    raise ValueError("grouped score must also appear in scores")
                if grouped.score_vectors.name not in vector_name_set:
                    raise ValueError("grouped score vectors must also appear in score_vectors")
            if len(grouped_names) != len(set(grouped_names)):
                raise ValueError("grouped score names must be unique")
        pit_diagnostics = tuple(self.pit_diagnostics)
        if pit_diagnostics:
            score_by_name = {score.name: score for score in scores}
            vector_by_name = {vector.name: vector for vector in score_vectors}
            pit_names = []
            for diagnostic in pit_diagnostics:
                if not isinstance(diagnostic, PITDiagnostics):
                    raise ValueError("pit_diagnostics must contain PITDiagnostics objects")
                pit_names.append(diagnostic.name)
                if diagnostic.name not in score_by_name:
                    raise ValueError("PIT diagnostic name must also appear in scores")
                score = score_by_name[diagnostic.name]
                if diagnostic.n != score.n:
                    raise ValueError("PIT diagnostic row count must match score n")
                vector = vector_by_name.get(diagnostic.name)
                if vector is not None and diagnostic.dimension != vector.dimension:
                    raise ValueError("PIT diagnostic dimension must match score-vector dimension")
            if len(pit_names) != len(set(pit_names)):
                raise ValueError("PIT diagnostic names must be unique")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "score_vectors", score_vectors)
        object.__setattr__(self, "grouped_scores", grouped_scores)
        object.__setattr__(self, "pit_diagnostics", pit_diagnostics)

    def score(self, name):
        """Return the named `GaussianScore`."""
        for score in self.scores:
            if score.name == name:
                return score
        raise KeyError(name)

    def score_vector(self, name):
        """Return the named per-row score vectors."""
        for vector in self.score_vectors:
            if vector.name == name:
                return vector
        raise KeyError(name)

    def pit_diagnostic(self, name):
        """Return the named PIT calibration diagnostic."""
        for diagnostic in self.pit_diagnostics:
            if diagnostic.name == name:
                return diagnostic
        raise KeyError(name)

    def nll_improvement(self, baseline, candidate):
        """Positive means `candidate` has lower mean NLL than `baseline`."""
        return self.score(baseline).mean_nll - self.score(candidate).mean_nll


def _parse_name_list(text):
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    if not names:
        raise ValueError("name list must contain at least one name")
    if len(names) != len(set(names)):
        raise ValueError(f"name list contains duplicates: {text!r}")
    return tuple(names)


def _normalize_baselines(baselines):
    if baselines is None:
        baselines = DEFAULT_NLL_BASELINES
    if isinstance(baselines, str):
        return _parse_name_list(baselines)
    names = tuple(str(name).strip() for name in baselines if str(name).strip())
    if not names:
        raise ValueError("nll_baselines must contain at least one score name")
    if len(names) != len(set(names)):
        raise ValueError(f"nll_baselines contains duplicates: {names!r}")
    return names


def _bootstrap_settings(n_boot, seed, confidence):
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)):
        raise ValueError("n_boot must be a nonnegative integer")
    n_boot = int(n_boot)
    if n_boot < 0:
        raise ValueError("n_boot must be nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return n_boot, seed, confidence


def _bootstrap_nll_diff(diff, baseline, candidate, n_boot, seed, confidence):
    n_boot, seed, confidence = _bootstrap_settings(n_boot, seed, confidence)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive for a bootstrap interval")
    diff = _readonly_vector("nll difference", diff)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, diff.size, size=diff.size)
        boot[i] = float(np.mean(diff[idx]))
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = [float(x) for x in np.quantile(boot, [alpha, 1.0 - alpha])]
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "observed_mean_gain": float(np.mean(diff)),
        "bootstrap_mean_gain": float(np.mean(boot)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
        "n_boot": n_boot,
        "seed": seed,
        "n": int(diff.size),
        "method": "paired_row_resample",
    }


def paired_bootstrap_nll_improvement(result, baseline, candidate, n_boot=1000, seed=0, confidence=0.95):
    """Paired bootstrap CI for mean NLL gain, `baseline - candidate`.

    Rows are resampled with replacement from the shared evaluation split, preserving
    the paired comparison between two prediction families. Positive gain means the
    candidate has lower held-out NLL than the baseline.
    """
    baseline_vec = result.score_vector(baseline)
    candidate_vec = result.score_vector(candidate)
    if baseline_vec.n != candidate_vec.n:
        raise ValueError("baseline and candidate row-score vectors must have the same length")
    return _bootstrap_nll_diff(
        baseline_vec.nll - candidate_vec.nll,
        baseline,
        candidate,
        n_boot,
        seed,
        confidence,
    )


def _decode_score_names(value):
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("score_names must be a nonempty 1-D array")
    names = []
    for item in arr:
        if isinstance(item, bytes):
            name = item.decode("utf-8")
        else:
            name = str(item)
        if not name:
            raise ValueError("score_names must not contain empty names")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("score_names must be unique")
    return tuple(names)


def _score_row_nll_inputs(score_names, score_row_nll):
    names = _decode_score_names(score_names)
    rows = np.asarray(score_row_nll, dtype=float)
    if rows.ndim != 2:
        raise ValueError("score_row_nll must be a 2-D array shaped (models, rows)")
    if rows.shape[0] != len(names):
        raise ValueError("score_row_nll first dimension must match score_names")
    if rows.shape[1] == 0:
        raise ValueError("score_row_nll must contain at least one held-out row")
    if not np.isfinite(rows).all():
        raise ValueError("score_row_nll must be finite")
    return names, rows


def paired_bootstrap_nll_improvement_from_score_rows(
    score_names,
    score_row_nll,
    baseline,
    candidate,
    n_boot=1000,
    seed=0,
    confidence=0.95,
):
    """Paired bootstrap CI from stored row-level NLL artifacts."""
    names, rows = _score_row_nll_inputs(score_names, score_row_nll)
    name_to_idx = {name: i for i, name in enumerate(names)}
    if baseline not in name_to_idx:
        raise ValueError(f"baseline {baseline!r} is not present in score_names")
    if candidate not in name_to_idx:
        raise ValueError(f"candidate {candidate!r} is not present in score_names")
    diff = rows[name_to_idx[baseline]] - rows[name_to_idx[candidate]]
    return _bootstrap_nll_diff(diff, baseline, candidate, n_boot, seed, confidence)


def bootstrap_nll_improvements_from_score_rows(
    score_names,
    score_row_nll,
    n_boot=1000,
    seed=0,
    confidence=0.95,
    baselines=DEFAULT_NLL_BASELINES,
):
    """Return JSON-ready paired bootstrap maps from stored row-level NLL artifacts."""
    names, rows = _score_row_nll_inputs(score_names, score_row_nll)
    baselines = _normalize_baselines(baselines)
    out = {}
    for baseline in baselines:
        if baseline not in names:
            continue
        key = f"nll_improvement_bootstrap_vs_{baseline}"
        out[key] = {
            candidate: paired_bootstrap_nll_improvement_from_score_rows(
                names,
                rows,
                baseline,
                candidate,
                n_boot=n_boot,
                seed=seed,
                confidence=confidence,
            )
            for candidate in names
            if candidate != baseline
        }
    return out


def score_gaussian_prediction_vectors(name, target_state, mean, covariance, jitter=1e-9):
    """Return per-row Gaussian score vectors against held-out target states.

    `target_state` and `mean` must be `(n, d)` row matrices. `covariance` is one
    shared `(d, d)` covariance for the prediction family, matching the current
    Product-Kalman calibration/update contract.
    """
    target = _as_2d("target_state", target_state)
    pred = _as_2d("mean", mean)
    if pred.shape != target.shape:
        raise ValueError(f"mean shape {pred.shape} must match target_state shape {target.shape}")
    cov = _as_covariance("covariance", covariance, target.shape[1])
    nll = np.array([gaussian_nll(target[i], pred[i], cov, jitter=jitter) for i in range(target.shape[0])])
    residual = target - pred
    score_cov = regularize_covariance(cov, jitter=jitter, name=f"{name} score covariance")
    solved = np.linalg.solve(score_cov, residual.T).T
    return GaussianScoreVectors(
        name=name,
        nll=nll,
        squared_error=np.sum(residual * residual, axis=1),
        squared_mahalanobis=np.sum(residual * solved, axis=1),
        dimension=target.shape[1],
    )


def score_gaussian_prediction_vectors_rowwise(name, target_state, mean, covariances, jitter=1e-9):
    """Return per-row Gaussian score vectors with one covariance per row.

    This is the scoring primitive needed by hop-conditioned or regime-conditioned
    error models. `target_state` and `mean` must be `(n, d)` row matrices, and
    `covariances` must be `(n, d, d)` in the same row order.
    """
    target = _as_2d("target_state", target_state)
    pred = _as_2d("mean", mean)
    if pred.shape != target.shape:
        raise ValueError(f"mean shape {pred.shape} must match target_state shape {target.shape}")
    covs = _as_row_covariances("covariances", covariances, target.shape[0], target.shape[1])
    residual = target - pred
    nll = np.empty(target.shape[0], dtype=float)
    squared_mahalanobis = np.empty(target.shape[0], dtype=float)
    for i in range(target.shape[0]):
        nll[i] = gaussian_nll(target[i], pred[i], covs[i], jitter=jitter)
        score_cov = regularize_covariance(covs[i], jitter=jitter, name=f"{name} row {i} score covariance")
        solved = np.linalg.solve(score_cov, residual[i])
        squared_mahalanobis[i] = float(residual[i] @ solved)
    return GaussianScoreVectors(
        name=name,
        nll=nll,
        squared_error=np.sum(residual * residual, axis=1),
        squared_mahalanobis=squared_mahalanobis,
        dimension=target.shape[1],
    )


def _score_from_vectors(vectors, covariance):
    cov = _as_covariance("covariance", covariance, vectors.dimension)
    mean_squared_mahalanobis = float(np.mean(vectors.squared_mahalanobis))
    q50, q90, q95 = [float(q) for q in np.quantile(vectors.squared_mahalanobis, [0.50, 0.90, 0.95])]
    return GaussianScore(
        name=vectors.name,
        mean_nll=float(np.mean(vectors.nll)),
        mse=float(np.mean(vectors.squared_error)),
        n=vectors.n,
        covariance_trace=float(np.trace(cov)),
        mean_squared_mahalanobis=mean_squared_mahalanobis,
        mahalanobis_per_dim=mean_squared_mahalanobis / float(vectors.dimension),
        squared_mahalanobis_q50=q50,
        squared_mahalanobis_q90=q90,
        squared_mahalanobis_q95=q95,
    )


def _score_from_vectors_rowwise(vectors, covariances):
    covs = _as_row_covariances("covariances", covariances, vectors.n, vectors.dimension)
    mean_squared_mahalanobis = float(np.mean(vectors.squared_mahalanobis))
    q50, q90, q95 = [float(q) for q in np.quantile(vectors.squared_mahalanobis, [0.50, 0.90, 0.95])]
    return GaussianScore(
        name=vectors.name,
        mean_nll=float(np.mean(vectors.nll)),
        mse=float(np.mean(vectors.squared_error)),
        n=vectors.n,
        covariance_trace=float(np.mean(np.trace(covs, axis1=1, axis2=2))),
        mean_squared_mahalanobis=mean_squared_mahalanobis,
        mahalanobis_per_dim=mean_squared_mahalanobis / float(vectors.dimension),
        squared_mahalanobis_q50=q50,
        squared_mahalanobis_q90=q90,
        squared_mahalanobis_q95=q95,
    )


def score_gaussian_predictions(name, target_state, mean, covariance, jitter=1e-9):
    """Return aggregate Gaussian prediction scores for one held-out split."""
    vectors = score_gaussian_prediction_vectors(name, target_state, mean, covariance, jitter=jitter)
    return _score_from_vectors(vectors, covariance)


def score_gaussian_predictions_rowwise(name, target_state, mean, covariances, jitter=1e-9):
    """Return aggregate Gaussian scores when each row has its own covariance.

    `GaussianScore.covariance_trace` is the mean trace across row covariances.
    """
    vectors = score_gaussian_prediction_vectors_rowwise(name, target_state, mean, covariances, jitter=jitter)
    return _score_from_vectors_rowwise(vectors, covariances)


def score_gaussian_predictions_grouped(
    name,
    calibration_target_state,
    calibration_mean,
    calibration_groups,
    evaluation_target_state,
    evaluation_mean,
    evaluation_groups,
    min_group_rows=None,
    shrinkage=0.0,
    jitter=1e-9,
    ddof=1,
    shrinkage_target="diagonal",
):
    """Fit grouped residual covariances on calibration rows and score evaluation rows.

    Means are supplied by the caller. This helper only replaces the shared
    Gaussian covariance with calibration-fitted group covariances, making it the
    direct bridge from discrete contexts such as hop labels to rowwise NLL and
    Mahalanobis diagnostics.
    """
    residual_covariances = fit_group_residual_covariances(
        calibration_mean,
        calibration_target_state,
        calibration_groups,
        min_group_rows=min_group_rows,
        shrinkage=shrinkage,
        jitter=jitter,
        ddof=ddof,
        shrinkage_target=shrinkage_target,
    )
    row_covariances = residual_covariances.row_covariances(evaluation_groups)
    vectors = score_gaussian_prediction_vectors_rowwise(
        name,
        evaluation_target_state,
        evaluation_mean,
        row_covariances,
        jitter=jitter,
    )
    return GroupedGaussianScore(
        score=_score_from_vectors_rowwise(vectors, row_covariances),
        score_vectors=vectors,
        residual_covariances=residual_covariances,
        row_covariances=row_covariances,
    )


def _check_split_ids(calibration_ids, evaluation_ids, n_calibration, n_evaluation):
    if calibration_ids is None and evaluation_ids is None:
        return
    if calibration_ids is None or evaluation_ids is None:
        raise ValueError("calibration_ids and evaluation_ids must be provided together")
    calibration_ids = list(calibration_ids)
    evaluation_ids = list(evaluation_ids)
    if len(calibration_ids) != n_calibration:
        raise ValueError("calibration_ids length must match calibration rows")
    if len(evaluation_ids) != n_evaluation:
        raise ValueError("evaluation_ids length must match evaluation rows")
    assert_disjoint_ids(calibration_ids, evaluation_ids)


def _check_group_labels(calibration_groups, evaluation_groups, n_calibration, n_evaluation):
    if calibration_groups is None and evaluation_groups is None:
        return None, None
    if calibration_groups is None or evaluation_groups is None:
        raise ValueError("calibration_groups and evaluation_groups must be provided together")
    return (
        _as_group_labels("calibration_groups", calibration_groups, n=n_calibration),
        _as_group_labels("evaluation_groups", evaluation_groups, n=n_evaluation),
    )


def _zero_cross_calibration(calibration):
    return ProductKalmanCalibration(
        state_covariance=calibration.state_covariance,
        observation_covariance=calibration.observation_covariance,
        cross_covariance=np.zeros_like(calibration.cross_covariance),
        H=calibration.H,
        n_samples=calibration.n_samples,
        ddof=calibration.ddof,
        shrinkage=calibration.shrinkage,
        shrinkage_target=calibration.shrinkage_target,
    )


def _has_identity_observation(calibration):
    if calibration.state_dim != calibration.observation_dim:
        return False
    return bool(np.allclose(calibration.H, np.eye(calibration.state_dim), atol=1e-12, rtol=1e-12))


def evaluate_product_kalman_holdout(
    calibration_prior_mean,
    calibration_measurement,
    calibration_target_state,
    evaluation_prior_mean,
    evaluation_measurement,
    evaluation_target_state,
    H=None,
    calibration_ids=None,
    evaluation_ids=None,
    calibration_groups=None,
    evaluation_groups=None,
    min_group_rows=None,
    shrinkage=0.0,
    jitter=1e-9,
    ddof=1,
    shrinkage_target="diagonal",
):
    """Fit calibration blocks on one split and score updates on a held-out split.

    The returned scores always include:
    - `prior`: evaluation prior mean with fitted prior-error covariance;
    - `independent_kalman`: same fitted `P`/`R`, but `C=0`;
    - `product_kalman`: fitted correlated Product-Kalman update.

    If `H` is identity and observation/state dimensions match, a `measurement`
    baseline is also scored. Non-identity observations are not directly target-
    coordinate predictions, so the measurement baseline is omitted rather than
    silently applying an inverse map.
    """
    cal_prior = _as_2d("calibration_prior_mean", calibration_prior_mean)
    cal_measure = _as_2d("calibration_measurement", calibration_measurement)
    cal_target = _as_2d("calibration_target_state", calibration_target_state)
    eval_prior = _as_2d("evaluation_prior_mean", evaluation_prior_mean)
    eval_measure = _as_2d("evaluation_measurement", evaluation_measurement)
    eval_target = _as_2d("evaluation_target_state", evaluation_target_state)
    _check_split_ids(calibration_ids, evaluation_ids, cal_target.shape[0], eval_target.shape[0])
    cal_groups, eval_groups = _check_group_labels(
        calibration_groups,
        evaluation_groups,
        cal_target.shape[0],
        eval_target.shape[0],
    )

    calibration = fit_product_kalman_calibration(
        cal_prior,
        cal_measure,
        cal_target,
        H=H,
        shrinkage=shrinkage,
        jitter=jitter,
        ddof=ddof,
        shrinkage_target=shrinkage_target,
    )
    independent = _zero_cross_calibration(calibration)
    correlated_update = apply_product_kalman_calibration(calibration, eval_prior, eval_measure, jitter=jitter)
    independent_update = apply_product_kalman_calibration(independent, eval_prior, eval_measure, jitter=jitter)

    score_inputs = [
        ("prior", eval_prior, calibration.state_covariance),
        ("independent_kalman", independent_update.mean, independent_update.covariance),
        ("product_kalman", correlated_update.mean, correlated_update.covariance),
    ]
    identity_observation = _has_identity_observation(calibration)
    if identity_observation:
        score_inputs.insert(1, ("measurement", eval_measure, calibration.observation_covariance))

    score_vectors = [
        score_gaussian_prediction_vectors(name, eval_target, mean, covariance, jitter=jitter)
        for name, mean, covariance in score_inputs
    ]
    scores = [
        _score_from_vectors(vectors, covariance)
        for vectors, (_, _, covariance) in zip(score_vectors, score_inputs)
    ]
    pit_diagnostics = [
        gaussian_marginal_pit_diagnostics(name, eval_target, mean, covariance, jitter=jitter)
        for name, mean, covariance in score_inputs
    ]

    grouped_scores = []
    if cal_groups is not None:
        cal_correlated_update = apply_product_kalman_calibration(calibration, cal_prior, cal_measure, jitter=jitter)
        cal_independent_update = apply_product_kalman_calibration(independent, cal_prior, cal_measure, jitter=jitter)
        grouped_inputs = [
            ("prior_grouped", cal_prior, eval_prior),
            ("independent_kalman_grouped", cal_independent_update.mean, independent_update.mean),
            ("product_kalman_grouped", cal_correlated_update.mean, correlated_update.mean),
        ]
        if identity_observation:
            grouped_inputs.insert(1, ("measurement_grouped", cal_measure, eval_measure))
        grouped_scores = [
            score_gaussian_predictions_grouped(
                name,
                cal_target,
                cal_mean,
                cal_groups,
                eval_target,
                eval_mean,
                eval_groups,
                min_group_rows=min_group_rows,
                shrinkage=shrinkage,
                jitter=jitter,
                ddof=ddof,
                shrinkage_target=shrinkage_target,
            )
            for name, cal_mean, eval_mean in grouped_inputs
        ]
        scores.extend(grouped.score for grouped in grouped_scores)
        score_vectors.extend(grouped.score_vectors for grouped in grouped_scores)
        grouped_eval_means = {name: eval_mean for name, _cal_mean, eval_mean in grouped_inputs}
        for grouped in grouped_scores:
            pit_diagnostics.append(
                gaussian_marginal_pit_diagnostics(
                    grouped.score.name,
                    eval_target,
                    grouped_eval_means[grouped.score.name],
                    grouped.row_covariances,
                    jitter=jitter,
                )
            )

    return ProductKalmanHoldoutEvaluation(
        calibration=calibration,
        independent_calibration=independent,
        correlated_update=correlated_update,
        independent_update=independent_update,
        scores=tuple(scores),
        score_vectors=tuple(score_vectors),
        grouped_scores=tuple(grouped_scores),
        pit_diagnostics=tuple(pit_diagnostics),
    )


def _prefix_update_arrays(prefix, update):
    return {
        f"{prefix}_mean": update.mean,
        f"{prefix}_covariance": update.covariance,
        f"{prefix}_gain": update.gain,
        f"{prefix}_innovation": update.innovation,
        f"{prefix}_innovation_covariance": update.innovation_covariance,
    }


def _string_keyed_mapping(name, mapping, value_fn):
    out = {}
    for label, value in mapping.items():
        key = str(label)
        if key in out:
            raise ValueError(f"{name} has duplicate stringified group label {key!r}")
        out[key] = value_fn(value)
    return dict(sorted(out.items()))


def _grouped_covariance_to_json_dict(grouped):
    residuals = grouped.residual_covariances
    return {
        "score": grouped.score.name,
        "min_group_rows": residuals.min_group_rows,
        "group_counts": _string_keyed_mapping("group_counts", residuals.group_counts, int),
        "fallback_covariance": residuals.fallback_covariance.tolist(),
        "covariance_by_group": _string_keyed_mapping(
            "covariance_by_group",
            residuals.covariance_by_group,
            lambda cov: np.asarray(cov, dtype=float).tolist(),
        ),
        "row_covariance_shape": [int(v) for v in grouped.row_covariances.shape],
    }


def evaluation_artifact_arrays(result):
    """Return compact NPZ-ready arrays for row-level evaluation diagnostics.

    The artifact intentionally stores result-side data: fitted covariance blocks,
    row-level posterior means/innovations, shared gains/covariances, and score
    vectors. The original input NPZ remains the source of the held-out targets,
    prior rows, and measurement rows.
    """
    score_names = np.array([score.name for score in result.scores])
    arrays = {
        "schema_version": np.array(1, dtype=np.int64),
        "score_names": score_names,
        "score_mean_nll": np.array([score.mean_nll for score in result.scores], dtype=float),
        "score_mse": np.array([score.mse for score in result.scores], dtype=float),
        "score_n": np.array([score.n for score in result.scores], dtype=np.int64),
        "score_covariance_trace": np.array([score.covariance_trace for score in result.scores], dtype=float),
        "score_mean_squared_mahalanobis": np.array(
            [score.mean_squared_mahalanobis for score in result.scores],
            dtype=float,
        ),
        "score_mahalanobis_per_dim": np.array([score.mahalanobis_per_dim for score in result.scores], dtype=float),
        "score_squared_mahalanobis_q50": np.array(
            [score.squared_mahalanobis_q50 for score in result.scores],
            dtype=float,
        ),
        "score_squared_mahalanobis_q90": np.array(
            [score.squared_mahalanobis_q90 for score in result.scores],
            dtype=float,
        ),
        "score_squared_mahalanobis_q95": np.array(
            [score.squared_mahalanobis_q95 for score in result.scores],
            dtype=float,
        ),
        "calibration_n_samples": np.array(result.calibration.n_samples, dtype=np.int64),
        "calibration_state_covariance": result.calibration.state_covariance,
        "calibration_observation_covariance": result.calibration.observation_covariance,
        "calibration_cross_covariance": result.calibration.cross_covariance,
        "calibration_H": result.calibration.H,
        "independent_cross_covariance": result.independent_calibration.cross_covariance,
    }
    if result.score_vectors:
        arrays.update({
            "score_row_nll": np.vstack([vectors.nll for vectors in result.score_vectors]),
            "score_row_squared_error": np.vstack([vectors.squared_error for vectors in result.score_vectors]),
            "score_row_squared_mahalanobis": np.vstack([
                vectors.squared_mahalanobis
                for vectors in result.score_vectors
            ]),
        })
    if result.pit_diagnostics:
        arrays.update({
            "pit_names": np.array([diagnostic.name for diagnostic in result.pit_diagnostics]),
            "pit_values": np.stack([diagnostic.pit for diagnostic in result.pit_diagnostics]),
            "pit_channel_ks": np.vstack([diagnostic.channel_ks for diagnostic in result.pit_diagnostics]),
            "pit_summary_json": np.array([
                json.dumps(diagnostic.to_json_dict(), sort_keys=True)
                for diagnostic in result.pit_diagnostics
            ]),
        })
    if result.grouped_scores:
        arrays.update({
            "grouped_score_names": np.array([grouped.score.name for grouped in result.grouped_scores]),
            "grouped_score_row_covariances": np.stack([
                grouped.row_covariances
                for grouped in result.grouped_scores
            ]),
            "grouped_score_summary_json": np.array([
                json.dumps(_grouped_covariance_to_json_dict(grouped), sort_keys=True)
                for grouped in result.grouped_scores
            ]),
        })
    arrays.update(_prefix_update_arrays("product_kalman", result.correlated_update))
    arrays.update(_prefix_update_arrays("independent_kalman", result.independent_update))
    return arrays


def write_evaluation_npz(path, result):
    """Write row-level Product-Kalman evaluation artifacts to an NPZ file."""
    np.savez(path, **evaluation_artifact_arrays(result))


def _require_npz_array(npz, path, key):
    if key not in npz.files:
        raise ValueError(f"{path} is missing required array {key!r}")
    return npz[key]


def _optional_npz_array(npz, key):
    return npz[key] if key in npz.files else None


def evaluation_npz_score_summary(artifact_npz):
    """Return score-order, mean-NLL, and row-count summaries from an evaluation artifact NPZ."""
    path = str(artifact_npz)
    with np.load(path, allow_pickle=False) as data:
        names = _decode_score_names(_require_npz_array(data, path, "score_names"))
        mean_nll = np.asarray(_require_npz_array(data, path, "score_mean_nll"), dtype=float)
        score_n = np.asarray(_require_npz_array(data, path, "score_n"), dtype=np.int64)
    if mean_nll.shape != (len(names),):
        raise ValueError("score_mean_nll length must match score_names")
    if score_n.shape != (len(names),):
        raise ValueError("score_n length must match score_names")
    if not np.isfinite(mean_nll).all():
        raise ValueError("score_mean_nll must be finite")
    if (score_n <= 0).any():
        raise ValueError("score_n values must be positive")
    return {
        "score_order": list(names),
        "mean_nll": {name: float(mean_nll[i]) for i, name in enumerate(names)},
        "n": {name: int(score_n[i]) for i, name in enumerate(names)},
    }


def bootstrap_nll_improvements_from_evaluation_npz(
    artifact_npz,
    n_boot=1000,
    seed=0,
    confidence=0.95,
    baselines=DEFAULT_NLL_BASELINES,
):
    """Return JSON-ready paired NLL bootstrap maps from an evaluation artifact NPZ."""
    path = str(artifact_npz)
    with np.load(path, allow_pickle=False) as data:
        return bootstrap_nll_improvements_from_score_rows(
            _require_npz_array(data, path, "score_names"),
            _require_npz_array(data, path, "score_row_nll"),
            n_boot=n_boot,
            seed=seed,
            confidence=confidence,
            baselines=baselines,
        )


def _score_to_json_dict(score):
    return {
        "name": score.name,
        "mean_nll": score.mean_nll,
        "mse": score.mse,
        "n": score.n,
        "covariance_trace": score.covariance_trace,
        "mean_squared_mahalanobis": score.mean_squared_mahalanobis,
        "mahalanobis_per_dim": score.mahalanobis_per_dim,
        "squared_mahalanobis_q50": score.squared_mahalanobis_q50,
        "squared_mahalanobis_q90": score.squared_mahalanobis_q90,
        "squared_mahalanobis_q95": score.squared_mahalanobis_q95,
    }


def _bootstrap_improvement_map(result, baseline, n_boot, seed, confidence):
    return {
        score.name: paired_bootstrap_nll_improvement(
            result,
            baseline,
            score.name,
            n_boot=n_boot,
            seed=seed,
            confidence=confidence,
        )
        for score in result.scores
        if score.name != baseline
    }


def _score_names(result):
    return {score.name for score in result.scores}


def _validate_result_baselines(result, baselines):
    baselines = _normalize_baselines(baselines)
    names = _score_names(result)
    missing = [baseline for baseline in baselines if baseline not in names]
    if missing:
        raise ValueError(f"nll_baselines are not present in scores: {missing!r}")
    return baselines


def _nll_improvement_maps(result, baselines):
    out = {}
    for baseline in _validate_result_baselines(result, baselines):
        baseline_nll = result.score(baseline).mean_nll
        out[f"nll_improvement_vs_{baseline}"] = {
            score.name: baseline_nll - score.mean_nll
            for score in result.scores
            if score.name != baseline
        }
    return out


def _bootstrap_improvement_maps(result, baselines, n_boot, seed, confidence):
    return {
        f"nll_improvement_bootstrap_vs_{baseline}": _bootstrap_improvement_map(
            result,
            baseline,
            n_boot,
            seed,
            confidence,
        )
        for baseline in _validate_result_baselines(result, baselines)
    }


def evaluation_to_json_dict(
    result,
    bootstrap_nll=0,
    bootstrap_seed=0,
    bootstrap_confidence=0.95,
    nll_baselines=DEFAULT_NLL_BASELINES,
):
    """Return a JSON-serializable summary for a holdout evaluation result."""
    bootstrap_nll, bootstrap_seed, bootstrap_confidence = _bootstrap_settings(
        bootstrap_nll,
        bootstrap_seed,
        bootstrap_confidence,
    )
    nll_baselines = _validate_result_baselines(result, nll_baselines)
    scores = {score.name: _score_to_json_dict(score) for score in result.scores}
    out = {
        "score_order": [score.name for score in result.scores],
        "scores": scores,
        "nll_baselines": list(nll_baselines),
        "calibration": {
            "n_samples": result.calibration.n_samples,
            "state_dim": result.calibration.state_dim,
            "observation_dim": result.calibration.observation_dim,
            "ddof": result.calibration.ddof,
            "shrinkage": result.calibration.shrinkage,
            "shrinkage_target": result.calibration.shrinkage_target,
            "H": result.calibration.H.tolist(),
            "state_covariance": result.calibration.state_covariance.tolist(),
            "observation_covariance": result.calibration.observation_covariance.tolist(),
            "cross_covariance": result.calibration.cross_covariance.tolist(),
        },
    }
    out.update(_nll_improvement_maps(result, nll_baselines))
    if result.grouped_scores:
        out["grouped_covariances"] = {
            grouped.score.name: _grouped_covariance_to_json_dict(grouped)
            for grouped in result.grouped_scores
        }
    if result.pit_diagnostics:
        out["pit_diagnostics"] = {
            diagnostic.name: diagnostic.to_json_dict()
            for diagnostic in result.pit_diagnostics
        }
    if bootstrap_nll:
        out.update(_bootstrap_improvement_maps(
            result,
            nll_baselines,
            bootstrap_nll,
            bootstrap_seed,
            bootstrap_confidence,
        ))
    return out


def run_product_kalman_holdout_npz(
    input_npz,
    calibration_prior_key="calibration_prior_mean",
    calibration_measurement_key="calibration_measurement",
    calibration_target_key="calibration_target_state",
    evaluation_prior_key="evaluation_prior_mean",
    evaluation_measurement_key="evaluation_measurement",
    evaluation_target_key="evaluation_target_state",
    H_key="H",
    calibration_ids_key="calibration_ids",
    evaluation_ids_key="evaluation_ids",
    calibration_groups_key="calibration_groups",
    evaluation_groups_key="evaluation_groups",
    min_group_rows=None,
    shrinkage=0.0,
    jitter=1e-9,
    ddof=1,
    shrinkage_target="diagonal",
):
    """Load a single NPZ fixture and run `evaluate_product_kalman_holdout`.

    Required arrays default to the key names in the argument list. Optional `H`,
    split-ID arrays, and group-label arrays are used when present. Store scalar
    channels as explicit `(n, 1)` matrices, matching the calibration API.
    """
    path = str(input_npz)
    with np.load(path, allow_pickle=False) as data:
        H = _optional_npz_array(data, H_key) if H_key else None
        calibration_ids = _optional_npz_array(data, calibration_ids_key) if calibration_ids_key else None
        evaluation_ids = _optional_npz_array(data, evaluation_ids_key) if evaluation_ids_key else None
        calibration_groups = _optional_npz_array(data, calibration_groups_key) if calibration_groups_key else None
        evaluation_groups = _optional_npz_array(data, evaluation_groups_key) if evaluation_groups_key else None
        return evaluate_product_kalman_holdout(
            _require_npz_array(data, path, calibration_prior_key),
            _require_npz_array(data, path, calibration_measurement_key),
            _require_npz_array(data, path, calibration_target_key),
            _require_npz_array(data, path, evaluation_prior_key),
            _require_npz_array(data, path, evaluation_measurement_key),
            _require_npz_array(data, path, evaluation_target_key),
            H=H,
            calibration_ids=calibration_ids,
            evaluation_ids=evaluation_ids,
            calibration_groups=calibration_groups,
            evaluation_groups=evaluation_groups,
            min_group_rows=min_group_rows,
            shrinkage=shrinkage,
            jitter=jitter,
            ddof=ddof,
            shrinkage_target=shrinkage_target,
        )


def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Run a split-safe Product-Kalman holdout evaluation from one NPZ fixture.",
    )
    ap.add_argument("input_npz", help="NPZ with calibration/evaluation row matrices")
    ap.add_argument("--output-json", help="write JSON summary to this path instead of stdout")
    ap.add_argument("--output-npz", help="write row-level prediction/covariance artifacts to this NPZ path")
    ap.add_argument("--calibration-prior-key", default="calibration_prior_mean")
    ap.add_argument("--calibration-measurement-key", default="calibration_measurement")
    ap.add_argument("--calibration-target-key", default="calibration_target_state")
    ap.add_argument("--evaluation-prior-key", default="evaluation_prior_mean")
    ap.add_argument("--evaluation-measurement-key", default="evaluation_measurement")
    ap.add_argument("--evaluation-target-key", default="evaluation_target_state")
    ap.add_argument("--H-key", default="H", help="optional observation-matrix key; use '' to disable")
    ap.add_argument(
        "--calibration-ids-key",
        default="calibration_ids",
        help="optional calibration ID key; use '' to disable",
    )
    ap.add_argument(
        "--evaluation-ids-key",
        default="evaluation_ids",
        help="optional evaluation ID key; use '' to disable",
    )
    ap.add_argument(
        "--calibration-groups-key",
        default="calibration_groups",
        help="optional calibration group-label key; use '' to disable grouped covariance scores",
    )
    ap.add_argument(
        "--evaluation-groups-key",
        default="evaluation_groups",
        help="optional evaluation group-label key; use '' to disable grouped covariance scores",
    )
    ap.add_argument("--min-group-rows", type=int, help="minimum calibration rows before fitting a group covariance")
    ap.add_argument("--shrinkage", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=1e-9)
    ap.add_argument("--ddof", type=int, default=1)
    ap.add_argument("--shrinkage-target", default="diagonal", choices=("diagonal", "scaled_identity"))
    ap.add_argument(
        "--nll-baselines",
        default=",".join(DEFAULT_NLL_BASELINES),
        help="comma-separated score names used as NLL-gain baselines",
    )
    ap.add_argument("--bootstrap-nll", type=int, default=0, help="paired bootstrap replicates for NLL gains; 0 disables")
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--bootstrap-confidence", type=float, default=0.95)
    ap.add_argument("--indent", type=int, default=2, help="JSON indentation; use 0 for compact output")
    return ap


def main(argv=None):
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    try:
        result = run_product_kalman_holdout_npz(
            args.input_npz,
            calibration_prior_key=args.calibration_prior_key,
            calibration_measurement_key=args.calibration_measurement_key,
            calibration_target_key=args.calibration_target_key,
            evaluation_prior_key=args.evaluation_prior_key,
            evaluation_measurement_key=args.evaluation_measurement_key,
            evaluation_target_key=args.evaluation_target_key,
            H_key=args.H_key or None,
            calibration_ids_key=args.calibration_ids_key or None,
            evaluation_ids_key=args.evaluation_ids_key or None,
            calibration_groups_key=args.calibration_groups_key or None,
            evaluation_groups_key=args.evaluation_groups_key or None,
            min_group_rows=args.min_group_rows,
            shrinkage=args.shrinkage,
            jitter=args.jitter,
            ddof=args.ddof,
            shrinkage_target=args.shrinkage_target,
        )
    except ValueError as exc:
        ap.error(str(exc))
    try:
        data = evaluation_to_json_dict(
            result,
            bootstrap_nll=args.bootstrap_nll,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_confidence=args.bootstrap_confidence,
            nll_baselines=_parse_name_list(args.nll_baselines),
        )
    except ValueError as exc:
        ap.error(str(exc))
    indent = None if args.indent == 0 else args.indent
    text = json.dumps(data, indent=indent, sort_keys=True) + "\n"
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    if args.output_npz:
        write_evaluation_npz(args.output_npz, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
