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
from dataclasses import dataclass
import json
import sys

import numpy as np

try:
    from .product_kalman import gaussian_nll, regularize_covariance
    from .product_kalman_calibration import (
        ProductKalmanCalibration,
        apply_product_kalman_calibration,
        assert_disjoint_ids,
        fit_product_kalman_calibration,
    )
except ImportError:  # direct script execution from prototypes/mu_cosine
    from product_kalman import gaussian_nll, regularize_covariance
    from product_kalman_calibration import (
        ProductKalmanCalibration,
        apply_product_kalman_calibration,
        assert_disjoint_ids,
        fit_product_kalman_calibration,
    )


__all__ = [
    "GaussianScore",
    "ProductKalmanHoldoutEvaluation",
    "evaluate_product_kalman_holdout",
    "evaluation_artifact_arrays",
    "evaluation_to_json_dict",
    "run_product_kalman_holdout_npz",
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

    def __post_init__(self):
        if not isinstance(self.calibration, ProductKalmanCalibration):
            raise ValueError("calibration must be a ProductKalmanCalibration")
        if not isinstance(self.independent_calibration, ProductKalmanCalibration):
            raise ValueError("independent_calibration must be a ProductKalmanCalibration")
        scores = tuple(self.scores)
        names = [s.name for s in scores]
        if len(names) != len(set(names)):
            raise ValueError("score names must be unique")
        object.__setattr__(self, "scores", scores)

    def score(self, name):
        """Return the named `GaussianScore`."""
        for score in self.scores:
            if score.name == name:
                return score
        raise KeyError(name)

    def nll_improvement(self, baseline, candidate):
        """Positive means `candidate` has lower mean NLL than `baseline`."""
        return self.score(baseline).mean_nll - self.score(candidate).mean_nll


def score_gaussian_predictions(name, target_state, mean, covariance, jitter=1e-9):
    """Score row-wise Gaussian predictions against held-out target states.

    `target_state` and `mean` must be `(n, d)` row matrices. `covariance` is one
    shared `(d, d)` covariance for the prediction family, matching the current
    Product-Kalman calibration/update contract.
    """
    target = _as_2d("target_state", target_state)
    pred = _as_2d("mean", mean)
    if pred.shape != target.shape:
        raise ValueError(f"mean shape {pred.shape} must match target_state shape {target.shape}")
    cov = _as_covariance("covariance", covariance, target.shape[1])
    nll = [gaussian_nll(target[i], pred[i], cov, jitter=jitter) for i in range(target.shape[0])]
    residual = target - pred
    score_cov = regularize_covariance(cov, jitter=jitter, name=f"{name} score covariance")
    solved = np.linalg.solve(score_cov, residual.T).T
    squared_mahalanobis = np.sum(residual * solved, axis=1)
    mse = float(np.mean(np.sum(residual * residual, axis=1)))
    mean_squared_mahalanobis = float(np.mean(squared_mahalanobis))
    q50, q90, q95 = [float(q) for q in np.quantile(squared_mahalanobis, [0.50, 0.90, 0.95])]
    return GaussianScore(
        name=name,
        mean_nll=float(np.mean(nll)),
        mse=mse,
        n=int(target.shape[0]),
        covariance_trace=float(np.trace(cov)),
        mean_squared_mahalanobis=mean_squared_mahalanobis,
        mahalanobis_per_dim=mean_squared_mahalanobis / float(target.shape[1]),
        squared_mahalanobis_q50=q50,
        squared_mahalanobis_q90=q90,
        squared_mahalanobis_q95=q95,
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

    scores = [
        score_gaussian_predictions("prior", eval_target, eval_prior, calibration.state_covariance, jitter=jitter),
        score_gaussian_predictions(
            "independent_kalman",
            eval_target,
            independent_update.mean,
            independent_update.covariance,
            jitter=jitter,
        ),
        score_gaussian_predictions(
            "product_kalman",
            eval_target,
            correlated_update.mean,
            correlated_update.covariance,
            jitter=jitter,
        ),
    ]
    if _has_identity_observation(calibration):
        scores.insert(
            1,
            score_gaussian_predictions(
                "measurement",
                eval_target,
                eval_measure,
                calibration.observation_covariance,
                jitter=jitter,
            ),
        )

    return ProductKalmanHoldoutEvaluation(
        calibration=calibration,
        independent_calibration=independent,
        correlated_update=correlated_update,
        independent_update=independent_update,
        scores=tuple(scores),
    )


def _prefix_update_arrays(prefix, update):
    return {
        f"{prefix}_mean": update.mean,
        f"{prefix}_covariance": update.covariance,
        f"{prefix}_gain": update.gain,
        f"{prefix}_innovation": update.innovation,
        f"{prefix}_innovation_covariance": update.innovation_covariance,
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


def evaluation_to_json_dict(result):
    """Return a JSON-serializable summary for a holdout evaluation result."""
    scores = {score.name: _score_to_json_dict(score) for score in result.scores}
    prior = result.score("prior").mean_nll
    improvements = {name: prior - score["mean_nll"] for name, score in scores.items() if name != "prior"}
    out = {
        "score_order": [score.name for score in result.scores],
        "scores": scores,
        "nll_improvement_vs_prior": improvements,
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
    if "independent_kalman" in scores:
        independent = scores["independent_kalman"]["mean_nll"]
        out["nll_improvement_vs_independent_kalman"] = {
            name: independent - score["mean_nll"]
            for name, score in scores.items()
            if name != "independent_kalman"
        }
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
    shrinkage=0.0,
    jitter=1e-9,
    ddof=1,
    shrinkage_target="diagonal",
):
    """Load a single NPZ fixture and run `evaluate_product_kalman_holdout`.

    Required arrays default to the key names in the argument list. Optional `H`,
    `calibration_ids`, and `evaluation_ids` arrays are used when present. Store
    scalar channels as explicit `(n, 1)` matrices, matching the calibration API.
    """
    path = str(input_npz)
    with np.load(path, allow_pickle=False) as data:
        H = _optional_npz_array(data, H_key) if H_key else None
        calibration_ids = _optional_npz_array(data, calibration_ids_key) if calibration_ids_key else None
        evaluation_ids = _optional_npz_array(data, evaluation_ids_key) if evaluation_ids_key else None
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
    ap.add_argument("--shrinkage", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=1e-9)
    ap.add_argument("--ddof", type=int, default=1)
    ap.add_argument("--shrinkage-target", default="diagonal", choices=("diagonal", "scaled_identity"))
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
            shrinkage=args.shrinkage,
            jitter=args.jitter,
            ddof=args.ddof,
            shrinkage_target=args.shrinkage_target,
        )
    except ValueError as exc:
        ap.error(str(exc))
    data = evaluation_to_json_dict(result)
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
