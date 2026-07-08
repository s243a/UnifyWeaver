#!/usr/bin/env python3
"""Monte Carlo checks for product-space nonlinear covariance propagation.

CPU is the default so the diagnostic is easy to run anywhere:

    python3 test_product_space_monte_carlo.py

For larger runs on a machine with CUDA:

    python3 test_product_space_monte_carlo.py --device cuda --samples 200000

This is a local sanity check for the small-noise regime where first-order covariance
propagation should be accurate. It is not a calibration claim for real graph/judge data.
"""

import argparse
import sys

try:
    import torch
except ModuleNotFoundError:
    print("skipped: torch is not installed")
    sys.exit(0)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is false")
    return torch.device(name)


def empirical_cov(x):
    centered = x - x.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(x.shape[0] - 1)


def source_covariance(sigmas, rho):
    d = sigmas.numel()
    corr = torch.full((d, d), float(rho), dtype=sigmas.dtype, device=sigmas.device)
    corr.fill_diagonal_(1.0)
    return corr * sigmas[:, None] * sigmas[None, :]


def max_relative_error(empirical, predicted, floor=1e-10):
    scale = torch.clamp(predicted.abs(), min=floor)
    return float(((empirical - predicted).abs() / scale).max().cpu())


def assert_delta_cov(name, empirical, predicted, rel_tol):
    rel = max_relative_error(empirical, predicted)
    print(f"{name}: max relative covariance error = {rel:.4f}")
    if rel > rel_tol:
        raise AssertionError(f"{name} delta covariance error {rel:.4f} exceeds {rel_tol:.4f}")


def run(samples, device, seed, rel_tol):
    dtype = torch.float64
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    mean = torch.tensor([0.31, 0.55, 0.74], dtype=dtype, device=device)
    sigmas = torch.tensor([0.018, 0.015, 0.012], dtype=dtype, device=device)
    cov = source_covariance(sigmas, rho=0.35)
    chol = torch.linalg.cholesky(cov)
    mu = mean + torch.randn(samples, mean.numel(), dtype=dtype, device=device) @ chol.T
    mu = mu.clamp(1e-6, 1.0 - 1e-6)

    log_j = 1.0 / mean
    log_pred = log_j[:, None] * cov * log_j[None, :]
    assert_delta_cov("log(mu)", empirical_cov(torch.log(mu)), log_pred, rel_tol)

    logit = torch.log(mu) - torch.log1p(-mu)
    logit_j = 1.0 / (mean * (1.0 - mean))
    logit_pred = logit_j[:, None] * cov * logit_j[None, :]
    assert_delta_cov("logit(mu)", empirical_cov(logit), logit_pred, rel_tol)

    lower = mu.prod(dim=1)
    upper_complement = (1.0 - mu).prod(dim=1)
    upper = 1.0 - upper_complement
    interval = torch.stack([lower, upper, upper - lower], dim=1)

    lower0 = mean.prod()
    upper_complement0 = (1.0 - mean).prod()
    lower_j = lower0 / mean
    upper_j = upper_complement0 / (1.0 - mean)
    interval_j = torch.stack([lower_j, upper_j, upper_j - lower_j])
    interval_pred = interval_j @ cov @ interval_j.T
    assert_delta_cov("lower/upper/width product proxies", empirical_cov(interval), interval_pred, rel_tol)

    print(f"all product-space Monte Carlo checks passed on {device} with {samples} samples")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto; default: cpu")
    ap.add_argument("--samples", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--rel-tol", type=float, default=0.18)
    args = ap.parse_args()
    if args.samples < 1000:
        raise SystemExit("--samples should be at least 1000 for a meaningful covariance check")
    run(args.samples, resolve_device(args.device), args.seed, args.rel_tol)


if __name__ == "__main__":
    main()
