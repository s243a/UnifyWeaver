#!/usr/bin/env python3
"""Benchmark compiled dense-gain and square-root/QR conditioners.

This separates design compilation (covariance decorrelation, whitening, and
``geqrf``) from conditioning an observation batch (one compact-reflector
``ormqr`` plus triangular solve).  CUDA timings include warmups and explicit
synchronisation.

The current mu-cosine D/S state has ``n=2``.  At batch size one that problem is
expected to be launch/latency-bound.  For any fixed design the cached dense
gain is the primary throughput baseline; QR is intended for workloads that
must retain and update the precision root across later blocks.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from joint_square_root_conditioner_torch import (
    CompiledCorrelatedConditionerTorch,
    CompiledDenseGainConditionerTorch,
    precision_root_from_covariance_torch,
)


def _csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _devices(specification: str) -> list[torch.device]:
    names = [part.strip() for part in specification.split(",") if part.strip()]
    if names == ["auto"]:
        names = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    devices = []
    for name in names:
        if name == "cuda" and not torch.cuda.is_available():
            print("# skipping cuda: torch.cuda.is_available() is false")
            continue
        devices.append(torch.device(name))
    if not devices:
        raise SystemExit("no requested benchmark device is available")
    return devices


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(function, repeats: int, warmups: int, device: torch.device) -> float:
    for _ in range(warmups):
        function()
    _synchronise(device)
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    _synchronise(device)
    return 1000.0 * (time.perf_counter() - start) / repeats


def _problem(n: int, m: int, batch: int, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device=device)
    generator.manual_seed(1729 + 1009 * n + 917 * m + batch)
    eye_n = torch.eye(n, dtype=dtype, device=device)
    eye_m = torch.eye(m, dtype=dtype, device=device)
    p_factor = torch.randn(n, n, generator=generator, dtype=dtype, device=device)
    r_factor = torch.randn(m, m, generator=generator, dtype=dtype, device=device)
    P = p_factor @ p_factor.mT / max(n, 1) + 0.75 * eye_n
    R = r_factor @ r_factor.mT / max(m, 1) + 0.75 * eye_m

    # Construct nonzero C while guaranteeing the joint covariance is SPD:
    # C = Lp B Lr.T and ||B||_2 < 1 implies R - C.T P^-1 C > 0.
    Lp = torch.linalg.cholesky(P)
    Lr = torch.linalg.cholesky(R)
    bridge = torch.randn(n, m, generator=generator, dtype=dtype, device=device)
    bridge = 0.25 * bridge / torch.clamp(
        torch.linalg.matrix_norm(bridge, ord=2), min=torch.finfo(dtype).eps
    )
    C = Lp @ bridge @ Lr.mT
    H = torch.randn(m, n, generator=generator, dtype=dtype, device=device) / math.sqrt(n)
    mean = torch.randn(batch, n, generator=generator, dtype=dtype, device=device)
    observation = torch.randn(batch, m, generator=generator, dtype=dtype, device=device)
    root = precision_root_from_covariance_torch(P)
    return root, H, R, C, mean, observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dims", type=_csv_ints, default=_csv_ints("2,16,64"))
    parser.add_argument("--measurement-dims", type=_csv_ints, default=_csv_ints("2,16"))
    parser.add_argument("--batch-sizes", type=_csv_ints, default=_csv_ints("1,256,4096"))
    parser.add_argument("--devices", default="auto", help="auto or comma-separated cpu,cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--compile-repeats",
        type=int,
        default=5,
        help="repetitions for the more expensive design-compilation timing",
    )
    parser.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args()
    if min(args.warmups, args.repeats, args.compile_repeats) < 1:
        parser.error("warmups and repeat counts must be positive")
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)

    dtype = _dtype(args.dtype)
    print("# Fixed-design correlated conditioning: compiled QR versus compiled dense gain")
    print("# n=2 (current D/S state) and batch=1 is expected to be latency-bound on CUDA")
    print("algorithm,device,dtype,n,m,batch,compile_ms,condition_ms,rows_per_second")
    with torch.inference_mode():
        for device in _devices(args.devices):
            for n in args.state_dims:
                for m in args.measurement_dims:
                    for batch in args.batch_sizes:
                        root, H, R, C, mean, observation = _problem(
                            n, m, batch, device, dtype
                        )

                        algorithms = (
                            ("qr", CompiledCorrelatedConditionerTorch),
                            ("dense_gain", CompiledDenseGainConditionerTorch),
                        )
                        for algorithm, conditioner_type in algorithms:
                            def compile_design():
                                return conditioner_type.compile(root, H, R, C)

                            compile_ms = _time_ms(
                                compile_design,
                                args.compile_repeats,
                                min(args.warmups, args.compile_repeats),
                                device,
                            )
                            compiled = compile_design()

                            def condition_batch():
                                return compiled.condition(mean, observation)

                            condition_ms = _time_ms(
                                condition_batch, args.repeats, args.warmups, device
                            )
                            rows_per_second = 1000.0 * batch / condition_ms
                            print(
                                f"{algorithm},{device.type},{args.dtype},{n},{m},{batch},"
                                f"{compile_ms:.6f},{condition_ms:.6f},"
                                f"{rows_per_second:.3f}"
                            )


if __name__ == "__main__":
    main()
