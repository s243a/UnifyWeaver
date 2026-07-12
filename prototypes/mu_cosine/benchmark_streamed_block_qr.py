#!/usr/bin/env python3
"""Benchmark batched sequential information updates with 32-judge blocks.

The four workload axes are deliberately separate:

``n``
    latent state dimension;
``m_block``
    measurements (for example judges/channels) in one likelihood block;
``T``
    blocks assimilated into one posterior trajectory; and
``B``
    independent posterior systems processed in parallel.

The synthetic likelihood has a dense covariance *within* each block and zero
conditional covariance *between* blocks.  Each block is Cholesky-whitened
before timing.  The benchmark therefore measures the information-conditioner
core, not covariance estimation or whitening.

Algorithms:

* ``full_stacked_qr``: one Householder QR of all ``T * m_block`` rows;
* ``streamed_qr``: one Householder QR per block, carrying ``(U, z)``;
* ``dense_recompute``: accumulate normal equations and refactor them after
  every block (a less stable but useful sequential baseline);
* ``cached_dense_full``: compile the final normal equations once and apply
  only new right-hand sides.  This is valid only for a shared, fixed design
  and does not provide intermediate sequential roots.  No matched cached-QR
  chain is timed here, so static repeated-throughput claims must use the
  earlier fixed-design benchmark instead.

Timings are warmup followed by independent trials and are reported as
median/MAD.  CUDA compute uses explicit synchronization.  Host/device input
and output transfers are measured separately and excluded from trajectory
latency.  Peak CUDA memory is the maximum incremental allocation observed in
the timed trials; persistent compiled storage that predates a conditioning
trial is not included. Algorithm execution failures are emitted as CSV rows
after problem/reference construction, while generation, transfer, or OOM
failures remain the responsibility of the outer job runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import platform
import statistics
import time
from typing import Callable, Iterable

import torch

from joint_square_root_conditioner_torch import (
    compile_information_update_torch,
    householder_information_update_torch,
    precision_root_from_covariance_torch,
)


ALGORITHMS = (
    "full_stacked_qr",
    "streamed_qr",
    "dense_recompute",
    "cached_dense_full",
)


@dataclass(frozen=True)
class Problem:
    root: torch.Tensor
    blocks: torch.Tensor
    rhs: torch.Tensor
    shared_design: bool

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.root, self.blocks, self.rhs


@dataclass(frozen=True)
class Result:
    root: torch.Tensor
    information_rhs: torch.Tensor
    solution: torch.Tensor


@dataclass(frozen=True)
class Timing:
    median_ms: float
    mad_ms: float
    peak_cuda_mib: float


@dataclass(frozen=True)
class CachedDenseFull:
    root: torch.Tensor
    cholesky: torch.Tensor
    blocks: torch.Tensor

    def condition(self, rhs: torch.Tensor) -> Result:
        batch, block_count, block_size = rhs.shape
        stacked_rhs = rhs.reshape(batch, block_count * block_size)
        stacked_blocks = self.blocks.reshape(
            block_count * block_size, self.blocks.shape[-1]
        )
        eta = torch.einsum("mn,bm->bn", stacked_blocks, stacked_rhs)
        solution = torch.cholesky_solve(
            eta.transpose(0, 1).contiguous(), self.cholesky
        ).transpose(0, 1)
        information_rhs = torch.linalg.solve_triangular(
            self.cholesky,
            eta.transpose(0, 1).contiguous(),
            upper=False,
        ).transpose(0, 1)
        return Result(self.root, information_rhs, solution)


def _csv_positive_ints(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _csv_positive_floats(value: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive finite numbers")
    return values


def _csv_choices(value: str, choices: Iterable[str]) -> list[str]:
    allowed = set(choices)
    values = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [item for item in values if item not in allowed]
    if not values or invalid:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated values from {sorted(allowed)}; invalid={invalid}"
        )
    return values


def _dtypes(value: str) -> list[str]:
    return _csv_choices(value, ("float32", "float64"))


def _design_modes(value: str) -> list[str]:
    return _csv_choices(value, ("shared", "distinct"))


def _algorithms(value: str) -> list[str]:
    return _csv_choices(value, ALGORITHMS)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _devices(specification: str) -> list[torch.device]:
    names = [part.strip() for part in specification.split(",") if part.strip()]
    if names == ["auto"]:
        names = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    invalid = [name for name in names if name not in ("cpu", "cuda")]
    if not names or invalid:
        raise argparse.ArgumentTypeError(
            f"devices must be auto or comma-separated cpu,cuda; invalid={invalid}"
        )
    devices: list[torch.device] = []
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


def _median_mad(values: list[float]) -> tuple[float, float]:
    median = statistics.median(values)
    return median, statistics.median(abs(value - median) for value in values)


def _time_callable(
    function: Callable[[], object],
    *,
    device: torch.device,
    warmups: int,
    trials: int,
    inner_repeats: int,
) -> Timing:
    for _ in range(warmups):
        output = function()
        del output
    _synchronise(device)

    elapsed_ms: list[float] = []
    peak_bytes: list[int] = []
    for _ in range(trials):
        if device.type == "cuda":
            _synchronise(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline = 0
        start = time.perf_counter()
        for _ in range(inner_repeats):
            output = function()
            # Avoid retaining a previous result while the next repeat
            # allocates; that would inflate peak memory for inner_repeats > 1.
            del output
        _synchronise(device)
        elapsed_ms.append(
            1000.0 * (time.perf_counter() - start) / inner_repeats
        )
        if device.type == "cuda":
            peak_bytes.append(
                max(0, torch.cuda.max_memory_allocated(device) - baseline)
            )
    median, mad = _median_mad(elapsed_ms)
    return Timing(median, mad, max(peak_bytes, default=0) / (1024.0**2))


def _time_transfer(
    function: Callable[[], object],
    *,
    device: torch.device,
    trials: int,
) -> tuple[float, float, object]:
    if device.type != "cuda":
        return 0.0, 0.0, function()
    # Exclude one-time CUDA context/allocation initialization from transfer
    # latency just as compute timing excludes warmups.
    output = function()
    _synchronise(device)
    del output
    values: list[float] = []
    output: object | None = None
    for _ in range(trials):
        start = time.perf_counter()
        output = function()
        _synchronise(device)
        values.append(1000.0 * (time.perf_counter() - start))
        del output
    median, mad = _median_mad(values)
    output = function()
    _synchronise(device)
    return median, mad, output


def _make_problem(
    *,
    n: int,
    m_block: int,
    block_count: int,
    batch: int,
    shared_design: bool,
    condition_number: float,
    noise_scale: float,
    seed: int,
) -> Problem:
    """Generate a float64 CPU problem with block-diagonal conditional noise."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        seed
        + 1009 * n
        + 917 * m_block
        + 101 * block_count
        + 17 * batch
        + int(not shared_design)
    )
    def spd(shape: tuple[int, ...], dimension: int, scale: float) -> torch.Tensor:
        factor = torch.randn(
            *shape, dimension, dimension, generator=generator, dtype=torch.float64
        )
        orthogonal, _ = torch.linalg.qr(factor)
        half_log_condition = 0.5 * math.log10(condition_number)
        eigenvalues = torch.logspace(
            -half_log_condition,
            half_log_condition,
            dimension,
            dtype=torch.float64,
        )
        return (
            orthogonal
            @ torch.diag(eigenvalues * scale)
            @ orthogonal.mT
        )

    covariance = spd((), n, 1.0)
    root = precision_root_from_covariance_torch(covariance, jitter=0.0)

    design_shape = (
        (block_count, m_block, n)
        if shared_design
        else (batch, block_count, m_block, n)
    )
    design = torch.randn(*design_shape, generator=generator, dtype=torch.float64)
    design = design / math.sqrt(n)
    covariance_batch_shape = (
        (block_count,) if shared_design else (batch, block_count)
    )
    conditional_covariance = spd(
        covariance_batch_shape, m_block, noise_scale
    )
    cholesky = torch.linalg.cholesky(conditional_covariance)
    blocks = torch.linalg.solve_triangular(cholesky, design, upper=False)

    latent = torch.randn(batch, n, generator=generator, dtype=torch.float64)
    noise = torch.randn(
        batch, block_count, m_block, generator=generator, dtype=torch.float64
    )
    if shared_design:
        rhs = torch.einsum("tmn,bn->btm", blocks, latent) + noise
    else:
        rhs = torch.einsum("btmn,bn->btm", blocks, latent) + noise
    return Problem(root=root, blocks=blocks, rhs=rhs, shared_design=shared_design)


def _to_device(problem: Problem, device: torch.device, dtype: torch.dtype) -> Problem:
    return Problem(
        *(value.to(device=device, dtype=dtype) for value in problem.tensors()),
        shared_design=problem.shared_design,
    )


def _full_stacked_qr(problem: Problem) -> Result:
    batch, block_count, block_size = problem.rhs.shape
    n = problem.root.shape[-1]
    if problem.shared_design:
        blocks = problem.blocks.reshape(block_count * block_size, n)
    else:
        blocks = problem.blocks.reshape(batch, block_count * block_size, n)
    rhs = problem.rhs.reshape(batch, block_count * block_size)
    result = householder_information_update_torch(
        problem.root,
        torch.zeros(batch, n, dtype=problem.root.dtype, device=problem.root.device),
        blocks,
        rhs,
    )
    root = result.precision_root[0] if problem.shared_design else result.precision_root
    return Result(root, result.information_rhs, result.solution)


def _streamed_qr(problem: Problem) -> Result:
    batch, block_count, _ = problem.rhs.shape
    n = problem.root.shape[-1]
    information_rhs = torch.zeros(
        batch, n, dtype=problem.root.dtype, device=problem.root.device
    )
    if problem.shared_design:
        root = problem.root
        solution = torch.zeros_like(information_rhs)
        for block_index in range(block_count):
            compiled = compile_information_update_torch(
                root, problem.blocks[block_index]
            )
            update = compiled.apply_vectors(
                information_rhs, problem.rhs[:, block_index]
            )
            root = compiled.precision_root
            information_rhs = update.information_rhs
            solution = update.solution
        return Result(root, information_rhs, solution)

    root = problem.root.expand(batch, n, n).contiguous()
    solution = torch.zeros_like(information_rhs)
    for block_index in range(block_count):
        update = householder_information_update_torch(
            root,
            information_rhs,
            problem.blocks[:, block_index],
            problem.rhs[:, block_index],
        )
        root = update.precision_root
        information_rhs = update.information_rhs
        solution = update.solution
    return Result(root, information_rhs, solution)


def _dense_recompute(problem: Problem) -> Result:
    batch, block_count, _ = problem.rhs.shape
    n = problem.root.shape[-1]
    precision = problem.root.mT @ problem.root
    if not problem.shared_design:
        precision = precision.expand(batch, n, n).clone()
    eta = torch.zeros(batch, n, dtype=problem.root.dtype, device=problem.root.device)
    solution = torch.zeros_like(eta)
    root = problem.root
    information_rhs = eta
    for block_index in range(block_count):
        block = (
            problem.blocks[block_index]
            if problem.shared_design
            else problem.blocks[:, block_index]
        )
        precision = precision + block.mT @ block
        eta = eta + torch.einsum(
            "...mn,...m->...n", block, problem.rhs[:, block_index]
        )
        cholesky = torch.linalg.cholesky(precision)
        root = cholesky.mT
        if problem.shared_design:
            information_rhs = torch.linalg.solve_triangular(
                cholesky,
                eta.transpose(0, 1).contiguous(),
                upper=False,
            ).transpose(0, 1)
            solution = torch.cholesky_solve(
                eta.transpose(0, 1).contiguous(), cholesky
            ).transpose(0, 1)
        else:
            information_rhs = torch.linalg.solve_triangular(
                cholesky, eta.unsqueeze(-1), upper=False
            ).squeeze(-1)
            solution = torch.cholesky_solve(
                eta.unsqueeze(-1), cholesky
            ).squeeze(-1)
    return Result(root, information_rhs, solution)


def _compile_cached_dense_full(problem: Problem) -> CachedDenseFull:
    if not problem.shared_design:
        raise ValueError("cached_dense_full requires a shared fixed design")
    block_count, block_size, n = problem.blocks.shape
    stacked = problem.blocks.reshape(block_count * block_size, n)
    precision = problem.root.mT @ problem.root + stacked.mT @ stacked
    cholesky = torch.linalg.cholesky(precision)
    return CachedDenseFull(cholesky.mT, cholesky, problem.blocks)


def _reference(problem: Problem) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Float64 dense full-trajectory reference on CPU."""
    batch, block_count, block_size = problem.rhs.shape
    n = problem.root.shape[-1]
    if problem.shared_design:
        blocks = problem.blocks.reshape(block_count * block_size, n)
        precision = problem.root.mT @ problem.root + blocks.mT @ blocks
        eta = torch.einsum(
            "mn,bm->bn", blocks, problem.rhs.reshape(batch, block_count * block_size)
        )
        solution = torch.linalg.solve(
            precision, eta.transpose(0, 1)
        ).transpose(0, 1)
    else:
        blocks = problem.blocks.reshape(batch, block_count * block_size, n)
        precision = (
            problem.root.mT @ problem.root
        ).expand(batch, n, n) + blocks.mT @ blocks
        eta = torch.einsum(
            "bmn,bm->bn", blocks, problem.rhs.reshape(batch, block_count * block_size)
        )
        solution = torch.linalg.solve(precision, eta.unsqueeze(-1)).squeeze(-1)
    root = torch.linalg.cholesky(precision).mT
    return precision, root, solution


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().cpu().to(torch.float64)
    expected = expected.detach().cpu().to(torch.float64)
    numerator = torch.linalg.vector_norm(actual - expected)
    denominator = torch.clamp(torch.linalg.vector_norm(expected), min=1e-30)
    return float(numerator / denominator)


def _errors(
    result: Result,
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[float, float, float]:
    expected_precision, expected_root, expected_solution = reference
    root = result.root.detach().cpu().to(torch.float64)
    actual_precision = root.mT @ root
    return (
        _relative_error(result.solution, expected_solution),
        _relative_error(root, expected_root),
        _relative_error(actual_precision, expected_precision),
    )


def _reverse_blocks(problem: Problem) -> Problem:
    block_axis = 0 if problem.shared_design else 1
    return Problem(
        root=problem.root,
        blocks=torch.flip(problem.blocks, dims=(block_axis,)),
        rhs=torch.flip(problem.rhs, dims=(1,)),
        shared_design=problem.shared_design,
    )


def _order_errors(result: Result, reversed_result: Result) -> tuple[float, float]:
    root = result.root.detach().cpu().to(torch.float64)
    reversed_root = reversed_result.root.detach().cpu().to(torch.float64)
    return (
        _relative_error(reversed_result.solution, result.solution),
        _relative_error(reversed_root.mT @ reversed_root, root.mT @ root),
    )


def _output_transfer(result: Result) -> tuple[torch.Tensor, ...]:
    return tuple(
        value.detach().to(device="cpu", copy=True)
        for value in (result.root, result.information_rhs, result.solution)
    )


def _print_header() -> None:
    print(
        "algorithm,design_mode,device,dtype,n,m_block,T,B,M_total,"
        "target_condition,noise_scale,"
        "setup_median_ms,setup_mad_ms,trajectory_median_ms,trajectory_mad_ms,"
        "setup_plus_one_ms,h2d_median_ms,h2d_mad_ms,d2h_median_ms,d2h_mad_ms,"
        "peak_cuda_mib,one_shot_systems_per_second,relative_solution_error,"
        "relative_root_error,relative_precision_gram_error,"
        "reverse_order_solution_delta,reverse_order_precision_delta,status"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dims", type=_csv_positive_ints)
    parser.add_argument("--measurements-per-block", type=int, default=32)
    parser.add_argument("--block-counts", type=_csv_positive_ints)
    parser.add_argument("--batch-sizes", type=_csv_positive_ints)
    parser.add_argument("--design-modes", type=_design_modes)
    parser.add_argument("--dtypes", type=_dtypes)
    parser.add_argument("--condition-numbers", type=_csv_positive_floats)
    parser.add_argument("--noise-scales", type=_csv_positive_floats)
    parser.add_argument("--algorithms", type=_algorithms, default=list(ALGORITHMS))
    parser.add_argument("--devices", default="auto", help="auto or cpu,cuda")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--inner-repeats", type=int, default=1)
    parser.add_argument("--transfer-trials", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help=(
            "fill unspecified axes with n=2,32,128; T=1,2,4,8,16; "
            "B=1,8,32,128; shared,distinct; float32,float64"
        ),
    )
    args = parser.parse_args()

    if args.measurements_per_block <= 0:
        parser.error("--measurements-per-block must be positive")
    if min(args.warmups, args.trials, args.inner_repeats, args.transfer_trials) <= 0:
        parser.error("warmups, trials, repeats, and transfer trials must be positive")
    if args.cpu_threads is not None:
        if args.cpu_threads <= 0:
            parser.error("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)

    if args.full_grid:
        state_dims = args.state_dims or [2, 32, 128]
        block_counts = args.block_counts or [1, 2, 4, 8, 16]
        batch_sizes = args.batch_sizes or [1, 8, 32, 128]
        design_modes = args.design_modes or ["shared", "distinct"]
        dtype_names = args.dtypes or ["float32", "float64"]
        condition_numbers = args.condition_numbers or [10.0]
        noise_scales = args.noise_scales or [1.0]
    else:
        state_dims = args.state_dims or [128]
        block_counts = args.block_counts or [1, 4]
        batch_sizes = args.batch_sizes or [1, 128]
        design_modes = args.design_modes or ["distinct"]
        dtype_names = args.dtypes or ["float32"]
        condition_numbers = args.condition_numbers or [10.0]
        noise_scales = args.noise_scales or [1.0]

    if any(value < 1.0 for value in condition_numbers):
        parser.error("--condition-numbers values must be at least 1")

    devices = _devices(args.devices)
    print("# Streamed block square-root/information benchmark")
    print(f"# host={platform.platform()}")
    print(f"# torch={torch.__version__}; cuda_runtime={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"# cuda_device={torch.cuda.get_device_name(0)}")
    print(
        "# Rc is dense within each block and conditionally block diagonal across T; "
        "whitening is excluded"
    )
    print(
        f"# warmups={args.warmups}; trials={args.trials}; "
        f"inner_repeats={args.inner_repeats}; seed={args.seed}"
    )
    _print_header()

    with torch.inference_mode():
        for n in state_dims:
            for block_count in block_counts:
                for batch in batch_sizes:
                    for condition_number in condition_numbers:
                        for noise_scale in noise_scales:
                            for design_mode in design_modes:
                                shared_design = design_mode == "shared"
                                cpu_problem = _make_problem(
                                    n=n,
                                    m_block=args.measurements_per_block,
                                    block_count=block_count,
                                    batch=batch,
                                    shared_design=shared_design,
                                    condition_number=condition_number,
                                    noise_scale=noise_scale,
                                    seed=args.seed,
                                )
                                reference = _reference(cpu_problem)
                                for dtype_name in dtype_names:
                                    dtype = _dtype(dtype_name)
                                    # Cast on the host for both CPU and CUDA.
                                    # H2D then measures transfer only, not a
                                    # CUDA-only dtype-conversion surcharge.
                                    host_problem = _to_device(
                                        cpu_problem, torch.device("cpu"), dtype
                                    )
                                    for device in devices:
                                        h2d_median, h2d_mad, transferred = _time_transfer(
                                            lambda: _to_device(host_problem, device, dtype),
                                            device=device,
                                            trials=args.transfer_trials,
                                        )
                                        problem = transferred
                                        assert isinstance(problem, Problem)
                                        for algorithm in args.algorithms:
                                            if algorithm == "cached_dense_full" and not shared_design:
                                                continue

                                            total_measurements = block_count * args.measurements_per_block
                                            prefix = (
                                                f"{algorithm},{design_mode},{device.type},{dtype_name},"
                                                f"{n},{args.measurements_per_block},{block_count},{batch},"
                                                f"{total_measurements},{condition_number:.6g},"
                                                f"{noise_scale:.6g}"
                                            )
                                            try:
                                                setup = Timing(0.0, 0.0, 0.0)
                                                if algorithm == "full_stacked_qr":
                                                    function = lambda: _full_stacked_qr(problem)
                                                elif algorithm == "streamed_qr":
                                                    function = lambda: _streamed_qr(problem)
                                                elif algorithm == "dense_recompute":
                                                    function = lambda: _dense_recompute(problem)
                                                elif algorithm == "cached_dense_full":
                                                    setup = _time_callable(
                                                        lambda: _compile_cached_dense_full(problem),
                                                        device=device,
                                                        warmups=args.warmups,
                                                        trials=args.trials,
                                                        inner_repeats=args.inner_repeats,
                                                    )
                                                    compiled = _compile_cached_dense_full(problem)
                                                    function = lambda: compiled.condition(problem.rhs)
                                                else:  # pragma: no cover - argparse validates choices
                                                    raise AssertionError(algorithm)

                                                timing = _time_callable(
                                                    function,
                                                    device=device,
                                                    warmups=args.warmups,
                                                    trials=args.trials,
                                                    inner_repeats=args.inner_repeats,
                                                )
                                                result = function()
                                                _synchronise(device)
                                                assert isinstance(result, Result)
                                                solution_error, root_error, precision_error = _errors(
                                                    result, reference
                                                )
                                                reversed_problem = _reverse_blocks(problem)
                                                if algorithm == "full_stacked_qr":
                                                    reversed_result = _full_stacked_qr(reversed_problem)
                                                elif algorithm == "streamed_qr":
                                                    reversed_result = _streamed_qr(reversed_problem)
                                                elif algorithm == "dense_recompute":
                                                    reversed_result = _dense_recompute(reversed_problem)
                                                else:
                                                    reversed_compiled = _compile_cached_dense_full(
                                                        reversed_problem
                                                    )
                                                    reversed_result = reversed_compiled.condition(
                                                        reversed_problem.rhs
                                                    )
                                                order_solution, order_precision = _order_errors(
                                                    result, reversed_result
                                                )
                                                d2h_median, d2h_mad, copied = _time_transfer(
                                                    lambda: _output_transfer(result),
                                                    device=device,
                                                    trials=args.transfer_trials,
                                                )
                                                del copied
                                                setup_plus_one = (
                                                    setup.median_ms + timing.median_ms
                                                )
                                                one_shot_systems_per_second = (
                                                    1000.0 * batch / setup_plus_one
                                                )
                                                print(
                                                    f"{prefix},{setup.median_ms:.6f},"
                                                    f"{setup.mad_ms:.6f},{timing.median_ms:.6f},"
                                                    f"{timing.mad_ms:.6f},"
                                                    f"{setup_plus_one:.6f},"
                                                    f"{h2d_median:.6f},{h2d_mad:.6f},"
                                                    f"{d2h_median:.6f},{d2h_mad:.6f},"
                                                    f"{max(setup.peak_cuda_mib, timing.peak_cuda_mib):.6f},"
                                                    f"{one_shot_systems_per_second:.3f},"
                                                    f"{solution_error:.9e},"
                                                    f"{root_error:.9e},{precision_error:.9e},"
                                                    f"{order_solution:.9e},{order_precision:.9e},ok"
                                                )
                                                del result, reversed_result
                                            except Exception as exc:
                                                try:
                                                    _synchronise(device)
                                                except Exception:
                                                    pass
                                                message = str(exc).replace("\n", " ").replace(",", ";")
                                                failure_name = type(exc).__name__.lstrip("_")
                                                print(
                                                    f"# failed {algorithm}/{design_mode}/{device.type}/"
                                                    f"{dtype_name}: {failure_name}: {message}"
                                                )
                                                print(
                                                    f"{prefix},nan,nan,nan,nan,nan,"
                                                    f"{h2d_median:.6f},{h2d_mad:.6f},"
                                                    "nan,nan,nan,nan,nan,nan,nan,nan,nan,"
                                                    f"failed_{failure_name}"
                                                )


if __name__ == "__main__":
    main()
