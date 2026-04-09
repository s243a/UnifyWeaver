#!/usr/bin/env python3
"""
Run semantic distance benchmarks at scale.

Precomputes edge weights, generates Prolog benchmarks, and runs:
  - prolog-semantic-min: shortest weighted path (Dijkstra)
  - prolog-eff-semantic: effective semantic distance (power-mean)
  - prolog-accumulated: existing hop-count effective distance (baseline)

Designed for constrained environments (e.g., Termux on Android).
Falls back to hash-based embeddings if sentence-transformers unavailable.

Usage:
    python run_semantic_distance_benchmark.py [--scales dev,300,1k] [--real-embeddings]
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
MIN_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_min_semantic_distance_benchmark.pl"
EFF_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_effective_semantic_distance_benchmark.pl"
PROLOG_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_effective_distance_benchmark.pl"


def hash_embed(text: str, dim: int = 128) -> np.ndarray:
    """Deterministic hash-based embedding (fallback when no ML model available)."""
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def real_embed_factory(model_name: str = "all-MiniLM-L6-v2"):
    """Return an embedding function using transformers (with scipy/sklearn stubs for Termux)."""
    import os, types
    from importlib.machinery import ModuleSpec
    os.environ['HF_HUB_DISABLE_XET'] = '1'

    # Stub scipy/sklearn if broken (common on Termux/Android)
    for name in ['scipy', 'scipy.optimize', 'scipy.sparse', 'scipy.special',
                 'scipy._lib', 'scipy._lib._ccallback', 'scipy.stats']:
        if name not in sys.modules:
            m = types.ModuleType(name); m.__spec__ = ModuleSpec(name, None)
            sys.modules[name] = m
    sys.modules.setdefault('scipy', types.ModuleType('scipy')).__dict__.setdefault('__version__', '1.0.0')
    if hasattr(sys.modules.get('scipy.optimize'), 'linear_sum_assignment') is False:
        sys.modules['scipy.optimize'].linear_sum_assignment = lambda *a: ([], [])
    for name in ['sklearn', 'sklearn.metrics', 'sklearn.metrics.pairwise',
                 'sklearn.cluster', 'sklearn.preprocessing']:
        if name not in sys.modules:
            m = types.ModuleType(name); m.__spec__ = ModuleSpec(name, None)
            sys.modules[name] = m
    if not hasattr(sys.modules.get('sklearn.metrics', None), 'roc_curve'):
        sys.modules['sklearn.metrics'].roc_curve = lambda *a, **k: ([], [], [])

    import torch
    from transformers import AutoTokenizer, AutoModel

    full_name = f"sentence-transformers/{model_name}"
    print(f"  Loading {full_name}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(full_name)
    model = AutoModel.from_pretrained(full_name)
    model.eval()
    cache: dict[str, np.ndarray] = {}

    def embed(text: str) -> np.ndarray:
        if text not in cache:
            clean = text.replace("_", " ")
            inputs = tokenizer([clean], padding=True, truncation=True,
                             return_tensors='pt', max_length=128)
            with torch.no_grad():
                embs = model(**inputs).last_hidden_state.mean(dim=1)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            cache[text] = embs.numpy()[0]
        return cache[text]

    return embed


def precompute_weights(edges_tsv: Path, output_dir: Path, use_real: bool) -> Path:
    """Precompute edge weights and write edge_weights.pl."""
    weights_path = output_dir / "edge_weights.pl"
    if weights_path.exists():
        print(f"  Using cached: {weights_path}", file=sys.stderr)
        return weights_path

    # Load edges
    edges = []
    with open(edges_tsv) as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))

    nodes = sorted(set(n for e in edges for n in e))
    print(f"  {len(nodes)} nodes, {len(edges)} edges", file=sys.stderr)

    # Choose embedding function
    if use_real:
        try:
            embed = real_embed_factory()
            print("  Using sentence-transformers (real embeddings)", file=sys.stderr)
        except ImportError:
            print("  sentence-transformers not available, using hash embeddings", file=sys.stderr)
            embed = hash_embed
    else:
        embed = hash_embed
        print("  Using hash embeddings (deterministic)", file=sys.stderr)

    # Compute embeddings
    t0 = time.perf_counter()
    emb_map = {n: embed(n) for n in nodes}
    embed_time = time.perf_counter() - t0
    print(f"  Embeddings computed in {embed_time:.1f}s", file=sys.stderr)

    # Compute weights
    weighted = []
    for child, parent in edges:
        a, b = emb_map[child], emb_map[parent]
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        w = max(0.001, 1.0 - sim)  # clamp to avoid zero weights
        weighted.append((child, parent, round(w, 6)))

    weights = [w for _, _, w in weighted]
    print(f"  Weights: min={min(weights):.4f} max={max(weights):.4f} "
          f"mean={sum(weights)/len(weights):.4f}", file=sys.stderr)

    # Write
    with open(weights_path, "w") as f:
        f.write("%% Precomputed semantic edge weights\n")
        for child, parent, w in weighted:
            c = child.replace("'", "\\'")
            p = parent.replace("'", "\\'")
            f.write(f"edge_weight('{c}', '{p}', {w}).\n")

    print(f"  Wrote: {weights_path}", file=sys.stderr)
    return weights_path


def run_prolog(cmd: list[str], timeout: int = 300) -> tuple[float, str]:
    """Run a Prolog command, return (elapsed_seconds, stdout)."""
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0
    return elapsed, result.stdout


def generate_and_run(generator: str, facts: Path, weights: Path,
                     output: Path, label: str, reps: int = 3,
                     extra_args: list[str] | None = None,
                     timeout: int = 300) -> dict:
    """Generate a Prolog benchmark and run it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["swipl", "-q", "-s", generator, "--", str(facts), str(weights), str(output)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    times = []
    stdout = ""
    for i in range(reps):
        try:
            elapsed, stdout = run_prolog(["swipl", "-q", "-s", str(output)], timeout=timeout)
            times.append(elapsed)
        except subprocess.TimeoutExpired:
            print(f"  {label}: timeout at rep {i+1} ({timeout}s)", file=sys.stderr)
            times.append(float("inf"))
            break

    rows = len(stdout.strip().split("\n")) if stdout.strip() else 0
    return {
        "label": label,
        "times": times,
        "median": statistics.median(times) if times and all(t < float("inf") for t in times) else float("inf"),
        "rows": rows,
    }


def run_hop_count_baseline(facts: Path, output: Path, scale: str,
                           reps: int = 3, timeout: int = 300) -> dict | None:
    """Run the hop-count accumulated baseline if generator exists."""
    if not PROLOG_GENERATOR.exists():
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["swipl", "-q", "-s", str(PROLOG_GENERATOR), "--",
             str(facts), str(output), "accumulated"],
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError:
        return None

    times = []
    stdout = ""
    for i in range(reps):
        try:
            elapsed, stdout = run_prolog(["swipl", "-q", "-s", str(output)], timeout=timeout)
            times.append(elapsed)
        except subprocess.TimeoutExpired:
            times.append(float("inf"))
            break

    rows = len(stdout.strip().split("\n")) if stdout.strip() else 0
    return {
        "label": "hop-accumulated",
        "times": times,
        "median": statistics.median(times) if times and all(t < float("inf") for t in times) else float("inf"),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="dev,300", help="Comma-separated scales")
    parser.add_argument("--real-embeddings", action="store_true", help="Use sentence-transformers")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per benchmark")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per run in seconds")
    args = parser.parse_args()

    scales = [s.strip() for s in args.scales.split(",")]

    print("=" * 70)
    print("Semantic Distance Benchmark")
    print(f"Scales: {scales}, Reps: {args.reps}, Timeout: {args.timeout}s")
    print(f"Embeddings: {'real (sentence-transformers)' if args.real_embeddings else 'hash-based (deterministic)'}")
    print("=" * 70)

    for scale in scales:
        scale_dir = BENCH_DIR / scale
        facts = scale_dir / "facts.pl"
        edges_tsv = scale_dir / "category_parent.tsv"

        if not facts.exists() or not edges_tsv.exists():
            print(f"\n[{scale}] SKIP — data not found", file=sys.stderr)
            continue

        print(f"\n{'='*70}")
        print(f"Scale: {scale}")
        print(f"{'='*70}")

        # Precompute weights
        print("\nPrecomputing edge weights...", file=sys.stderr)
        weights = precompute_weights(edges_tsv, scale_dir, args.real_embeddings)

        tmp = Path(f"/data/data/com.termux/files/home/tmp/bench_{scale}")
        tmp.mkdir(parents=True, exist_ok=True)

        results = []

        # Min semantic distance
        print(f"\n[{scale}] Running min semantic distance...", file=sys.stderr)
        r = generate_and_run(
            str(MIN_GENERATOR), facts, weights,
            tmp / "min_semantic.pl", "semantic-min",
            reps=args.reps, timeout=args.timeout)
        results.append(r)

        # Effective semantic distance
        print(f"\n[{scale}] Running effective semantic distance (N=5)...", file=sys.stderr)
        r = generate_and_run(
            str(EFF_GENERATOR), facts, weights,
            tmp / "eff_semantic.pl", "eff-semantic-N5",
            reps=args.reps, timeout=args.timeout)
        results.append(r)

        # Hop-count baseline
        print(f"\n[{scale}] Running hop-count accumulated baseline...", file=sys.stderr)
        hop_result = run_hop_count_baseline(
            facts, tmp / "hop_accumulated.pl", scale,
            reps=args.reps, timeout=args.timeout)
        if hop_result:
            results.append(hop_result)

        # Print results table
        print(f"\n--- Results: {scale} ---")
        print(f"{'Target':<25} {'Median (s)':>10} {'Rows':>8}")
        print("-" * 45)
        for r in results:
            med = f"{r['median']:.3f}" if r["median"] < float("inf") else "TIMEOUT"
            print(f"{r['label']:<25} {med:>10} {r['rows']:>8}")

        # Speedup comparison
        if len(results) >= 2 and results[0]["median"] < float("inf"):
            base = results[0]
            for r in results[1:]:
                if r["median"] < float("inf") and r["median"] > 0:
                    ratio = r["median"] / base["median"]
                    print(f"  {r['label']} vs {base['label']}: {ratio:.2f}x")

    print(f"\n{'='*70}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
