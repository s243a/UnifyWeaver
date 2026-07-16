#!/usr/bin/env python3
"""Build and validate frozen, revision-pinned node embedding caches.

Evaluation code only consumes the three deterministic files written under an
output prefix.  Model loading is confined to the CLI and defaults to
``local_files_only=True`` so tests and covariance evaluation never download.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingModelSpec:
    model_id: str
    revision: str
    task_prefix: str
    dimension: int
    trust_remote_code: bool
    text_transform: str = "underscores_to_spaces"
    normalization: str = "l2"


MODEL_SPECS = {
    "minilm": EmbeddingModelSpec(
        "sentence-transformers/all-MiniLM-L6-v2",
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "",
        384,
        False,
    ),
    "nomic": EmbeddingModelSpec(
        "nomic-ai/nomic-embed-text-v1.5",
        "e9b6763023c676ca8431644204f50c2b100d9aab",
        "clustering: ",
        768,
        True,
    ),
}


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _content_record(value):
    return {"size_bytes": len(value), "sha256": _sha256_bytes(value)}


def canonical_names(names):
    """Stable unique title order; empty titles and normalization collisions fail."""
    names = [str(value) for value in names]
    if any(not value.strip() for value in names):
        raise ValueError("node names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("node names must be unique")
    return tuple(sorted(names))


def embedding_texts(names, spec):
    names = canonical_names(names)
    if spec.text_transform != "underscores_to_spaces":
        raise ValueError(f"unsupported text transform: {spec.text_transform}")
    return tuple(spec.task_prefix + value.replace("_", " ") for value in names)


def validate_embeddings(names, embeddings, spec):
    names = tuple(names)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (len(names), spec.dimension):
        raise ValueError(
            f"embedding shape {embeddings.shape} does not match "
            f"({len(names)}, {spec.dimension})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("embeddings must be nonzero")
    normalized = embeddings / norms[:, None].astype(np.float32)
    if not np.allclose(norms, 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError("cache contract requires L2-normalized embeddings")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def encode_with_model(names, spec, model, *, batch_size=256):
    """Encode canonical node titles using an injected SentenceTransformer-like model."""
    names = canonical_names(names)
    texts = embedding_texts(names, spec)
    embeddings = model.encode(
        list(texts),
        batch_size=int(batch_size),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return names, texts, validate_embeddings(names, embeddings, spec)


def _npy_bytes(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        stream.write(value)
    os.replace(temporary, path)


def cache_paths(prefix):
    prefix = Path(prefix)
    return {
        "nodes": Path(str(prefix) + ".nodes.json"),
        "embeddings": Path(str(prefix) + ".embeddings.npy"),
        "manifest": Path(str(prefix) + ".manifest.json"),
    }


def write_embedding_cache(prefix, names, embeddings, spec, *, package_versions=None):
    """Write a deterministic three-file cache and return its portable manifest."""
    names = tuple(names)
    if canonical_names(names) != names:
        raise ValueError("names must already be canonical and sorted before writing")
    embeddings = validate_embeddings(names, embeddings, spec)
    texts = embedding_texts(names, spec)
    nodes_bytes = (
        json.dumps(list(names), ensure_ascii=False, indent=2, separators=(",", ": ")) + "\n"
    ).encode("utf-8")
    embeddings_bytes = _npy_bytes(embeddings)
    text_bytes = (json.dumps(list(texts), ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    manifest = {
        "schema_version": 1,
        "model": asdict(spec),
        "node_count": len(names),
        "array": {"shape": list(embeddings.shape), "dtype": str(embeddings.dtype)},
        "input_text_sha256": _sha256_bytes(text_bytes),
        "artifacts": {
            "nodes": _content_record(nodes_bytes),
            "embeddings": _content_record(embeddings_bytes),
        },
        "package_versions": dict(sorted((package_versions or {}).items())),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    paths = cache_paths(prefix)
    _atomic_write(paths["nodes"], nodes_bytes)
    _atomic_write(paths["embeddings"], embeddings_bytes)
    _atomic_write(paths["manifest"], manifest_bytes)
    return manifest, _content_record(manifest_bytes)


@dataclass(frozen=True)
class EmbeddingCache:
    names: tuple
    embeddings: np.ndarray
    metadata: dict
    manifest_sha256: str

    def by_name(self):
        return {name: self.embeddings[row] for row, name in enumerate(self.names)}


def load_embedding_cache(prefix, *, expected_spec=None, required_names=()):
    paths = cache_paths(prefix)
    raw = {name: path.read_bytes() for name, path in paths.items()}
    manifest = json.loads(raw["manifest"].decode("utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported embedding-cache schema")
    for role in ("nodes", "embeddings"):
        if manifest["artifacts"][role] != _content_record(raw[role]):
            raise ValueError(f"{role} content hash/size does not match the manifest")
    names = tuple(json.loads(raw["nodes"].decode("utf-8")))
    if canonical_names(names) != names:
        raise ValueError("cached names must be canonical, unique, and sorted")
    spec = EmbeddingModelSpec(**manifest["model"])
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("embedding model specification does not match the required frozen spec")
    text_bytes = (
        json.dumps(
            list(embedding_texts(names, spec)),
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
    ).encode("utf-8")
    if manifest.get("input_text_sha256") != _sha256_bytes(text_bytes):
        raise ValueError("input text hash does not match names and the frozen text contract")
    embeddings = np.load(io.BytesIO(raw["embeddings"]), allow_pickle=False)
    embeddings = validate_embeddings(names, embeddings, spec)
    if manifest["array"] != {"shape": list(embeddings.shape), "dtype": str(embeddings.dtype)}:
        raise ValueError("embedding array metadata does not match its content")
    missing = sorted(set(required_names) - set(names))
    if missing:
        raise ValueError(f"embedding cache misses {len(missing)} required nodes; first={missing[0]!r}")
    return EmbeddingCache(
        names,
        embeddings,
        manifest,
        _sha256_bytes(raw["manifest"]),
    )


def _read_nodes(path):
    with open(path, encoding="utf-8") as stream:
        return canonical_names(line.rstrip("\n") for line in stream if line.strip())


def read_pair_nodes(path):
    """Canonical endpoint titles from the first two columns of a TSV campaign."""
    nodes = set()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 2 or not columns[0] or not columns[1]:
                raise ValueError(f"malformed pair row at line {line_number}")
            nodes.update(columns[:2])
    if not nodes:
        raise ValueError("pair TSV contains no data rows")
    return canonical_names(nodes)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--nodes", help="one exact node title per line")
    source.add_argument("--pairs-tsv", help="campaign TSV; first two columns are endpoint titles")
    parser.add_argument("--preset", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow model retrieval; default evaluation contract is local cache only",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    from sentence_transformers import SentenceTransformer, __version__ as sentence_transformers_version
    import transformers

    spec = MODEL_SPECS[args.preset]
    model = SentenceTransformer(
        spec.model_id,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=not args.allow_download,
        device=args.device,
    )
    input_names = _read_nodes(args.nodes) if args.nodes else read_pair_nodes(args.pairs_tsv)
    names, _texts, embeddings = encode_with_model(
        input_names, spec, model, batch_size=args.batch_size
    )
    manifest, manifest_record = write_embedding_cache(
        args.out_prefix,
        names,
        embeddings,
        spec,
        package_versions={
            "sentence_transformers": sentence_transformers_version,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
    )
    print(json.dumps({
        "model": manifest["model"],
        "node_count": manifest["node_count"],
        "manifest_sha256": manifest_record["sha256"],
        "out_prefix": os.path.abspath(args.out_prefix),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
