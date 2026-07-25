#!/usr/bin/env python3
"""Full process identity — step 1 of the encoder handoff's delegable sequence.

``DESIGN_expression_encoder_future.md`` §2 keeps identity *outside* the learned
bottleneck.  A latent, a position vector, or any other derived representation is
a conditioning feature; it is never a process identifier, cache identity,
provenance key, or proof of equality.  The lossless identity contract is:

    canonical AST bytes
    + exact REGISTRY_VERSION
    + full 64-hex AST digest
    + factory/manifest fingerprint

This module supplies that contract.  It changes no P0 behavior:
``process_cards.ast_sha`` keeps returning its compact 16-hex convenience key and
stays valid for the existing card caches.  What this module adds is the stricter
identity required for handoff artifacts, split joins, residual lookup, and
provenance — and it *fails closed* on a compact-only identity rather than
widening it.

Grammar-valid synthetic expressions need not have executable factories.  They
carry a ``synthetic_sample_digest`` binding the full AST digest to an immutable
``generator_spec_sha256`` and are marked ``synthetic_only``.  A synthetic row can
never be promoted to a deployed identity without a separately verified
factory/manifest fingerprint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re
import sys
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process_cards import REGISTRY_VERSION, canonical, parse, validate

#: The digest convention frozen by ``PROTOCOL_process_expression_p1.md``.
IDENTITY_DIGEST_ALGORITHM = "sha256"
IDENTITY_DIGEST_HEX_LENGTH = 64
IDENTITY_PREIMAGE = "REGISTRY_VERSION|canonical_identity_string"

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_COMPACT_HEX = re.compile(r"\A[0-9a-f]{16}\Z")


class ProcessIdentityError(ValueError):
    """A process identity is malformed, truncated, or unbound."""


def canonical_identity_string(node) -> str:
    """The lossless V3-plus-resolved-defaults rendering used for identity."""

    validate(node)
    return canonical(node)


def canonical_identity_bytes(node) -> bytes:
    """Callers must retain these bytes, not only a digest (§2)."""

    return canonical_identity_string(node).encode("utf-8")


def full_ast_digest(node) -> str:
    """``sha256(REGISTRY_VERSION + "|" + canonical_identity_string)``, 64 hex.

    This is the digest already frozen by the P1 protocol, kept byte-identical
    here on purpose: this handoff does not introduce a second convention.
    """

    preimage = f"{REGISTRY_VERSION}|{canonical_identity_string(node)}"
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def full_ast_digest_for_expression(expression: str) -> str:
    return full_ast_digest(parse(expression))


def require_full_digest(value: Any, label: str = "process digest") -> str:
    """Reject a truncated identity loudly instead of widening it."""

    if not isinstance(value, str):
        raise ProcessIdentityError(f"{label} must be a string")
    if _COMPACT_HEX.fullmatch(value):
        raise ProcessIdentityError(
            f"{label} is a compact 16-hex convenience key; handoff artifacts "
            "require the full 64-hex digest bound to a factory fingerprint"
        )
    if not _HEX64.fullmatch(value):
        raise ProcessIdentityError(
            f"{label} must be {IDENTITY_DIGEST_HEX_LENGTH} lowercase hex characters"
        )
    return value


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProcessIdentityError(f"{label} must be a nonempty string")
    return value


@dataclass(frozen=True)
class ProcessIdentity:
    """A deployed process identity: canonical bytes plus both fingerprints."""

    canonical_bytes: bytes
    registry_version: str
    full_digest: str
    factory_fingerprint: str

    def as_record(self) -> dict[str, Any]:
        return {
            "canonical_identity_string": self.canonical_bytes.decode("utf-8"),
            "registry_version": self.registry_version,
            "digest_algorithm": IDENTITY_DIGEST_ALGORITHM,
            "digest_preimage": IDENTITY_PREIMAGE,
            "full_process_digest": self.full_digest,
            "factory_fingerprint": self.factory_fingerprint,
            "synthetic_only": False,
        }

    @property
    def identity_key(self) -> str:
        """The residual-lookup key: digest *and* factory, never digest alone."""

        return f"{self.full_digest}|{self.factory_fingerprint}"


def deployed_identity(node, *, factory_fingerprint: str) -> ProcessIdentity:
    """Bind a validated AST to the factory/manifest that realizes it."""

    _require_nonempty_str(factory_fingerprint, "factory_fingerprint")
    return ProcessIdentity(
        canonical_bytes=canonical_identity_bytes(node),
        registry_version=REGISTRY_VERSION,
        full_digest=full_ast_digest(node),
        factory_fingerprint=factory_fingerprint,
    )


@dataclass(frozen=True)
class SyntheticSampleIdentity:
    """A grammar-valid sample with no executable factory (§2, §6)."""

    canonical_bytes: bytes
    registry_version: str
    full_digest: str
    generator_spec_sha256: str
    synthetic_sample_digest: str

    def as_record(self) -> dict[str, Any]:
        return {
            "canonical_identity_string": self.canonical_bytes.decode("utf-8"),
            "registry_version": self.registry_version,
            "digest_algorithm": IDENTITY_DIGEST_ALGORITHM,
            "full_process_digest": self.full_digest,
            "generator_spec_sha256": self.generator_spec_sha256,
            "synthetic_sample_digest": self.synthetic_sample_digest,
            "synthetic_only": True,
        }


def synthetic_sample_digest(full_digest: str, generator_spec_sha256: str) -> str:
    """Bind a sample's AST digest to the immutable generator specification."""

    require_full_digest(full_digest, "full_process_digest")
    require_full_digest(generator_spec_sha256, "generator_spec_sha256")
    preimage = f"{full_digest}|{generator_spec_sha256}"
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def synthetic_identity(node, *, generator_spec_sha256: str) -> SyntheticSampleIdentity:
    digest = full_ast_digest(node)
    return SyntheticSampleIdentity(
        canonical_bytes=canonical_identity_bytes(node),
        registry_version=REGISTRY_VERSION,
        full_digest=digest,
        generator_spec_sha256=generator_spec_sha256,
        synthetic_sample_digest=synthetic_sample_digest(digest, generator_spec_sha256),
    )


def verify_identity_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute an identity record from its own canonical bytes.

    A filename, branch name, compact hash, or self-reported manifest field is
    not provenance (§9): everything checkable is recomputed here.
    """

    if not isinstance(record, Mapping):
        raise ProcessIdentityError("identity record must be an object")
    expression = _require_nonempty_str(
        record.get("canonical_identity_string"), "canonical_identity_string"
    )
    declared_registry = _require_nonempty_str(
        record.get("registry_version"), "registry_version"
    )
    if declared_registry != REGISTRY_VERSION:
        raise ProcessIdentityError(
            f"identity was minted under registry {declared_registry!r}; this "
            f"process runs {REGISTRY_VERSION!r} and an identity-version "
            "migration is required rather than a silent reinterpretation"
        )
    if record.get("digest_algorithm") not in (None, IDENTITY_DIGEST_ALGORITHM):
        raise ProcessIdentityError("unsupported identity digest algorithm")

    node = parse(expression)
    recomputed_canonical = canonical_identity_string(node)
    if recomputed_canonical != expression:
        raise ProcessIdentityError(
            "canonical_identity_string is not canonical under this registry"
        )
    recomputed_digest = full_ast_digest(node)
    declared_digest = require_full_digest(
        record.get("full_process_digest"), "full_process_digest"
    )
    if declared_digest != recomputed_digest:
        raise ProcessIdentityError("full_process_digest does not match its own bytes")

    synthetic_only = record.get("synthetic_only")
    if not isinstance(synthetic_only, bool):
        raise ProcessIdentityError("synthetic_only must be a boolean")
    if synthetic_only:
        spec = require_full_digest(
            record.get("generator_spec_sha256"), "generator_spec_sha256"
        )
        expected = synthetic_sample_digest(recomputed_digest, spec)
        if record.get("synthetic_sample_digest") != expected:
            raise ProcessIdentityError(
                "synthetic_sample_digest does not bind this AST to its generator spec"
            )
        if record.get("factory_fingerprint") is not None:
            raise ProcessIdentityError(
                "a synthetic sample cannot carry a factory fingerprint; promotion "
                "requires separate verification"
            )
    else:
        _require_nonempty_str(record.get("factory_fingerprint"), "factory_fingerprint")
        if record.get("generator_spec_sha256") is not None:
            raise ProcessIdentityError(
                "a deployed identity is not a generated sample"
            )
    return dict(record)


def promote_synthetic(record: Mapping[str, Any], *, factory_fingerprint: str):
    """Refuse promotion without separate factory verification (§2)."""

    verify_identity_record(record)
    raise ProcessIdentityError(
        "synthetic rows cannot be promoted to deployed process identities here; "
        "mint a deployed identity from a separately verified factory/manifest "
        "fingerprint instead"
    )
