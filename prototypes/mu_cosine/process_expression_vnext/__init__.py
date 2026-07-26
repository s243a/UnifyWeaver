#!/usr/bin/env python3
"""Experimental vNext process-expression frontend kernel.

Proves the parser/elaborator seam described in
``DESIGN_process_expression_patterns.md``, and one edge of its §1 state machine:

```text
functional expression
    -> elaboration-preserving source AST  (functional_parser)
    -> registry-driven elaboration        (elaborator)
    -> in-memory typed term               (ast)

PatternAST --ground(bindings)--> GroundAST   (patterns)
```

The source tree keeps what elaboration and diagnostics need; it does not
promise exact source reconstruction.

**This is not vNext activation.** It creates no identity-bearing bytes, no
canonical ``pe-typed-ast-v1`` serialization, no digests of any kind including a
pattern digest, no receipts, and no deployed processes. Registry ``v0.3``,
``pec-v2``, and both sealed golden bundles are untouched frozen contracts, not
scaffolding this package extends.

The public seam is deliberately small. ``interpret``, ``represent`` and
``verify_factory_receipt`` are the *later* edges of the state machine and are
absent, as are hashing, identity, deployment, resolution and verification —
exporting them would invite exactly the premature identity-minting §1 forbids.
"""
from __future__ import annotations

from .elaborator import (
    ElaborationError,
    elaborate,
    elaborate_ground,
    elaborate_pattern,
    ground_surface,
)
from .functional_parser import NotImplementedInMilestone, ParseError, parse_functional
from .patterns import (
    GroundAST,
    GroundingError,
    PatternAST,
    PatternVar,
    alpha_equivalent,
    ground,
    is_ground,
)
from .registry import Registry, RegistryError, load_registry

__all__ = [
    "parse_functional",
    "elaborate",
    "elaborate_ground",
    "elaborate_pattern",
    "ground",
    "ground_surface",
    "alpha_equivalent",
    "is_ground",
    "load_registry",
    "Registry",
    "PatternAST",
    "PatternVar",
    "GroundAST",
    "ParseError",
    "NotImplementedInMilestone",
    "ElaborationError",
    "GroundingError",
    "RegistryError",
]
