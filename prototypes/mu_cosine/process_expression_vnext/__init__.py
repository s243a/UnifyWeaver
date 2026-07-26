#!/usr/bin/env python3
"""Experimental vNext process-expression frontend kernel.

Proves the parser/elaborator seam described in
``DESIGN_process_expression_patterns.md``:

```text
functional expression
    -> elaboration-preserving source AST  (functional_parser)
    -> registry-driven elaboration        (elaborator)
    -> in-memory typed ground term        (ast)
```

The source tree keeps what elaboration and diagnostics need; it does not
promise exact source reconstruction.

**This is not vNext activation.** It creates no identity-bearing bytes, no
canonical ``pe-typed-ast-v1`` serialization, no digests, no receipts, and no
deployed processes. Registry ``v0.3``, ``pec-v2``, and both sealed golden
bundles are untouched frozen contracts, not scaffolding this package extends.

The public seam is deliberately two functions. Hashing, identity, deployment,
resolution, and verification are *not* exported, because exporting them would
invite exactly the premature identity-minting the state machine in §1 forbids.
"""
from __future__ import annotations

from .elaborator import ElaborationError, elaborate
from .functional_parser import NotImplementedInMilestone, ParseError, parse_functional
from .registry import Registry, RegistryError, load_registry

__all__ = [
    "parse_functional",
    "elaborate",
    "load_registry",
    "Registry",
    "ParseError",
    "NotImplementedInMilestone",
    "ElaborationError",
    "RegistryError",
]
