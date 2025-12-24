"""
UnifyWeaver Code Generation

This package contains compilers that generate target language code from
declarative Prolog specifications.

Modules:
    lda_smoothing_policy_compiler: Compiles smoothing policy to Python/Go/Rust
"""

from .lda_smoothing_policy_compiler import compile_policy, PolicyConfig

__all__ = ['compile_policy', 'PolicyConfig']
