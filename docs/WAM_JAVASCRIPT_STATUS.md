<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# WAM JavaScript (Node) Target — Status

Living summary of the hybrid WAM-JavaScript backend
(`wam_javascript_target.pl` + the Node WAM runtime under
`templates/targets/javascript_wam/`).

Companion docs:

- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md) — the six
  conventions this runtime honours.
- [`WAM_HASKELL_STATUS.md`](WAM_HASKELL_STATUS.md),
  [`WAM_RUST_STATUS.md`](WAM_RUST_STATUS.md),
  [`WAM_CPP_STATUS.md`](WAM_CPP_STATUS.md) — sibling backends.

## Role

**Ubiquitous-runtime interpreter tier.** Prolog → shared WAM bytecode →
a self-contained Node.js WAM interpreter. Zero external npm
dependencies; runs anywhere Node runs. Introduced by the WAMJS-1
research spike.

This is the WAM-target quartet analogue of the reference backends,
implemented in the compact, template-driven style of `wam_lua_target`
(JavaScript, like Lua, is dynamically typed, so terms and instructions
are plain tagged objects).

## Codegen surface

| Module | Role |
|---|---|
| `src/unifyweaver/targets/wam_javascript_target.pl` | emitter (WAM text → JS instruction vector + label table) |
| `templates/targets/javascript_wam/runtime.js.mustache` | the WAM virtual machine (static, `{{date}}` only) |
| `templates/targets/javascript_wam/program.js.mustache` | instruction vector + labels + intern seed + CLI shim |
| `src/unifyweaver/bindings/javascript_wam_bindings.pl` | interpreter builtin catalogue; reserved lowered-tier map |

Exported quartet (analogous to the reference backends):

- `compile_wam_predicate_to_javascript/4`
- `compile_wam_runtime_to_javascript/2`
- `write_wam_javascript_project/3`
- `javascript_wam_resolve_emit_mode/2` → `interpreter`

## Emit tiers

| Tier | State |
|---|---|
| `interpreter` | **Shipped.** Node VM executes the WAM instruction stream. |
| lowered (per-predicate JS functions) | Out of scope for WAMJS-1. `javascript_wam_bindings.pl` reserves the mapping table. |
| FFI / foreign kernels | Out of scope for WAMJS-1. |

`javascript_wam_resolve_emit_mode/2` accepts only `interpreter`; any
other mode raises a `domain_error` rather than silently degrading.

## Project layout produced by `write_wam_javascript_project/3`

```
<ProjectDir>/
  js/
    wam_runtime.js          # the VM (module.exports = { Runtime, V })
    generated_program.js    # data + `module.exports = { program, Runtime, V }`
                            # + CLI: node generated_program.js <pred>/<arity> [args...]
```

The CLI prints `true`/`false` and exits `0`/`1`, so it drops straight
into the conformance harness's process-per-query contract (no build
step — Node is interpreted, like the Python arm).

## The six conventions checklist

Verified against the conformance fixtures (a two-plus-element recursive
list program and a depth-≥2 arithmetic expression), all **PASS**:

- [x] **§1 cons cells have two spellings.** `put_list` and
  `put_structure [|]/2` both intern the **same** functor id, so a
  list cell built either way is the identical `{tag:"struct", fid:[|],
  args:[h,t]}` and unifies directly; the empty list is the atom `[]`
  (interned at a fixed id shared with the runtime). `get_list` accepts
  any 2-arg `[|]` struct.
- [x] **§2 functor `name/arity`, name may contain `/`.**
  `js_parse_functor_arity/3` strips only the trailing `/<digits>`, so
  `///2` → (`//`, 2) and `//2` → (`/`, 2). `cbi_arith` (`17 // 5`)
  evaluates correctly; `=\=` survives escaping into the JS builtin
  dispatch.
- [x] **§3 nested terms built outer-first; placeholder bound/trailed.**
  Write mode uses a **reserve-slot** model: `get`/`put_structure`/`_list`
  allocates the struct with placeholder vars up front and binds
  (trails) the target register's placeholder. A later nested
  `put_structure` into a placeholder binds it, so the outer term sees
  the inner one. Bind-through is gated: the get-family always binds
  (caller output arg); the put-family binds only X/Y (index ≥ 101),
  never A registers.
- [x] **§4 deref before every type test.** `get_structure`/`get_list`
  deref the register before choosing read vs write; `unify`/`exactEqual`
  deref first.
- [x] **§5 `is/2` integer result typing.** `evalArith` carries an
  `isInt` flag; integral results become `{tag:"int"}` and unify with a
  ground integer (`fib`, `ack`).
- [x] **§6 unhandled instruction ⇒ real no-op, never dropped/thrown.**
  Every `switch_on_*` indexing hint and any unrecognised instruction
  emits a one-slot `{op:"NoOp"}`, so label PCs stay aligned and
  backtracking chains (`retry_me_else`/`trust_me`) are never skipped.
  Falling through to the `try_me_else` chain is always correct.

Additional runtime notes:

- **Permanent variables.** Y registers live in per-clause environment
  frames (`allocate`/`deallocate`), so recursive calls do **not**
  clobber a caller's `Y_n` (required by `fib`/`ack`). A/X registers use
  the flat register table.
- **Read/write nesting.** A read-context stack lets a `get_structure`
  nested inside a `get_list` resume the outer cons **tail** afterwards
  (the interleaved head shape).

## Conformance results

Measured with a scratch conformance arm wired exactly as the proposed
integration patch (0-arity wrappers, `node generated_program.js`),
`CONFORMANCE_TARGETS=javascript`, SWI-Prolog 9.0.4 + Node v22:

| Program | Result |
|---|---|
| member | PASS (5/5) |
| append | PASS (4/4) |
| reverse | PASS (4/4) |
| fib | PASS (3/3) |
| ack | PASS (3/3) |
| builtins | PASS (6/6) |
| wide | PASS (3/3) |
| nested | PASS (10/10) |
| buildnest | PASS (2/2) |
| repeatvar | PASS (3/3) |
| emptylist | PASS (5/5) |

**48/48 queries pass — full green, no `ct_xfail` / `ct_skip`.**

## Known gaps

- **Lowered / FFI tiers** not implemented (interpreter only) — by
  design for the spike.
- **Runtime-parser capability** (`runtime_parser(...)`) not wired: the
  CLI parses ground atoms/ints/lists/compounds for convenience, but
  there is no `parser_dependent_body_goal` gating, `read_term`, or the
  native/compiled parser modes the mature backends carry. Predicates
  that need term parsing at runtime are unsupported.
- **Broader builtin surface.** Only the builtins exercised by the
  conformance suite plus the common arithmetic/type family are
  implemented (see `javascript_wam_bindings.pl`). `findall`/`bagof`/
  `setof`/aggregates, `functor/3`, `arg/3`, `=../2`, `copy_term/2`,
  `write/1`, `\+/1` metacall, cut via `get_level`/`cut` beyond the
  simple `cut_ite`/`!` paths, fact-stream / indexed-fact / foreign
  opcodes — all present in the Lua model — are **not** ported. The
  interpreter no-ops or fails cleanly on anything outside its
  catalogue.
- **Indexing is not implemented, only tolerated.** `switch_on_*` are
  no-ops; correctness comes from the try/retry/trust chain. Fine for
  correctness, leaves first-argument-indexing performance on the table.
- **Not yet registered** in `core/target_registry.pl` or the real
  conformance harness — see
  `docs/proposals/integration_patches/WAMJS-1_INTEGRATION_PATCH.md`
  (this card does not edit those files).
