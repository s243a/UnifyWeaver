# T7 route 2 — embedded-aggregate input threading

Status: **implemented**. Completes the last open T7 item — parallelising an
embedded aggregate whose inner goal reads an enclosing-clause input, instead of
declining it to sequential.

## Problem

An embedded aggregate (one goal inside a larger clause body) like

```prolog
eg_p(N, R) :- eg_guard(N), findall(D, (eg_link(N, X), eg_dn(X, D)), R).
```

reads `N` (a head arg) inside the aggregate. Route 2 drives the synthesised
`__par_enum`/`__par_body` helpers by label on cloned machines. Originally it set
the helper's `A1` to the unbound tuple var, so the enumerator ran with `N` free
and enumerated *every* `eg_link` — wrong results. The interim fix DECLINED such
aggregates (compile sequentially, correct but no parallelism). This change
threads the inputs so they parallelise correctly.

## Design

Mirrors the whole-body route-1 input threading (closures binding `a<i>.clone()`),
adapted to route 2's label-driven helpers.

1. **Helpers gain leading params.** `rust_embedded_par_aggregate` computes the
   external inputs (`rust_embedded_aggregate_inputs/4`: inner-goal vars the
   enclosing clause binds, minus the result, in inner-goal first-appearance
   order) and calls `parallel_aggregate_transform/6` with them. The helpers
   become `__par_enum(In1..InK, Tuple)` / `__par_body(In1..InK, Tuple, Value)`.

2. **The instruction carries the input registers.** At the WAM-text splice,
   `rust_block_input_regs/4` recovers the external-input registers from the
   `begin_aggregate..end_aggregate` block: Y-registers **read** in the block but
   **not written** there (so bound by the enclosing clause), minus the value and
   result registers, in first-read order. That order matches the helpers'
   leading-param order, so register *i* feeds param *i*. The emitted line is
   `par_aggregate Type, Enum/EnumArity, Body/BodyArity, ResultReg, Reg1, ...`.
   If the register count ≠ the transform's K, the splice leaves the block
   sequential (a safe fallback rather than mis-threading).

3. **The handler captures values and threads them.**
   `Instruction::ParAggregate(agg_type, enum_label, body_label, result_reg,
   input_regs)` reads each `input_regs` value from the container (Y-aware,
   dereferenced) and passes the vector to `par_collect_labels(.., &input_vals)`.
   `collect_inputs_labeled` / `run_body_labeled` bind `A1..AK` to those values,
   then the tuple (and value) in the trailing arg registers. For an input-less
   embedded aggregate `input_regs` is empty and the tuple/value land in `A1`/`A2`
   exactly as before.

## Files

- `src/unifyweaver/targets/wam_rust_target.pl` — gate now threads (no decline);
  `rust_splice_par_aggregate/5` + `rust_block_input_regs/4` + `rust_line_rw/3` +
  `rust_is_yreg/1`; `wam_line_to_rust_instr` par_aggregate clause emits the
  register vec; `ParAggregate` handler captures input values.
- `templates/targets/rust_wam/instructions.rs.mustache` — `ParAggregate` gains a
  `Vec<String>` input-regs field.
- `templates/targets/rust_wam/par_aggregate.rs.mustache` — `par_collect_labels` /
  `collect_inputs_labeled` / `run_body_labeled` / `map_bodies_labeled` thread
  `input_vals`.
- `templates/targets/rust_wam/state.rs.mustache` — runtime text parser accepts the
  5-field par_aggregate line.

## Tests

- `tests/test_wam_rust_embedded_input_threading.pl` — fast codegen checks
  (no cargo): input-taking ⇒ helper arity `/2,/3` + `vec!["Y2"...]`; input-less ⇒
  `/1,/2` + `vec![]`.
- `tests/test_wam_rust_embedded_input_threading_exec.pl` — cargo-gated:
  `eg_p(1,R) = [[3,2,1]]` (one input) and a two-input order-sensitive case
  (`tw_p(1,2,R) = [[3,2,1]]`) proving param↔register alignment.

Replaces the interim decline test `test_wam_rust_embedded_input_decline_exec.pl`.
