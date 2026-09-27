<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# G-P4 Integration Patch (for INT-0)

**Task:** G-P4 — wire component **emission** into the TypeScript pattern
target's `compile_module/3`, and revive the orphaned `custom_chart` component.
**Worktree:** `agent-aa7611c5aaa2ea56d`
**Shared-file rule:** this agent did **NOT** edit `core/target_registry.pl`,
`docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`, `glue/js_glue.pl`,
`vanilla_js_target.pl`, `annotated_js_target.pl`, or any `wam_*` file.

## Files changed (all inside the allowed set — no central wiring required)

| File | Change |
|------|--------|
| `src/unifyweaver/targets/typescript_target.pl` | Added `compile_collected_components/1` (mirrors `python_target.pl:~195`), exported it, and called it from `compile_module/3` so declared+collected components are compiled via `component_registry:compile_component/4` and appended to the module output. Added `:- use_module('typescript_runtime/custom_chart', [])` so its component type self-registers (was orphaned). |
| `src/unifyweaver/targets/typescript_runtime/custom_chart.pl` | Narrowed the `component_registry` import to `[register_component_type/4]` (silences three "local definition overrides weak import" load warnings). Fixed a latent bug surfaced by reviving the module: the title-display flag was an `( _ -> _ ; _ )` term passed **unevaluated** into `format/3`, emitting invalid TS (`display: Demo\= -> true;false`); it is now precomputed to `"true"`/`"false"`. |
| `tests/core/test_typescript_target.pl` | Added `component_emission_includes_declared` (asserts a `custom_typescript` raw-inject component AND a `custom_chart` component appear in the emitted module, and that the display flag is a real boolean) and `component_free_module_unchanged` (asserts a module with no components carries no component markers — behavior-preserving). |

**No central wiring is required.** All edited files are in the allowed set. The
fix propagates to `annotated_js` / `vanilla_js` automatically: their
`compile_module/3` delegates to `typescript_target:compile_module/3` and then
rewrites (JSDoc / type-strip), so emitted components flow through unchanged.

## Behavior preservation

`compile_collected_components/1` returns `''` when nothing was collected;
`compile_module/3` then sets `Body = PredsSection` and formats exactly as before,
so a module that declares no components is byte-for-byte identical to the
pre-patch output. Verified by `component_free_module_unchanged` and by the
existing `module_multi_predicate` test (still green).

## Note for INT-0 — vanilla_js type-strip coverage (NOT a blocker)

The punchlist flagged a risk that `vanilla_js`'s centralized type-strip might
**mangle** emitted component TS. It does **not** mangle: a `custom_typescript`
raw-inject component strips to clean valid JS
(`export const v = (input) => { return input * 2; };`), and the `custom_chart`
component's *logic* is left intact (`display: true`, correct booleans, `void`
return removed, tuple param annotations removed).

What the regex stripper does **not** cover, for the chart component only, is a
handful of **arbitrary custom-type annotations** it was never taught about:
`: Chart` / `: ChartConfiguration` return-and-var types, the
`CanvasRenderingContext2D | HTMLCanvasElement` union param, and the
`{ labels: ...; datasets: any[] }` object-literal param type. These survive into
the vanilla-JS output as leftover annotations. That is an incompleteness of the
stripper, not corruption, and it applies to any arbitrary TS — not something
specific to this patch.

Fully covering those types would require **editing `vanilla_js_target.pl`**
(`vanilla_js_type_strip/2` / `js_strip_rules/2`), which is outside this task's
allowed file set and owned by INT-0. Suggested follow-up rules (add to
`js_strip_rules/2`, before the generic keyword rule): strip return/var
annotations for bare capitalized identifiers not ending in `Fact`, strip union
type annotations (`: A | B`), and strip inline object-literal type annotations
(`: { ... }`). Deferred; the raw-inject `custom_typescript` path — the common
component case — already yields valid JS today.
