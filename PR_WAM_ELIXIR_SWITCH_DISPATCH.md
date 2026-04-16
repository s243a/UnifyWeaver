# Title
feat(wam-elixir): implement clause indexing and global predicate dispatcher

# Description
This PR finalizes two significant architectural features for the lowered Elixir WAM target, fulfilling the next steps identified on the `main` branch roadmap.

### Key Enhancements

*   **First & Second Argument Indexing (`switch_on_constant`)**:
    *   Implemented full lowering for `switch_on_constant` and `switch_on_constant_a2`.
    *   These instructions now emit highly optimized Elixir `case` statements that branch directly to the matching clause function, bypassing unnecessary choice points.
    *   The design elegantly integrates with the existing idiomatic `try/catch` backtracking model.
    *   **Fixes applied from code review**: 
        *   Unmatched constants now properly throw `:fail` instead of silently falling through, respecting WAM backtracking semantics.
        *   The `"default"` label dispatch case now correctly falls through to the immediately following instruction (usually `try_me_else`), enabling standard WAM fallback behaviour.
        *   Fixed missing variable dereferencing in the interpreter path for `switch_on_constant`.
        *   Choice point (`try_me_else`/`retry_me_else`) wrappers correctly catch and bubble up `{:return, result}` throws.
*   **Global Predicate Dispatcher**:
    *   Introduced `WamDispatcher` generation within `write_wam_elixir_project/3`. This module acts as a global router for all compiled predicates.
    *   Updated the `call` and `execute` instructions in the lowered emitter to route through `WamDispatcher.call("pred/arity", state)` rather than hardcoding direct static module calls.
    *   This unlocks support for dynamic calls and higher-order predicates (`call/N`) that cannot be resolved to a specific module at compile time.
    *   **Note**: The interpreter mode currently only supports intra-module calls via labels. This limitation has been explicitly documented inline.

### Testing
*   Expanded `test_elixir_baseline.pl` to compile a multi-clause predicate (`test_animal/2`) that triggers `switch_on_constant` indexing.
*   Added assertions to explicitly verify:
    *   The generation of the `WamDispatcher` module.
    *   The updated lowering of the `call` instruction delegating to `WamDispatcher`.
    *   The correct syntax and early-return throw (`throw({:return, ...})`) mechanism of `switch_on_constant`.

### Future Work
*   **Performance Benchmarking**: The lowered emitter is now functionally complete for core WAM operations. A formal benchmark suite comparing Interpreted Elixir vs. Lowered Elixir is the recommended next step to quantify these optimizations.

---
*Co-authored-By: Claude Opus 4.6 (1M context)*