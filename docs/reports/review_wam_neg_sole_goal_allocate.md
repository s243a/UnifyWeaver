# Pre-merge review packet — WAM sole-`\+` allocate fix + pkg_resolver deb wiring

> **Self-contained.** You do not have repo access; every source section you need is inlined
> below (Appendix A–D). File paths are given only for traceability. This is UnifyWeaver, a
> Prolog→WAM transpiler that emits code for several backend targets. The change makes a new
> C++ target lane of the `pkg_resolver` benchmark pass its 51-scenario contract corpus against
> SWI-Prolog as the oracle (46/51 → 51/51). Two fixes. **Fix 1 is in the SHARED WAM compiler**
> (`wam_target.pl`) and changes emitted code for *every* target compiled with
> `ite_use_y_level(true)` — cpp, go, rust, python, lua, haskell, javascript, llvm — so its
> blast radius is fleet-wide. **Fix 2 is example-local** (one build script) with zero blast
> radius. Both root causes were established with a WAM instruction dump + live gdb.
> `resolver.pl` is the FROZEN spec and was NOT modified.

> **Update (post-review, commit `a6e0baca5`).** After the initial fix (`231262180`) a
> follow-up refined it and generalized it fleet-wide: (1) the guard is tightened to
> `wam_ite_use_y_level_enabled, wam_inline_not_enabled, G = \+(_)` — a frame is forced *only*
> when `\+` is actually inlined to the soft-cut (the `inline_not_as_failure(false)`
> runtime-builtin path emits no Y-level barrier, so needs none), and the redundant `not(_)`
> disjunct is dropped (`not/1` already reaches the callable arm); (2) the same frame-isolation
> hardening was extended across the js / lua / python / haskell lowered backends; (3) regression
> tests were added — `test_wam_sole_negation_frame_gate` (compiler-level: inlined y-level
> allocates; legacy *and* runtime-builtin modes do not) and `cpp_e2e_not_callee_frame_isolation`
> (behavioral). The diffs and Appendix A below show the **final, refined** form.

---

## FIX 1 (SHARED, fleet-wide) — force an environment frame for sole-`\+`/`not` clauses under `ite_use_y_level`

### The diff (`src/unifyweaver/targets/wam_target.pl`, in `goals_contain_call_or_aggregate/1`)
```prolog
     ; wam_inline_bagof_setof_enabled, G = bagof(_, _, _)
     ; wam_inline_bagof_setof_enabled, G = setof(_, _, _)
+    % Under ite_use_y_level(true), `\+ G` compiles to the soft-cut form
+    % `(G -> fail ; true)`, whose get_level/cut barrier needs a permanent
+    % Y-register. A clause whose ONLY body goal is such a negation would
+    % otherwise skip allocation (length(Goals) == 1 and `\+` is treated as
+    % a builtin below), so its barrier Y-register aliases the CALLER's
+    % frame and corrupts a shared register cell. Force a frame here.
+    % (`not/1` already forces allocation via the callable catch-all below;
+    % `\+` is exempted there by is_builtin_goal((\+)), hence this case.)
+    ; wam_ite_use_y_level_enabled, wam_inline_not_enabled, G = \+(_)
     ; callable(G), functor(G, F, _), \+ is_builtin_goal(F)
     ),
     !.
```

### Root cause (gdb-verified)
1. A single-goal clause emits `allocate` (an environment frame, which backs the permanent
   **Y-registers**) only if `length(Goals) > 1` **or** `goals_contain_call_or_aggregate/1`
   is true — see `compile_single_clause_wam/3` in Appendix A.
2. `goals_contain_call_or_aggregate/1` treated `\+` as a no-frame builtin, because
   `is_builtin_goal((\+))` is a fact (Appendix A).
3. But under `ite_use_y_level(true)`, the negation compiler rewrites `\+ G` into the soft-cut
   if-then-else `(G -> fail ; true)` (Appendix B), whose `get_level`/`cut` barrier **requires a
   permanent Y-register**.
4. So a clause whose *entire body* is one `\+ G` got **no `allocate`**, yet its barrier still
   emitted `get_level Y1` / `cut Y1`.
5. In the C++ runtime, a Y-register is resolved through `env_stack.back().y_regs` (Appendix C).
   With no frame pushed, `env_stack.back()` is the **caller's** frame, so `get_level Y1`
   binds the *caller's* `Y1` cell. Registers are passed by **shared `CellPtr`** (Appendix C,
   `set_cell(a,c); set_cell(b,c)`), so that one mutation corrupts both a caller variable and
   the callee's own argument at once.
6. Concretely for `satisfies(Ver, gte(G)) :- \+ version_lt(Ver, G).` (Appendix D): the barrier
   clobbered the caller's `BV`; `version_lt` then saw a corrupted integer and wrongly failed;
   so `\+ version_lt(...)` wrongly succeeded, `\+ satisfies(...)` wrongly failed, and the
   outer if-then-else in `blocked_acc` took the wrong (ELSE) branch — the observed corpus bug.

### Why the fix is correct and contained
- Forcing the clause into the `allocate` branch (Appendix A) makes `compile_goals` run with
  `HasEnv = yes`; the if-then-else compiler then pairs a `deallocate` before `proceed` on the
  clause's exit (Appendix B) — so the frame is properly opened and closed by the existing
  machinery. The barrier Y-register now lives in the clause's *own* frame.
- **Guarded on `wam_ite_use_y_level_enabled`**, so legacy (non-y-level) codegen is byte-
  identical. The only behavioral change under y-level mode: a clause whose sole body goal is
  `\+`/`not` gains an `allocate`/`deallocate` pair it previously lacked.
- `not/1` already forced allocation via the `callable(G), \+ is_builtin_goal(F)` catch-all
  (it is absent from `is_builtin_goal/1`); the new clause adds the `\+` spelling and harmlessly
  restates `not`.

---

## FIX 2 (EXAMPLE-LOCAL, zero blast) — pull `predsort` stdlib into the pkg_resolver C++ build

### The change (`examples/pkg_resolver/cpp/build.pl`)
Add `include_stdlib([predsort])` to the options passed to `write_wam_cpp_project/3`.

### Root cause (verified by scratch rebuild)
`sort_versions_desc/2` (Appendix D) sorts non-`v(_,_,_)` versions (i.e. every Debian `deb(...)`
term) via `predsort(cmp_ver, ...)`. `predsort/3`'s helper clauses are provided by a load-time
`assertz` in the compiler, not as textual clauses, and are only compiled into the output when
the caller passes `include_stdlib(...)`. The pkg_resolver build didn't, so the generated C++
contained a real `Call("predsort/3", 3)` site but **no `predsort/3` label** — at runtime the
call found no label, fell through the builtin dispatch (no case), and returned `false`
silently. The three `deb` corpus scenarios are the only ones that reach `predsort`, which is
exactly the residual 3-case failure set after Fix 1. Adding `include_stdlib([predsort])` made
`resolve(deb_tilde_catalog,[app],Sel)` return `[app-deb_10, lib-deb_10]`, matching SWI.
`include_stdlib` is an established, tested option (used throughout the generator test suite).

---

## Test evidence (all on this box)

| Check | Result |
|---|---|
| pkg_resolver C++ corpus — before any fix | 46 / 51 |
| after Fix 1 (shared) | 48 / 51 (both non-deb blocked cases fixed; minimal reproducer fixed) |
| after Fix 2 (deb wiring) | **51 / 51** — byte-identical to the SWI oracle |
| `tests/test_wam_target.pl` (shared-compiler WAM-level unit suite, incl. allocate/deallocate cases) | ALL PASS |
| `tests/test_wam_go_negation_cut.pl` (Go negation/cut — a second `ite_use_y_level` target) | ALL PASS |
| `tests/test_wam_cpp_generator.pl` — **404 `cpp_e2e_*` units** (the C++ end-to-end regression) | **404 / 404 PASS, 0 failures** |

### Regression tests (landed with the refinement `a6e0baca5`, addressing codex's finding)
| Test | Location | Coverage |
|---|---|---|
| `test_wam_sole_negation_frame_gate` | `tests/test_wam_target.pl` | Compiler-level, target-agnostic: an inlined sole-`\+` clause emits `allocate`/`deallocate` under `ite_use_y_level`; legacy mode **and** the runtime-builtin (`inline_not_as_failure(false)`) mode do **not** — pinning the tightened guard. Fails on the pre-fix compiler. Protects all lanes. |
| `cpp_e2e_not_callee_frame_isolation` | `tests/test_wam_cpp_generator.pl` | Behavioral: a caller retains a value across a sole-`\+` callee (the frame-isolation case). |

**Suggested additional coverage (not yet landed):** a *backtracking* case — a caller that
backtracks over the sole-`\+` callee to a later success (independently verified during this
review to fail on the pre-fix compiler and pass post-fix). The landed e2e test calls the callee
once; a backtracking variant exercises the choice-point/trail interaction of the barrier. Left
to the backend owner to add, to avoid concurrent edits to the same test file.

Post-merge, each target lane owner re-runs its own gates (corpus / term differential / store
differential). The Rust perf lane is aware benchmark baselines may shift slightly from added
frames.

---

## Requested review focus (exploration-oriented — the mechanism is gdb-established; the risk is *breadth*)

1. **Per-target safety.** For each `ite_use_y_level(true)` target (cpp/go/rust/python/lua/
   haskell/js/llvm), is emitting `allocate`/`deallocate` on a sole-`\+`/`not` clause correct
   and runnable? Any target whose runtime assumes such clauses are frameless (e.g. flat-heap
   runtimes)? (Note: a stale comment in Appendix A calls y-level "currently LLVM only" — in
   fact every listed target now passes `ite_use_y_level(true)` at its call site.)
2. **Clause-shape edges.** sole `\+ G` where G is itself an if-then-else / aggregate / another
   `\+` (double negation `\+ \+`); `not/1` vs `\+`; `p :- \+ true.` / `p :- \+ fail.`; a
   sole-`\+` clause that also carries a cut; several such clauses sharing a functor (first-arg
   indexing interaction).
3. **Deallocate discipline.** Confirm `deallocate` is emitted on **all** exit paths of the
   now-framed sole-`\+` clause — including the failure/backtrack path — with no leak or
   double-pop. (Appendix B shows the success-path deallocate; probe the fail path.)
4. **Barrier numbering.** With a frame now present, is the soft-cut barrier Y-register given a
   fresh slot beyond the clause's permanent vars (no residual collision)?
5. **Legacy mode.** Confirm the `wam_ite_use_y_level_enabled` guard leaves non-y-level codegen
   byte-identical.
6. **Perf.** Could the extra frame on these clauses time out any perf-sensitive test
   (a differential or large-catalog resolve)?
7. **Fix 2 completeness.** Does `include_stdlib([predsort])` pull every transitive helper
   `predsort` needs, or could a deeper stdlib dependency still be missing?

---

# Appendix A — the allocate decision (compiler)

`src/unifyweaver/targets/wam_target.pl`:
```prolog
%% compile_single_clause_wam(+Clause, +Options, -Code)
compile_single_clause_wam(Head-Body, Options, Code) :-
    set_clause_binding_context(Head, Body),
    set_clause_visited_context(Head),
    Head =.. [_|Args],
    normalize_goals(Body, Goals),
    empty_varmap(V0),
    expand_aggregate_goals_for_perm_vars(Goals, ExpandedGoals),
    (   ( length(ExpandedGoals, N), N > 1
        ; goals_contain_call_or_aggregate(Goals)      % <-- the decision the fix touches
        )
    ->  % allocate branch: pre-assign Y-registers, emit `allocate` before head,
        % and compile the body with HasEnv = yes.
        pre_assign_permanent_vars(ExpandedGoals, V0, V0a),
        compile_head_arguments(Args, 1, V0a, V1, HeadCode),
        compile_goals(Goals, V1, yes, _, GoalsCode),
        format(string(Code), "    allocate~n~w~n~w", [HeadCode, GoalsCode])
    ;   % no-frame branch: body compiled without an environment (HasEnv absent)
        compile_head_arguments(Args, 1, V0, V1, HeadCode),
        (   Goals == []
        ->  BodyCode = "    proceed"
        ;   compile_body_goals(Goals, V1, Options, BodyCode)
        ),
        format(string(Code), "~w~n~w", [HeadCode, BodyCode])
    ).

%% goals_contain_call_or_aggregate(+Goals)  -- with the fix applied
goals_contain_call_or_aggregate(Goals) :-
    member(G, Goals),
    ( G = aggregate_all(_, _, _)
    ; G = findall(_, _, _)
    ; wam_inline_bagof_setof_enabled, G = bagof(_, _, _)
    ; wam_inline_bagof_setof_enabled, G = setof(_, _, _)
    ; wam_ite_use_y_level_enabled, wam_inline_not_enabled, G = \+(_)   % <-- the fix (refined form)
    ; callable(G), functor(G, F, _), \+ is_builtin_goal(F)
    ),
    !.

wam_ite_use_y_level_enabled :-
    catch(b_getval(wam_ite_use_y_level, true), _, fail).

% `\+` is listed here, which is why a sole-`\+` clause skipped allocation:
is_builtin_goal(is).      is_builtin_goal(=).       is_builtin_goal(\=).
is_builtin_goal(>).       is_builtin_goal(<).       is_builtin_goal(>=).
is_builtin_goal(=<).      is_builtin_goal(=:=).     is_builtin_goal(=\=).
is_builtin_goal(true).    is_builtin_goal(fail).    is_builtin_goal(write).
is_builtin_goal(display). is_builtin_goal(nl).      is_builtin_goal(format).
is_builtin_goal(member).  is_builtin_goal(length).  is_builtin_goal(is_list).
is_builtin_goal(append).  is_builtin_goal((\+)).    is_builtin_goal(atom_length).
is_builtin_goal(atom_codes). is_builtin_goal(atom_chars). is_builtin_goal(char_code).
is_builtin_goal(between).
% (note: `not` is NOT in this list, so `not/1` already forces allocation via the catch-all)

% How ite_use_y_level is turned on for a compilation (STALE comment kept verbatim —
% every listed target now opts in at its own call site):
%   % M17: emit if-then-else with get_level Y_n / cut Y_n bytecode ... Targets that opt
%   % in (currently LLVM) ... Default off so existing C++/Go/Lua/etc. paths don't see
%   % new opcodes they don't parse.
(   option(ite_use_y_level(true), Options) -> b_setval(wam_ite_use_y_level, true)
;   b_setval(wam_ite_use_y_level, false) ).
```

# Appendix B — the `\+` → soft-cut if-then-else rewrite, and the ITE deallocate (compiler)

`src/unifyweaver/targets/wam_target.pl`, inside `compile_goals/5`:
```prolog
    % If-then-else: (Cond -> Then ; Else)
    ;   Goal = (CondGoal -> ThenGoal ; ElseGoal)
    ->  compile_if_then_else(CondGoal, ThenGoal, ElseGoal, V0, V1, HasEnv, GoalCode),
        (   Rest == []
        ->  Vf = V1,
            (   HasEnv == yes
            ->  format(string(Code), "~w~n    deallocate~n    proceed", [GoalCode])
            ;   format(string(Code), "~w~n    proceed", [GoalCode])
            )
        ;   compile_goals(Rest, V1, HasEnv, Vf, RestCode),
            format(string(Code), "~w~n~w", [GoalCode, RestCode])
        )
    % Negation-as-failure: \+ G
    ;   nonvar(Goal), Goal = \+(NotGoal), wam_inline_not_enabled
    ->  (   wam_ite_use_y_level_enabled
        % M17 targets (get_level/cut): soft-cut form (NotGoal -> fail ; true).
        % `cut Yn` truncates choicepoints to the snapshot taken before the
        % negation's try_me_else, scoped to the negation alone.
        ->  compile_goals([((NotGoal -> fail ; true)) | Rest], V0, HasEnv, Vf, Code)
        % Legacy targets: hard-cut rewrite ((G, !, fail) ; true).
        ;   compile_goals([((NotGoal, !, fail) ; true) | Rest], V0, HasEnv, Vf, Code)
        )
```
So under y-level, `p :- \+ q.` becomes the ITE `((q -> fail ; true))`, compiled by
`compile_if_then_else`, which emits the `get_level`/`cut` barrier on a Y-register. With the fix,
the clause is in the `allocate` branch (Appendix A), so `HasEnv = yes` and a `deallocate`
precedes `proceed`.

# Appendix C — runtime Y-register resolution (why a missing frame corrupts)

`src/unifyweaver/targets/wam_cpp_target.pl` (embedded C++ runtime). Each environment frame owns
its own `y_regs` map; registers are shared by pointer:
```cpp
// EnvFrame { std::unordered_map<std::string, CellPtr> y_regs; ... };  b0, etc.

CellPtr WamState::get_cell(const std::string& name) {
    if (is_y_reg(name)) {
        if (env_stack.empty()) env_stack.emplace_back();
        auto& y = env_stack.back().y_regs;          // <-- the CALLER's frame if callee didn't allocate
        auto it = y.find(name);
        if (it != y.end()) return it->second;
        CellPtr c = make_cell(Value::Unbound("_U_" + name));
        y[name] = c;
        return c;
    }
    int i = reg_index(name);
    ...
    return regs[i];
}
void WamState::set_cell(const std::string& name, CellPtr c) {
    if (is_y_reg(name)) {
        if (env_stack.empty()) env_stack.emplace_back();
        env_stack.back().y_regs[name] = std::move(c);   // <-- writes into that frame
        return;
    }
    ...
}
// registers are aliased by pointer (put_value Y1,A1 then get_variable X1,A1 share one Cell):
void WamState::put_variable_reg(const std::string& a, const std::string& b) {
    CellPtr c = make_cell(Value::Unbound("_V" + std::to_string(var_counter++)));
    set_cell(a, c);
    set_cell(b, c);
}
```

# Appendix D — the FROZEN resolver clauses involved (`examples/pkg_resolver/resolver.pl`)

```prolog
% version comparison
version_lt(v(A, B, C), v(D, E, F)) :-
    ( A < D -> true ; A =:= D, B < E -> true ; A =:= D, B =:= E, C < F ).
version_lt(deb(E1, U1, R1), deb(E2, U2, R2)) :-
    ( E1 < E2 -> true
    ; E1 =:= E2, segs_lt(U1, U2) -> true
    ; E1 =:= E2, \+ segs_lt(U1, U2), \+ segs_lt(U2, U1), segs_lt(R1, R2) ).

% the SOLE-`\+` clauses that triggered Fix 1's bug:
satisfies(Ver, gte(G)) :- \+ version_lt(Ver, G).
satisfies(Ver, lte(G)) :- \+ version_lt(G, Ver).

% the predsort path that triggered Fix 2's bug (deb versions only):
is_v3(v(_, _, _)).
sort_versions_desc(Vs, Desc) :-
    (   maplist(is_v3, Vs)
    ->  sort(Vs, Asc), reverse(Asc, Desc)
    ;   predsort(cmp_ver, Vs, Asc), reverse(Asc, Desc)   % deb -> predsort
    ).
cmp_ver(<, A, B) :- version_lt(A, B), !.
cmp_ver(>, A, B) :- version_lt(B, A), !.
cmp_ver(=, _, _).
```
