:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_cross_target_conformance.pl
%
% Cross-target WAM conformance harness. Compiles the SHARED classic
% programs (wam_conformance_fixtures.pl) to each WAM backend that has a
% toolchain on PATH, runs every backend against the SAME query vectors,
% and asserts the results match the single shared spec.
%
% Why: per-target classic-program tests already exist, but each
% re-declares its own fixtures, so a backend can silently diverge from
% the others without any test noticing. This harness is the safety net
% for exactly that: the Haskell `Proceed` and WAT `allocate`
% first-argument-indexing bugs (member/2 wrongly succeeding) are the kind
% of divergence it catches.
%
% INVOCATION is per-target (each adapter owns build + run):
%   - scala  passes the query args straight to GeneratedProgram (its
%     runtime parses "[a,b,c]" etc). It is NOT given 0-arity wrappers:
%     the Scala backend currently loops compiling 0-arity predicates.
%   - elixir / wat synthesise a ground 0-arity wrapper per query
%     (`ctw_N :- pred(args).`) and ask whether ctw_N succeeds — the shape
%     their own runtime tests use, sidestepping per-backend quirks in
%     feeding list args into a predicate entry point.
%
% SPEED. Builds are PER PROGRAM (one small project per program per
% backend) and the query set can be random-sampled to keep CI cheap while
% coverage accumulates across runs. Knobs (environment variables):
%   CONFORMANCE_TARGETS  = scala,elixir   limit which backends run
%   CONFORMANCE_PROGRAMS = member,fib     limit which programs run
%   CONFORMANCE_SAMPLE   = N              random N queries per program
%   CONFORMANCE_SEED     = N              seed the sampler (reproducible)
%
% Adding a target = implement ct_toolchain/2 + ct_build/4 + ct_run/5
% (+ ct_teardown/2) and register it in conformance_target/1. Missing
% toolchains skip rather than fail. Known-divergent (target, program)
% pairs are tracked in ct_xfail/2 so the suite stays green while the
% underlying gap is tracked separately.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(random)).
:- use_module('helpers/smoke_paths', [tmp_root/1]).
:- use_module('wam_conformance_fixtures').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_scala_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_target').
:- use_module('../src/unifyweaver/targets/wam_wat_target').
:- use_module('../src/unifyweaver/targets/wam_haskell_target',
              [write_wam_haskell_project/3]).
:- use_module('../src/unifyweaver/targets/wam_python_target',
              [write_wam_python_project/3]).
:- use_module('../src/unifyweaver/targets/wam_go_target',
              [write_wam_go_project/3]).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).
:- use_module('../src/unifyweaver/targets/wam_c_target',
              [write_wam_c_project/3]).
% library(pairs) is loaded before wam_cpp_target so the latter's load-time
% "lists_extra stdlib" directive (guarded by current_predicate(pairs_keys/2))
% short-circuits: without pairs_keys present it would assertz user:intersection
% etc., which clash with library(lists)' static predicates and raise a
% permission error on load.
:- use_module(library(pairs)).
:- use_module('../src/unifyweaver/targets/wam_cpp_target',
              [write_wam_cpp_project/3]).
:- use_module('../src/unifyweaver/targets/wam_kotlin_target',
              [write_wam_kotlin_project/3]).
:- use_module('../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).

% ============================================================
% Target registry + known-divergence (xfail) registry
% ============================================================

conformance_target(scala).
conformance_target(elixir).
conformance_target(wat).
conformance_target(haskell).
conformance_target(python).
conformance_target(go).
conformance_target(rust).
conformance_target(c).
conformance_target(cpp).
conformance_target(kotlin).
conformance_target(kotlin_functions).
conformance_target(fsharp).
conformance_target(fsharp_functions).

%% ct_default_target(Target): runs unless CONFORMANCE_TARGETS overrides.
%  Only scala and elixir run by default. Every other registered backend
%  (wat, haskell, python, go, rust, c, cpp, kotlin, kotlin_functions,
%  fsharp, fsharp_functions) is opt-in via CONFORMANCE_TARGETS because it
%  builds a per-program project with an external toolchain. Kotlin/F# are
%  especially slow (gradle / dotnet compile per program) and stay opt-in.
ct_default_target(scala).
ct_default_target(elixir).

% Per-target adapter clauses are grouped by target, not by predicate.
:- discontiguous ct_build/4.
:- discontiguous ct_run/5.
:- discontiguous ct_teardown/2.
%% ct_xfail/2 and ct_skip/2 may legitimately have zero clauses (every
%% target/program is now conformant). Declare them dynamic so calls fail
%% cleanly instead of raising existence_error when no entries remain.
:- dynamic ct_xfail/2.
:- dynamic ct_skip/2.

%% ct_xfail(Target, ProgramName)
%  (Target, program) pairs known to diverge from the shared spec.
%  Mismatches are tolerated (logged, not failed); an unexpected full
%  match (xpass) is logged so the entry can be retired once the gap is
%  fixed.
%
%  WAT member/builtins USED to be xfail. Both are fixed in
%  wam_wat_target.pl (and templates/targets/wat_wam/state.wat.mustache):
%   - member: read-mode argument unification was unimplemented (unify_*
%     read-mode branches were nops; no S register), so get_structure/
%     get_list matched only the functor and element mismatches went
%     undetected. Added an S register (heap arg pointer at 65568),
%     wired get_structure/get_list to set it and unify_variable/
%     unify_value/unify_constant to consume successive arg cells, and
%     made get_list also accept a tag-3 [|]/2 compound as a list cell
%     (the compiler emits the outer list as put_list/tag-4 but nested
%     cons as put_structure [|]/2/tag-3 — the same ./2-vs-[|]/2 split
%     handled in the other backends), via a generated $cons_op1 global.
%   - builtins: eval_arith lacked // and mod, AND functor_arity_of
%     mis-parsed '///2' (split on '/' expecting two parts) to arity 0,
%     so // structures were skipped by eval_arith's arity-2 dispatch.
%     Fixed functor_arity_of to take the last '/'-component and added
%     //, mod (with zero guards and floored mod). cmp/eq already worked.
%
%  WAT fib USED to be xfail: cfib(10,54) wrongly succeeded. Root cause
%  (now fixed in wam_wat_target.pl): peephole_fused_arith rewrites
%  `F is F1+F2` into a single fused_is_add(Dest,Src1,Src2), and the WAT
%  handlers for the fused is/* forms (add/sub/mul and the _const variants)
%  called $bind_reg_deref to STORE the result into Dest unconditionally,
%  never checking an already-bound Dest. So a recursive predicate result
%  check (F bound to the queried value) always succeeded by clobbering F.
%  Added $is_unify_int (bind if unbound, else integer-equality check,
%  mirroring builtin_is) and routed all five fused forms through it.
%  (The non-fused is/2 path was already correct, which is why cbi_arith
%  passed — fib was the only program hitting the fused result-check.)

%  (Elixir append/reverse used to be xfail here: a freshly-constructed
%  list ("./2", from put_list) would not unify against an already-GROUND
%  list compound ("[|]/2", from put_structure) in a clause head, so
%  capp([a],[b],[a,b]) returned false while capp([a],[b],X), X=[a,b]
%  succeeded. Root cause: unify/3's compound clause demanded identical
%  functor names, never applying the ./2 <-> [|]/2 cons-cell aliasing
%  that the get_structure match path already used. Fixed in
%  wam_elixir_target.pl; both programs are now conformant and the xfails
%  are removed.)

%  Haskell builtins. The conformance driver
%  (tests/fixtures/haskell_conformance_driver.hs) runs a 0-arity wrapper
%  whose atoms are baked into the compiled instruction stream, which
%  removed the atom-interning mismatch that sank the earlier throwaway
%  driver (fib/ack discriminate correctly).
%
%  member/append/reverse USED to be xfail here, all failing on heap-built
%  cons lists. Root causes (now fixed in wam_haskell_target.pl):
%   1. unifyVal/unifyValues did NO structural unification (identity +
%      var-binding only) — added a shared structural unifyTerms.
%   2. the cons functor "[|]/2" interned to its own id, distinct from
%      atomDot — intern_struct_functor/2 now folds every cons spelling
%      onto atomDot (the ./2-vs-[|]/2 class, same as the Elixir bug).
%   3. GetValue had its own inline unify bypassing the above — now routed
%      through unifyVal.
%   4. THE multi-element bug: building [a|X] with X a set_variable tail
%      placeholder emitted VList [a, X] (X as a 2nd ELEMENT) instead of a
%      cons cell, and put_structure filling X did not bind the embedded
%      var. addToBuilder now emits Str atomDot [hd, tl] for a partial
%      tail and binds the placeholder var on finalize. This fixed member,
%      append, AND reverse on lists of any length; the xfails are removed.
%
%  builtins USED to be xfail (=/2 and //,mod). Both fixed in
%  wam_haskell_target.pl:
%   - // (interns as "///2") and / ("//2") evaluated to Nothing because
%     the arity-stripper used takeWhile (/= '/'), which truncates any
%     operator that contains '/'. Replaced with bareArithOp, which strips
%     only a trailing /<digits>. (cbi_arith's 17//5 + 17 mod 5 now folds.)
%   - =/2 is emitted by the compiler as BuiltinCall "=/2" but had no
%     handler, so it fell to the default branch and always failed
%     (cbi_eq(foo) -> false). Added a handler routing through unifyVal.
%  fib/ack already passed, so +/-/comparison and is/2 bound-LHS checking
%  were fine; with these two fixes the Haskell adapter is fully green.

%  Go (now fully conformant; the WAM runtime is exercised here via
%  prefer_wam(true), since its default strategy is the dataflow/stream
%  backend, not the shared WAM pipeline). Onboarding surfaced four gaps,
%  all since fixed:
%   - is/2 produced a Float for every result, and Unify is type-strict
%     (Integer never unifies with Float), so `R is N + 1` failed whenever
%     R was bound to a ground Integer. Integral results now wrap as Integer
%     (state.go.mustache) — fixed fib/ack.
%   - nested-arithmetic (cbi_arith): the compiler builds `+(+(A,B),C)`
%     outer-first, emitting a set_variable placeholder into the outer arg
%     and then a put_structure into that placeholder register; put_structure
%     overwrote the register but never bound the embedded placeholder, so
%     the outer arg stayed unbound and evalArithmetic gave up at depth >= 2.
%     PutStructure now binds an unbound placeholder it overwrites
%     (wam_go_target.pl) — this also fixes list TAIL cells built the same
%     way (the ./2-vs-[|]/2 class).
%   - member/reverse (cons-cell traversal): two further GetList bugs —
%     (a) it recognised only the outer *List cell, not the inner "[|]/2"
%     *Structure tail cells (added consHeadTail in state.go.mustache), and
%     (b) it tested isUnbound BEFORE deref, so a *bound* variable tail was
%     mistaken for unbound and sent into write mode (now derefs first).

%  Rust is now fully conformant. The Rust WAM runtime had the same gap
%  family as Go, plus a missing =/2; the fixes (wam_rust_target.pl /
%  templates/targets/rust_wam/state.rs.mustache):
%   - =/2 had NO handler in execute_builtin (the compiler emits `X = Y` as
%     builtin_call =/2), so even `a = a` returned false. Added a handler
%     routing through unify(). This made fib and builtins pass.
%   - set_variable did not write the fresh var into the current
%     structure/list arg slot (unlike set_value/set_constant), so terms
%     with a variable argument had their args misaligned — now calls
%     set_heap_or_list.
%   - nested arithmetic: put_structure into a placeholder register did not
%     bind the embedded placeholder, and eval_arith did not deref Unbound.
%     Both fixed; nested cbi_arith now evaluates.
%   - cons-cell aliasing added to get_list (accepts "[|]/2"/"./2" Str and
%     Ref, derefs before the unbound test) and to unify (List <-> cons Str,
%     empty-list <-> "[]" atom).
%   - put_constant/get_constant emitted Value::Atom("28") for the integer
%     28 (unlike set_/unify_constant, which already used Value::Integer), so
%     `R is <expr>` with R bound to a ground integer (the head arg) failed —
%     is/2's result match handles Unbound/Integer/Float, not Atom. Now route
%     through rust_const_value; this made builtins conformant.
%   - switch_on_constant_fallthrough had NO codegen, so it fell to the
%     unknown-instruction comment and was dropped from the emitted vector.
%     That shifted every later label PC by one: backtracking into a clause
%     chain landed one instruction PAST retry_me_else, so the choice point's
%     next_pc was never advanced and execution looped on the same clause
%     (returning false at the step_limit). Any predicate indexed on a
%     constant first argument with 3+ clauses hit this — fib (cfib(0/1/N))
%     and ack (cack(0,_)/cack(M,_)). Added the instruction with proper
%     fallthrough semantics (jump on a table hit; advance to the try_me_else
%     chain on unbound/miss — never fail). fib and ack are now conformant.
%   - the list model (member/append/reverse) needed four fixes:
%     (a) a partial list whose tail is still an unbound variable
%         ([a|X2] then X2=[b|X3]) was materialised as Value::List([a,X2]),
%         treating the tail var as a SECOND ELEMENT; get_list then peeled a
%         wrong tail and member(z,...) bound the var and wrongly succeeded.
%         set_heap_or_list now keeps such a cell as a "[|]/2" cons so the
%         tail var derefs to the rest of the list.
%     (b) switch_on_term had no codegen and was dropped (same label-shift
%         class as switch_on_constant_fallthrough: a skipped trust_me made
%         the choice point loop). Unknown instructions now emit a real NoOp,
%         preserving PC alignment; falling through to the try chain is
%         correct for indexing hints.
%     (c) get_constant [] now matches an empty Value::List (append's base
%         case capp([],L,L) sees the peeled tail as Value::List([])).
%     (d) get_value did only raw equality + unbound-binding; it now routes
%         through unify(), so it follows heap Refs and structurally unifies
%         the accumulated result list (append/reverse output).
%  The runner driver keeps vm.step_limit as a guard against accidental
%  non-termination.

%% ct_skip(Target, ProgramName)
%  Stronger than xfail: do NOT even build/run this (target, program).
%  Used when *generation itself* is unusable, not just the answer.
%
%  WAT append/reverse USED to be skipped: the parser had no working clause
%  for the `switch_on_term` first-arg index that list-recursive predicates
%  emit (its `parse_term_entries` expected an old operand format), so it
%  fell to the `unrecognized instruction -> allocate` fallback and either
%  looped on warnings or silently failed to generate. Fixed by emitting an
%  empty (unindexed) switch_on_term header on register 0 -- the try_me_else
%  chain alone is correct -- mirroring switch_on_term_a2. With generation
%  working, the output-list unification then needed the same fix as the
%  read path: $unify_regs did only SHALLOW (tag+payload) equality, so a
%  constructed cons (tag-3 [|]/2) would not match an already-ground list
%  cell (tag-4) and append/reverse returned false on correct answers. It
%  is now recursive and cons-aware ($unify_addrs). Both are conformant;
%  the skips are removed.

%  Kotlin (CONF-KOTLIN, 2026-07-12). Adapter registered (opt-in). All classic
%  programs green after KT-ARITH-SLASH-FUNCTOR + KT-LIST-BACKTRACK +
%  KT-Y-ENV-RECURSION (no remaining kotlin ct_xfail entries).

%  F# (CONF-FSHARP, 2026-07-15). Adapter registered (opt-in). Measured:
%   - Green: member, fib, ack, builtins, append, reverse on interpreter.
%   - append / reverse ALSO green under emit_mode(functions) after
%     FS-LIST-PARTIAL-TAIL (2026-07-15): GetValue routes through
%     unifyVal/unifyTerms so ground Str("[|]",…) result spines unify with
%     compact VList. Builder was already correct; open-tail
%     `capp([a],[b],X),X=[a,b]` already passed via =/2.
%   - fsharp_functions + builtins: FIXED FS-FUNCTIONS-BUILTINS-LOWER
%     (2026-07-15). Stall was not cbi_eq/=/2 — parse_functor_fs soft-cut
%     on the first "/" so put_structure `///2` (integer-div) failed mid
%     emit of cbi_arith and lower_all_fs never finished. Last-slash parse
%     (Scala/R/Lua shape) unblocks all three builtins lowers; =/2 already
%     delegated to step via emit_one_fs(builtin_call). No remaining
%     fsharp / fsharp_functions ct_xfail or ct_skip.

% ============================================================
% Toolchain probes
% ============================================================

ct_toolchain(scala,  [scalac]).
ct_toolchain(elixir, [elixir]).
ct_toolchain(wat,    [wat2wasm, node]).
ct_toolchain(haskell, [ghc, cabal]).
ct_toolchain(python, [python3]).
ct_toolchain(go,     [go]).
ct_toolchain(rust,   [cargo]).
ct_toolchain(c,      [gcc]).
ct_toolchain(cpp,    ['g++']).
ct_toolchain(kotlin, [gradle]).
ct_toolchain(kotlin_functions, [gradle]).
ct_toolchain(fsharp, [dotnet]).
ct_toolchain(fsharp_functions, [dotnet]).

ct_available(scala) :-
    ct_enabled(scala),
    exe_on_path(scalac),
    (   exe_on_path(java), scala_runtime_jars(_)
    ;   exe_on_path(scala)
    ),
    !.
ct_available(Target) :-
    Target \= scala,
    ct_enabled(Target),
    ct_toolchain(Target, Exes),
    forall(member(E, Exes), exe_on_path(E)).

ct_enabled(Target) :-
    (   getenv('CONFORMANCE_TARGETS', Spec), Spec \== ''
    ->  split_string(Spec, ",", " ", Parts), atom_string(Target, TS),
        memberchk(TS, Parts)
    ;   ct_default_target(Target) ).

exe_on_path(Exe) :-
    catch(
        ( process_create(path(Exe), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, _) ),
        _, fail).

% ============================================================
% Tests — one per target, looping the (sampled) shared programs
% ============================================================

:- begin_tests(wam_cross_target_conformance).

test(scala,  [condition(ct_available(scala))])  :- run_target_conformance(scala).
test(elixir, [condition(ct_available(elixir))]) :- run_target_conformance(elixir).
test(wat,    [condition(ct_available(wat))])    :- run_target_conformance(wat).
test(python, [condition(ct_available(python))]) :- run_target_conformance(python).
test(go,     [condition(ct_available(go))])     :- run_target_conformance(go).
test(rust,   [condition(ct_available(rust))])   :- run_target_conformance(rust).
test(c,      [condition(ct_available(c))])      :- run_target_conformance(c).
test(cpp,    [condition(ct_available(cpp))])    :- run_target_conformance(cpp).
test(kotlin, [condition(ct_available(kotlin))]) :- run_target_conformance(kotlin).
test(kotlin_functions, [condition(ct_available(kotlin_functions))]) :-
    run_target_conformance(kotlin_functions).
test(fsharp, [condition(ct_available(fsharp))]) :- run_target_conformance(fsharp).
test(fsharp_functions, [condition(ct_available(fsharp_functions))]) :-
    run_target_conformance(fsharp_functions).

:- end_tests(wam_cross_target_conformance).

% ============================================================
% Driver
% ============================================================

run_target_conformance(Target) :-
    seed_sampler,
    selected_programs(Programs),
    nb_setval(ct_wrapper_counter, 0),
    foldl(run_program(Target), Programs, [], Failures),
    (   Failures == []
    ->  true
    ;   throw(error(conformance_failures(Target, Failures), _))
    ).

seed_sampler :-
    (   getenv('CONFORMANCE_SEED', S), S \== ''
    ->  (   atom_number(S, Seed), integer(Seed)
        ->  set_random(seed(Seed)),
            log_conformance_seed_once(Seed)
        ;   throw(error(domain_error(conformance_seed, S), _))
        )
    ;   true
    ).

log_conformance_seed_once(Seed) :-
    (   nb_current(ct_conformance_seed_logged, Seed)
    ->  true
    ;   format(user_error, '  [conformance] CONFORMANCE_SEED=~w~n', [Seed]),
        nb_setval(ct_conformance_seed_logged, Seed)
    ).

selected_programs(Programs) :-
    findall(P, conformance_program(P, _), All0),
    list_to_set(All0, All),
    (   getenv('CONFORMANCE_PROGRAMS', Spec), Spec \== ''
    ->  split_string(Spec, ",", " ", Parts),
        include([P]>>( atom_string(P, PS), memberchk(PS, Parts) ), All, Programs)
    ;   Programs = All ).

run_program(Target, Program, In, Out) :-
    (   ct_skip(Target, Program)
    ->  log_line(Target, Program, "SKIP (generation unusable, tracked)", []),
        Out = In
    ;   run_program_(Target, Program, In, Out) ).

run_program_(Target, Program, In, Out) :-
    conformance_program(Program, Preds),
    sampled_queries(Program, Queries),      % [q(K,A,E)...]
    (   Queries == []
    ->  Out = In
    ;   (   catch(
                setup_call_cleanup(
                    ct_build(Target, Preds, Queries, Ctx),
                    run_queries(Target, Ctx, Program, Queries, In, Out),
                    ct_teardown(Target, Ctx)),
                Err,
                build_error(Target, Program, Err, In, Out))
        ->  true
        ;   build_error(Target, Program, build_failed, In, Out) )
    ).

build_error(Target, Program, Err, In, Out) :-
    (   ct_xfail(Target, Program)
    ->  log_line(Target, Program, "XFAIL (build/run error, tolerated): ~w", [Err]),
        Out = In
    ;   format(atom(F), '~w/~w: build/run error: ~w', [Target, Program, Err]),
        Out = [F|In] ).

run_queries(Target, Ctx, Program, Queries, In, Out) :-
    foldl(run_query(Target, Ctx, Program), Queries, In, Out).

run_query(Target, Ctx, Program, q(K, A, Expected), In, Out) :-
    (   catch(ct_run(Target, Ctx, K, A, Got), E, ( Got = error(E), true ))
    ->  true ; Got = error(run_failed) ),
    (   Got == Expected
    ->  (   ct_xfail(Target, Program)
        ->  log_line(Target, Program, "XPASS ~w(~w) matched under xfail", [K, A]),
            Out = In
        ;   Out = In )
    ;   (   ct_xfail(Target, Program)
        ->  log_line(Target, Program,
                     "xfail ~w(~w): expected ~w got ~w (tolerated)",
                     [K, A, Expected, Got]),
            Out = In
        ;   format(atom(F), '~w/~w: ~w(~w) expected ~w got ~w',
                   [Target, Program, K, A, Expected, Got]),
            Out = [F|In] ) ).

log_line(Target, Program, Fmt, Args) :-
    format(string(Body), Fmt, Args),
    format(user_error, '  [conformance] ~w/~w: ~w~n', [Target, Program, Body]).

% ============================================================
% Query sampling
% ============================================================

sampled_queries(Program, Picked) :-
    findall(q(K, A, E), conformance_query(Program, K, A, E), Qs),
    (   getenv('CONFORMANCE_SAMPLE', NS), atom_number(NS, N), integer(N), N >= 0
    ->  random_subset(Qs, N, Picked)
    ;   Picked = Qs ).

random_subset(List, N, Picked) :-
    length(List, Len),
    (   N >= Len -> Picked = List
    ;   random_permutation(List, Shuffled),
        length(Picked, N), append(Picked, _, Shuffled) ).

% ============================================================
% Shared helpers
% ============================================================

%% render_arg(+Term, -Atom): textual form the arg-passing drivers parse:
%  ints -> '10', atoms -> 'a', atom-lists -> '[a,b,c]'.
render_arg(Term, Atom) :- format(atom(Atom), '~w', [Term]).

bool_of_string(S, B) :-
    string_to_atom(S, A),
    ( A == true -> B = true ; A == false -> B = false ; B = A ).

ct_tmp_dir(Prefix, Dir) :-
    get_time(T), Stamp is floor(T * 1000000),
    tmp_root(Root),
    format(atom(Base), '~w_~w', [Prefix, Stamp]),
    directory_file_path(Root, Base, Dir),
    make_directory_path(Dir).

cleanup_dir(Dir) :-
    ( atom(Dir), exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

strip_pred(_Mod:N/A, N/A) :- !.
strip_pred(N/A, N/A).

run_proc(Exe, Args, Cwd, Exit, ErrStr) :-
    process_create(path(Exe), Args,
                   [cwd(Cwd), stdout(null), stderr(pipe(Err)), process(Pid)]),
    read_string(Err, _, ErrStr), close(Err), process_wait(Pid, exit(Exit)).

run_proc_out(Exe, Args, Cwd, Exit, OutStr) :-
    process_create(path(Exe), Args,
                   [cwd(Cwd), stdout(pipe(Out)), stderr(null), process(Pid)]),
    read_string(Out, _, OutStr), close(Out), process_wait(Pid, exit(Exit)).

run_proc_out_err(Exe, Args, Cwd, Exit, OutStr, ErrStr) :-
    process_create(path(Exe), Args,
                   [cwd(Cwd), stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out), close(Err),
    process_wait(Pid, exit(Exit)).

%% Wrapper synthesis (elixir / wat). One ground 0-arity wrapper per
%  query, asserted into user as `ctw_N :- pred(Args).`. The global
%  counter keeps names unique across programs in one run. Returns the
%  wrapper predicate indicators and a (K-A)->WName lookup map.
synth_wrappers([], [], []).
synth_wrappers([q(K, A, _E)|Rest], [WName/0|WPs], [(K-A)-WName|Map]) :-
    nb_getval(ct_wrapper_counter, I0), I is I0 + 1,
    nb_setval(ct_wrapper_counter, I),
    atom_concat(ctw_, I, WName),
    pred_name_of_key(K, PredName),
    Goal =.. [PredName|A],
    assertz(user:(WName :- Goal)),
    synth_wrappers(Rest, WPs, Map).

abolish_wrappers([]).
abolish_wrappers([(_K-_A)-WName|Rest]) :-
    ( current_predicate(user:WName/0) -> abolish(user:WName/0) ; true ),
    abolish_wrappers(Rest).

pred_name_of_key(Key, Name) :-
    atomic_list_concat([NameA|_], '/', Key), atom_string(Name, NameA).

% ============================================================
% Adapter: Scala  (direct args: GeneratedProgram <predkey> <args>)
% ============================================================

ct_build(scala, Preds, _Queries, scala_ctx(Dir)) :-
    ct_tmp_dir('tmp_ct_scala', Dir),
    Opts = [ package('generated.wam_ct.core'),
             runtime_package('generated.wam_ct.core'), module_name('wam-ct') ],
    write_wam_scala_project(Preds, Opts, Dir),
    absolute_file_name(Dir, Abs),
    directory_file_path(Abs, 'classes', ClassDir), make_directory_path(ClassDir),
    directory_file_path(Abs, 'src', SrcDir),
    findall(F, ( directory_member(SrcDir, RelF,
                     [extensions([scala]), recursive(true)]),
                 directory_file_path(SrcDir, RelF, F) ), Sources),
    Sources \= [],
    run_proc(scalac, ['-d', ClassDir | Sources], Abs, Exit, Err),
    ( Exit =:= 0 -> true ; throw(scala_compile_failed(Exit, Err)) ).

ct_run(scala, scala_ctx(Dir), PredKey, Args, Bool) :-
    absolute_file_name(Dir, Abs),
    directory_file_path(Abs, 'classes', ClassDir),
    atom_string(PredKey, PredStr),
    maplist(render_arg, Args, ArgAtoms),
    maplist([X,S]>>atom_string(X,S), ArgAtoms, ArgStrs),
    scala_run_command(ClassDir, 'generated.wam_ct.core.GeneratedProgram', PredStr,
                      ArgStrs, Exe, ProcArgs),
    run_proc_out_err(Exe, ProcArgs, Abs, Exit, OutStr, ErrStr),
    (   Exit =:= 0
    ->  normalize_space(string(Out), OutStr), bool_of_string(Out, Bool)
    ;   throw(scala_run_failed(Exit, PredKey, Args, ErrStr))
    ).

scala_run_command(ClassDir, MainClass, PredStr, ArgStrs, java, ProcArgs) :-
    scala_runtime_classpath(ClassDir, ClassPath),
    !,
    append(['-cp', ClassPath, MainClass, PredStr], ArgStrs, ProcArgs).
scala_run_command(ClassDir, MainClass, PredStr, ArgStrs, scala, ProcArgs) :-
    append(['-classpath', ClassDir, MainClass, PredStr], ArgStrs, ProcArgs).

scala_runtime_classpath(ClassDir, ClassPath) :-
    scala_runtime_jars(Jars),
    path_list_separator(Sep),
    atomic_list_concat([ClassDir|Jars], Sep, ClassPath).

scala_runtime_jars(Jars) :-
    findall(Jar, scala_runtime_jar(Jar), Jars0),
    sort(Jars0, Jars),
    Jars \= [].

path_list_separator(';') :- current_prolog_flag(windows, true), !.
path_list_separator(':').

scala_runtime_jar(Jar) :-
    scala_lang_maven_root(Root),
    member(Artifact, ['scala-library', 'scala3-library_3', 'tasty-core_3']),
    directory_file_path(Root, Artifact, ArtifactDir),
    exists_directory(ArtifactDir),
    directory_member(ArtifactDir, Rel,
                     [extensions([jar]), recursive(true)]),
    directory_file_path(ArtifactDir, Rel, Jar).

scala_lang_maven_root(Root) :-
    getenv('SCALA_MAVEN_ROOT', Raw),
    Raw \== '',
    exists_directory(Raw),
    !,
    Root = Raw.
scala_lang_maven_root(Root) :-
    getenv('PREFIX', Prefix),
    Prefix \== '',
    directory_file_path(Prefix, 'opt/scala/maven2/org/scala-lang', Root),
    exists_directory(Root).

ct_teardown(scala, scala_ctx(Dir)) :- cleanup_dir(Dir).

% ============================================================
% Adapter: Elixir  (0-arity wrapper via run_classic.exs <Module> <wname>/0)
% ============================================================

ct_build(elixir, Preds, Queries, elixir_ctx(Dir, Map)) :-
    ct_tmp_dir('tmp_ct_elixir', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds),
    ct_compile_preds_to_wam(AllPreds, [], PredWamPairs),
    write_wam_elixir_project(PredWamPairs,
        [ module_name('wam_ct'), emit_mode(lowered), source_module(user) ], Dir),
    classic_elixir_driver(DriverSrc),
    directory_file_path(Dir, 'run_classic.exs', DriverDst),
    copy_file(DriverSrc, DriverDst).

ct_run(elixir, elixir_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    absolute_file_name(Dir, Abs),
    format(atom(PredKey), '~w/0', [WName]), atom_string(PredKey, PredStr),
    run_proc_out(elixir, ['run_classic.exs', "WamCt", PredStr], Abs, _Exit, OutStr),
    normalize_space(string(Out), OutStr), bool_of_string(Out, Bool).

ct_teardown(elixir, elixir_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

ct_compile_preds_to_wam([], _Opts, []).
ct_compile_preds_to_wam([Pred|Rest], Opts, [Name/Arity-WamCode|RestPairs]) :-
    strip_pred(Pred, Name/Arity),
    (   wam_target:compile_predicate_to_wam(Name/Arity, Opts, WamCode) -> true
    ;   wam_target:compile_predicate_to_wam(user:Name/Arity, Opts, WamCode) -> true
    ;   throw(wam_compile_failed(Pred))
    ),
    ct_compile_preds_to_wam(Rest, Opts, RestPairs).

:- ( prolog_load_context(directory, Dir),
     directory_file_path(Dir, 'elixir_e2e/run_classic.exs', P),
     assertz(classic_elixir_driver_fact(P))
   ; true ).
classic_elixir_driver(P) :- classic_elixir_driver_fact(P).

% ============================================================
% Adapter: WAT  (0-arity wrapper -> wasm export <wname>_0 -> 1/0)
% ============================================================

ct_build(wat, Preds, Queries, wat_ctx(Dir, HarnessJs, WasmFile, Map)) :-
    ct_tmp_dir('tmp_ct_wat', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds),
    directory_file_path(Dir, 'prog.wat', WatFile),
    write_wam_wat_project(AllPreds, [module_name(wam_ct)], WatFile),
    directory_file_path(Dir, 'prog.wasm', WasmFile),
    run_proc(wat2wasm, [WatFile, '-o', WasmFile], Dir, CExit, CErr),
    ( CExit =:= 0 -> true ; throw(wat2wasm_failed(CExit, CErr)) ),
    wat_write_harness(Dir, HarnessJs).

ct_run(wat, wat_ctx(_Dir, HarnessJs, WasmFile, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(Export), '~w_0', [WName]), atom_string(Export, ExportStr),
    run_proc_out(node, [HarnessJs, WasmFile, ExportStr], '.', _Exit, OutStr),
    normalize_space(string(Out), OutStr),
    ( Out == "1" -> Bool = true ; Out == "0" -> Bool = false ; atom_string(Bool, Out) ).

ct_teardown(wat, wat_ctx(Dir, _, _, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

wat_write_harness(Dir, HarnessJs) :-
    directory_file_path(Dir, 'harness.js', HarnessJs),
    Src = "const fs=require('fs');\n\c
const [,,wasmPath,exportName]=process.argv;\n\c
const buf=fs.readFileSync(wasmPath);\n\c
const imports={env:{print_i64:()=>{},print_char:()=>{},print_newline:()=>{}}};\n\c
WebAssembly.instantiate(buf,imports).then(({instance})=>{\n\c
  const fn=instance.exports[exportName];\n\c
  if(typeof fn!=='function'){console.log('ERR');process.exit(1);}\n\c
  console.log(fn());\n\c
}).catch(e=>{console.log('TRAP');process.exit(2);});\n",
    open(HarnessJs, write, S), write(S, Src), close(S).

% ============================================================
% Adapter: Haskell  (0-arity wrapper via `cabal run <key>`)
%
% write_wam_haskell_project/3 compiles predicate INDICATORS itself
% (unlike elixir, which takes WAM pairs), so this mirrors the WAT
% adapter shape. The driver (haskell_conformance_driver.hs) runs one
% wrapper key and prints true/false. Crucially it wires the context
% with compileTimeAtomTable and runs a 0-arity wrapper whose atoms are
% all baked into the compiled instruction stream — so there is NO
% runtime atom parsing and hence none of the atom-interning mismatch
% that sank the earlier throwaway driver.
%
% The build is a cabal compile (done once in ct_build); ct_run then
% `cabal run`s per query (fast, no recompile). Opt-in only.
% ============================================================

ct_build(haskell, Preds, Queries, haskell_ctx(Dir, Map)) :-
    ct_tmp_dir('tmp_ct_haskell', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds0),
    maplist(qualify_user, AllPreds0, AllPreds),
    write_wam_haskell_project(AllPreds,
        [no_kernels(true), module_name('uw-hs-ct'), use_hashmap(false)], Dir),
    classic_haskell_driver(DriverSrc),
    directory_file_path(Dir, 'src/Main.hs', DriverDst),
    copy_file(DriverSrc, DriverDst),
    run_proc(cabal, ['build'], Dir, BExit, BErr),
    ( BExit =:= 0 -> true ; throw(haskell_build_failed(BExit, BErr)) ).

ct_run(haskell, haskell_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    absolute_file_name(Dir, Abs),
    format(atom(KeyAtom), '~w/0', [WName]), atom_string(KeyAtom, KeyStr),
    run_proc_out(cabal, ['run', '-v0', 'uw-hs-ct', '--', KeyStr], Abs, _Exit, OutStr),
    normalize_space(string(Out), OutStr), bool_of_string(Out, Bool).

ct_teardown(haskell, haskell_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

% ============================================================
% Adapter: Python  (0-arity wrapper -> `python3 main.py <key>` -> bool)
%
% write_wam_python_project/3 compiles predicate INDICATORS itself (like
% the Haskell adapter), so this mirrors that shape: synthesise one ground
% 0-arity wrapper per query, generate the project, and run each wrapper.
% Python is interpreted, so there is NO build step — generation in
% ct_build is the whole cost, and ct_run is a fast `python3 main.py` per
% query. The generated main.py prints `A_i = ...` register dumps on
% success and the literal `false.` on failure (a 0-arity wrapper has no
% meaningful output registers, so success is detected by the ABSENCE of
% `false.`). Opt-in via CONFORMANCE_TARGETS=python.
% ============================================================

ct_build(python, Preds, Queries, python_ctx(Dir, Map)) :-
    ct_tmp_dir('tmp_ct_python', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds0),
    maplist(qualify_user, AllPreds0, AllPreds),
    write_wam_python_project(AllPreds, [module_name(wam_ct)], Dir).

ct_run(python, python_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(KeyAtom), '~w/0', [WName]), atom_string(KeyAtom, KeyStr),
    run_proc_out(python3, ['main.py', KeyStr], Dir, _Exit, OutStr),
    (   sub_string(OutStr, _, _, _, "false.")
    ->  Bool = false
    ;   sub_string(OutStr, _, _, _, "Unknown predicate")
    ->  Bool = error(unknown_predicate)
    ;   Bool = true
    ).

ct_teardown(python, python_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

% ============================================================
% Adapter: Go  (0-arity wrapper -> `./bin <key>` -> true/false)
%
% write_wam_go_project/3 compiles predicate INDICATORS itself, but its
% DEFAULT strategy is the dataflow/stream Go backend (a stdin filter),
% NOT the WAM pipeline the other backends share. prefer_wam(true) forces
% every predicate through compile_predicate_to_wam, so Go is tested on the
% same WAM lowering as scala/elixir/wat/haskell/python. We generate with
% package_name(main) (which emits a domain-specific benchmark main.go) and
% then OVERWRITE main.go with a tiny query-runner driver: it looks up the
% wrapper label in SharedWamLabels, sets vm.PC, and runs vm.Run() — which
% returns true only when the machine halts (Proceed with CP<=0), false on
% backtrack exhaustion. Go compiles fast to a single binary, so ct_build
% does the `go build` and ct_run is a cheap exec per query. Opt-in via
% CONFORMANCE_TARGETS=go.
% ============================================================

go_runner_driver('package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: runner <pred/arity>")
		os.Exit(2)
	}
	key := os.Args[1]
	pc, ok := SharedWamLabels[key]
	if !ok {
		fmt.Fprintln(os.Stderr, "unknown predicate: "+key)
		os.Exit(3)
	}
	ctx := NewWamContext(SharedWamCode, SharedWamLabels)
	vm := NewWamStateFromCtx(ctx)
	setupSharedForeignPredicates(vm)
	vm.PC = pc
	if vm.Run() {
		fmt.Println("true")
	} else {
		fmt.Println("false")
	}
}
').

ct_build(go, Preds, Queries, go_ctx(Dir, Map)) :-
    ct_tmp_dir('tmp_ct_go', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds0),
    maplist(qualify_user, AllPreds0, AllPreds),
    write_wam_go_project(AllPreds,
        [package_name(main), module_name('uw_go_ct'), prefer_wam(true)], Dir),
    go_runner_driver(DriverSrc),
    directory_file_path(Dir, 'main.go', MainPath),
    setup_call_cleanup(open(MainPath, write, S), write(S, DriverSrc), close(S)),
    run_proc(go, ['build', '-o', 'runner', '.'], Dir, BExit, BErr),
    ( BExit =:= 0 -> true ; throw(go_build_failed(BExit, BErr)) ).

ct_run(go, go_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(KeyAtom), '~w/0', [WName]), atom_string(KeyAtom, KeyStr),
    % Execute via `go run` rather than exec'ing the built binary directly:
    % process_create(path(AbsPath)) fails to launch a binary under some
    % $TMPDIR layouts, whereas `go` is resolved on PATH and runs the program
    % from its own (warm) build cache. ct_build already gated compilation.
    run_proc_out(go, ['run', '.', KeyStr], Dir, _Exit, OutStr),
    normalize_space(string(Out), OutStr), bool_of_string(Out, Bool).

ct_teardown(go, go_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

% ============================================================
% Adapter: Rust  (0-arity wrapper -> `cargo run -- <key>` -> true/false)
%
% Like Go, write_wam_rust_project/3 attempts native lowering first and
% falls back to the shared WAM pipeline; the conformance predicates all
% take the WAM path. We generate the project, overwrite the emitted
% benchmark src/main.rs with a tiny query-runner driver (look up the
% wrapper label in shared_wam_program(), set vm.pc, run vm.run() — which
% returns true when the machine halts with pc==0), gate compilation with
% `cargo build`, and run each query with `cargo run` (cargo execs the
% binary from its own target dir, sidestepping the $TMPDIR exec issue the
% Go adapter hit). The generated crate has no external deps, so --offline
% needs no network. Opt-in via CONFORMANCE_TARGETS=rust.
% ============================================================

rust_runner_driver('use std::env;
use uw_rust_ct::state::WamState;
use uw_rust_ct::{shared_wam_program, setup_foreign_predicates};

fn main() {
    let key = match env::args().nth(1) {
        Some(k) => k,
        None => { eprintln!("usage: runner <pred/arity>"); std::process::exit(2); }
    };
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels.clone());
    setup_foreign_predicates(&mut vm);
    // Bound execution so a not-yet-conformant (xfailed) program that loops
    // returns false instead of hanging the harness. 5M is generous for the
    // legitimate conformance computations (fib(10)/ack(2,3) are well under
    // it) while bounding the runaway backtracking some programs still hit.
    vm.step_limit = 5_000_000;
    match labels.get(&key) {
        Some(&pc) => {
            vm.pc = pc;
            println!("{}", if vm.run() { "true" } else { "false" });
        }
        None => { eprintln!("unknown predicate: {}", key); std::process::exit(3); }
    }
}
').

ct_build(rust, Preds, Queries, rust_ctx(Dir, Map)) :-
    ct_tmp_dir('tmp_ct_rust', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds0),
    maplist(qualify_user, AllPreds0, AllPreds),
    write_wam_rust_project(AllPreds,
        [module_name('uw_rust_ct'), no_kernels(true)], Dir),
    rust_runner_driver(DriverSrc),
    directory_file_path(Dir, 'src/main.rs', MainPath),
    setup_call_cleanup(open(MainPath, write, S), write(S, DriverSrc), close(S)),
    run_proc(cargo, ['build', '--offline', '-q'], Dir, BExit, BErr),
    ( BExit =:= 0 -> true ; throw(rust_build_failed(BExit, BErr)) ).

ct_run(rust, rust_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(KeyAtom), '~w/0', [WName]), atom_string(KeyAtom, KeyStr),
    run_proc_out(cargo, ['run', '--offline', '-q', '--bin', bench, '--', KeyStr],
                 Dir, _Exit, OutStr),
    normalize_space(string(Out), OutStr), bool_of_string(Out, Bool).

ct_teardown(rust, rust_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

% ============================================================
% Adapter: C  (0-arity wrapper -> compiled binary, result via exit code)
%
% write_wam_c_project/3 emits wam_runtime.c + lib.c (and reuses the static
% wam_runtime.h shipped in the targets tree, which we copy in). Each
% predicate gets a setup_<pred>_<arity>() that registers its code; the
% driver calls them all, then wam_run_predicate(state, "ctw_N/0", ...).
% wam_run returns 0 on success (machine halted) and WAM_HALT on failure,
% which the driver maps to process exit status 0/1. We exec via shell/2
% (the pattern the C target's own tests use) rather than
% process_create(path(AbsBinary)), which is unreliable under $TMPDIR.
% Opt-in via CONFORMANCE_TARGETS=c.
% ============================================================

ct_build(c, Preds, Queries, c_ctx(Dir, Map)) :-
    ct_tmp_dir('tmp_ct_c', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds0),
    maplist(qualify_user, AllPreds0, AllPreds),
    write_wam_c_project(AllPreds, [no_kernels(true)], Dir),
    c_copy_runtime_header(Dir),
    c_runner_driver(AllPreds, DriverSrc),
    directory_file_path(Dir, 'driver.c', DriverPath),
    setup_call_cleanup(open(DriverPath, write, S), write(S, DriverSrc), close(S)),
    run_proc(gcc, ['-O1', '-o', 'runner', 'driver.c', 'lib.c', 'wam_runtime.c'],
             Dir, BExit, BErr),
    ( BExit =:= 0 -> true ; throw(c_build_failed(BExit, BErr)) ).

ct_run(c, c_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(KeyAtom), '~w/0', [WName]),
    absolute_file_name(Dir, AbsDir),
    directory_file_path(AbsDir, runner, Exe),
    % Result is carried by the process exit status: 0 = true, 1 = false.
    format(atom(Cmd), 'timeout 30 ~w ~w', [Exe, KeyAtom]),
    shell(Cmd, Status),
    ( Status =:= 0 -> Bool = true
    ; Status =:= 1 -> Bool = false
    ; throw(c_run_failed(KeyAtom, Status)) ).

ct_teardown(c, c_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

% ============================================================
% Adapter: C++  (0-arity wrapper -> compiled binary, result via exit code)
%
% write_wam_cpp_project/3 with emit_main(true) materialises a self-contained
% project under <Dir>/cpp/ (wam_runtime.{h,cpp}, generated_program.cpp, and
% a main.cpp CLI shim). The shim runs vm.query(argv[1], parsed_args),
% printing true/false and exiting 0/1 — so we need no hand-written driver.
% We build the three .cpp files with g++ and exec via shell/2 (matching the
% C adapter; process_create(path(tmpbin)) is unreliable under $TMPDIR).
% Opt-in via CONFORMANCE_TARGETS=cpp.
% ============================================================

ct_build(cpp, Preds, Queries, cpp_ctx(Dir, Map)) :-
    ct_tmp_dir('tmp_ct_cpp', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds0),
    maplist(qualify_user, AllPreds0, AllPreds),
    write_wam_cpp_project(AllPreds, [emit_main(true)], Dir),
    directory_file_path(Dir, cpp, CppDir),
    run_proc('g++', ['-O1', '-std=c++17', '-o', 'runner',
                     'wam_runtime.cpp', 'generated_program.cpp', 'main.cpp'],
             CppDir, BExit, BErr),
    ( BExit =:= 0 -> true ; throw(cpp_build_failed(BExit, BErr)) ).

ct_run(cpp, cpp_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(KeyAtom), '~w/0', [WName]),
    absolute_file_name(Dir, AbsDir),
    directory_file_path(AbsDir, 'cpp/runner', Exe),
    % Result carried by the process exit status: 0 = true, 1 = false.
    format(atom(Cmd), 'timeout 30 ~w ~w', [Exe, KeyAtom]),
    shell(Cmd, Status),
    ( Status =:= 0 -> Bool = true
    ; Status =:= 1 -> Bool = false
    ; throw(cpp_run_failed(KeyAtom, Status)) ).

ct_teardown(cpp, cpp_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

% ============================================================
% Adapter: Kotlin  (0-arity wrapper -> gradle run --args=<key> -> true/false)
%
% write_wam_kotlin_project/3 emits a Gradle Kotlin project. The human-facing
% Main prints "Ran <pred>" + registers; for conformance we pass
% conformance_main(true) so Main uses WamRuntime.tryRun and prints true/false
% (additive — default Main output for existing e2e tests is unchanged).
%
% Two opt-in targets share this adapter:
%   kotlin            — emit_mode(interpreter)
%   kotlin_functions  — emit_mode(functions) (native-first + WAM fallback)
% Classic multi-clause programs with last-call execute (member/append/acc-reverse)
% and deterministic mid-body call + arithmetic (fib/ack) lower under
% kotlin_functions (EMIT-KOTLIN-4/5). Nondeterministic mid-body call declines.
% ============================================================

ct_build(kotlin, Preds, Queries, kotlin_ctx(Dir, Map)) :-
    kotlin_ct_build(interpreter, Preds, Queries, Dir, Map).

ct_build(kotlin_functions, Preds, Queries, kotlin_ctx(Dir, Map)) :-
    kotlin_ct_build(functions, Preds, Queries, Dir, Map).

kotlin_ct_build(EmitMode, Preds, Queries, Dir, Map) :-
    ct_tmp_dir('tmp_ct_kotlin', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds0),
    maplist(qualify_user, AllPreds0, AllPreds),
    write_wam_kotlin_project(AllPreds,
        [module_name(wam_ct),
         emit_mode(EmitMode),
         conformance_main(true)], Dir),
    run_proc(gradle, ['-q', 'compileKotlin'], Dir, BExit, BErr),
    ( BExit =:= 0 -> true ; throw(kotlin_build_failed(BExit, BErr)) ).

ct_run(kotlin, Ctx, K, A, Bool) :-
    kotlin_ct_run(Ctx, K, A, Bool).
ct_run(kotlin_functions, Ctx, K, A, Bool) :-
    kotlin_ct_run(Ctx, K, A, Bool).

kotlin_ct_run(kotlin_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(KeyAtom), '~w/0', [WName]), atom_string(KeyAtom, KeyStr),
    format(atom(ArgsOpt), '--args=~w', [KeyStr]),
    run_proc_out(gradle, ['-q', 'run', ArgsOpt], Dir, _Exit, OutStr),
    normalize_space(string(Out), OutStr),
    bool_of_string(Out, Bool).

ct_teardown(kotlin, kotlin_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).
ct_teardown(kotlin_functions, kotlin_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

% ============================================================
% Adapter: F#  (0-arity wrapper -> `dotnet run -- <key>` -> true/false)
%
% write_wam_fsharp_project/3 emits a .NET F# exe. The default Program.fs is
% a TSV/LMDB benchmark driver; for conformance we pass conformance_main(true)
% so Program.fs uses tryRun (dispatchCall) and prints true/false (additive —
% default human-facing benchmark output for existing smokes is unchanged).
%
% Two opt-in targets share this adapter:
%   fsharp            — emit_mode(interpreter)
%   fsharp_functions  — emit_mode(functions) (native-first + WAM fallback)
% Opt-in via CONFORMANCE_TARGETS=fsharp[,fsharp_functions].
% ============================================================

ct_build(fsharp, Preds, Queries, fsharp_ctx(Dir, Map)) :-
    fsharp_ct_build(interpreter, Preds, Queries, Dir, Map).

ct_build(fsharp_functions, Preds, Queries, fsharp_ctx(Dir, Map)) :-
    fsharp_ct_build(functions, Preds, Queries, Dir, Map).

fsharp_ct_build(EmitMode, Preds, Queries, Dir, Map) :-
    setenv('DOTNET_CLI_TELEMETRY_OPTOUT', '1'),
    setenv('DOTNET_NOLOGO', '1'),
    ct_tmp_dir('tmp_ct_fsharp', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds0),
    maplist(qualify_user, AllPreds0, AllPreds),
    write_wam_fsharp_project(AllPreds,
        [module_name(wam_ct),
         no_kernels(true),
         emit_mode(EmitMode),
         conformance_main(true)], Dir),
    run_proc(dotnet, ['build', '--nologo', '-v', 'q'], Dir, BExit, BErr),
    ( BExit =:= 0 -> true ; throw(fsharp_build_failed(BExit, BErr)) ).

ct_run(fsharp, Ctx, K, A, Bool) :-
    fsharp_ct_run(Ctx, K, A, Bool).
ct_run(fsharp_functions, Ctx, K, A, Bool) :-
    fsharp_ct_run(Ctx, K, A, Bool).

fsharp_ct_run(fsharp_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(KeyAtom), '~w/0', [WName]), atom_string(KeyAtom, KeyStr),
    % --no-build: ct_build already gated compilation. Do NOT pass --nologo
    % here: on some SDK versions it is forwarded as argv[0] even after `--`,
    % which makes tryRun look up the wrong predicate key. DOTNET_NOLOGO=1
    % (set in fsharp_ct_build) suppresses the banner instead.
    run_proc_out(dotnet, ['run', '--no-build', '--', KeyStr],
                 Dir, _Exit, OutStr),
    normalize_space(string(Out), OutStr),
    bool_of_string(Out, Bool).

ct_teardown(fsharp, fsharp_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).
ct_teardown(fsharp_functions, fsharp_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

%% c_copy_runtime_header(+Dir) — copy the static wam_runtime.h into Dir.
c_copy_runtime_header(Dir) :-
    module_property(wam_c_target, file(TargetFile)),
    file_directory_name(TargetFile, TargetsDir),
    directory_file_path(TargetsDir, 'wam_c_runtime/wam_runtime.h', HdrSrc),
    directory_file_path(Dir, 'wam_runtime.h', HdrDst),
    copy_file(HdrSrc, HdrDst).

%% c_runner_driver(+QualifiedPreds, -CSource)
%  A main() that registers every predicate (setup_<pred>_<arity>) then runs
%  the predicate named on argv[1], exiting 0 (true) / 1 (false).
c_runner_driver(Preds, Src) :-
    findall(Decl-Call,
        ( member(_M:N/A, Preds),
          format(atom(Decl), 'void setup_~w_~w(WamState*);', [N, A]),
          format(atom(Call), '    setup_~w_~w(&state);', [N, A]) ),
        Pairs),
    pairs_keys_values(Pairs, Decls, Calls),
    atomic_list_concat(Decls, '\n', DeclS),
    atomic_list_concat(Calls, '\n', CallS),
    format(atom(Src),
'#include <string.h>\n#include <stdio.h>\n#include "wam_runtime.h"\n~w\n\c
int main(int argc, char **argv) {\n\c
    WamState state; wam_state_init(&state);\n~w\n\c
    if (argc < 2) { return 2; }\n\c
    WamValue args[1];\n\c
    int rc = wam_run_predicate(&state, argv[1], args, 0);\n\c
    wam_free_state(&state);\n\c
    return (rc == 0) ? 0 : 1;\n\c
}\n', [DeclS, CallS]).

qualify_user(_M:N/A, user:N/A) :- !.
qualify_user(N/A, user:N/A).

:- ( prolog_load_context(directory, Dir),
     directory_file_path(Dir, 'fixtures/haskell_conformance_driver.hs', P),
     assertz(classic_haskell_driver_fact(P))
   ; true ).
classic_haskell_driver(P) :- classic_haskell_driver_fact(P).
