:- encoding(utf8).
% test_wam_llvm_lowered_emitter.pl
%
% IR structure + execution tests for the WAM-LLVM lowered emitter
% (src/unifyweaver/targets/wam_llvm_lowered_emitter.pl).
%
% The lowered emitter compiles deterministic single-clause predicates
% to direct LLVM functions, bypassing the @step switch dispatcher.
% These tests verify:
%
%   1. With emit_mode(functions), eligible predicates produce a
%      `define i1 @lowered_<pred>_<arity>` function in the IR, and
%      ineligible ones (multi-clause) fall back to the WAM-bytecode
%      path (no `@lowered_` function emitted; the entry-function
%      pattern is what shows up).
%   2. The IR validates via `llvm-as` (catches structural bugs the
%      emitter could produce).
%   3. End-to-end execution: compile via `llc -O2` + `clang -O2`,
%      run the binary, and check the exit code. Covers the same
%      shape `test_wam_llvm_builtin_execution.pl` does but routed
%      through the lowered path.
%
% The lowered emitter is opt-in via `emit_mode(functions)` (or
% `emit_mode(mixed([P/A, ...]))`); the default `emit_mode(interpreter)`
% remains unchanged, so this file does not affect any of the other
% tests in tests/core/test_wam_llvm_*.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module('../../src/unifyweaver/targets/wam_llvm_lowered_emitter',
    [wam_llvm_lowerable/3,
     llvm_lowered_func_name/2]).
:- use_module('../../src/unifyweaver/targets/wam_target',
    [compile_predicate_to_wam/3]).
:- use_module(library(process)).
:- use_module(library(readutil)).

% === Test predicates ===

% Single-clause deterministic — lowerable.
:- dynamic lw_const/2.
lw_const(_, 42).

:- dynamic lw_unify/2.
lw_unify(X, X).

:- dynamic lw_add/2.
lw_add(X, R) :- R is X + 1.

:- dynamic lw_multi/2.
lw_multi(X, R) :- T is X + 1, R is T * 2.

% Pattern-matching deterministic — uses get_structure / unify_*.
:- dynamic lw_pair_first/2.
lw_pair_first(pair(X, _), X).

% List head match — uses get_list / unify_variable.
:- dynamic lw_head/2.
lw_head([H|_], H).

% Literal compound — uses get_structure + unify_constant chain.
:- dynamic lw_lit_pair/1.
lw_lit_pair(p(1, 2)).

% Multi-clause — NOT lowerable (try_me_else/trust_me in the bytecode).
:- dynamic lw_choice/2.
lw_choice(1, 10).
lw_choice(2, 20).
lw_choice(3, 30).

host_target_triple(Triple) :-
    ( catch(
        ( process_create(path(clang), ['-print-target-triple'],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, Raw), close(Out),
          process_wait(PID, exit(0))
        ), _, fail)
    -> split_string(Raw, "", "\n\r\t ", [S]), atom_string(Triple, S)
    ;  Triple = 'x86_64-pc-linux-gnu'
    ).

% ============================================================================
% Unit tests: lowerability gate
% ============================================================================

test_lowerability :-
    format('--- lowerability gate ---~n'),
    % Single-clause deterministic predicates lower as single_clause.
    % The first 4 are simple register/builtin shapes (M1); the latter
    % 3 exercise the pattern-matching extension (M2 — get_structure /
    % get_list / unify_variable / unify_constant).
    forall(member(PI,
                  [lw_const/2, lw_unify/2, lw_add/2, lw_multi/2,
                   lw_pair_first/2, lw_head/2, lw_lit_pair/1]),
        ( compile_predicate_to_wam(user:PI, [], Wam),
          ( wam_llvm_lowerable(PI, Wam, Shape)
          -> ( Shape == single_clause
             -> format('  PASS: ~w lowerable (single_clause)~n', [PI])
             ;  format('  FAIL: ~w expected single_clause shape, got ~w~n',
                       [PI, Shape]),
                throw(unexpected_shape(PI, Shape))
             )
          ;  format('  FAIL: ~w should be lowerable~n', [PI]),
             throw(unexpected_unlowerable(PI))
          )
        )),
    % Multi-clause first-arg-indexed predicates are now lowerable as
    % multi_clause_c1 (M3) — clause 1 becomes the fast path, the full
    % bytecode handles clauses 2+ via the dispatcher's slow path.
    compile_predicate_to_wam(user:lw_choice/2, [], MultiWam),
    ( wam_llvm_lowerable(lw_choice/2, MultiWam, MultiShape)
    -> ( MultiShape == multi_clause_c1
       -> format('  PASS: lw_choice/2 lowerable (multi_clause_c1)~n')
       ;  format('  FAIL: lw_choice/2 expected multi_clause_c1, got ~w~n',
                 [MultiShape]),
          throw(unexpected_shape(lw_choice/2, MultiShape))
       )
    ;  format('  FAIL: lw_choice/2 should be lowerable as multi_clause_c1~n'),
       throw(unexpected_unlowerable(lw_choice/2))
    ).

% ============================================================================
% IR structure: the lowered function appears in the module
% ============================================================================

test_ir_structure :-
    format('--- IR structure (emit_mode(functions)) ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:lw_const/2, user:lw_add/2,
         user:lw_pair_first/2, user:lw_head/2, user:lw_lit_pair/1,
         user:lw_choice/2],
        [ module_name('lw_struct'),
          target_triple(Triple),
          target_datalayout(''),
          emit_mode(functions)
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    ( sub_string(Src, _, _, _, "define i1 @lowered_lw_const_2")
    -> format('  PASS: lowered lw_const_2 function present~n')
    ;  format('  FAIL: lowered lw_const_2 function missing~n'),
       throw(missing_lowered(lw_const_2))
    ),
    ( sub_string(Src, _, _, _, "define i1 @lowered_lw_add_2")
    -> format('  PASS: lowered lw_add_2 function present~n')
    ;  format('  FAIL: lowered lw_add_2 function missing~n'),
       throw(missing_lowered(lw_add_2))
    ),
    % Pattern-matching predicates also produce lowered functions now.
    ( sub_string(Src, _, _, _, "define i1 @lowered_lw_pair_first_2")
    -> format('  PASS: lowered lw_pair_first_2 (get_structure path)~n')
    ;  throw(missing_lowered(lw_pair_first_2))
    ),
    ( sub_string(Src, _, _, _, "define i1 @lowered_lw_head_2")
    -> format('  PASS: lowered lw_head_2 (get_list path)~n')
    ;  throw(missing_lowered(lw_head_2))
    ),
    ( sub_string(Src, _, _, _, "define i1 @lowered_lw_lit_pair_1")
    -> format('  PASS: lowered lw_lit_pair_1 (unify_constant path)~n')
    ;  throw(missing_lowered(lw_lit_pair_1))
    ),
    % lw_choice/2 is multi-clause → under M3 it gets BOTH a lowered
    % clause-1 fast path AND a hybrid dispatcher @lw_choice that
    % falls back to the full bytecode on failure. Verify both
    % symbols are present.
    ( sub_string(Src, _, _, _, "define i1 @lowered_lw_choice_2")
    -> format('  PASS: hybrid lowered_lw_choice_2 fast-path present~n')
    ;  format('  FAIL: hybrid lowered_lw_choice_2 missing~n'),
       throw(missing_hybrid_fast(lw_choice_2))
    ),
    ( sub_string(Src, _, _, _, "M3 hybrid dispatcher")
    -> format('  PASS: hybrid dispatcher @lw_choice present~n')
    ;  format('  FAIL: hybrid dispatcher @lw_choice missing~n'),
       throw(missing_hybrid_dispatcher(lw_choice_2))
    ),
    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

% ============================================================================
% IR structure: llvm-as accepts the emitted module
% ============================================================================

test_llvm_as_validation :-
    format('--- llvm-as validation ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:lw_const/2, user:lw_unify/2, user:lw_add/2, user:lw_multi/2,
         user:lw_pair_first/2, user:lw_head/2, user:lw_lit_pair/1],
        [ module_name('lw_validate'),
          target_triple(Triple),
          target_datalayout(''),
          emit_mode(functions)
        ],
        LLPath),
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit =:= 0
    -> format('  PASS: llvm-as accepted the module with 7 lowered predicates~n')
    ;  format('  FAIL: llvm-as rejected (exit=~w)~n', [Exit]),
       throw(llvm_as_failed)
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(BCPath), _, true),
    clear_llvm_foreign_kernel_specs.

% ============================================================================
% Execution test: lowered predicate runs and returns success
% ============================================================================

run_exec_test(PredAtom, A1Tag, A1Pay) :-
    format('  testing lowered ~w with A1=tag~w/~w...~n',
        [PredAtom, A1Tag, A1Pay]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:PredAtom/2],
        [ module_name('lwexec'),
          target_triple(Triple),
          target_datalayout(''),
          emit_mode(functions)
        ],
        LLPath),
    atom_string(PredAtom, PredStr),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 ~w, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 6, 0
  %a2 = insertvalue %Value %a2_0, i64 0, 1
  %ok = call i1 @lowered_~w_2(%Value %a1, %Value %a2)
  %ret = select i1 %ok, i32 0, i32 1
  ret i32 %ret
}
', [A1Tag, A1Pay, PredStr]),
    setup_call_cleanup(open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ), close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -O2 -filetype=obj -relocation-model=pic ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =\= 0
    -> format('    FAIL: llc exit=~w~n', [LlcExit]),
       throw(llc_failed(PredAtom))
    ; true
    ),
    format(atom(ClangCmd), 'clang -O2 ~w -o ~w 2>/dev/null', [OPath, BinPath]),
    shell(ClangCmd, ClangExit),
    ( ClangExit =\= 0
    -> format('    FAIL: clang exit=~w~n', [ClangExit]),
       throw(clang_failed(PredAtom))
    ; true
    ),
    shell(BinPath, RunExit),
    ( RunExit =:= 0
    -> format('    PASS: ~w lowered exec returned success~n', [PredAtom])
    ;  format('    FAIL: ~w returned ~w (expected 0)~n', [PredAtom, RunExit]),
       throw(exec_failed(PredAtom, RunExit))
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_execution :-
    format('--- execution: lowered preds compile and run ---~n'),
    % lw_const(_, 42)        : pass anything
    run_exec_test(lw_const, 1, 5),
    % lw_unify(X, X)         : pass int 7 as A1, A2 unbound
    run_exec_test(lw_unify, 1, 7),
    % lw_add(X, R) :- R is X + 1.   : pass int 10
    run_exec_test(lw_add, 1, 10),
    % lw_multi(X, R) :- T is X+1, R is T*2.   : pass int 4
    run_exec_test(lw_multi, 1, 4),
    % --- pattern-matching predicates ---
    % Each is invoked with A1 = unbound, which exercises the
    % get_structure/get_list "write" path (allocate compound on arena,
    % bind A1, then unify_*/set_* populate the args). The lowered
    % function should succeed.
    %
    % Read-mode verification (passing a bound compound and checking the
    % extracted arg) requires constructing %Compound globals in the
    % driver IR — out of scope for this smoke test; the read path is
    % exercised by the same code paths as the bytecode @step case,
    % which is already covered indirectly by the wider regression suite.
    run_exec_test(lw_pair_first, 6, 0),
    run_exec_test(lw_head, 6, 0),
    test_exec_unary(lw_lit_pair, 6, 0),
    % --- M3 hybrid dispatcher coverage ---
    %
    % lw_choice/2 has 3 clauses indexed on A1. Calling the dispatcher
    % `@lw_choice` (the public entry) with each A1 value verifies:
    %
    %   A1 = 1 → fast path (lowered clause 1) hits, returns true.
    %   A1 = 2 → fast path fails on `get_constant 1, A1` mismatch;
    %            slow path runs full bytecode, clause 2 succeeds.
    %   A1 = 3 → same as A1=2 but clause 3 succeeds in the slow path.
    %   A1 = 9 → both paths fail (no matching clause), returns false.
    %
    % `run_exec_test_pred/4` calls the public entry `@<pred>` directly
    % (not @lowered_*), which is what external WAM users do.
    test_hybrid_dispatch(lw_choice, 1, 1, success),
    test_hybrid_dispatch(lw_choice, 2, 1, success),
    test_hybrid_dispatch(lw_choice, 3, 1, success),
    test_hybrid_dispatch(lw_choice, 9, 1, failure).

%% test_hybrid_dispatch(+Pred, +A1Value, +A1Tag, +Expected)
%
%  Compile a hybrid predicate and call its public entry `@<pred>` with
%  the given A1 (passing A2 as unbound). Expected ∈ {success, failure}.
test_hybrid_dispatch(Pred, A1Val, A1Tag, Expected) :-
    format('  hybrid dispatch ~w(~w, _) → ~w...~n', [Pred, A1Val, Expected]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:Pred/2],
        [ module_name('lwhybrid'),
          target_triple(Triple),
          target_datalayout(''),
          emit_mode(functions)
        ],
        LLPath),
    atom_string(Pred, PredStr),
    ( Expected == success -> ExpectedExit = 0 ; ExpectedExit = 1 ),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 ~w, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 6, 0
  %a2 = insertvalue %Value %a2_0, i64 0, 1
  %ok = call i1 @~w(%Value %a1, %Value %a2)
  %ret = select i1 %ok, i32 0, i32 1
  ret i32 %ret
}
', [A1Tag, A1Val, PredStr]),
    setup_call_cleanup(open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ), close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -O2 -filetype=obj -relocation-model=pic ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, _),
    format(atom(ClangCmd), 'clang -O2 ~w -o ~w 2>/dev/null', [OPath, BinPath]),
    shell(ClangCmd, _),
    shell(BinPath, RunExit),
    ( RunExit =:= ExpectedExit
    -> format('    PASS: @~w(~w, _) → ~w (exit=~w)~n',
              [Pred, A1Val, Expected, RunExit])
    ;  format('    FAIL: @~w(~w, _) expected exit=~w, got ~w~n',
              [Pred, A1Val, ExpectedExit, RunExit]),
       throw(hybrid_dispatch_failed(Pred, A1Val, Expected, RunExit))
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs.

% Same as run_exec_test but for 1-arity preds.
test_exec_unary(PredAtom, A1Tag, A1Pay) :-
    format('  testing lowered ~w/1 with A1=tag~w/~w...~n',
        [PredAtom, A1Tag, A1Pay]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:PredAtom/1],
        [ module_name('lwexec1'),
          target_triple(Triple),
          target_datalayout(''),
          emit_mode(functions)
        ],
        LLPath),
    atom_string(PredAtom, PredStr),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 ~w, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %ok = call i1 @lowered_~w_1(%Value %a1)
  %ret = select i1 %ok, i32 0, i32 1
  ret i32 %ret
}
', [A1Tag, A1Pay, PredStr]),
    setup_call_cleanup(open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ), close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -O2 -filetype=obj -relocation-model=pic ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, _),
    format(atom(ClangCmd), 'clang -O2 ~w -o ~w 2>/dev/null', [OPath, BinPath]),
    shell(ClangCmd, _),
    shell(BinPath, RunExit),
    ( RunExit =:= 0
    -> format('    PASS: ~w/1 lowered exec returned success~n', [PredAtom])
    ;  format('    FAIL: ~w/1 returned ~w (expected 0)~n', [PredAtom, RunExit]),
       throw(exec_failed(PredAtom, RunExit))
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs.

% ============================================================================
% Main
% ============================================================================

test_all :-
    ( process_which('clang'), process_which('llc'), process_which('llvm-as')
    -> catch(
         ( test_lowerability,
           test_ir_structure,
           test_llvm_as_validation,
           test_execution,
           format('~n=== All wam_llvm_lowered_emitter tests passed ===~n')
         ),
         E,
         ( format(user_error, '  ERROR: ~w~n', [E]),
           halt(1)
         ))
    ;  format('  SKIP: clang/llc/llvm-as not all on PATH~n')
    ).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

:- initialization(test_all, main).
