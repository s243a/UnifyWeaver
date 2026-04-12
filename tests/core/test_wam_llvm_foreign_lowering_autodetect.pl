:- encoding(utf8).
% test_wam_llvm_foreign_lowering_autodetect.pl
% Verifies M5.6c: clause-shape auto-detection of foreign kernels.
% Mirrors the Rust target's rust_recursive_kernel_detector/2 registry.
%
% The user writes a normal recursive Prolog predicate matching a
% known kernel shape (here: transitive_distance3). When they pass
% `foreign_lowering(true)` in the options list, the compile pipeline
% walks each predicate's clauses, runs them through the detectors,
% and asserts a llvm_foreign_kernel_spec/3 for any that match.
% Downstream behavior is identical to paths (a) and (b).

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     llvm_foreign_kernel_spec/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).

% Edge predicate — facts the kernel's fact-table emitter will read.
:- dynamic edge/2.
edge(p, q).
edge(q, r).
edge(r, s).
edge(p, t).

% A predicate matching the transitive_distance3 clause shape:
%   base: pred(Start, Target, 1) :- edge(Start, Target).
%   rec:  pred(Start, Target, Depth) :-
%             edge(Start, Mid),
%             pred(Mid, Target, Prev),
%             Depth is Prev + 1.
:- dynamic reach/3.
reach(Start, Target, 1) :- edge(Start, Target).
reach(Start, Target, Depth) :-
    edge(Start, Mid),
    reach(Mid, Target, Prev),
    Depth is Prev + 1.

test_autodetect_populates_spec :-
    format('--- auto-detect registers td3 from clause shape ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project(
        [user:reach/3],
        [ module_name('td3auto_test'),
          foreign_lowering(true)
        ],
        LLPath),
    ( llvm_foreign_kernel_spec(reach/3, transitive_distance3, Config)
    -> format('  PASS: auto-detect registered reach/3 as td3, config=~w~n', [Config])
    ;  format('  FAIL: auto-detect did not match reach/3~n'),
       throw(no_autodetect_match)
    ),
    read_file_to_string(LLPath, Src, []),
    % First registered td3 spec (reach/3) gets instance_id=0.
    ( sub_string(Src, _, _, _, '%Instruction { i32 30, i64 4, i64 0 }')
    -> format('  PASS: call_foreign tag=30 kind=4 instance=0 emitted~n')
    ;  format('  FAIL: no call_foreign instruction~n')
    ),
    ( sub_string(Src, _, _, _, 'define i1 @wam_td3_kernel_impl(%WamState* %vm, i32 %instance)')
    -> format('  PASS: concrete td3 impl spliced in~n')
    ;  format('  FAIL: concrete impl missing~n')
    ),
    ( sub_string(Src, _, _, _, '@td3_inst_reach_0_edges')
    -> format('  PASS: instance edge table global present~n')
    ;  format('  FAIL: fact table missing~n')
    ),
    ( process_which('llvm-as')
    -> atom_concat(LLPath, '.bc', BCPath),
       format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
       shell(Cmd, Exit),
       ( Exit == 0
       -> format('  PASS: llvm-as accepted the auto-detected module~n')
       ;  format('  FAIL: llvm-as exit=~w~n', [Exit])
       ),
       ( process_which('opt')
       -> format(atom(VCmd),
           'opt -passes=verify -disable-output ~w 2>&1', [BCPath]),
          shell(VCmd, VExit),
          ( VExit == 0
          -> format('  PASS: opt -passes=verify accepted bitcode~n')
          ;  format('  FAIL: opt -passes=verify exit=~w~n', [VExit])
          )
       ;  format('  SKIP: opt not found~n')
       ),
       catch(delete_file(BCPath), _, true)
    ;  format('  SKIP: llvm-as not found~n')
    ),
    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_autodetect_off_by_default :-
    format('--- auto-detect off by default (no foreign_lowering flag) ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    % Same td3-shaped reach/3 predicate, but WITHOUT foreign_lowering(true).
    % The auto-detector should not run, so reach/3 falls through to
    % the normal WAM compilation path.
    write_wam_llvm_project(
        [user:reach/3],
        [module_name('td3auto_off_test')],
        LLPath),
    ( llvm_foreign_kernel_spec(reach/3, _, _)
    -> format('  FAIL: spec registered without foreign_lowering(true)~n'),
       throw(spec_leaked)
    ;  format('  PASS: no spec registered when flag is off~n')
    ),
    read_file_to_string(LLPath, Src, []),
    ( sub_string(Src, _, _, _, '%Instruction { i32 30, i64 4,')
    -> format('  FAIL: call_foreign emitted despite flag off~n'),
       throw(call_foreign_leaked)
    ;  format('  PASS: no call_foreign emitted (normal WAM path taken)~n')
    ),
    catch(delete_file(LLPath), _, true).

test_non_matching_predicate_ignored :-
    format('--- non-matching predicate is not auto-registered ---~n'),
    clear_llvm_foreign_kernel_specs,
    assertz((user:unrelated(X, Y) :- X = Y)),
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project(
        [user:unrelated/2],
        [ module_name('td3auto_nomatch_test'),
          foreign_lowering(true)
        ],
        LLPath),
    ( llvm_foreign_kernel_spec(unrelated/2, _, _)
    -> format('  FAIL: non-td3 predicate was wrongly auto-registered~n'),
       throw(false_positive)
    ;  format('  PASS: unrelated/2 did not match any detector~n')
    ),
    retractall(user:unrelated(_, _)),
    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _),
          close(Out),
          process_wait(PID, exit(0))
        ),
        _,
        fail).

test_all :-
    catch(test_autodetect_populates_spec, E1,
        format('  ERROR: ~w~n', [E1])),
    catch(test_autodetect_off_by_default, E2,
        format('  ERROR: ~w~n', [E2])),
    catch(test_non_matching_predicate_ignored, E3,
        format('  ERROR: ~w~n', [E3])).

:- initialization(test_all, main).
