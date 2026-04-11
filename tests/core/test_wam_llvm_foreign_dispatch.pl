:- encoding(utf8).
% test_wam_llvm_foreign_dispatch.pl
% Verifies the M3 foreign dispatch scaffolding:
%  - call_foreign text parser produces a valid %Instruction literal
%  - Tag 30 appears in generated LLVM IR
%  - @wam_execute_foreign_predicate helper is emitted
%  - llvm-as accepts the complete module
%
% This PR only adds the dispatch infrastructure — no actual kernels are
% registered yet. M5 will replace the stubs in @wam_execute_foreign_predicate.
%
% The pure-WAM path (foreign_lowering off by default) is preserved: no
% call_foreign instructions are emitted unless the user explicitly opts in.

:- use_module('../../src/unifyweaver/targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [compile_wam_predicate_to_llvm/4, write_wam_llvm_project/3, wam_llvm_foreign_kind_id/2]).
:- use_module(library(process)).

% --- Test 1: pure WAM path emits no call_foreign (default behavior) ---
:- dynamic simple_fact/1.
simple_fact(a).
simple_fact(b).
simple_fact(c).

test_pure_wam_no_foreign :-
    format('--- pure WAM path emits no call_foreign ---~n'),
    compile_predicate_to_wam(user:simple_fact/1, [], WamCode),
    compile_wam_predicate_to_llvm(simple_fact/1, WamCode, [], LLVMCode),
    ( sub_atom_or_string(LLVMCode, _, _, _, 'i32 30')
    -> format('  FAIL: pure WAM path emitted call_foreign (tag 30)~n'),
       throw(unexpected_call_foreign)
    ;  format('  PASS: no call_foreign in pure WAM output~n')
    ).

% --- Test 2: hand-crafted WAM text with call_foreign parses correctly ---
test_call_foreign_parses :-
    format('--- call_foreign text parser ---~n'),
    % Hand-craft a minimal WAM text that uses call_foreign.
    % This is what the M5 compile path will emit for foreign predicates.
    atom_string(FakeWam,
'fake_pred/3:
    call_foreign transitive_distance3, 3
    proceed'),
    compile_wam_predicate_to_llvm(fake_pred/3, FakeWam, [], LLVMCode),
    ( sub_atom_or_string(LLVMCode, _, _, _, 'i32 30')
    -> format('  PASS: LLVM output contains tag 30 (call_foreign)~n')
    ;  format('  FAIL: LLVM output missing tag 30~n'),
       throw(missing_call_foreign_tag)
    ),
    % transitive_distance3 has kind ID 4
    ( sub_atom_or_string(LLVMCode, _, _, _, 'i32 30, i64 4')
    -> format('  PASS: call_foreign encodes transitive_distance3 kind ID = 4~n')
    ;  format('  FAIL: kind ID not encoded correctly~n'),
       throw(wrong_kind_id)
    ).

% --- Test 3: kind ID registry is consistent ---
test_kind_registry :-
    format('--- foreign kind ID registry ---~n'),
    ExpectedKinds = [
        category_ancestor-0,
        countdown_sum2-1,
        list_suffix2-2,
        transitive_closure2-3,
        transitive_distance3-4,
        weighted_shortest_path3-5,
        astar_shortest_path4-6
    ],
    forall(member(Kind-ExpectedId, ExpectedKinds),
        ( wam_llvm_foreign_kind_id(Kind, ActualId),
          ( ActualId == ExpectedId
          -> format('  PASS: ~w -> ~w~n', [Kind, ActualId])
          ;  format('  FAIL: ~w expected ~w got ~w~n', [Kind, ExpectedId, ActualId]),
             throw(wrong_kind_id(Kind, ExpectedId, ActualId))
          )
        )).

% --- Test 4: full module with call_foreign validates via llvm-as ---
test_full_module_validates :-
    format('--- Full module with call_foreign validates with llvm-as ---~n'),
    ( process_which('llvm-as')
    -> test_llvm_as_with_foreign
    ;  format('  SKIP: llvm-as not found on PATH~n')
    ).

test_llvm_as_with_foreign :-
    % Hand-craft a complete predicate module containing call_foreign.
    % We bypass the compile_predicate_to_wam path since M3 doesn't yet
    % have the foreign pattern matcher wired in.
    atom_string(FakeWam,
'demo_fp/3:
    call_foreign weighted_shortest_path3, 3
    proceed'),
    compile_wam_predicate_to_llvm(demo_fp/3, FakeWam, [], PredCode),
    format('  Predicate compiles to ~w chars of LLVM~n', [PredCode]),
    % Build a minimal complete module that includes this predicate
    % alongside a second pure-WAM predicate for good measure.
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:simple_fact/1], [module_name('m3_test')], LLPath),
    % Append our hand-crafted foreign predicate
    setup_call_cleanup(
        open(LLPath, append, Out),
        write(Out, PredCode),
        close(Out)),
    format('  Wrote: ~w~n', [LLPath]),
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit == 0
    -> format('  PASS: llvm-as accepted the module with call_foreign~n')
    ;  format('  FAIL: llvm-as exit=~w~n', [Exit])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(BCPath), _, true).

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

sub_atom_or_string(Haystack, Before, Length, After, Needle) :-
    ( atom(Haystack) -> sub_atom(Haystack, Before, Length, After, Needle)
    ; string(Haystack) -> sub_string(Haystack, Before, Length, After, Needle)
    ; atom_string(Atom, Haystack), sub_atom(Atom, Before, Length, After, Needle)
    ).

test_all :-
    test_pure_wam_no_foreign,
    test_call_foreign_parses,
    test_kind_registry,
    catch(test_full_module_validates, E,
        format('  ERROR in llvm-as test: ~w~n', [E])).

:- initialization(test_all, main).
