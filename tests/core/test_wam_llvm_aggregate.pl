:- encoding(utf8).
% test_wam_llvm_aggregate.pl
% Verifies that the LLVM WAM target can compile predicates using
% aggregate_all/3 without producing invalid IR.

:- use_module('../../src/unifyweaver/targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [compile_wam_predicate_to_llvm/4, write_wam_llvm_project/3]).
:- use_module(library(process)).

% --- Test fixture: a predicate using aggregate_all(min, ...) ---
:- dynamic number_fact/1.
number_fact(5).
number_fact(3).
number_fact(8).
number_fact(1).
number_fact(7).

% p(Y) :- aggregate_all(min(X), number_fact(X), Y).
:- dynamic p/1.
p(Y) :- aggregate_all(min(X), number_fact(X), Y).

test_aggregate_compiles :-
    format('--- aggregate_all compilation ---~n'),
    compile_predicate_to_wam(user:p/1, [], WamCode),
    format('  WAM code generated (~w chars)~n', [WamCode]),
    ( sub_atom_or_string(WamCode, _, _, _, 'begin_aggregate')
    -> format('  PASS: WAM contains begin_aggregate~n')
    ;  format('  FAIL: WAM does not contain begin_aggregate~n'),
       throw(missing_begin_aggregate)
    ),
    ( sub_atom_or_string(WamCode, _, _, _, 'end_aggregate')
    -> format('  PASS: WAM contains end_aggregate~n')
    ;  format('  FAIL: WAM does not contain end_aggregate~n'),
       throw(missing_end_aggregate)
    ),
    compile_wam_predicate_to_llvm(p/1, WamCode, [], LLVMCode),
    format('  LLVM code generated (~w chars)~n', [LLVMCode]),
    % Tag 28: begin_aggregate
    ( sub_atom_or_string(LLVMCode, _, _, _, 'i32 28')
    -> format('  PASS: LLVM output contains tag 28 (begin_aggregate)~n')
    ;  format('  FAIL: LLVM output missing tag 28~n'),
       throw(missing_begin_agg_tag)
    ),
    % Tag 29: end_aggregate
    ( sub_atom_or_string(LLVMCode, _, _, _, 'i32 29')
    -> format('  PASS: LLVM output contains tag 29 (end_aggregate)~n')
    ;  format('  FAIL: LLVM output missing tag 29~n'),
       throw(missing_end_agg_tag)
    ).

test_full_module_validates :-
    format('--- Full module validates with llvm-as ---~n'),
    ( process_which('llvm-as')
    -> test_llvm_as
    ;  format('  SKIP: llvm-as not found on PATH~n')
    ).

test_llvm_as :-
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:p/1], [module_name('test_agg')], LLPath),
    format('  Wrote: ~w~n', [LLPath]),
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit == 0
    -> format('  PASS: llvm-as accepted the generated IR~n')
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
    test_aggregate_compiles,
    catch(test_full_module_validates, E,
        format('  ERROR in llvm-as test: ~w~n', [E])).

:- initialization(test_all, main).
