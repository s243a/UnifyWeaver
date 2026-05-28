:- encoding(utf8).
% test_wam_llvm_effective_distance_smoke.pl
%
% Smoke test: the WAM-LLVM hybrid target must produce VALID LLVM IR
% for a non-trivial Prolog program (effective_distance.pl + dev-scale
% Wikipedia category facts). Pre-this-test there was an off-by-one in
% @module_code's declared length whenever a put_structure / call /
% execute targeted a functor whose name contained `/` (e.g. the
% integer-division operator `//2`): the literal emitter fell through
% to a `; TODO: ...` comment, leaving the array one entry short of
% the declared `[N x %Instruction]` type. `llvm-as` rejected the
% module.
%
% The fix: split_functor_arity/3 in wam_llvm_target.pl now splits on
% the LAST `/` rather than the first, so `//2` parses as Name="/" +
% Arity=2 instead of triggering the fallback.
%
% This test compiles the effective-distance workload + dev-scale facts
% to LLVM IR and runs `llvm-as` against the result. End-to-end runtime
% execution of the benchmark is NOT in scope — many high-level
% predicates (setof/3, findall/3, msort/2, format/2) aren't supported
% by the bytecode interpreter yet, and the foreign-kernel path that
% does work for category_ancestor needs its own driver IR. This test
% verifies the CODEGEN stays valid as the M-series matures.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0,
     split_functor_arity/3]).
:- use_module(library(process)).
:- use_module(library(readutil)).

% --- Unit tests for split_functor_arity/3 itself ---

test_split_functor_arity :-
    format('--- split_functor_arity/3 unit tests ---~n'),
    % atomic_list_concat/3 returns atoms, so expected names below are atoms.
    forall(member(In-ExpectedName-ExpectedArity, [
                "foo/2"    - foo    - 2,
                "bar/0"    - bar    - 0,
                "//2"      - '/'    - 2,
                "+/2"      - '+'    - 2,
                "is/2"     - is     - 2,
                "category_ancestor/4" - category_ancestor - 4,
                "/3"       - ''     - 3,
                "a/b/4"    - 'a/b'  - 4
            ]),
        ( split_functor_arity(In, GotName, GotArity),
          ( GotName == ExpectedName, GotArity == ExpectedArity
          -> format('  PASS: ~q → ~q / ~w~n', [In, GotName, GotArity])
          ;  format('  FAIL: ~q → ~q/~w (expected ~q/~w)~n',
                    [In, GotName, GotArity, ExpectedName, ExpectedArity]),
             throw(split_mismatch(In, GotName, GotArity,
                                  ExpectedName, ExpectedArity))
          )
        )),
    % Malformed input should throw.
    ( catch(split_functor_arity("noslash", _, _), _, true)
    -> format('  PASS: malformed input throws~n')
    ;  format('  FAIL: malformed input did not throw~n'),
       throw(malformed_did_not_throw)
    ).

% --- End-to-end: compile effective_distance.pl to LLVM IR ---

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

test_effective_distance_compiles :-
    format('~n--- compile effective_distance.pl + dev facts to LLVM IR ---~n'),
    clear_llvm_foreign_kernel_specs,
    % Resolve paths relative to this test file so the test works
    % regardless of the caller's CWD.
    source_file(test_effective_distance_compiles, ThisFile),
    file_directory_name(ThisFile, ThisDir),
    directory_file_path(ThisDir, '../../examples/benchmark/effective_distance.pl', WorkloadPath),
    directory_file_path(ThisDir, '../../data/benchmark/dev/facts.pl', FactsPath),
    consult(WorkloadPath),
    consult(FactsPath),
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    % Register category_ancestor as a foreign kernel (the rest are
    % WAM-fallback predicates — they compile to bytecode and run via
    % @run_loop; some use unsupported builtins like setof/findall and
    % will fail at runtime, but they STRUCTURALLY compile).
    write_wam_llvm_project(
        [user:dimension_n/1,
         user:max_depth/1,
         user:category_ancestor/4,
         user:path_to_root/3,
         user:effective_distance/3],
        [ module_name('ed_smoke'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              category_ancestor/4 - category_ancestor -
                  [edge_pred(category_parent/2), max_depth(10)]
          ])
        ],
        LLPath),
    format('  PASS: effective_distance.pl emitted to LLVM IR~n'),
    % The critical check: llvm-as must accept the module. Before the
    % split_functor_arity fix this would fail with
    % "constant expression type mismatch: got type '[N x %Instruction]'
    %  but expected '[N+1 x %Instruction]'" because `put_structure //2`
    % emitted a `; TODO: ...` comment instead of a literal.
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit =:= 0
    -> format('  PASS: llvm-as accepted the effective_distance module~n')
    ;  format('  FAIL: llvm-as rejected (exit=~w)~n', [Exit]),
       throw(llvm_as_rejected)
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(BCPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_all :-
    ( process_which('clang'), process_which('llvm-as')
    -> catch(
         ( test_split_functor_arity,
           test_effective_distance_compiles,
           format('~n=== effective-distance smoke + split_functor_arity tests passed ===~n')
         ),
         E,
         ( format(user_error, '  ERROR: ~w~n', [E]),
           halt(1)
         ))
    ;  format('  SKIP: clang/llvm-as not all on PATH~n')
    ).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

:- initialization(test_all, main).
