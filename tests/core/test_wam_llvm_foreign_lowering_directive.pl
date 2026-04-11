:- encoding(utf8).
% test_wam_llvm_foreign_lowering_directive.pl
% Verifies M5.6b: the Prolog directive entry point for foreign kernel
% lowering (path (a) in the M5.6 design).
%
% The user writes `:- foreign_kernel(Pred/Arity, Kind, Config)` at
% load time in their Prolog source, and the directive asserts a
% llvm_foreign_kernel_spec/3 fact that the compile pipeline picks up.
% Downstream of that assertion the behavior is identical to the
% options-list path tested in M5.6a.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     foreign_kernel/3,
     llvm_foreign_kernel_spec/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).

% Edge predicate used as the kernel's fact source.
:- dynamic edge/2.
edge(x, y).
edge(y, z).
edge(z, w).

% The predicate we want to lower. No real body — the compile pipeline
% replaces it with a call_foreign instruction.
:- dynamic trav/3.
trav(_, _, _) :- fail.

% Clear any stale specs from other tests, then use the directive form
% to register trav/3 as a foreign td3 kernel.
:- clear_llvm_foreign_kernel_specs.
:- foreign_kernel(trav/3, transitive_distance3, [edge_pred(edge/2)]).

test_directive_populated_spec_table :-
    format('--- :- foreign_kernel(...) directive asserts spec ---~n'),
    ( llvm_foreign_kernel_spec(trav/3, transitive_distance3, Config)
    -> format('  PASS: spec present, config=~w~n', [Config])
    ;  format('  FAIL: directive did not populate spec table~n'),
       throw(directive_failed)
    ).

test_directive_path_generates_module :-
    format('--- directive-path compile produces valid module ---~n'),
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project(
        [user:trav/3],
        [module_name('td3dir_test')],
        LLPath),
    read_file_to_string(LLPath, Src, []),

    ( sub_string(Src, _, _, _, 'M5.6 concrete td3 impl')
    -> format('  PASS: concrete td3 impl spliced in~n')
    ;  format('  FAIL: concrete td3 impl missing~n'),
       throw(no_concrete_impl)
    ),
    ( sub_string(Src, _, _, _, '%Instruction { i32 30, i64 4, i64 3 }')
    -> format('  PASS: call_foreign tag=30 kind=4 arity=3 emitted~n')
    ;  format('  FAIL: call_foreign instruction not emitted~n'),
       throw(no_call_foreign)
    ),
    ( sub_string(Src, _, _, _, '@foreign_td3_edge_2')
    -> format('  PASS: fact table global present~n')
    ;  format('  FAIL: fact table global missing~n')
    ),

    ( process_which('llvm-as')
    -> atom_concat(LLPath, '.bc', BCPath),
       format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
       shell(Cmd, Exit),
       ( Exit == 0
       -> format('  PASS: llvm-as accepted the directive-driven module~n')
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
    catch(delete_file(LLPath), _, true).

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
    test_directive_populated_spec_table,
    catch(test_directive_path_generates_module, E,
        format('  ERROR: ~w~n', [E])),
    clear_llvm_foreign_kernel_specs.

:- initialization(test_all, main).
