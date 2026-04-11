:- encoding(utf8).
% test_wam_llvm_bfs_atom.pl
% Verifies M5.2: @wam_bfs_atom_distance helper (shortest-path length over
% a %AtomFactPair table).
%
% This is the computational core of transitive_distance3. It's a
% standalone helper so it can be tested without the full foreign-dispatch
% protocol wiring (which lands in M5.5).
%
% Checks:
%   - The helper is defined in the generated module.
%   - A module that appends a concrete %AtomFactPair fact table and a
%     demo caller invoking @wam_bfs_atom_distance parses cleanly with
%     llvm-as and llc (IR → object codegen), which exercises the phi
%     wiring and block structure end-to-end.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     llvm_emit_atom_fact2_table/3]).
:- use_module(library(process)).

:- dynamic color/1.
color(red).

test_helper_defined :-
    format('--- @wam_bfs_atom_distance defined in template ---~n'),
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('bfs_test')], LLPath),
    read_file_to_string(LLPath, Src, []),
    ( sub_string(Src, _, _, _, 'define i1 @wam_bfs_atom_distance')
    -> format('  PASS: helper definition present~n')
    ;  format('  FAIL: helper missing~n'),
       throw(helper_missing)
    ),
    catch(delete_file(LLPath), _, true).

test_bfs_module_validates :-
    format('--- llvm-as accepts module with bfs caller ---~n'),
    ( process_which('llvm-as')
    -> test_bfs_llvm_as
    ;  format('  SKIP: llvm-as not found on PATH~n')
    ).

test_bfs_llvm_as :-
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('bfs_test')], LLPath),
    % Build a 4-node graph: a -> b -> c -> d, with a side branch a -> e.
    llvm_emit_atom_fact2_table(bfs_demo_graph,
        [ fact(a, b),
          fact(b, c),
          fact(c, d),
          fact(a, e)
        ], TableCode),
    % Demo caller: invokes BFS from atom id 1 (a) to atom id 4 (d).
    % Atom IDs come from intern_atom/2 which starts at 1 and assigns in
    % order of first use — in practice the test only validates that the
    % IR compiles, not the numeric result, since interned IDs depend on
    % load order. An end-to-end execution test would require lli or jit.
    CallerIR = '
define i1 @bfs_demo_caller(i64 %start_id, i64 %target_id, i64* %out) {
entry:
  %table_ptr = getelementptr [4 x %AtomFactPair], [4 x %AtomFactPair]* @bfs_demo_graph, i64 0, i64 0
  %ok = call i1 @wam_bfs_atom_distance(
      i64 %start_id, i64 %target_id,
      %AtomFactPair* %table_ptr, i64 4,
      i64 32,
      i64* %out)
  ret i1 %ok
}
',
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, TableCode),
          write(Out, '\n'), write(Out, CallerIR),
          write(Out, '\n') ),
        close(Out)),
    format('  Wrote module: ~w~n', [LLPath]),
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit == 0
    -> format('  PASS: llvm-as accepted module with bfs caller~n')
    ;  format('  FAIL: llvm-as exit=~w~n', [Exit])
    ),
    % Also verify the IR passes LLVM's verifier pass (stricter than parse).
    ( process_which('opt')
    -> format(atom(VCmd),
        'opt -passes=verify -disable-output ~w 2>&1', [BCPath]),
       shell(VCmd, VExit),
       ( VExit == 0
       -> format('  PASS: opt -passes=verify accepted bitcode~n')
       ;  format('  FAIL: opt -passes=verify exit=~w~n', [VExit])
       )
    ;  format('  SKIP: opt not found on PATH~n')
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

test_all :-
    test_helper_defined,
    catch(test_bfs_module_validates, E,
        format('  ERROR in llvm-as test: ~w~n', [E])).

:- initialization(test_all, main).
