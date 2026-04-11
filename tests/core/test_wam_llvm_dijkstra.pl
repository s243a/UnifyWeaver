:- encoding(utf8).
% test_wam_llvm_dijkstra.pl
% Verifies M5.3: @wam_dijkstra_weighted_distance helper.
%
% This is the computational core of weighted_shortest_path3. Uses linear
% scan extract-min over a best[] array — O(V^2), fine for small graphs.
%
% Checks:
%   - Helper is defined in the generated module.
%   - A module with a concrete %WeightedFact table and a demo caller
%     that invokes the helper parses with llvm-as AND passes
%     opt -passes=verify (dominance/phi/SSA check).

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     llvm_emit_weighted_edge_table/3]).
:- use_module(library(process)).

:- dynamic color/1.
color(red).

test_helper_defined :-
    format('--- @wam_dijkstra_weighted_distance defined in template ---~n'),
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('dijk_test')], LLPath),
    read_file_to_string(LLPath, Src, []),
    ( sub_string(Src, _, _, _, 'define i1 @wam_dijkstra_weighted_distance')
    -> format('  PASS: helper definition present~n')
    ;  format('  FAIL: helper missing~n'),
       throw(helper_missing)
    ),
    ( sub_string(Src, _, _, _, '0x7FF0000000000000')
    -> format('  PASS: +inf sentinel 0x7FF0000000000000 present~n')
    ;  format('  FAIL: +inf sentinel missing~n')
    ),
    catch(delete_file(LLPath), _, true).

test_dijkstra_module_validates :-
    format('--- llvm-as + opt verify accept dijkstra caller ---~n'),
    ( process_which('llvm-as')
    -> test_dijkstra_llvm_as
    ;  format('  SKIP: llvm-as not found on PATH~n')
    ).

test_dijkstra_llvm_as :-
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('dijk_test')], LLPath),
    % 4-node graph: a -0.5-> b -1.5-> c -0.3-> d
    %               a -2.0-> d   (direct but slower)
    llvm_emit_weighted_edge_table(dijk_demo_graph,
        [ edge(a, b, 0.5),
          edge(b, c, 1.5),
          edge(c, d, 0.3),
          edge(a, d, 2.0)
        ], TableCode),
    CallerIR = '
define i1 @dijk_demo_caller(i64 %start_id, i64 %target_id, double* %out) {
entry:
  %table_ptr = getelementptr [4 x %WeightedFact], [4 x %WeightedFact]* @dijk_demo_graph, i64 0, i64 0
  %ok = call i1 @wam_dijkstra_weighted_distance(
      i64 %start_id, i64 %target_id,
      %WeightedFact* %table_ptr, i64 4,
      i64 32,
      double* %out)
  ret i1 %ok
}
',
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, TableCode),
          write(Out, '\n'), write(Out, CallerIR),
          write(Out, '\n') ),
        close(Out)),
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit == 0
    -> format('  PASS: llvm-as accepted dijkstra caller module~n')
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
    catch(test_dijkstra_module_validates, E,
        format('  ERROR in llvm-as test: ~w~n', [E])).

:- initialization(test_all, main).
