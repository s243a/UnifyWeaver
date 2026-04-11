:- encoding(utf8).
% test_wam_llvm_astar.pl
% Verifies M5.4: @wam_astar_weighted_distance helper.
%
% A* over a %WeightedFact table with a precomputed heuristic double[]
% indexed by atom ID. Same structure as Dijkstra but orders the
% frontier by f(n) = g(n) + h(n).

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     llvm_emit_weighted_edge_table/3]).
:- use_module(library(process)).

:- dynamic color/1.
color(red).

test_helper_defined :-
    format('--- @wam_astar_weighted_distance defined in template ---~n'),
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('astar_test')], LLPath),
    read_file_to_string(LLPath, Src, []),
    ( sub_string(Src, _, _, _, 'define i1 @wam_astar_weighted_distance')
    -> format('  PASS: helper definition present~n')
    ;  format('  FAIL: helper missing~n'),
       throw(helper_missing)
    ),
    ( sub_string(Src, _, _, _, '%heuristic')
    -> format('  PASS: heuristic parameter present~n')
    ;  format('  FAIL: heuristic parameter missing~n')
    ),
    catch(delete_file(LLPath), _, true).

test_astar_module_validates :-
    format('--- llvm-as + opt verify accept astar caller ---~n'),
    ( process_which('llvm-as')
    -> test_astar_llvm_as
    ;  format('  SKIP: llvm-as not found on PATH~n')
    ).

test_astar_llvm_as :-
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('astar_test')], LLPath),
    llvm_emit_weighted_edge_table(astar_demo_graph,
        [ edge(a, b, 0.5),
          edge(b, c, 1.5),
          edge(c, d, 0.3),
          edge(a, d, 2.0)
        ], TableCode),
    % Demo caller provides a 33-entry heuristic array (indices 0..32).
    % The numbers are arbitrary — we only check that the IR compiles,
    % not that A* finds the right answer for this particular heuristic.
    HeuristicIR = '
@astar_demo_heuristic = private constant [33 x double] [
  double 0.0, double 1.8, double 1.5, double 0.3, double 0.0,
  double 0.0, double 0.0, double 0.0, double 0.0, double 0.0,
  double 0.0, double 0.0, double 0.0, double 0.0, double 0.0,
  double 0.0, double 0.0, double 0.0, double 0.0, double 0.0,
  double 0.0, double 0.0, double 0.0, double 0.0, double 0.0,
  double 0.0, double 0.0, double 0.0, double 0.0, double 0.0,
  double 0.0, double 0.0, double 0.0
]
',
    CallerIR = '
define i1 @astar_demo_caller(i64 %start_id, i64 %target_id, double* %out) {
entry:
  %table_ptr = getelementptr [4 x %WeightedFact], [4 x %WeightedFact]* @astar_demo_graph, i64 0, i64 0
  %h_ptr = getelementptr [33 x double], [33 x double]* @astar_demo_heuristic, i64 0, i64 0
  %ok = call i1 @wam_astar_weighted_distance(
      i64 %start_id, i64 %target_id,
      %WeightedFact* %table_ptr, i64 4,
      double* %h_ptr,
      i64 32,
      double* %out)
  ret i1 %ok
}
',
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, TableCode),
          write(Out, '\n'), write(Out, HeuristicIR),
          write(Out, '\n'), write(Out, CallerIR),
          write(Out, '\n') ),
        close(Out)),
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit == 0
    -> format('  PASS: llvm-as accepted astar caller module~n')
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
    catch(test_astar_module_validates, E,
        format('  ERROR in llvm-as test: ~w~n', [E])).

:- initialization(test_all, main).
