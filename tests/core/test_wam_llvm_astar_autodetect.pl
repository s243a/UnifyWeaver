:- encoding(utf8).
% test_wam_llvm_astar_autodetect.pl
% Verifies auto-detect recognizes astar_shortest_path4 clause shape.
%
% Tests:
%   1. my_astar_path/4 matched as astar_shortest_path4 with weight_pred.
%   2. wsp3 auto-detect still works (regression).

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     llvm_foreign_kernel_spec/3,
     clear_llvm_foreign_kernel_specs/0]).

% A* clause shape (arity 4 with visited list).
:- dynamic my_astar_path/4.
my_astar_path(X, Y, W, _Vis) :- my_aweight(X, Y, W).
my_astar_path(X, Y, Cost, Vis) :-
    my_aweight(X, Z, W),
    my_astar_path(Z, Y, RC, [Z|Vis]),
    Cost is W + RC.

:- dynamic my_aweight/3.
my_aweight(a, b, 3.0).
my_aweight(b, c, 4.0).

% wsp3 clause shape (arity 3, no visited list).
:- dynamic my_wsp/3.
my_wsp(X, Y, W) :- my_wedge(X, Y, W).
my_wsp(X, Y, Cost) :-
    my_wedge(X, Z, W),
    my_wsp(Z, Y, RC),
    Cost is W + RC.

:- dynamic my_wedge/3.
my_wedge(x, y, 1.0).

test_astar4_autodetect :-
    format('--- astar4 auto-detect ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project(
        [user:my_astar_path/4],
        [ module_name('astar_ad'),
          target_triple('x86_64-pc-linux-gnu'),
          target_datalayout(''),
          foreign_lowering(true)
        ],
        LLPath),
    ( llvm_foreign_kernel_spec(my_astar_path/4, astar_shortest_path4, Config)
    -> format('  PASS: auto-detect matched astar4, config=~w~n', [Config])
    ;  format('  FAIL: auto-detect did not match astar4~n'),
       throw(astar4_autodetect_failed)
    ),
    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_wsp3_autodetect :-
    format('--- wsp3 auto-detect (regression) ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project(
        [user:my_wsp/3],
        [ module_name('wsp_ad'),
          target_triple('x86_64-pc-linux-gnu'),
          target_datalayout(''),
          foreign_lowering(true)
        ],
        LLPath),
    ( llvm_foreign_kernel_spec(my_wsp/3, weighted_shortest_path3, _)
    -> format('  PASS: auto-detect matched wsp3~n')
    ;  format('  FAIL: auto-detect did not match wsp3~n'),
       throw(wsp3_autodetect_failed)
    ),
    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_all :-
    catch(test_astar4_autodetect, E1,
        format('  ERROR: ~w~n', [E1])),
    catch(test_wsp3_autodetect, E2,
        format('  ERROR: ~w~n', [E2])).

:- initialization(test_all, main).
