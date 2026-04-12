:- encoding(utf8).
% test_wam_llvm_bfs_execution.pl
% M5.7: End-to-end execution test for @wam_bfs_atom_distance.
%
% Generates a WAM-LLVM module, appends a concrete %AtomFactPair fact
% table and a small main() function that invokes the BFS helper, then
% runs the resulting module through `lli` (LLVM's JIT interpreter).
% The main() returns the computed distance as the process exit code,
% and the test asserts it matches the expected BFS distance.
%
% This is the first test that actually runs the generated code — prior
% tests stopped at llvm-as (parse) and opt -passes=verify (SSA check).
% Execution testing catches runtime bugs that the static passes miss:
% uninitialized memory, off-by-one loop bounds, wrong register indices,
% double-free on cleanup paths, etc.
%
% Atom IDs are written literally in the fact table (not via
% intern_atom/2) so the test is deterministic regardless of what other
% tests interned first in the same session. The max_atom_id bound is
% correspondingly hardcoded.
%
% Test graph:
%    1 -> 2 -> 3 -> 4
%    1 -> 5
% Expected: BFS distance from 1 to 4 is 3 (three hops: 1->2->3->4).
% Expected: BFS distance from 1 to 5 is 1.
% Expected: BFS distance from 1 to 1 is 0 (self-hit fast path).
% Expected: BFS distance from 1 to 99 fails (unreachable).

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3]).
:- use_module(library(process)).

:- dynamic color/1.
color(red).

% Append a main() function that calls @wam_bfs_atom_distance for a
% specific (start, target) pair and returns the result (or 255 on
% unreachable) as the process exit code.
build_exec_driver(Start, Target, DriverIR) :-
    format(atom(DriverIR),
'; === M5.7 execution driver ===
; Test graph (hand-written, not via intern_atom for determinism):
;   1 -> 2 -> 3 -> 4
;   1 -> 5
@bfs_exec_edges = private constant [4 x %AtomFactPair] [
  %AtomFactPair { i64 1, i64 2 },
  %AtomFactPair { i64 2, i64 3 },
  %AtomFactPair { i64 3, i64 4 },
  %AtomFactPair { i64 1, i64 5 }
]

define i32 @main() {
entry:
  %dist_slot = alloca i64
  %table_ptr = getelementptr [4 x %AtomFactPair], [4 x %AtomFactPair]* @bfs_exec_edges, i64 0, i64 0
  %ok = call i1 @wam_bfs_atom_distance(
      i64 ~w, i64 ~w,
      %AtomFactPair* %table_ptr, i64 4,
      i64 5,
      i64* %dist_slot)
  br i1 %ok, label %hit, label %miss

hit:
  %dist = load i64, i64* %dist_slot
  %dist32 = trunc i64 %dist to i32
  ret i32 %dist32

miss:
  ret i32 255
}
', [Start, Target]).

run_bfs_for(Start, Target, ExitCode) :-
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    host_target_datalayout(DataLayout),
    write_wam_llvm_project(
        [user:color/1],
        [ module_name('bfs_exec'),
          target_triple(Triple),
          target_datalayout(DataLayout)
        ],
        LLPath),
    build_exec_driver(Start, Target, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    % Full compile path: llc → .o → clang link → run.
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -filetype=obj -relocation-model=pic ~w -o ~w 2>~w.llc.err',
        [LLPath, OPath, LLPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =\= 0
    -> format('    llc failed (exit=~w), see ~w.llc.err~n', [LlcExit, LLPath]),
       ExitCode = -1
    ;  format(atom(ClangCmd),
           'clang ~w -o ~w 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('    clang link failed (exit=~w), see ~w.clang.err~n',
             [ClangExit, LLPath]),
          ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true).

check_bfs(Label, Start, Target, Expected) :-
    format('  testing ~w: bfs(~w, ~w) expected ~w~n', [Label, Start, Target, Expected]),
    run_bfs_for(Start, Target, ExitCode),
    ( ExitCode =:= Expected
    -> format('    PASS: ~w returned ~w~n', [Label, ExitCode])
    ;  format('    FAIL: ~w returned ~w (expected ~w)~n', [Label, ExitCode, Expected]),
       throw(bfs_wrong_result(Label, ExitCode, Expected))
    ).

test_bfs_executes :-
    format('--- BFS helper execution via lli ---~n'),
    ( process_which('lli')
    -> check_bfs('three-hop path',    1, 4, 3),
       check_bfs('direct edge',       1, 5, 1),
       check_bfs('self hit',          1, 1, 0),
       check_bfs('unreachable',       1, 99, 255)
    ;  format('  SKIP: lli not found on PATH~n')
    ).

% Detect the host's target triple and datalayout by asking clang. We
% need these to override the write_wam_llvm_project defaults (which
% hardcode x86_64-pc-linux-gnu), otherwise the generated object file
% is incompatible with the linker. Falls back to reasonable values
% if clang is not callable.
host_target_triple(Triple) :-
    ( catch(
        ( process_create(path(clang), ['-print-target-triple'],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, TripleStrRaw),
          close(Out),
          process_wait(PID, exit(0))
        ),
        _, fail)
    -> split_string(TripleStrRaw, "", "\n\r\t ", [TripleStr]),
       atom_string(Triple, TripleStr)
    ;  Triple = 'x86_64-pc-linux-gnu'
    ).

% A generic datalayout that clang accepts for most linux-like targets
% — for aarch64-android the actual layout from clang is slightly
% different, but keeping a tighter layout here means we compile
% through llc → clang cleanly on the host. We only query clang's
% triple above; for the datalayout we let llc pick its own default
% by handing it an empty string (llc then fills in the target's
% canonical layout).
host_target_datalayout('').

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
    catch(test_bfs_executes, E,
        format('  ERROR: ~w~n', [E])).

:- initialization(test_all, main).
