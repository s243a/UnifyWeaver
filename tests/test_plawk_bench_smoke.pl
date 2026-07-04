:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Keeps the benchmark harness from rotting: a tiny-N run must pass its
% own correctness gate (plawk output identical to the system awk on
% every text workload) and produce the report table. Timings at this
% size are meaningless and are not asserted.

:- use_module(library(plunit)).
:- use_module(library(process)).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_bench_smoke, [condition(clang_available)]).

test(bench_harness_runs_and_gates_correctness) :-
    process_create(path(sh), ['examples/plawk/bench/bench.sh'],
        [environment(['N'='2000']), stdout(pipe(S)), stderr(std),
         process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(sub_string(Out, _, _, _, "correctness gate: all outputs identical")),
    assertion(sub_string(Out, _, _, _, "W4 group-by")),
    !.

:- end_tests(plawk_bench_smoke).
