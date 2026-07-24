:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% NR-guarded rules in an assoc (group-by) program -- the skip-a-header idiom
% `NR > 1 { c[$1]++ } END { for (k in c) print k, c[k] }`. The assoc rule chain's
% pattern guard reads %current_nr, but the assoc-END drivers did not emit the
% per-record counter (only the scalar/print chain and the `END { print arr[k] }`
% driver did), so %current_nr was undefined and clang failed -- a miscompile.
%
% Fix: the plain and multi for-in-print and the accumulate assoc-END drivers now
% emit the conditional record counter (via plawk_assoc_record_counter) when a rule
% references NR -- byte-identical when it does not. The decode-into-struct driver,
% which does not emit the counter, declines cleanly on NR instead of miscompiling.
% for-in order is hash-dependent, so outputs are compared as sorted sets.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_nr_assoc_guard).

% Skip the header record, group-by the rest (single for-in END).
test(nr_guard_group_by, [condition(clang_available)]) :-
    gdir(Dir),
    build_run_sorted(Dir, 'g1',
        "NR > 1 { c[$1]++ } END { for (k in c) print k, c[k] }\n",
        "hdr\na\nb\na\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["a 2", "b 1"]),
    !.

% NR guard combined with a multi-table for-in END dump.
test(nr_guard_multi_forin, [condition(clang_available)]) :-
    gdir(Dir),
    build_run_sorted(Dir, 'g2',
        "NR > 1 { c[$1]++; d[$2]++ } END { for (a in c) print a; for (b in d) print b }\n",
        "h h\na x\nb x\na y\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["a", "b", "x", "y"]),
    !.

% An NR range guard (`NR==2, NR==3`) selecting a window of records.
test(nr_range_guard, [condition(clang_available)]) :-
    gdir(Dir),
    build_run_sorted(Dir, 'g3',
        "NR==2, NR==3 { c[$1]++ } END { for (k in c) print k }\n",
        "a\nb\nc\nd\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["b", "c"]),
    !.

% NR guard with an accumulate-then-print END (a different assoc-END driver).
test(nr_guard_accumulate, [condition(clang_available)]) :-
    gdir(Dir),
    build_run_sorted(Dir, 'g4',
        "NR > 1 { c[$1] += $2 } END { for (k in c) s += c[k]; print s }\n",
        "hdr 0\na 5\nb 3\na 2\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["10"]),
    !.

:- end_tests(plawk_nr_assoc_guard).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

gdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_nr_assoc_guard', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run_sorted(Dir, Name, Src, Input, SortedLines, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, ['-'],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)),
    split_string(Out, "\n", "", Parts0),
    exclude(==(""), Parts0, Parts),
    msort(Parts, SortedLines).
