:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Multiple plain prints in an END block (`END { print a; print b }`), scalar
% chain. The multi-statement END parser change made these parse; now a driver
% lowers them by emitting each print independently (fields + ORS newline) and
% per-print renaming its `end_`-prefixed SSA/globals so they stay unique -- print
% 0 is byte-identical to the single-print path. Each END statement must be a
% plain `print`; a for-in or a mixed for-in/print END is a follow-on (declines).
% An assoc-table program with a multi-print END also declines (scalar chain).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_multi_print_end).

% --- parsing ----------------------------------------------------------------

test(two_print_end_parses) :-
    plawk_parse_string("{ n++ } END { print \"a\"; print \"b\" }\n",
        program([], [rule(always, [inc(var(n))])],
            [end([print([string("a")]), print([string("b")])])])),
    !.

% --- runtime ----------------------------------------------------------------

% Two scalar prints: a sum then a count, one per line.
test(two_scalar_prints, [condition(clang_available)]) :-
    edir(Dir),
    build_run(Dir, 'two', "{ s += $1; n++ } END { print s; print n }\n",
        "10\n20\n30\n", Out, St),
    assertion(St == 0), assertion(Out == "60\n3\n"), !.

% Two string-literal prints.
test(two_string_prints, [condition(clang_available)]) :-
    edir(Dir),
    build_run(Dir, 'str', "{ n++ } END { print \"a\"; print \"b\" }\n",
        "x\ny\n", Out, St),
    assertion(St == 0), assertion(Out == "a\nb\n"), !.

% A scalar print then a string print (mixed kinds).
test(scalar_then_string, [condition(clang_available)]) :-
    edir(Dir),
    build_run(Dir, 'mix', "{ s += $1; n++ } END { print n; print \"done\" }\n",
        "5\n5\n", Out, St),
    assertion(St == 0), assertion(Out == "2\ndone\n"), !.

% Three prints, two of them a concat (`print "label:", value`).
test(three_prints_with_concat, [condition(clang_available)]) :-
    edir(Dir),
    build_run(Dir, 'three',
        "{ s += $1; n++ } END { print \"sum:\", s; print \"count:\", n; print s + n }\n",
        "4\n6\n", Out, St),
    assertion(St == 0), assertion(Out == "sum: 10\ncount: 2\n12\n"), !.

% NR is available in a multi-print END (the record counter is still emitted).
test(nr_in_multi_print, [condition(clang_available)]) :-
    edir(Dir),
    build_run(Dir, 'nr', "{ s += $1 } END { print NR; print s }\n",
        "3\n4\n5\n", Out, St),
    assertion(St == 0), assertion(Out == "3\n12\n"), !.

% A single-print END is unchanged (still lowers via the single-print driver).
test(single_print_unchanged, [condition(clang_available)]) :-
    edir(Dir),
    build_run(Dir, 'single', "{ s += $1 } END { print s }\n",
        "1\n2\n", Out, St),
    assertion(St == 0), assertion(Out == "3\n"), !.

% An assoc-table program with a multi-print END is a clean not-yet (scalar chain
% only); it declines rather than mis-lowering.
test(assoc_multi_print_declines, [condition(clang_available)]) :-
    edir(Dir),
    build_status(Dir, 'assoc',
        "{ c[$1]++ } END { print \"a\"; print \"b\" }\n", St),
    assertion(St \== 0), !.

% A mixed for-in + plain print END declines (a follow-on).
test(mixed_forin_print_declines, [condition(clang_available)]) :-
    edir(Dir),
    build_status(Dir, 'mixfp',
        "{ c[$1]++ } END { for (k in c) print k; print \"done\" }\n", St),
    assertion(St \== 0), !.

:- end_tests(plawk_multi_print_end).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

edir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_multi_print_end', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Src, Bin) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin).

build_status(Dir, Name, Src, Status) :-
    write_prog(Dir, Name, Src, Bin),
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    write_prog(Dir, Name, Src, Bin),
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
