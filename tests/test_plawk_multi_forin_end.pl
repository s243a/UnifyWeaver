:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Multiple for-in loops in an END block -- the "dump several tables" idiom:
%   { c[$1]++; d[$2]++ } END { for (a in c) print a, c[a]; for (b in d) print b, d[b] }
% Previously the END block parsed only a single statement. Now a general
% multi-statement END clause parses a sequence of end_actions (separated by `;`
% or newline), and a new whole-program driver chains the for-in loops: each loop
% walks its own table with index-suffixed labels/SSA (loop 0 keeps the historical
% unsuffixed names, so single-loop IR is byte-identical), branching into the next,
% and the last frees every table and returns.
%
% v1 scope: each END statement is a for-in whose body is a single print of the
% loop key and/or `arr[k]` lookups (the common table-dump). A for-in print with a
% string literal, or a multi plain-print END, declines cleanly (follow-ons).
% for-in order is hash-dependent, so outputs are compared as sorted sets.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_multi_forin_end).

% --- parsing ----------------------------------------------------------------

test(two_forin_end_parses) :-
    plawk_parse_string(
        "{ c[$1]++; d[$2]++ } END { for (a in c) print a; for (b in d) print b }\n",
        program([],
            [rule(always, [inc_assoc(var(c), field(1)), inc_assoc(var(d), field(2))])],
            [end([for_in(var(a), var(c), [print([var(a)])]),
                  for_in(var(b), var(d), [print([var(b)])])])])),
    !.

% --- runtime ----------------------------------------------------------------

% Two tables, each dumped key + count.
test(two_tables, [condition(clang_available)]) :-
    mdir(Dir),
    build_run_sorted(Dir, 'two',
        "{ c[$1]++; d[$2]++ } END { for (a in c) print a, c[a]; for (b in d) print b, d[b] }\n",
        "x p\ny p\nx q\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["p 2", "q 1", "x 2", "y 1"]),
    !.

% Three tables, key-only dumps.
test(three_tables, [condition(clang_available)]) :-
    mdir(Dir),
    build_run_sorted(Dir, 'three',
        "{ a[$1]++; b[$2]++; e[$3]++ } END { for (k in a) print k; for (k in b) print k; for (k in e) print k }\n",
        "p q r\np q s\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["p", "q", "r", "s"]),
    !.

% Two loops over the SAME table (both resolve to one table index; freed once).
test(same_table_twice, [condition(clang_available)]) :-
    mdir(Dir),
    build_run_sorted(Dir, 'same',
        "{ c[$1]++ } END { for (a in c) print a, c[a]; for (b in c) print b }\n",
        "m\nn\nm\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["m", "m 2", "n", "n 1"]),
    !.

% A summed table dumped twice: key+value, then value-only.
test(value_sum_dump, [condition(clang_available)]) :-
    mdir(Dir),
    build_run_sorted(Dir, 'val',
        "{ tot[$1] += $2 } END { for (k in tot) print k, tot[k]; for (k in tot) print tot[k] }\n",
        "x 5\ny 3\nx 2\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["3", "7", "x 7", "y 3"]),
    !.

% A for-in print with a string literal is a clean not-yet (keeps END string-global
% naming simple): the program declines rather than mis-lowering.
test(string_literal_in_forin_declines, [condition(clang_available)]) :-
    mdir(Dir),
    build_status(Dir, 'strlit',
        "{ c[$1]++; d[$2]++ } END { for (a in c) print \"c\", a; for (b in d) print \"d\", b }\n",
        St),
    assertion(St \== 0),
    !.

% A multi plain-print END (no for-in) is a separate follow-on; it now PARSES but
% no driver lowers it, so it declines cleanly rather than compiling.
test(multi_plain_print_declines, [condition(clang_available)]) :-
    mdir(Dir),
    build_status(Dir, 'pp',
        "{ n++ } END { print \"a\"; print \"b\" }\n", St),
    assertion(St \== 0),
    !.

:- end_tests(plawk_multi_forin_end).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

mdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_multi_forin_end', Dir),
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

build_run_sorted(Dir, Name, Src, Input, SortedLines, RunStatus) :-
    write_prog(Dir, Name, Src, Bin),
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
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
