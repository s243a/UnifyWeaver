:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Field-vs-string-literal inequality guard `$N != "str"` -- the common
% skip-a-value idiom (`$1 != "header"`). Awk string `!=`; the parser rewrites it
% directly to not_pat(field_eq(N, "str")), so the existing `$N == "str"`
% (field_eq) codegen and its combinator / if-condition / for-in handling apply
% with no new lowering. A numeric RHS keeps the field_i64 path (`$N != 5`); only
% a quoted-string RHS takes this rule. String ORDERING as a field pattern
% (`$N < "str"`) needs a real lexical compare and remains a follow-on.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_field_str_ne).

% --- parsing ----------------------------------------------------------------

% `$1 != "str"` parses to the negation of the `==` guard.
test(field_ne_parses) :-
    plawk_parse_string("$1 != \"skip\" { print $1 }\n",
        program([], [rule(not_pat(field_eq(1, "skip")), [print([field(1)])])], [])),
    !.

% In an `if` condition, the same leaf pattern is reused (via condition_pattern).
test(field_ne_if_parses) :-
    plawk_parse_string("{ if ($2 != \"x\") print $1 }\n",
        program([], [rule(always,
            [if(not_pat(field_eq(2, "x")), [print([field(1)])], [])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% Per-record filter: `$1 != "skip"` passes every non-skip record through.
test(field_ne_filters, [condition(clang_available)]) :-
    ndir(Dir),
    build_run(Dir, 'flt', "$1 != \"skip\" { print $1 }\n",
        "a\nskip\nb\nskip\nc\n", Out, St),
    assertion(St == 0), assertion(Out == "a\nb\nc\n"), !.

% As a scalar-counter guard (the count of non-matching records).
test(field_ne_counts, [condition(clang_available)]) :-
    ndir(Dir),
    build_run(Dir, 'cnt', "$1 != \"x\" { n++ } END { print n }\n",
        "a\nx\nb\nx\n", Out, St),
    assertion(St == 0), assertion(Out == "2\n"), !.

% The same leaf pattern in an `if` condition.
test(field_ne_if_cond, [condition(clang_available)]) :-
    ndir(Dir),
    build_run(Dir, 'if', "{ if ($1 != \"hdr\") n++ } END { print n }\n",
        "hdr\na\nb\n", Out, St),
    assertion(St == 0), assertion(Out == "2\n"), !.

% Composes with `&&` and an `==` conjunct.
test(field_ne_combinator, [condition(clang_available)]) :-
    ndir(Dir),
    build_run(Dir, 'comb',
        "$1 != \"x\" && $2 == \"y\" { n++ } END { print n }\n",
        "a y\nx y\nb z\na y\n", Out, St),
    assertion(St == 0), assertion(Out == "2\n"), !.

% A numeric RHS still takes the field_i64 (`$N != INT`) path, unchanged.
test(field_ne_numeric_unchanged, [condition(clang_available)]) :-
    ndir(Dir),
    build_run(Dir, 'num', "$1 != 5 { n++ } END { print n }\n",
        "5\n6\n7\n5\n", Out, St),
    assertion(St == 0), assertion(Out == "2\n"), !.

% Group-by guarded by `!=`: skip records then count the rest by key (for-in
% order is hash-dependent, compared as a sorted set).
test(field_ne_group_by, [condition(clang_available)]) :-
    ndir(Dir),
    build_run_sorted(Dir, 'grp',
        "$1 != \"skip\" { c[$1]++ } END { for (k in c) print k, c[k] }\n",
        "a\nskip\na\nb\nskip\n", Lines, St),
    assertion(St == 0), assertion(Lines == ["a 2", "b 1"]), !.

:- end_tests(plawk_field_str_ne).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ndir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_field_str_ne', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Src, Bin) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    write_prog(Dir, Name, Src, Bin),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).

build_run_sorted(Dir, Name, Src, Input, SortedLines, RunStatus) :-
    write_prog(Dir, Name, Src, Bin),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)),
    split_string(Out, "\n", "", Parts0),
    exclude(==(""), Parts0, Parts),
    msort(Parts, SortedLines).
