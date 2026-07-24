:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Field-vs-string-literal ORDERING guards `$N < "str"` / `<=` / `>` / `>=`.
% awk compares a field against a string CONSTANT lexically -- the constant forces
% a string comparison, never numeric -- so `$1 < "10"` is byte-wise (`"9" > "10"`)
% while `$1 < 10` stays the numeric field path. The guard lowers to a memcmp of
% the field slice against the interned literal (runtime @wam_atom_field_str_cmp_value,
% op codes shared with the i64 field compare), so it composes with `!`/`&&`/`||`,
% `if` conditions, and for-in-END group-by. `==` / `!=` remain the field_eq /
% field_str_ne rules.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_field_str_order).

% --- parsing ----------------------------------------------------------------

test(field_lt_parses) :-
    plawk_parse_string("$1 < \"m\" { print $1 }\n",
        program([], [rule(field_str_cmp(1, lt, "m"), [print([field(1)])])], [])),
    !.

test(field_ge_parses) :-
    plawk_parse_string("$2 >= \"k\" { print $2 }\n",
        program([], [rule(field_str_cmp(2, ge, "k"), [print([field(2)])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% `<` selects values that sort lexically before the literal (per-record order).
test(field_lt_filters, [condition(clang_available)]) :-
    odir(Dir),
    build_run(Dir, 'lt', "$1 < \"m\" { print $1 }\n",
        "apple\nzebra\nmango\nbanana\n", Out, St),
    assertion(St == 0), assertion(Out == "apple\nbanana\n"), !.

% `>=` is the complement.
test(field_ge_filters, [condition(clang_available)]) :-
    odir(Dir),
    build_run(Dir, 'ge', "$1 >= \"m\" { print $1 }\n",
        "apple\nzebra\nmango\nbanana\n", Out, St),
    assertion(St == 0), assertion(Out == "zebra\nmango\n"), !.

% `>` excludes an equal value; `<=` includes it.
test(field_gt_excludes_equal, [condition(clang_available)]) :-
    odir(Dir),
    build_run(Dir, 'gt', "$1 > \"mango\" { print $1 }\n",
        "apple\nmango\nzebra\n", Out, St),
    assertion(St == 0), assertion(Out == "zebra\n"), !.

test(field_le_includes_equal, [condition(clang_available)]) :-
    odir(Dir),
    build_run(Dir, 'le', "$1 <= \"mango\" { print $1 }\n",
        "apple\nmango\nzebra\n", Out, St),
    assertion(St == 0), assertion(Out == "apple\nmango\n"), !.

% Comparison against a string constant is LEXICAL even for numeric-looking data:
% "9" > "10" > "100"... so `$1 < "10"` matches nothing here.
test(field_lexical_not_numeric, [condition(clang_available)]) :-
    odir(Dir),
    build_run(Dir, 'lex', "$1 < \"10\" { print $1 }\n",
        "9\n10\n2\n100\n", Out, St),
    assertion(St == 0), assertion(Out == ""), !.

% A numeric RHS still takes the numeric field path (`$1 < 10`): 9 and 2 match.
test(field_numeric_rhs_unchanged, [condition(clang_available)]) :-
    odir(Dir),
    build_run(Dir, 'num', "$1 < 10 { print $1 }\n",
        "9\n10\n2\n100\n", Out, St),
    assertion(St == 0), assertion(Out == "9\n2\n"), !.

% Works as an `if` condition (shared condition_pattern grammar).
test(field_order_if_cond, [condition(clang_available)]) :-
    odir(Dir),
    build_run(Dir, 'if', "{ if ($1 < \"m\") n++ } END { print n }\n",
        "apple\nzebra\nmango\n", Out, St),
    assertion(St == 0), assertion(Out == "1\n"), !.

% Composes with `&&` (a lexical range).
test(field_order_combinator, [condition(clang_available)]) :-
    odir(Dir),
    build_run(Dir, 'comb', "$1 >= \"a\" && $1 < \"n\" { print $1 }\n",
        "apple\nzebra\nmango\n", Out, St),
    assertion(St == 0), assertion(Out == "apple\nmango\n"), !.

% Drives a for-in-END group-by (multi-rule guard path; sorted set).
test(field_order_group_by, [condition(clang_available)]) :-
    odir(Dir),
    build_run_sorted(Dir, 'grp',
        "$1 >= \"m\" { c[$1]++ } END { for (k in c) print k, c[k] }\n",
        "apple\nzebra\nmango\nzebra\n", Lines, St),
    assertion(St == 0), assertion(Lines == ["mango 1", "zebra 2"]), !.

:- end_tests(plawk_field_str_order).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_field_str_order', Dir),
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
