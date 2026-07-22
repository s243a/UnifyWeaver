:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Expression patterns `NF OP int` and `length OP int`: bare rule patterns that
% fire on the current record's field count (`NF > 3 { … }`) or byte length
% (`length > 80 { … }`, `length($0) <= 40 { … }`). Both parse to
% special_cmp(Special, Op, Value) and lower through the pattern-guard path (NF
% via @wam_atom_field_count_value, length via strlen of $0), so they compose
% with `!`, `&&`, `||`, and the other base patterns. Both operand orders parse:
% `SPECIAL OP int` and `int OP SPECIAL` (the reversed form swaps the operator).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_expr_pattern).

% --- parsing ----------------------------------------------------------------

test(nf_gt_pattern_parses) :-
    plawk_parse_string("NF > 3 { print $0 }\n",
        program([], [rule(special_cmp('NF', gt, 3), [print([field(0)])])], [])),
    !.

test(nf_eq_pattern_parses) :-
    plawk_parse_string("NF == 0 { print \"blank\" }\n",
        program([], [rule(special_cmp('NF', eq, 0), [print([string("blank")])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% NF > 2 selects records with more than two fields.
test(nf_gt_selects, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'ng', "NF > 2 { print $0 }\n", "a b c\nd e\nf g h i\n", Out),
    assertion(Out == "a b c\nf g h i\n"), !.

% NF == 0 matches a blank record.
test(nf_eq_zero_blank, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'nz', "NF == 0 { print \"blank\" }\n", "x y\n\nz\n", Out),
    assertion(Out == "blank\n"), !.

% Two NF-pattern rules in one program (multi-rule guard, unique count temps).
test(nf_multi_rule, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'nm',
        "NF == 0 { print \"blank\" }\nNF >= 2 { print \"multi:\", $1 }\n",
        "x y\n\nz\n", Out),
    assertion(Out == "multi: x\nblank\n"), !.

% Negated NF pattern via the combinator path.
test(nf_negated, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'nn', "!(NF > 2) { print $0 }\n", "a b c\nd e\n", Out),
    assertion(Out == "d e\n"), !.

% NF pattern combines with a field-equality guard.
test(nf_combined_with_field_eq, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'nc', "NF >= 2 && $1 == \"x\" { print $2 }\n", "x y\nx\nz w\n", Out),
    assertion(Out == "y\n"), !.

% length OP int parses to special_cmp(length, …).
test(length_gt_pattern_parses) :-
    plawk_parse_string("length > 3 { print $0 }\n",
        program([], [rule(special_cmp(length, gt, 3), [print([field(0)])])], [])),
    !.

% length($0) parses to the same length special.
test(length_paren_pattern_parses) :-
    plawk_parse_string("length($0) <= 2 { print $0 }\n",
        program([], [rule(special_cmp(length, le, 2), [print([field(0)])])], [])),
    !.

% Reversed `int OP NF` swaps the operator.
test(reversed_nf_pattern_parses) :-
    plawk_parse_string("3 < NF { print $1 }\n",
        program([], [rule(special_cmp('NF', gt, 3), [print([field(1)])])], [])),
    !.

% length > 3 selects records longer than three bytes.
test(length_gt_selects, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'lg', "length > 3 { print $0 }\n",
        "ab\nabcdef\nxy\nhello world\n", Out),
    assertion(Out == "abcdef\nhello world\n"), !.

% length($0) <= 2 selects short records.
test(length_le_selects, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'll', "length($0) <= 2 { print $0 }\n",
        "ab\nabcdef\nxy\n", Out),
    assertion(Out == "ab\nxy\n"), !.

% Reversed `3 < NF` matches records with more than three fields.
test(reversed_nf_selects, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rn', "3 < NF { print $1 }\n", "a b c d\ne f\n", Out),
    assertion(Out == "a\n"), !.

% Reversed `5 >= length` matches records of at most five bytes.
test(reversed_length_selects, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rl', "5 >= length { print $0 }\n",
        "ab\nabcdef\nxy\n", Out),
    assertion(Out == "ab\nxy\n"), !.

% Field-vs-field pattern `$I OP $J` parses to field_cmp2.
test(field_field_pattern_parses) :-
    plawk_parse_string("$1 > $2 { print $0 }\n",
        program([], [rule(field_cmp2(1, gt, 2), [print([field(0)])])], [])),
    !.

% `$1 > $2` compares the two fields numerically.
test(field_field_numeric, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'ff', "$1 > $2 { print $0 }\n", "5 3\n2 9\n10 10\n", Out),
    assertion(Out == "5 3\n"), !.

% `$1 == $2` compares by strnum (lexical for non-numeric, numeric for numbers).
test(field_field_equality, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'fe', "$1 == $2 { print \"eq\" }\n", "abc abc\nx y\n10 10\n", Out),
    assertion(Out == "eq\neq\n"), !.

% A field beyond NF compares as the empty string (awk semantics).
test(field_field_missing, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'fm', "$1 < $3 { print $1 }\n", "1 x 2\n5 y\n", Out),
    assertion(Out == "1\n"), !.

% Composes with a && combinator.
test(field_field_combined, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'fc2', "$1 > $2 && $2 > 0 { print $0 }\n", "5 3\n5 -1\n1 2\n", Out),
    assertion(Out == "5 3\n"), !.

:- end_tests(plawk_expr_pattern).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_nf_pattern', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out) :-
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
    process_wait(RPid, exit(0)).
