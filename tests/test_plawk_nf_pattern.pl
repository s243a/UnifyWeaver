:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Expression pattern `NF OP int`: a bare rule pattern that fires on the current
% record's field count, e.g. `NF > 3 { … }`, `NF == 0 { … }`. It parses to
% special_cmp('NF', Op, Value) and lowers through the pattern-guard path by
% counting fields with @wam_atom_field_count_value and comparing to the literal,
% so it composes with `!`, `&&`, `||`, and the other base patterns. (A reversed
% `int OP NF` and a `length OP int` pattern are follow-ons.)

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_nf_pattern).

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

:- end_tests(plawk_nf_pattern).

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
