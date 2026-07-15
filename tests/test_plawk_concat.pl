:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk string concatenation by juxtaposition (awk). `print $1 $2`, `print "x"
% $1`, `print $1 "-" $2` output the operands ADJACENT, with no field separator
% (unlike the comma list `print $1, $2`). Two or more operands separated by
% whitespace parse to concat([...]); arithmetic binds tighter than
% concatenation. Print-only (assignment concat would need string-valued
% scalars, a separate feature).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_concat).

% --- parsing / precedence ---------------------------------------------------

test(concat_two_fields_parses) :-
    plawk_parse_string("{ print $1 $2 }\n",
        program([], [rule(always, [print([concat([field(1), field(2)])])])], [])),
    !.

test(concat_string_and_field_parses) :-
    plawk_parse_string("{ print \"x\" $1 }\n",
        program([], [rule(always, [print([concat([string("x"), field(1)])])])], [])),
    !.

% A single operand keeps its plain form (no concat wrapper).
test(single_field_unchanged) :-
    plawk_parse_string("{ print $1 }\n",
        program([], [rule(always, [print([field(1)])])], [])),
    !.

% A comma still splits print args (not concatenation).
test(comma_splits_not_concat) :-
    plawk_parse_string("{ print $1, $2 }\n",
        program([], [rule(always, [print([field(1), field(2)])])], [])),
    !.

% Arithmetic binds tighter than concatenation: `$1 + $2` stays arithmetic;
% `$1 $2 + $3` is concat($1, $2 + $3).
test(arithmetic_tighter_than_concat) :-
    plawk_parse_string("{ print $1 + $2 }\n",
        program([], [rule(always, [print([add_i64(field(1), field(2))])])], [])),
    plawk_parse_string("{ print $1 $2 + $3 }\n",
        program([], [rule(always,
            [print([concat([field(1), add_i64(field(2), field(3))])])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

test(concat_two_fields, [condition(clang_available)]) :-
    cdir(Dir),
    build_run(Dir, 'c2', "{ print $1 $2 }\n", "a b c\n", Out, St),
    assertion(St == 0),
    assertion(Out == "ab\n"),
    !.

test(concat_with_string_literal, [condition(clang_available)]) :-
    cdir(Dir),
    build_run(Dir, 'cs', "{ print $1 \"-\" $2 }\n", "a b c\n", Out, St),
    assertion(St == 0),
    assertion(Out == "a-b\n"),
    !.

test(concat_prefix_string, [condition(clang_available)]) :-
    cdir(Dir),
    build_run(Dir, 'cp', "{ print \"row: \" $1 }\n", "hello world\n", Out, St),
    assertion(St == 0),
    assertion(Out == "row: hello\n"),
    !.

% Concatenation composes with the comma list: `print $1 $2, $3`.
test(concat_then_comma, [condition(clang_available)]) :-
    cdir(Dir),
    build_run(Dir, 'cc', "{ print $1 $2, $3 }\n", "a b c\n", Out, St),
    assertion(St == 0),
    assertion(Out == "ab c\n"),
    !.

% Concatenation with a scalar variable.
test(concat_with_scalar, [condition(clang_available)]) :-
    cdir(Dir),
    build_run(Dir, 'cv', "{ n++; print \"line\" n }\n", "a\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "line1\nline2\n"),
    !.

% Concatenation with NR (record counter) inside the concat.
test(concat_with_nr, [condition(clang_available)]) :-
    cdir(Dir),
    build_run(Dir, 'cnr', "{ print NR \": \" $1 }\n", "x\ny\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1: x\n2: y\n"),
    !.

% Concatenation in an END block (`print "total: " sum " done"`).
test(concat_in_end, [condition(clang_available)]) :-
    cdir(Dir),
    Src = "{ s += $1 }\nEND { print \"total: \" s \" done\" }\n",
    build_run(Dir, 'ce', Src, "10\n20\n", Out, St),
    assertion(St == 0),
    assertion(Out == "total: 30 done\n"),
    !.

:- end_tests(plawk_concat).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

cdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_concat', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
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
