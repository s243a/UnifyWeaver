:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk string-valued scalars: assignment concat `x = $1 $2` and string-literal
% assignment `x = "text"`. A string scalar's slot holds an interned atom id (an
% i64, so it threads through the same SSA/phi machinery as numeric scalars); the
% RHS is built into a buffer and interned at assignment (the mirror of the print
% concat), and a read / `print` resolves the id back to text (id 0 = unset =
% empty). Numeric and string scalars coexist in one program. v1 concat parts are
% fields and string literals; a scalar-var operand (`x = x $1` accumulation) is
% a follow-on.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_strscalar).

% --- parsing ----------------------------------------------------------------

test(concat_assign_parses) :-
    plawk_parse_string("{ x = $1 $2 }\n",
        program([], [rule(always, [set(var(x), concat([field(1), field(2)]))])], [])),
    !.

test(literal_assign_parses) :-
    plawk_parse_string("{ x = \"hi\" }\n",
        program([], [rule(always, [set(var(x), string("hi"))])], [])),
    !.

test(arithmetic_assign_unchanged) :-
    plawk_parse_string("{ x = $1 + $2 }\n",
        program([], [rule(always, [set(var(x), add_i64(field(1), field(2)))])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% `x = $1 $2` concatenates adjacent fields (no separator); END prints the last.
test(concat_to_end, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ce', "{ x = $1 $2 }\nEND { print x }\n", "ab cd\nef gh\n", Out, St),
    assertion(St == 0), assertion(Out == "efgh\n"), !.

% a string literal in the middle is kept verbatim.
test(concat_with_literal, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'cl', "{ x = $1 \"-\" $2 }\nEND { print x }\n", "foo bar\n", Out, St),
    assertion(St == 0), assertion(Out == "foo-bar\n"), !.

% three-field concat.
test(concat_three, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'c3', "{ x = $1 $2 $3 }\nEND { print x }\n", "a b c\n", Out, St),
    assertion(St == 0), assertion(Out == "abc\n"), !.

% a bare string literal assignment.
test(literal_assign, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'li', "{ x = \"hello\" }\nEND { print x }\n", "a\n", Out, St),
    assertion(St == 0), assertion(Out == "hello\n"), !.

% a literal prefix + field.
test(literal_prefix, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'lp', "{ x = \"id:\" $1 }\nEND { print x }\n", "42\n", Out, St),
    assertion(St == 0), assertion(Out == "id:42\n"), !.

% printing the string scalar in the per-record body.
test(concat_body_print, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'bp', "{ x = $1 $2; print x }\n", "ab cd\nef gh\n", Out, St),
    assertion(St == 0), assertion(Out == "abcd\nefgh\n"), !.

% a numeric scalar and a string scalar coexist and print together.
test(mixed_numeric_and_string, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'mx',
        "{ n = n + 1; label = $1 $2 }\nEND { print \"count:\", n, \"last:\", label }\n",
        "a b\nc d\n", Out, St),
    assertion(St == 0), assertion(Out == "count: 2 last: cd\n"), !.

% a string scalar printed with a leading label in the same print.
test(string_with_label, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'wl', "{ name = $1 $2 }\nEND { print \"full:\", name }\n", "John Doe\n", Out, St),
    assertion(St == 0), assertion(Out == "full: JohnDoe\n"), !.

% a plain numeric accumulator is unaffected by the string-scalar machinery.
test(numeric_unaffected, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nu', "{ s = s + $1 }\nEND { print s }\n", "3\n7\n2\n", Out, St),
    assertion(St == 0), assertion(Out == "12\n"), !.

% --- string accumulation (x = x $1) -----------------------------------------

% `acc = acc $1` accumulates across records; an unset string scalar starts empty.
test(accumulate, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ac', "{ acc = acc $1 }\nEND { print acc }\n", "a\nb\nc\n", Out, St),
    assertion(St == 0), assertion(Out == "abc\n"), !.

% accumulate with a trailing literal separator (CSV build).
test(accumulate_with_separator, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'as', "{ s = s $1 \",\" }\nEND { print s }\n", "x\ny\nz\n", Out, St),
    assertion(St == 0), assertion(Out == "x,y,z,\n"), !.

% accumulate with literal delimiters around each field.
test(accumulate_wrapped, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'aw', "{ log = log \"[\" $1 \"]\" }\nEND { print log }\n", "a\nb\n", Out, St),
    assertion(St == 0), assertion(Out == "[a][b]\n"), !.

:- end_tests(plawk_strscalar).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_strscalar', Dir),
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
