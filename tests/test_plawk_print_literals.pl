:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Constant print fields in the single-pass driver. awk prints a number as its
% text, and `print "literal"` is standard; both were rejected before. Two
% pieces close the gap: a print-specific grammar lowers a bare integer literal
% to the string of its digits (`print 1` == `print "1"` -- print-only, so
% `emit`/arithmetic keep a bare integer numeric), and a string-literal print
% field now emits (a missing plawk_print_expr_output_ir clause). field_expr is
% tried first, so `print 1 + 2` stays arithmetic.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_print_literals).

% A bare integer print field parses to a string of its digits (print-specific).
test(bare_integer_parses) :-
    plawk_parse_string("{ print 1 }\n",
        program([], [rule(always, [print([string("1")])])], [])),
    !.

% A bare integer print field does NOT change field_expr elsewhere: an emit of a
% bare integer stays the integer (int(1)), not a string.
test(emit_bare_integer_stays_int) :-
    plawk_parse_string("gen { emit 1 } as g\n",
        program_passes([], [gen_block(name(g), none, [emit(int(1))])], [])),
    !.

% Arithmetic is unaffected (field_expr tried before the literal fallback).
test(print_arithmetic_unchanged) :-
    plawk_parse_string("{ print 1 + 2 }\n",
        program([], [rule(always, [print([add_i64(int(1), int(2))])])], [])),
    !.

% `print 1` compiles and prints the constant once per record.
test(print_integer_literal_runs, [condition(clang_available)]) :-
    run_prog("{ print 1 }\n", "a\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1\n1\n"),
    !.

% `print "hi"` -- a string literal print field -- compiles and prints.
test(print_string_literal_runs, [condition(clang_available)]) :-
    run_prog("{ print \"hi\" }\n", "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "hi\n"),
    !.

% Literals mixed with fields, separated by the output separator.
test(print_mixed_literal_and_field_runs, [condition(clang_available)]) :-
    run_prog("{ print 1, $1, 42 }\n", "a\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1 a 42\n"),
    !.

% Arithmetic over literals still computes (not printed verbatim).
test(print_arithmetic_runs, [condition(clang_available)]) :-
    run_prog("{ print 1 + 2 }\n", "a\n", Out, St),
    assertion(St == 0),
    assertion(Out == "3\n"),
    !.

% BEGIN literals: `BEGIN { print ... }` routes through the same print grammar,
% so a bare integer and a string literal both compile and print once, before
% the records. (Locks in the BEGIN/END follow-on of the print-literal fix.)
test(begin_integer_literal_runs, [condition(clang_available)]) :-
    run_prog("BEGIN { print 1 }\n{ print $1 }\n", "a\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1\na\n"),
    !.

test(begin_string_and_mixed_literal_runs, [condition(clang_available)]) :-
    run_prog("BEGIN { print \"n\", 5 }\n{ print $1 }\n", "a\n", Out, St),
    assertion(St == 0),
    assertion(Out == "n 5\na\n"),
    !.

% END literals: `END { print ... }` prints after the records; a bare integer,
% a string, and a mix of literal + a computed scalar all print.
test(end_integer_literal_runs, [condition(clang_available)]) :-
    run_prog("{ n++ }\nEND { print 42 }\n", "a\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "42\n"),
    !.

test(end_string_literal_runs, [condition(clang_available)]) :-
    run_prog("{ n++ }\nEND { print \"done\" }\n", "a\n", Out, St),
    assertion(St == 0),
    assertion(Out == "done\n"),
    !.

:- end_tests(plawk_print_literals).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

pdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_print_literals', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build a program, run it over Input, capture stdout + exit status.
run_prog(Src, Input, Out, RunStatus) :-
    pdir(Dir),
    directory_file_path(Dir, 'p.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    directory_file_path(Dir, 'in.txt', InPath),
    setup_call_cleanup(open(InPath, write, IS, [encoding(utf8)]),
        write(IS, Input), close(IS)),
    directory_file_path(Dir, 'p_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [InPath],
        [stdout(pipe(RS)), stderr(std), process(RPid)]),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
