:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: `printf` in a BEGIN block -- `BEGIN { printf "%-8s %s\n", "name", "qty" }`,
% the column-header idiom.
%
% `printf` was not in the BEGIN action grammar, so `BEGIN { printf "hdr\n" }` was
% a PARSE ERROR. BEGIN runs before the record loop, so there is neither a current
% record nor any scalar slot: the arguments are literals (string / integer /
% float). Everything after the arguments -- argument kinds, the awk->C format
% rewrite, the format global, the `@printf` call -- is the shared
% plawk_printf_from_arg_pairs/4, the same consumer the record-context and END
% argument producers feed, so the format rewrite is identical in all three.
%
% Adding it exposed a PRE-EXISTING bug: the BEGIN emitter picked the first
% `print` with member/2 and emitted only that, so `BEGIN { print "a"; print "b" }`
% silently dropped the second line (and a BEGIN whose only output was a printf
% would have been dropped entirely). BEGIN now emits every output statement in
% source order, and a statement that cannot be lowered fails the driver rather
% than falling through to the stores-only path and quietly omitting output.
%
% gawk 5.2 is the oracle for every expectation here.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Two records, so BEGIN output is distinguishable from per-record output.
input("a 1\nb 2\n").

:- begin_tests(plawk_begin_printf).

% --- parsing ---------------------------------------------------------------

test(begin_printf_parses) :-
    plawk_parse_string("BEGIN { printf \"%d\\n\", 1 }\n{ print $1 }\n",
        program([begin([printf(string("%d\n"), [int(1)])])],
            [rule(always, [print([field(1)])])], [])),
    !.

% `printf` must be tried before `print` in the BEGIN grammar too, or `print`
% matches the keyword prefix and leaves a stray `f`.
test(begin_print_and_printf_both_parse) :-
    plawk_parse_string("BEGIN { print \"a\"; printf \"b\\n\" }\n{ print $1 }\n",
        program([begin([print([string("a")]), printf(string("b\n"), [])])],
            [rule(always, [print([field(1)])])], [])),
    !.

% A BEGIN assignment beside a printf still parses as an assignment.
test(begin_assignment_beside_printf_parses) :-
    plawk_parse_string("BEGIN { FS = \",\"; printf \"h\\n\" }\n{ print $1 }\n",
        program([begin([set(var('FS'), string(",")),
                        printf(string("h\n"), [])])],
            [rule(always, [print([field(1)])])], [])),
    !.

% --- output, against gawk --------------------------------------------------

test(begin_printf_no_args, [condition(clang_available)]) :-
    run("BEGIN { printf \"hdr\\n\" }\n{ print $1 }\n", "hdr\na\nb\n"),
    !.

test(begin_printf_string_args, [condition(clang_available)]) :-
    run("BEGIN { printf \"%s|%s\\n\", \"a\", \"b\" }\n{ print $1 }\n",
        "a|b\na\nb\n"),
    !.

test(begin_printf_integer_arg, [condition(clang_available)]) :-
    run("BEGIN { printf \"%d\\n\", 42 }\n{ print $1 }\n", "42\na\nb\n"),
    !.

test(begin_printf_float_arg, [condition(clang_available)]) :-
    run("BEGIN { printf \"%.2f\\n\", 3.5 }\n{ print $1 }\n", "3.50\na\nb\n"),
    !.

% The column-header idiom this feature exists for: flags and widths.
test(begin_printf_width_and_flags, [condition(clang_available)]) :-
    run("BEGIN { printf \"%-6s|%5s|\\n\", \"ab\", \"cd\" }\n{ print $1 }\n",
        "ab    |   cd|\na\nb\n"),
    !.

% printf appends no ORS, so the first record runs onto the BEGIN line.
test(begin_printf_appends_no_newline, [condition(clang_available)]) :-
    run("BEGIN { printf \"x\" }\n{ print $1 }\n", "xa\nb\n"),
    !.

test(begin_printf_percent_escape, [condition(clang_available)]) :-
    run("BEGIN { printf \"100%%\\n\" }\n{ print $1 }\n", "100%\na\nb\n"),
    !.

% A BEGIN printf coexists with a BEGIN assignment: FS still takes effect.
test(begin_printf_with_fs_assignment, [condition(clang_available)]) :-
    run("BEGIN { FS = \",\"; printf \"hdr\\n\" }\n{ print $1 }\n",
        "hdr\na 1\nb 2\n"),
    !.

% --- multiple BEGIN output statements (the pre-existing bug) ---------------

% Two prints: the second used to be silently dropped.
test(begin_two_prints_both_emitted, [condition(clang_available)]) :-
    run("BEGIN { print \"a\"; print \"b\" }\n{ print $1 }\n", "a\nb\na\nb\n"),
    !.

test(begin_print_then_printf, [condition(clang_available)]) :-
    run("BEGIN { print \"a\"; printf \"b\\n\" }\n{ print $1 }\n", "a\nb\na\nb\n"),
    !.

% The printf leaves the line open and the following print closes it.
test(begin_printf_then_print, [condition(clang_available)]) :-
    run("BEGIN { printf \"a \"; print \"b\" }\n{ print $1 }\n", "a b\na\nb\n"),
    !.

test(begin_three_statements, [condition(clang_available)]) :-
    run("BEGIN { print \"a\"; printf \"%d \", 1; print \"c\" }\n{ print $1 }\n",
        "a\n1 c\na\nb\n"),
    !.

% --- clean declines --------------------------------------------------------

% BEGIN has no current record, so a field argument cannot be lowered. gawk
% prints an empty line ($1 is empty in BEGIN); plawk declines (status 3) rather
% than slicing a record that does not exist.
test(begin_printf_field_arg_declines) :-
    build_status("BEGIN { printf \"%s\\n\", $1 }\n{ print $1 }\n", 3),
    !.

% Likewise a scalar variable: no slot exists before the record loop.
test(begin_printf_scalar_arg_declines) :-
    build_status("BEGIN { printf \"%d\\n\", n }\n{ n++ }\n", 3),
    !.

% An unsupported BEGIN output statement must DECLINE, never be silently
% dropped -- the stores-only fallback clause must not swallow it.
test(begin_unsupported_print_field_declines) :-
    build_status("BEGIN { print $1 }\n{ print $1 }\n", 3),
    !.

% --- structure -------------------------------------------------------------

% Output statements are collected in source order, both kinds.
test(begin_output_statements_in_source_order) :-
    Actions = [set(var('FS'), string(",")),
               print([string("a")]),
               printf(string("b\n"), []),
               print([string("c")])],
    plawk_native_codegen:plawk_begin_output_statements(Actions, Statements),
    assertion(Statements == [print([string("a")]),
                             printf(string("b\n"), []),
                             print([string("c")])]),
    !.

% A BEGIN with no output statement yields none (the assignment is handled by its
% own startup-store path, not as output).
test(begin_assignment_only_has_no_output_statements) :-
    plawk_native_codegen:plawk_begin_output_statements(
        [set(var('FS'), string(","))], Statements),
    assertion(Statements == []),
    !.

% The globals and the body come from ONE call, so they cannot disagree about a
% name or an index -- they used to be two independent walks over the clause,
% each numbering string globals from 0 on its own.
test(begin_globals_and_body_agree_on_names) :-
    Clauses = [begin([print([string("a")]), print([string("b")])])],
    plawk_native_codegen:plawk_begin_clause_outputs_ir(Clauses, [32], GlobalIR,
        BodyIR),
    % Statement 1's names are suffixed, so both statements' globals exist and
    % each is referenced by the matching body statement.
    assertion(once(sub_atom(GlobalIR, _, _, _, '@.plawk_begin_print_string_0'))),
    assertion(once(sub_atom(GlobalIR, _, _, _, '@.plawk_begin1_print_string_0'))),
    assertion(once(sub_atom(BodyIR, _, _, _, '@.plawk_begin_print_string_0'))),
    assertion(once(sub_atom(BodyIR, _, _, _, '@.plawk_begin1_print_string_0'))),
    !.

% --- IR shape --------------------------------------------------------------

% The BEGIN printf emits its own format global and call, distinct from the END
% printf's, and runs in the entry block (before the record loop).
test(begin_printf_ir_emits_own_format_global) :-
    plawk_parse_string("BEGIN { printf \"%d\\n\", 7 }\n{ print $1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_begin_printf_fmt'))),
    !.

% Two BEGIN printfs get distinct names.
test(begin_two_printfs_get_distinct_names) :-
    plawk_parse_string(
        "BEGIN { printf \"%d,\", 1; printf \"%d\\n\", 2 }\n{ print $1 }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_begin_printf_fmt'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_begin1_printf_fmt'))),
    !.

:- end_tests(plawk_begin_printf).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_begin_printf', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build Src, run it over the shared input, require Expected byte-for-byte
% (printf output is separator-sensitive, so no sorting or trimming).
run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'bp_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'bp', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', Bin], 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    assertion(Out == Expected).

% Build only, asserting the CLI status (3 = parses but outside the compilable
% surface -- a clean decline, distinct from 2 = parse error).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'bp_decline', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
