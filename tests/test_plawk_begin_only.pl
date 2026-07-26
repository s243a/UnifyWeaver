:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: BEGIN-ONLY programs -- `awk 'BEGIN { print 1 + 1 }'`, the calculator
% idiom, plus `exit` in BEGIN.
%
% Two gaps had to close together. `rules//1` required at least one rule, so a
% zero-rule program did not PARSE at all; and even parsed, the shared driver
% template opens a stream and runs a record loop.
%
% POSIX: "If an awk program consists of only actions with the pattern BEGIN, awk
% shall exit without reading its input." Verified against gawk 5.2: a BEGIN-only
% program IGNORES a file argument and does not consume stdin, while adding an END
% makes input read again (`BEGIN {…} END { print NR }` prints 0 with no stdin, 2
% with a two-line file). So the loop-free driver requires `[]` for BOTH rules and
% END; a zero-rule program WITH an END would have to read input with no rules to
% run, and declines cleanly.
%
% Because there is no record loop, `exit [N]` in BEGIN needs no branch -- the
% block runs straight into `ret`, so it is the same store-and-truncate as `exit`
% in an END block. That is exactly why BEGIN's exit is tractable HERE and not in
% a program with rules: there it would have to skip the loop yet still run END,
% and the template emits the BEGIN body inside `entry` ahead of that block's own
% terminator.
%
% gawk 5.2 is the oracle for every expectation here, exit STATUS included.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_begin_only).

% --- parsing: zero rules ---------------------------------------------------

test(begin_only_parses_with_no_rules) :-
    plawk_parse_string("BEGIN { print \"x\" }\n",
        program([begin([print([string("x")])])], [], [])),
    !.

test(begin_only_with_exit_parses) :-
    plawk_parse_string("BEGIN { print \"x\"; exit 3 }\n",
        program([begin([print([string("x")]), exit(int(3))])], [], [])),
    !.

% A zero-rule BEGIN/END pair parses too (codegen declines it -- see below).
test(begin_and_end_without_rules_parses) :-
    plawk_parse_string("BEGIN { print \"b\" }\nEND { print \"e\" }\n",
        program([begin([print([string("b")])])], [],
            [end([print([string("e")])])])),
    !.

% Programs that DO have rules are unaffected by the empty-rules clause.
test(rules_still_required_to_be_parsed_when_present) :-
    plawk_parse_string("BEGIN { print \"x\" }\n{ print $1 }\n",
        program([begin([print([string("x")])])],
            [rule(always, [print([field(1)])])], [])),
    !.

% The empty-rules clause must not turn malformed input into a zero-rule program:
% `eos` still has to be reached, so junk after the clauses is still an error.
test(malformed_programs_still_rejected) :-
    forall(member(Src, ["foo\n",
                        "{ print $1\n",
                        "BEGIN { print \"x\" }\n%%%\n",
                        "}\n",
                        "{ print $1 }\nzzz\n"]),
        ( plawk_parse_string(Src, P)
        -> format(user_error, "~nunexpectedly parsed: ~w -> ~q~n", [Src, P]), fail
        ;  true
        )),
    !.

% --- POSIX: no input is read ----------------------------------------------

% A file argument is IGNORED, not read -- the program prints only its own output.
test(begin_only_ignores_file_argument, [condition(clang_available)]) :-
    run_with_input("BEGIN { print \"x\" }\n", "a 1\nb 2\n", "x\n", 0),
    !.

% Nor is stdin consumed. Piping data in changes nothing.
test(begin_only_does_not_consume_stdin, [condition(clang_available)]) :-
    build_only("BEGIN { print \"x\" }\n", Bin),
    run_stdin(Bin, "a\nb\n", Out, Status),
    assertion(Out == "x\n"),
    assertion(Status == exit(0)),
    !.

% And with no stdin at all it must not block. run_stdin/4 with "" closes stdin
% immediately; a program that waited on input would hang the suite instead.
test(begin_only_does_not_block_without_stdin, [condition(clang_available)]) :-
    build_only("BEGIN { print \"x\" }\n", Bin),
    run_stdin(Bin, "", Out, Status),
    assertion(Out == "x\n"),
    assertion(Status == exit(0)),
    !.

% The IR proves it structurally: no stream is opened and no record loop exists.
test(begin_only_ir_has_no_stream_or_loop) :-
    plawk_parse_string("BEGIN { print \"x\" }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'wam_stream_open_value')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'wam_stream_read_line')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'loop:')),
    % main takes no argv, since there is no input path to consult
    assertion(once(sub_atom(DriverIR, _, _, _, 'define i32 @main() {'))),
    !.

% --- output ---------------------------------------------------------------

test(begin_only_print, [condition(clang_available)]) :-
    run("BEGIN { print \"x\" }\n", "x\n", 0),
    !.

test(begin_only_two_prints, [condition(clang_available)]) :-
    run("BEGIN { print \"a\"; print \"b\" }\n", "a\nb\n", 0),
    !.

test(begin_only_printf, [condition(clang_available)]) :-
    run("BEGIN { printf \"%d\\n\", 42 }\n", "42\n", 0),
    !.

% --- the calculator idiom -------------------------------------------------

test(begin_only_integer_arithmetic, [condition(clang_available)]) :-
    run("BEGIN { print 1 + 1 }\n", "2\n", 0),
    !.

test(begin_only_precedence, [condition(clang_available)]) :-
    run("BEGIN { print 2 * 3 + 4 }\n", "10\n", 0),
    !.

test(begin_only_modulo, [condition(clang_available)]) :-
    run("BEGIN { print 10 % 3 }\n", "1\n", 0),
    !.

% awk division is IEEE, so the tree is double-typed and prints as %g.
test(begin_only_division_is_float, [condition(clang_available)]) :-
    run("BEGIN { print 7 / 2 }\n", "3.5\n", 0),
    !.

test(begin_only_float_literal, [condition(clang_available)]) :-
    run("BEGIN { print 3.5 + 1 }\n", "4.5\n", 0),
    !.

% A label beside a computed value: OFS separates them, as in any print.
test(begin_only_label_and_value, [condition(clang_available)]) :-
    run("BEGIN { print \"sum:\", 1 + 2 }\n", "sum: 3\n", 0),
    !.

test(begin_only_printf_arithmetic, [condition(clang_available)]) :-
    run("BEGIN { printf \"%d\\n\", 1 + 1 }\n", "2\n", 0),
    !.

test(begin_only_printf_division_precision, [condition(clang_available)]) :-
    run("BEGIN { printf \"%.2f\\n\", 22 / 7 }\n", "3.14\n", 0),
    !.

% --- exit in BEGIN --------------------------------------------------------

test(begin_only_exit_sets_status, [condition(clang_available)]) :-
    run("BEGIN { print \"x\"; exit 3 }\n", "x\n", 3),
    !.

test(begin_only_bare_exit_is_zero, [condition(clang_available)]) :-
    run("BEGIN { exit }\n", "", 0),
    !.

% Statements after the exit are dead, as in END.
test(begin_only_exit_truncates, [condition(clang_available)]) :-
    run("BEGIN { print \"a\"; exit 1; print \"b\" }\n", "a\n", 1),
    !.

test(begin_only_calc_then_exit, [condition(clang_available)]) :-
    run("BEGIN { print 6 * 7; exit 0 }\n", "42\n", 0),
    !.

% --- clean declines -------------------------------------------------------

% A zero-rule program WITH an END: awk reads input in this case (END sees NR), so
% the loop-free driver must not claim it. No driver handles zero rules plus a
% record loop yet, so it declines -- a follow-on, not a silent wrong answer.
test(begin_and_end_without_rules_declines) :-
    build_status("BEGIN { print \"b\" }\nEND { print \"e\" }\n", 3),
    !.

test(end_only_without_rules_declines) :-
    build_status("END { print \"e\" }\n", 3),
    !.

% BEGIN has no record and no scalar slots, so a field or variable read declines
% rather than reading state that does not exist. (gawk prints empty for both --
% the same uninitialised-value gap tracked elsewhere.)
test(begin_only_field_read_declines) :-
    build_status("BEGIN { print $1 }\n", 3),
    !.

test(begin_only_variable_read_declines) :-
    build_status("BEGIN { print n }\n", 3),
    !.

% `exit` in a BEGIN block of a program that HAS rules: it would have to skip the
% record loop yet still run END, which the shared driver template cannot express.
% Declines rather than silently ignoring the exit.
test(begin_exit_with_rules_declines) :-
    build_status("BEGIN { exit 1 }\n{ print $1 }\n", 3),
    !.

% --- structure ------------------------------------------------------------

% Only literal-leaf expressions are BEGIN-evaluable; a field or variable leaf
% fails, which is what makes the declines above declines rather than bad IR.
test(begin_const_expr_accepts_only_literal_leaves) :-
    assertion(plawk_native_codegen:plawk_begin_const_expr(int(1))),
    assertion(plawk_native_codegen:plawk_begin_const_expr(float_const(7, 2))),
    assertion(plawk_native_codegen:plawk_begin_const_expr(
        add_i64(int(1), int(2)))),
    assertion(\+ plawk_native_codegen:plawk_begin_const_expr(field(1))),
    assertion(\+ plawk_native_codegen:plawk_begin_const_expr(var(n))),
    assertion(\+ plawk_native_codegen:plawk_begin_const_expr(
        add_i64(int(1), field(1)))),
    !.

% `exit` is collected as a BEGIN output statement so it keeps its position in the
% sequence -- that is what lets the emitter truncate at the right point.
test(begin_exit_is_an_output_statement) :-
    plawk_native_codegen:plawk_begin_output_statements(
        [set(var('FS'), string(",")), print([string("a")]), exit(int(1)),
         print([string("b")])],
        Statements),
    assertion(Statements == [print([string("a")]), exit(int(1)),
                             print([string("b")])]),
    !.

:- end_tests(plawk_begin_only).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_begin_only', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_only(Src, Bin) :-
    odir(Dir),
    directory_file_path(Dir, 'bo_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'bo', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    cli([build, Prog, '-o', Bin], 0).

% Run with NO arguments at all and stdin closed.
run(Src, Expected, ExpectedStatus) :-
    build_only(Src, Bin),
    run_stdin(Bin, "", Out, Status),
    assertion(Out == Expected),
    assertion(Status == exit(ExpectedStatus)).

% Run WITH a file argument, to prove the file is ignored rather than read.
run_with_input(Src, Input, Expected, ExpectedStatus) :-
    odir(Dir),
    build_only(Src, Bin),
    directory_file_path(Dir, 'bo_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, Status),
    assertion(Out == Expected),
    assertion(Status == exit(ExpectedStatus)).

% Run with no arguments, writing StdinText then closing stdin. Closing it is what
% makes "did it block on input?" observable: a reader would see EOF and finish,
% but the point is that our output must not depend on the data at all.
run_stdin(Bin, StdinText, Out, Status) :-
    process_create(Bin, [],
        [stdin(pipe(PIn)), stdout(pipe(POut)), stderr(std), process(Pid)]),
    write(PIn, StdinText),
    close(PIn),
    read_string(POut, _, Out),
    close(POut),
    process_wait(Pid, Status).

% Build only, asserting the CLI status (3 = parses but outside the compilable
% surface -- a clean decline, distinct from 2 = parse error).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'bo_decline', Prog0),
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
