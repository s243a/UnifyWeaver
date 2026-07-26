:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: `exit [N]` in an END block -- `END { print n; exit 1 }`, how an awk
% script reports failure to a pipeline or CI.
%
% `exit` was not in the END statement grammar, so this was a PARSE ERROR
% (exit 2) rather than a clean decline -- the third instance of that
% wrong-error-class shape after `printf` in END and in BEGIN.
%
% A rule-level `exit N` stores the code and branches to break_close_stream,
% which runs END and returns the code. In END none of that machinery is needed:
% the END block falls straight into `ret %plawk_exit_ec`, so `exit N` is just the
% store, and the statements AFTER it are dead. The emitter therefore stores and
% STOPS -- truncation is what makes the remainder unreachable, and it is valid
% because the END statement list is straight-line code with no branches.
%
% Because the store happens later than a rule-level `exit`, an END `exit`
% correctly OVERRIDES a rule-level one, and a rule-level `exit` survives an END
% block that has none -- both matching awk.
%
% Scope: the SCALAR END chain (`END { print n; exit 1 }`), where `exit` is
% actually used. The assoc / for-in END chains have their own emitters and
% decline cleanly; wiring them is a follow-on, pinned below.
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

% Three records, so `n` is 3 at END.
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_end_exit).

% --- parsing ---------------------------------------------------------------

test(end_exit_with_code_parses) :-
    plawk_parse_string("{ n++ } END { print n; exit 3 }\n",
        program([], [rule(always, [inc(var(n))])],
            [end([print([var(n)]), exit(int(3))])])),
    !.

% A bare `exit` defaults to status 0, same as at rule level.
test(end_bare_exit_parses_as_zero) :-
    plawk_parse_string("{ n++ } END { exit }\n",
        program([], [rule(always, [inc(var(n))])], [end([exit(int(0))])])),
    !.

% --- output and exit status, against gawk ---------------------------------

test(end_exit_sets_status, [condition(clang_available)]) :-
    run("{ n++ } END { print n; exit 3 }\n", "3\n", 3),
    !.

test(end_exit_zero, [condition(clang_available)]) :-
    run("{ n++ } END { print n; exit 0 }\n", "3\n", 0),
    !.

test(end_bare_exit_is_zero, [condition(clang_available)]) :-
    run("{ n++ } END { print n; exit }\n", "3\n", 0),
    !.

% `exit` as the ONLY END statement: no output, just the status. This shape is a
% single non-print statement, so it exercises the generalized END clause rather
% than the dedicated single-plain-print one.
test(end_exit_only, [condition(clang_available)]) :-
    run("{ n++ } END { exit 4 }\n", "", 4),
    !.

test(end_exit_after_printf, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%d\\n\", n; exit 2 }\n", "3\n", 2),
    !.

% A negative or >255 code is truncated to 8 bits by the shell, exactly as gawk's.
test(end_exit_negative_code, [condition(clang_available)]) :-
    run("{ n++ } END { exit -1 }\n", "", 255),
    !.

test(end_exit_large_code, [condition(clang_available)]) :-
    run("{ n++ } END { exit 200 }\n", "", 200),
    !.

% OFS still applies to the prints before the exit.
test(end_exit_after_ofs_print, [condition(clang_available)]) :-
    run("BEGIN { OFS = \"-\" } { n++ } END { print n, n; exit 6 }\n", "3-3\n", 6),
    !.

% --- statements after the exit are dead -----------------------------------

% gawk prints only "a": everything after `exit` in the END block is unreachable.
test(end_exit_truncates_following_statements, [condition(clang_available)]) :-
    run("{ n++ } END { print \"a\"; exit 1; print \"b\" }\n", "a\n", 1),
    !.

% `exit` first: nothing in the block runs.
test(end_exit_first_suppresses_all_output, [condition(clang_available)]) :-
    run("{ n++ } END { exit 5; print \"never\" }\n", "", 5),
    !.

% --- interaction with a rule-level exit -----------------------------------

% A rule-level `exit` runs END; an END `exit` stores later, so it WINS.
test(end_exit_overrides_rule_exit, [condition(clang_available)]) :-
    run("{ n++; exit 2 }\nEND { exit 3 }\n", "", 3),
    !.

% ...and a rule-level `exit` survives an END block that does not exit.
test(rule_exit_survives_end_without_exit, [condition(clang_available)]) :-
    run("{ n++; exit 2 }\nEND { print n }\n", "1\n", 2),
    !.

test(end_exit_overrides_rule_exit_after_print, [condition(clang_available)]) :-
    run("{ n++; exit 2 }\nEND { print n; exit 7 }\n", "1\n", 7),
    !.

% --- regressions: END without exit ----------------------------------------

test(end_without_exit_still_zero, [condition(clang_available)]) :-
    run("{ n++ } END { print n }\n", "3\n", 0),
    !.

test(end_multi_print_without_exit, [condition(clang_available)]) :-
    run("{ n++ } END { print n; print \"x\" }\n", "3\nx\n", 0),
    !.

% --- clean declines (follow-ons, not miscompiles) -------------------------

% The assoc END chain has its own emitter, so an `exit` beside an assoc read
% declines rather than being silently dropped.
test(end_exit_with_assoc_read_declines) :-
    build_status("{ c[$1]++ } END { print c[\"a\"]; exit 3 }\n", 3),
    !.

% `exit` AFTER a for-in used to decline (the for-in END driver took no statement
% list). The mixed END statement chain landed since, so it now works -- the
% status is checked here; the chain's output shapes are covered in
% tests/test_plawk_end_chain.pl. for-in order is hash-dependent, so compare the
% key lines as a sorted set.
test(end_exit_after_forin, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; exit 2 }\n",
        ["a", "b", "c"], 2),
    !.

% An `exit` inside an END `if` branch: the END-if surface takes prints only.
test(end_exit_inside_if_declines) :-
    build_status("{ n++ } END { if (n > 2) exit 1 }\n", 3),
    !.

% --- structure and IR shape ----------------------------------------------

% The state plan walks PAST the exit even though the emitter truncates at it, so
% a name mentioned only by a dead statement still gets a slot. The plan being a
% superset is the safe direction: an unused slot is harmless, whereas a plan
% missing a slot the emitter referenced would leave a dangling SSA name.
test(end_output_list_walks_past_exit) :-
    plawk_native_codegen:plawk_end_output_list(
        [print([var(n)]), exit(int(1)), print([var(m)])], Exprs),
    assertion(Exprs == [[var(n)], [], [var(m)]]),
    !.

% The END block stores the code and emits nothing for the dead statement.
test(end_exit_ir_stores_code_once) :-
    plawk_parse_string("{ n++ } END { print \"a\"; exit 1; print \"b\" }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        'store i32 1, i32* @plawk_exit_code'))),
    % the dead print's string global is never referenced from the END block
    assertion(\+ sub_atom(DriverIR, _, _, _, 'plawk_end1_print_string_0')),
    !.

% The status is read back at the single `ret`, as before -- END exit reuses the
% existing exit-code global rather than adding a second path out.
test(end_exit_ir_returns_exit_code_global) :-
    plawk_parse_string("{ n++ } END { exit 4 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '%plawk_exit_ec = load i32, i32* @plawk_exit_code'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'ret i32 %plawk_exit_ec'))),
    !.

:- end_tests(plawk_end_exit).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_exit', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build Src, run it over the shared input, require Expected output byte-for-byte
% AND ExpectedStatus as the process exit status.
run(Src, Expected, ExpectedStatus) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'ee_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ee', Prog0),
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
    process_wait(Pid, Status),
    assertion(Out == Expected),
    assertion(Status == exit(ExpectedStatus)).

% As run/3 but comparing the output lines as a SORTED set, for programs whose
% for-in iteration order is hash-dependent.
run_sorted(Src, ExpectedSortedLines, ExpectedStatus) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'ee_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ee', Prog0),
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
    process_wait(Pid, Status),
    split_string(Out, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines),
    msort(Lines, SortedLines),
    msort(ExpectedSortedLines, ExpectedSorted),
    assertion(SortedLines == ExpectedSorted),
    assertion(Status == exit(ExpectedStatus)).

% Build only, asserting the CLI status (3 = parses but outside the compilable
% surface -- a clean decline, distinct from 2 = parse error).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'ee_decline', Prog0),
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
