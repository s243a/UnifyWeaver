:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: `printf` in an END block -- `END { printf "%d items\n", n }`.
%
% Before this, `printf` was not in the END statement grammar at all, so even
% `END { printf "hi\n" }` was a PARSE ERROR (exit 2) rather than a clean
% decline -- plawk called a valid awk program malformed.
%
% END runs after the record loop, so there is no current record: an argument
% cannot slice `$0`/`$N`. Arguments read the FINAL scalar slot values, exactly
% as an END `print` field does -- i64 counters, double slots (%f/%g), string /
% strnum slots (an interned atom id resolved to text, id 0 printing empty), NR,
% string literals, integer and float literals, and scalar arithmetic over those.
%
% Everything after the arguments -- the awk->C format rewrite, the format
% global, the `@printf` call -- is shared with the record-context printf
% (plawk_printf_from_arg_pairs/4): two argument producers, one consumer, so a
% format fix cannot land in one context and not the other.
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

% Three records: "a 1" / "b 2" / "c 3".
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_end_printf).

% --- parsing ---------------------------------------------------------------

% `printf` is an END statement, sitting beside `print` in the same list.
test(end_printf_parses) :-
    plawk_parse_string("{ n++ } END { printf \"%d\\n\", n }\n",
        program([], [rule(always, [inc(var(n))])],
            [end([printf(string("%d\n"), [var(n)])])])),
    !.

% `printf` must be tried before `print` in the END grammar, or `print` matches
% the keyword prefix and leaves a stray `f`.
test(end_printf_not_confused_with_print) :-
    plawk_parse_string("{ n++ } END { print n; printf \"%d\\n\", n }\n",
        program([], [rule(always, [inc(var(n))])],
            [end([print([var(n)]), printf(string("%d\n"), [var(n)])])])),
    !.

% A lone plain print is untouched -- it still parses to the same single-print
% END shape its own (byte-identical) driver clause owns.
test(single_print_end_shape_unchanged) :-
    plawk_parse_string("{ n++ } END { print n }\n",
        program([], [rule(always, [inc(var(n))])], [end([print([var(n)])])])),
    !.

% --- output, against gawk --------------------------------------------------

% No arguments: the format IS the output, and printf appends no newline of its
% own (the "\n" here is the format's).
test(printf_no_args, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"hi\\n\" }\n", "hi\n"),
    !.

% The classic: report an accumulated counter.
test(printf_counter, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%d items\\n\", n }\n", "3 items\n"),
    !.

% printf does NOT append ORS -- two printfs run together on one line, and only
% the trailing "\n" in the second format ends it.
test(printf_appends_no_newline, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%d,\", n; printf \"%d\\n\", n + 1 }\n",
        "3,4\n"),
    !.

% String literal argument (%s) and a literal `%%`.
test(printf_string_literal_arg, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%s!\\n\", \"done\" }\n", "done!\n"),
    !.

test(printf_percent_escape, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"100%%\\n\" }\n", "100%\n"),
    !.

% Several arguments of mixed kinds in one call.
test(printf_mixed_args, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%s %d %s\\n\", \"a\", n, \"z\" }\n",
        "a 3 z\n"),
    !.

% NR in END is the final record count.
test(printf_nr, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"NR=%d\\n\", NR }\n", "NR=3\n"),
    !.

% Scalar arithmetic over the final slots, same operand surface `print` accepts.
test(printf_scalar_arithmetic, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%d\\n\", n + 1 }\n", "4\n"),
    !.

% A double-valued slot passes as a native double, so %f precision applies.
test(printf_double_slot, [condition(clang_available)]) :-
    run_end("{ d = $2 / 2 } END { printf \"%.2f\\n\", d }\n", "1.50\n"),
    !.

% Division promotes to double in awk, so `n / 2` with n=3 is 1.50, not 1.
test(printf_division_promotes_to_double, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%.2f\\n\", n / 2 }\n", "1.50\n"),
    !.

% A string scalar's slot holds an interned atom id; %s resolves it to text.
test(printf_string_scalar, [condition(clang_available)]) :-
    run_end("{ s = $1 } END { printf \"%s!\\n\", s }\n", "c!\n"),
    !.

% A string scalar that IS assigned but holds id 0 (the unset sentinel) prints
% empty rather than resolving to whatever atom 0 happens to be.
test(printf_empty_string_scalar_is_empty, [condition(clang_available)]) :-
    run_end("{ s = \"\" } END { printf \"[%s]\\n\", s }\n", "[]\n"),
    !.

% A NEVER-ASSIGNED variable has no scalar slot to read, so printf declines.
% gawk prints "[]" (an uninitialised value is the empty string in string
% context); plawk's END `print` prints "[0]" for the same program, because an
% unassigned name lands in an i64 slot defaulting to 0. Both are the same
% pre-existing uninitialised-scalar gap as the absent-array-element work, and
% neither matches gawk -- printf declines rather than committing to the wrong
% one of the two. Pinned so the follow-on that adds the dual value updates it.
test(printf_never_assigned_var_declines) :-
    build_status("{ n++ } END { printf \"[%s]\\n\", s }\n", 3),
    !.

% Flags / width / precision reach the C format unchanged.
test(printf_width_flag, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"[%5d]\\n\", n }\n", "[    3]\n"),
    !.

test(printf_hex_conversion, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%x\\n\", n }\n", "3\n"),
    !.

% --- print and printf in one END block ------------------------------------

test(print_then_printf, [condition(clang_available)]) :-
    run_end("{ n++ } END { print \"count:\"; printf \"%d\\n\", n }\n",
        "count:\n3\n"),
    !.

% The printf leaves the line open and the following print closes it -- proof the
% two statement kinds share one END block without either resetting the other.
test(printf_then_print, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%d \", n; print \"done\" }\n", "3 done\n"),
    !.

% --- integer literal printf arguments (both contexts) ---------------------

% A bare INTEGER literal argument. `print 1` renders a constant as its text,
% which is fine for print but wrong for a `%d`/`%x`/`%f` conversion, so a printf
% argument yields the numeric int(Value) leaf instead. A bare FLOAT literal was
% already accepted; this closes the integer half in BOTH contexts.
test(printf_integer_literal_arg_in_end, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%d\\n\", 42 }\n", "42\n"),
    !.

test(printf_integer_literal_arg_in_rule, [condition(clang_available)]) :-
    run_prog("{ printf \"%d\\n\", 42 }\n", "42\n42\n42\n"),
    !.

test(printf_float_literal_arg, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%.2f\\n\", 3.5 }\n", "3.50\n"),
    !.

% `print 1` still renders the constant as text -- unchanged by the above.
test(print_integer_constant_unchanged, [condition(clang_available)]) :-
    run_prog("{ print 1 }\n", "1\n1\n1\n"),
    !.

% --- record reads in an END printf ---------------------------------------

% This declined because END had no current record to slice. It now reads the
% RETAINED last record, like gawk: the record loop copies each record aside and the
% printf argument projects from that copy. tests/test_plawk_end_field_reads.pl owns
% the behaviour; the pin is inverted here rather than deleted so the transition
% stays visible in this file.
%
% The argument produces the same call-argument vocabulary a RECORD-context printf
% produces for a field -- a slice_len/slice_ptr pair -- so the format rewriter and
% the call renderer needed no new cases.
test(end_printf_field_arg_reads_the_last_record, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%s\\n\", $1 }\n", "c\n"),
    !.

test(end_printf_whole_record_arg_reads_the_last_record,
     [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%s\\n\", $0 }\n", "c 3\n"),
    !.

% NF counts the retained record too -- and in the END `if` and loop drivers it had
% been counting the `end_of_file` sentinel (printing 1), because the gate matched
% `field(_)` only and NF is not a `field(_)`.
test(end_printf_nf_arg_counts_the_last_record, [condition(clang_available)]) :-
    run_end("{ n++ } END { printf \"%d\\n\", NF }\n", "2\n"),
    !.

% --- clean declines (no miscompile) ---------------------------------------

% An assoc element as an END printf argument is not wired yet -- declines
% cleanly rather than reaching the printf call with a raw table value.
test(end_printf_assoc_arg_declines) :-
    build_status("{ c[$1]++ } END { printf \"%d\\n\", c[\"a\"] }\n", 3),
    !.

% --- IR shape --------------------------------------------------------------

% The END printf reads a FINAL slot value (established by the break-close phi),
% not a record slice, and emits no ORS terminator of its own.
test(end_printf_ir_reads_final_slot) :-
    plawk_parse_string("{ n++ } END { printf \"%d\\n\", n }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 %final_slot_0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'plawk_end_printf_fmt'))),
    !.

% Multiple END statements must not collide: each statement's `end_`-prefixed
% names are suffixed by its index, so the second printf's format global and
% call are distinct from the first's.
test(end_printf_statements_get_distinct_names) :-
    plawk_parse_string("{ n++ } END { printf \"%d,\", n; printf \"%d\\n\", n }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_end_printf_fmt'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_end1_printf_fmt'))),
    !.

:- end_tests(plawk_end_printf).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_printf', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build Src, run it over the shared input, and require Expected byte-for-byte
% (printf output is separator-sensitive, so no sorting or trimming here).
run_end(Src, Expected) :-
    run_prog(Src, Expected).

run_prog(Src, Expected) :-
    odir(Dir),
    input(Input),
    % Remove any binary a previous test left behind, so a build that unexpectedly
    % declines cannot silently run a stale executable and "pass".
    directory_file_path(Dir, 'ep_bin', StaleBin),
    ( exists_file(StaleBin) -> delete_file(StaleBin) ; true ),
    build(Dir, 'ep', Src, Bin, In, Input),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    assertion(Out == Expected).

build(Dir, Name, Src, Bin, In, Input) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)).

% Build only, asserting the CLI status (3 = parses but outside the compilable
% surface -- a clean decline, distinct from 2 = parse error).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'ep_decline', Prog0),
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
