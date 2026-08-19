:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: `$0` compared against a STRING LITERAL.
%
% This was the one comparison shape with no spelling at all -- a PARSE ERROR in
% both operand orders, as a rule pattern and inside a rule-body `if`:
%
%   $0 == "b 2" { … }        PARSE ERROR
%   "b 2" == $0 { … }        PARSE ERROR
%   { if ($0 == "b 2") … }   PARSE ERROR
%
% while `$0 ~ /re/` worked and every POSITIVE field compared fine ($1 == "b",
% ordering, reversed order, scalars, …).
%
% It gets its OWN term, record_str_cmp(Op, Value), rather than reusing
% field_eq/field_str_cmp with index 0. That is the whole point: the field
% comparators PROJECT A SLICE out of the record and answer false for index 0, so
% admitting `$0` into them yields WRONG OUTPUT rather than a decline -- which is
% exactly what happened when it was tried as a ternary condition (#4035, where
% `$0 == "b" ? 1 : 0` printed 0/0/0 against gawk's 0/1/0). A distinct term makes
% that mis-routing structurally impossible.
%
% The lowering is also simpler than a field's: the whole record is already a
% NUL-terminated string, so the guard is a plain strcmp against the literal with
% no slicing and no FS dependence. awk compares a field against a string constant
% AS STRINGS (the constant forces it), so a lexical strcmp is right for all six
% operators.
%
% The record pointer is re-resolved at the guard rather than reusing `%line_s`,
% because a `getline` can grow and relocate the shared transient buffer.
%
% Both operand orders go through the shared swap_cmp_op/2 and emit the SAME term,
% so nothing downstream ever sees a literal-first comparison.
%
% gawk 5.2 is the oracle for every expectation here.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records: "a 1" / "b 2" / "c 3".
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_whole_record_str_cmp).

% --- equality and inequality, both operand orders -------------------------

test(whole_record_equality, [condition(clang_available)]) :-
    run("$0 == \"b 2\" { print $1 }\n", "b\n"),
    !.

test(whole_record_equality_reversed, [condition(clang_available)]) :-
    run("\"b 2\" == $0 { print $1 }\n", "b\n"),
    !.

test(whole_record_inequality, [condition(clang_available)]) :-
    run("$0 != \"b 2\" { print $1 }\n", "a\nc\n"),
    !.

test(whole_record_inequality_reversed, [condition(clang_available)]) :-
    run("\"b 2\" != $0 { print $1 }\n", "a\nc\n"),
    !.

% A literal that matches nothing: the guard must be false for every record, not
% accidentally true (the failure mode a slice-based comparator had).
test(whole_record_equality_no_match, [condition(clang_available)]) :-
    run("$0 == \"nope\" { print $1 }\n", ""),
    !.

% --- all four ordering operators ------------------------------------------

test(whole_record_less_than, [condition(clang_available)]) :-
    run("$0 < \"b\" { print $1 }\n", "a\n"),
    !.

test(whole_record_less_equal, [condition(clang_available)]) :-
    run("$0 <= \"b 2\" { print $1 }\n", "a\nb\n"),
    !.

test(whole_record_greater_than, [condition(clang_available)]) :-
    run("$0 > \"b\" { print $1 }\n", "b\nc\n"),
    !.

test(whole_record_greater_equal, [condition(clang_available)]) :-
    run("$0 >= \"b 2\" { print $1 }\n", "b\nc\n"),
    !.

% Reversed ordering mirrors the operator: `"b" > $0` means `$0 < "b"`.
test(whole_record_ordering_reversed, [condition(clang_available)]) :-
    run("\"b\" > $0 { print $1 }\n", "a\n"),
    !.

% --- inside a rule-body `if` ----------------------------------------------

test(whole_record_in_if, [condition(clang_available)]) :-
    run("{ if ($0 == \"b 2\") print $1 }\n", "b\n"),
    !.

test(whole_record_in_if_else, [condition(clang_available)]) :-
    run("{ if ($0 == \"b 2\") print $1; else print \"x\" }\n", "x\nb\nx\n"),
    !.

% --- composition with the combinators -------------------------------------

test(whole_record_or, [condition(clang_available)]) :-
    run("$0 == \"b 2\" || $0 == \"c 3\" { print $1 }\n", "b\nc\n"),
    !.

test(whole_record_and_with_nr, [condition(clang_available)]) :-
    run("$0 == \"b 2\" && NR == 2 { print \"both\" }\n", "both\n"),
    !.

test(whole_record_negated, [condition(clang_available)]) :-
    run("!($0 == \"b 2\") { print $1 }\n", "a\nc\n"),
    !.

% --- regressions: the shapes that already worked --------------------------

test(positive_field_comparison_unchanged, [condition(clang_available)]) :-
    run("$1 == \"b\" { print $1 }\n", "b\n"),
    !.

test(reversed_field_comparison_unchanged, [condition(clang_available)]) :-
    run("\"b\" == $1 { print $1 }\n", "b\n"),
    !.

test(field_ordering_unchanged, [condition(clang_available)]) :-
    run("$1 < \"b\" { print $1 }\n", "a\n"),
    !.

test(whole_record_regex_unchanged, [condition(clang_available)]) :-
    run("$0 ~ /b 2/ { print $1 }\n", "b\n"),
    !.

% --- the TERNARY condition, now on this same emitter ----------------------

% This change was about the rule-pattern / `if`-guard surface; a ternary CONDITION
% goes through a different production and gate (plawk_ternary_cond_ok/3) and
% declined at the time. It no longer does: the follow-up wired that gate to the
% whole-record strcmp introduced here, so one emitter serves both surfaces. Pinned
% as compiling, in the suite that owns the emitter.
test(whole_record_ternary_condition_compiles, [condition(clang_available)]) :-
    build_status("{ x = $0 == \"b 2\" ? 1 : 0; print x }\n", 0),
    !.

% --- structure -------------------------------------------------------------

% Both operand orders produce the SAME term, so nothing downstream can tell them
% apart -- the property that keeps the reversed form from needing its own guard.
test(both_orders_produce_one_term) :-
    plawk_parse_string("$0 == \"b 2\" { print $1 }\n",
        program([], [rule(Forward, _)], [])),
    plawk_parse_string("\"b 2\" == $0 { print $1 }\n",
        program([], [rule(Reversed, _)], [])),
    assertion(Forward == Reversed),
    assertion(Forward = record_str_cmp(eq, "b 2")),
    !.

% The reversed ORDERING form mirrors its operator rather than emitting a
% literal-first term.
test(reversed_ordering_mirrors_the_operator) :-
    plawk_parse_string("\"b\" > $0 { print $1 }\n",
        program([], [rule(Pattern, _)], [])),
    assertion(Pattern = record_str_cmp(lt, "b")),
    !.

% `$0` is NOT routed into the field comparator -- a distinct term, so the
% index-0 slice bug cannot be reached.
test(whole_record_is_not_a_field_term) :-
    plawk_parse_string("$0 == \"b 2\" { print $1 }\n",
        program([], [rule(Pattern, _)], [])),
    assertion(\+ Pattern = field_eq(_, _)),
    assertion(\+ Pattern = field_str_cmp(_, _, _)),
    !.

% A positive field still parses to the field terms, unchanged.
test(positive_field_still_a_field_term) :-
    plawk_parse_string("$1 == \"b\" { print $1 }\n",
        program([], [rule(Pattern, _)], [])),
    assertion(Pattern = field_eq(1, "b")),
    !.

% --- IR shape --------------------------------------------------------------

% The guard is a strcmp of the whole record, with no field slicing.
test(guard_ir_strcmps_the_record) :-
    plawk_parse_string("$0 == \"b 2\" { print $1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '_reccmp = call i32 @strcmp'))),
    assertion(\+ sub_atom(DriverIR, _, _, _,
        '@wam_atom_field_str_cmp_value')),
    !.

% The record pointer is re-resolved at the guard (relocation-safe) rather than
% reusing the loop's %line_s.
test(guard_ir_reresolves_the_record_pointer) :-
    plawk_parse_string("$0 == \"b 2\" { print $1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '_rec_payload = call i64 @value_payload'))),
    !.

% A positive field still uses the field comparator -- the two lowerings stay
% distinct.
test(field_guard_ir_still_uses_the_field_comparator) :-
    plawk_parse_string("$1 == \"b\" { print $1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '_reccmp = call i32 @strcmp')),
    !.

:- end_tests(plawk_whole_record_str_cmp).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_whole_record_str_cmp', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'wrs_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'wrs', Prog0),
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
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

% Build only, asserting the CLI status (2 = parse error, 3 = parses but outside
% the compilable surface).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'wrs_reject', Prog0),
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
