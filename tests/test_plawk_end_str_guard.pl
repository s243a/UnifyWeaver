:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: STRING comparison guards in an END block.
%
% `END { if (s == "c") print s }` declined, while the IDENTICAL guard in a rule
% body compiled. NUMERIC END guards (`END { if (n > 2) print "many" }`) already
% worked, so this was one condition form missing from one context -- not END
% conditionals as a whole.
%
% Cause: the END-if lowering called plawk_while_cond_ir/7 directly, but the
% string-scalar comparison clauses live on plawk_if_cond_ir/8 -- they need a
% GLOBALS channel for the interned literal constant, which the while-condition
% emitter does not have. So END could reach the numeric forms and nothing else.
% END now asks plawk_if_cond_ir/8, the same predicate a rule-body `if` asks, so a
% condition form cannot compile in one context and decline in the other. Numeric
% conditions are untouched: that predicate's fall-through clause IS the
% plawk_while_cond_ir/7 call this replaced.
%
% A WRONG-OUTPUT bug fell out of enabling this, and is fixed here too. The
% string-scalar comparison is written TWICE, and the two copies differed in
% exactly one guard:
%
%   plawk_resolve_scalar_cmp/4 (the bare string-scalar PATTERN) required a
%       scalar_string / scalar_strnum slot;
%   plawk_if_cond_ir/8 (the `if` guard) matched on slot NAME alone.
%
% So a numeric COUNTER compared against a string literal compared a count against
% an interned atom id: `{ n++; if (n == "3") print "eq" }` printed nothing where
% gawk prints `eq` (awk compares a number against a string AS STRINGS). That was
% pre-existing in the rule body; routing END through the same emitter would have
% made it reachable in END too. Both `if` clauses now require
% plawk_slot_holds_text/1, so such a program DECLINES -- in both contexts.
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

% Three records: "a 1" / "b 2" / "c 3". After the loop `s = $1` holds "c".
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_end_str_guard).

% --- equality, on a field-assigned (strnum) scalar ------------------------

test(end_string_equality, [condition(clang_available)]) :-
    run("{ s = $1 } END { if (s == \"c\") print s }\n", "c\n"),
    !.

test(end_string_equality_with_else, [condition(clang_available)]) :-
    run("{ s = $1 } END { if (s == \"c\") print \"yes\"; else print \"no\" }\n",
        "yes\n"),
    !.

test(end_string_inequality, [condition(clang_available)]) :-
    run("{ s = $1 } END { if (s != \"c\") print \"no\"; else print \"yes\" }\n",
        "yes\n"),
    !.

% The false branch of an equality, so the else arm is exercised too.
test(end_string_equality_false, [condition(clang_available)]) :-
    run("{ s = $1 } END { if (s == \"zz\") print \"yes\"; else print \"no\" }\n",
        "no\n"),
    !.

% --- all four ordering operators ------------------------------------------

test(end_string_less_than, [condition(clang_available)]) :-
    run("{ s = $1 } END { if (s < \"d\") print \"lt\" }\n", "lt\n"),
    !.

test(end_string_less_equal, [condition(clang_available)]) :-
    run("{ s = $1 } END { if (s <= \"c\") print \"le\" }\n", "le\n"),
    !.

test(end_string_greater_than, [condition(clang_available)]) :-
    run("{ s = $1 } END { if (s > \"a\") print \"gt\" }\n", "gt\n"),
    !.

test(end_string_greater_equal, [condition(clang_available)]) :-
    run("{ s = $1 } END { if (s >= \"c\") print \"ge\" }\n", "ge\n"),
    !.

% --- a literal-assigned (string) scalar, not just a field copy ------------

test(end_guard_on_a_literal_assigned_scalar, [condition(clang_available)]) :-
    run("{ s = \"lit\" } END { if (s == \"lit\") print \"yes\" }\n", "yes\n"),
    !.

% --- composition ----------------------------------------------------------

% The guarded print is an ordinary END print, so ORS terminates it.
test(end_guard_print_honours_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { s = $1 } END { if (s == \"c\") print s }\n",
        "c|"),
    !.

% --- regressions: numeric END guards unchanged ---------------------------

test(end_numeric_guard_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { if (n > 2) print \"many\" }\n", "many\n"),
    !.

test(end_numeric_guard_with_else_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print \"three\"; else print \"other\" }\n",
        "three\n"),
    !.

% --- regressions: the same guards in a RULE BODY still work --------------

% The point of the change is that both contexts share one emitter, so the rule
% body must keep working identically.
test(rule_body_string_equality_unchanged, [condition(clang_available)]) :-
    run("{ s = $1; if (s == \"b\") print \"hit\" }\n", "hit\n"),
    !.

test(rule_body_string_ordering_unchanged, [condition(clang_available)]) :-
    run("{ s = $1; if (s < \"c\") print \"lt\" }\n", "lt\nlt\n"),
    !.

% --- the wrong-output bug, now a clean decline ---------------------------

% A numeric COUNTER against a string literal. awk compares a number against a
% string AS STRINGS (so `n == "3"` with n=3 is TRUE), but the id comparison
% answered false. Declines in BOTH contexts now, rather than mis-comparing.
test(counter_vs_string_literal_declines_in_end) :-
    build_status("{ n++ } END { if (n == \"3\") print \"eq\"; else print \"ne\" }\n",
        3),
    !.

test(counter_vs_string_literal_declines_in_rule_body) :-
    build_status("{ n++; if (n == \"3\") print \"eq\" }\n", 3),
    !.

test(counter_vs_string_literal_ordering_declines) :-
    build_status("{ n++ } END { if (n < \"9\") print \"lt\"; else print \"ge\" }\n",
        3),
    !.

% --- structure ------------------------------------------------------------

% Both string clauses of the `if` condition emitter require a text-holding slot.
% This is the guard whose ABSENCE was the bug, and its sibling emitter for the
% bare string-scalar PATTERN already had it.
test(text_slot_is_required_by_the_if_guard) :-
    Slots = [scalar_counter(n)],
    Values = ['%slot_0'],
    assertion(\+ plawk_native_codegen:plawk_if_cond_ir(
        scalar_if(cmp(var(n), eq, string("3"))), Slots, Values, [], 32,
        plawk_test, _CondValue, _Pair)),
    % ...and it succeeds for a text slot
    TextSlots = [scalar_strnum(s)],
    assertion(plawk_native_codegen:plawk_if_cond_ir(
        scalar_if(cmp(var(s), eq, string("c"))), TextSlots, Values, [], 32,
        plawk_test2, _CondValue2, _Pair2)),
    !.

% Both slot kinds that hold text are accepted; a counter is not.
test(slot_kinds_that_hold_text) :-
    assertion(plawk_native_codegen:plawk_slot_holds_text(scalar_string(s))),
    assertion(plawk_native_codegen:plawk_slot_holds_text(scalar_strnum(s))),
    assertion(\+ plawk_native_codegen:plawk_slot_holds_text(scalar_counter(n))),
    !.

% --- IR shape -------------------------------------------------------------

% NOTE: these pin the GUARD's own temporaries (`plawk_endif_…`), not bare
% `@wam_intern_atom` / `@strcmp`. Every driver already calls both -- it interns
% the input path and strcmps the EOF sentinel -- so a bare-name assertion would
% pass no matter what the guard emitted.

% An END equality guard interns the literal and compares atom ids.
test(end_equality_ir_compares_interned_ids) :-
    plawk_parse_string("{ s = $1 } END { if (s == \"c\") print s }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_endif_litid = call i64 @wam_intern_atom'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_endif_cond = icmp eq i64'))),
    !.

% An END ordering guard resolves the id to text and strcmps.
test(end_ordering_ir_uses_strcmp) :-
    plawk_parse_string("{ s = $1 } END { if (s < \"d\") print \"lt\" }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_endif_scmp = call i32 @strcmp'))),
    !.

% A numeric END guard emits neither -- it is still a plain icmp on the slot,
% which is what "byte-identical for numeric conditions" rests on.
test(end_numeric_guard_ir_is_still_an_icmp) :-
    plawk_parse_string("{ n++ } END { if (n > 2) print \"many\" }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_endif_cond = icmp sgt i64'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'plawk_endif_scmp')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'plawk_endif_litid')),
    !.

:- end_tests(plawk_end_str_guard).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_str_guard', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'esg_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'esg', Prog0),
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

% Build only, asserting the CLI status (3 = parses but outside the compilable
% surface -- a clean decline).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'esg_reject', Prog0),
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
