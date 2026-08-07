:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: ternary with STRING-valued branches.
%
% #4035 finished the ternary's CONDITION side (parenthesised conditions, string
% comparisons). Its BRANCHES stayed i64-only, so the string-valued form -- at
% least as common as the numeric one -- declined:
%
%   x = $2 > 1 ? "hi" : "lo"          DECLINED
%   print $1 == "b" ? "yes" : "no"    DECLINED
%
% Two contexts, two lowerings, one condition emitter:
%
%   print / printf   `select i8*` over two string constants, yielding the
%                    existing string(Base, PtrIR) print type -- so it prints via
%                    the same `%s` path a plain string literal does, and needed
%                    no new output emitter.
%   assignment       a scalar_string slot already holds an INTERNED ATOM ID as an
%                    i64, so the store is `select i64` over the two ids. Each id
%                    is built by calling the existing string-literal builder, so
%                    the interning idiom is written once.
%
% Both ask plawk_ternary_cond_ir/8 for the condition, so the set of lowerable
% CONDITIONS cannot differ by branch type. That is asserted here rather than
% assumed: a scalar-variable condition declines for BOTH branch types, so it is a
% pre-existing condition gap and not a string-branch limitation.
%
% BOTH branches must be string literals; a mixed `cond ? "hi" : 3` declines
% rather than guessing. awk would stringify the number, but that needs a runtime
% conversion on one arm -- a follow-on, pinned below.
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

% Three records: "a 1" / "b 2" / "c 3". $1 is a/b/c and $2 is 1/2/3.
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_ternary_str_branches).

% --- assignment context ---------------------------------------------------

test(assign_numeric_condition, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? \"hi\" : \"lo\"; print x }\n", "lo\nhi\nhi\n"),
    !.

test(assign_nr_condition, [condition(clang_available)]) :-
    run("{ x = NR == 2 ? \"a\" : \"b\"; print x }\n", "b\na\nb\n"),
    !.

test(assign_string_equality_condition, [condition(clang_available)]) :-
    run("{ x = $1 != \"a\" ? \"no\" : \"yes\"; print x }\n", "yes\nno\nno\n"),
    !.

test(assign_string_ordering_condition, [condition(clang_available)]) :-
    run("{ x = $1 <= \"b\" ? \"low\" : \"high\"; print x }\n",
        "low\nlow\nhigh\n"),
    !.

test(assign_strict_ordering_condition, [condition(clang_available)]) :-
    run("{ x = $1 < \"b\" ? \"low\" : \"high\"; print x }\n",
        "low\nhigh\nhigh\n"),
    !.

test(assign_parenthesised_condition, [condition(clang_available)]) :-
    run("{ x = ($2 > 1) ? \"hi\" : \"lo\"; print x }\n", "lo\nhi\nhi\n"),
    !.

% An EMPTY branch: the interned empty string, not a missing value.
test(assign_empty_string_branch, [condition(clang_available)]) :-
    run("{ x = ($1 == \"a\") ? \"\" : \"z\"; print x }\n", "\nz\nz\n"),
    !.

% The assigned scalar is an ordinary string scalar afterwards -- it prints beside
% other fields like any other.
test(assigned_scalar_prints_beside_a_field, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? \"hi\" : \"lo\"; print x, $1 }\n",
        "lo a\nhi b\nhi c\n"),
    !.

% --- print / printf context ----------------------------------------------

test(print_string_condition, [condition(clang_available)]) :-
    run("{ print $1 == \"b\" ? \"yes\" : \"no\" }\n", "no\nyes\nno\n"),
    !.

test(print_nr_condition, [condition(clang_available)]) :-
    run("{ print NR == 2 ? \"a\" : \"b\" }\n", "b\na\nb\n"),
    !.

test(print_parenthesised_condition, [condition(clang_available)]) :-
    run("{ print ($1 == \"b\") ? \"yes\" : \"no\" }\n", "no\nyes\nno\n"),
    !.

test(printf_string_branches, [condition(clang_available)]) :-
    run("{ printf \"%s\\n\", $1 == \"b\" ? \"yes\" : \"no\" }\n",
        "no\nyes\nno\n"),
    !.

% Beside another print field, in both orders.
test(print_beside_a_field, [condition(clang_available)]) :-
    run("{ print $1, $1 == \"b\" ? \"yes\" : \"no\" }\n",
        "a no\nb yes\nc no\n"),
    !.

test(print_after_a_literal, [condition(clang_available)]) :-
    run("{ print \"tag:\", $1 == \"b\" ? \"yes\" : \"no\" }\n",
        "tag: no\ntag: yes\ntag: no\n"),
    !.

% It is an ordinary print field, so ORS terminates it like any other.
test(print_honours_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { print $1 == \"b\" ? \"yes\" : \"no\" }\n",
        "no|yes|no|"),
    !.

% --- regressions: i64 branches unchanged ----------------------------------

test(i64_branches_unchanged, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? 10 : 20; print x }\n", "20\n10\n10\n"),
    !.

test(i64_print_branches_unchanged, [condition(clang_available)]) :-
    run("{ print $1 == \"b\" ? 1 : 0 }\n", "0\n1\n0\n"),
    !.

test(i64_arithmetic_branches_unchanged, [condition(clang_available)]) :-
    run("{ x = $2 + 1 > 2 ? $2 * 10 : 0; print x }\n", "0\n20\n30\n"),
    !.

% --- clean declines -------------------------------------------------------

% MIXED branches used to decline (this suite pinned that, reasoning that awk's
% stringification needs a runtime conversion). It does not for an INTEGER literal:
% the conversion is exactly its decimal digits, so the arm is folded to text at
% compile time -- covered in tests/test_plawk_ternary_mixed.pl. Asserted here as
% no-longer-declining, in both contexts, so the pair keeps moving together.
test(mixed_string_and_int_branches_compile_in_assignment,
        [condition(clang_available)]) :-
    build_status("{ x = $2 > 1 ? \"hi\" : 3; print x }\n", 0),
    !.

test(mixed_string_and_int_branches_compile_in_print,
        [condition(clang_available)]) :-
    build_status("{ print $2 > 1 ? \"hi\" : 3 }\n", 0),
    !.

% A NON-LITERAL numeric arm still declines -- that one does need a runtime
% conversion, which is what the original reasoning above actually applies to.
test(non_literal_numeric_arm_still_declines) :-
    build_status("{ x = $2 > 1 ? \"hi\" : $1; print x }\n", 3),
    build_status("{ x = $2 > 1 ? \"hi\" : 3.5; print x }\n", 3),
    !.

% A SCALAR-VARIABLE condition, for BOTH branch types. This is the asymmetry test:
% the condition gate is shared, so a condition form is never admitted for i64
% branches and refused for string branches. If one of these two ever changes without
% the other, the shared emitter has been bypassed.
%
% It was pinned as a DECLINE for both. Scalar-variable ternary conditions have since
% landed, so it is inverted -- still asserting both branch types together, which is
% the property that matters here, and now against gawk rather than a status code.
% Both were verified against gawk 5.2 on "a 1 / b 2 / c 3".
test(scalar_var_condition_compiles_for_both_branch_types,
        [condition(clang_available)]) :-
    run("{ n++; x = n > 1 ? 1 : 0; print x }\n", "0\n1\n1\n"),
    run("{ n++; x = n > 1 ? \"a\" : \"b\"; print x }\n", "b\na\na\n"),
    !.

% `$0` against a string literal now compiles as a condition, via a whole-record
% strcmp that is correct at index 0 -- and because the condition emitter is shared
% across branch types, it works with STRING branches without extra work. That
% sharing is the property being pinned here.
test(whole_record_condition_compiles_with_string_branches,
        [condition(clang_available)]) :-
    build_status("{ x = $0 == \"b 2\" ? \"y\" : \"n\"; print x }\n", 0),
    !.

% A parenthesised WHOLE ternary used to be a PARSE error (this suite pinned it as
% a gap found while building string branches). It now parses -- covered in
% tests/test_plawk_ternary_parens.pl -- for BOTH branch types, which is the pairing
% worth keeping: the parenthesised form is a grammar question, not a branch-type
% one.
test(parenthesised_whole_ternary_compiles, [condition(clang_available)]) :-
    build_status("{ x = ($2 > 1 ? \"hi\" : \"lo\"); print x }\n", 0),
    build_status("{ x = ($2 > 1 ? 10 : 20); print x }\n", 0),
    !.

% --- structure: one gate, one condition emitter ---------------------------

% The branch-type gate. A string pair, and now a MIXED pair (the integer literal
% folds to text) -- but NOT two integer literals, which must keep taking the
% ordinary i64 path rather than becoming a string-valued expression.
test(string_branch_gate) :-
    assertion(plawk_native_codegen:plawk_ternary_str_branches_ok(
        string("hi"), string("lo"))),
    assertion(plawk_native_codegen:plawk_ternary_str_branches_ok(
        string("hi"), int(3))),
    assertion(plawk_native_codegen:plawk_ternary_str_branches_ok(
        int(3), string("lo"))),
    assertion(\+ plawk_native_codegen:plawk_ternary_str_branches_ok(
        int(1), int(0))),
    !.

% The whole-ternary gate is the condition gate AND the branch gate -- so every
% condition plawk_ternary_cond_ok/3 admits is admitted with string branches too.
test(string_ternary_gate_reuses_the_condition_gate) :-
    forall(member(Cond, [cmp(field(2), gt, int(1)),
                         cmp(special('NR'), eq, int(2)),
                         cmp(field(1), eq, string("a")),
                         cmp(field(1), le, string("b"))]),
        ( assertion(plawk_native_codegen:plawk_ternary_str_ok(Cond,
              string("hi"), string("lo"))),
          Cond = cmp(L, Op, R),
          assertion(plawk_native_codegen:plawk_ternary_cond_ok(L, Op, R))
        )),
    % and a condition the gate rejects is rejected with string branches too
    assertion(\+ plawk_native_codegen:plawk_ternary_str_ok(
        cmp(string("a"), eq, string("b")), string("hi"), string("lo"))),
    !.

% Both admitting gates -- a print field and a scalar assignment -- go through
% that one predicate.
test(both_gates_admit_a_string_ternary) :-
    Ternary = ternary(cmp(field(1), eq, string("b")), string("yes"),
        string("no")),
    assertion(plawk_native_codegen:plawk_rule_body_print_field(Ternary)),
    assertion(plawk_native_codegen:plawk_scalar_action_update(
        set(var(x), Ternary), x,
        set_str(ternary_str(cmp(field(1), eq, string("b")), string("yes"),
            string("no"))))),
    !.

% --- IR shape -------------------------------------------------------------

% A print-context string ternary selects between two POINTERS.
test(print_context_selects_pointers) :-
    plawk_parse_string("{ print $1 == \"b\" ? \"yes\" : \"no\" }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'select i1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'i8*'))),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '@wam_atom_field_str_cmp_value'))),
    !.

% An assignment-context string ternary selects between two interned IDS, so the
% slot keeps holding an id like any other string scalar.
test(assignment_context_selects_interned_ids) :-
    plawk_parse_string("{ x = $2 > 1 ? \"hi\" : \"lo\"; print x }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_intern_atom'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'select i1'))),
    !.

% The NR condition must EMIT the record counter. Without it the IR references an
% undefined %current_nr and clang fails -- the bug this change had to fix, so it
% is pinned at the IR level as well as by the run above.
test(nr_condition_emits_the_record_counter) :-
    plawk_parse_string("{ x = NR == 2 ? \"a\" : \"b\"; print x }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%current_nr ='))),
    !.

% The NR walker sees the set_str payload by DELEGATING to the ternary clause, so
% both spellings of the same ternary agree.
test(nr_walker_sees_both_payload_shapes) :-
    Cond = cmp(special('NR'), eq, int(2)),
    assertion(plawk_native_codegen:plawk_expr_uses_nr(
        ternary(Cond, string("a"), string("b")))),
    assertion(plawk_native_codegen:plawk_expr_uses_nr(
        ternary_str(Cond, string("a"), string("b")))),
    % a ternary with no NR anywhere is not reported as using it
    assertion(\+ plawk_native_codegen:plawk_expr_uses_nr(
        ternary_str(cmp(field(2), gt, int(1)), string("a"), string("b")))),
    !.

:- end_tests(plawk_ternary_str_branches).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ternary_str_branches', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'tsb_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'tsb', Prog0),
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
    directory_file_path(Dir, 'tsb_reject', Prog0),
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
