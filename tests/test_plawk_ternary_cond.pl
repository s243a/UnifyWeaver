:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: ternary CONDITIONS -- parentheses and string comparisons.
%
% The ternary itself already worked in every context (print field, printf
% argument, scalar assignment RHS), lowering to an LLVM `select`. Two things about
% its CONDITION did not:
%
%   x = ($2 > 1) ? 10 : 20     was a PARSE ERROR  -- parenthesised condition
%   x = $1 == "a" ? 1 : 2      DECLINED           -- string-comparison condition
%
% The parenthesised form is how the ternary is most often written; both grammar
% forms now share ternary_cond//1, so they cannot drift.
%
% A string-comparison condition is not an i64 `icmp`, so it reuses the same
% field-vs-literal comparator the `$N OP "str"` rule guards use -- which yields an
% i1 directly and covers all SIX operators, equality and lexical ordering alike.
% The branches stay i64, so only the condition differs; the `select` is identical.
%
% Both gates that admit a ternary (a print field and a scalar assignment) now ask
% through one plawk_ternary_cond_ok/3, so a condition form cannot be accepted in
% one context and rejected in another.
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

% Three records: "a 1" / "b 2" / "c 3". $1 is a/b/c and $2 is 1/2/3.
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_ternary_cond).

% --- parenthesised conditions ---------------------------------------------

test(paren_numeric_condition, [condition(clang_available)]) :-
    run("{ x = ($2 > 1) ? 10 : 20; print x }\n", "20\n10\n10\n"),
    !.

test(paren_nr_condition, [condition(clang_available)]) :-
    run("{ x = (NR == 2) ? 1 : 0; print x }\n", "0\n1\n0\n"),
    !.

test(paren_string_condition, [condition(clang_available)]) :-
    run("{ x = ($1 == \"a\") ? 1 : 2; print x }\n", "1\n2\n2\n"),
    !.

% The parenthesised form works in every context the bare one does.
test(paren_condition_in_print, [condition(clang_available)]) :-
    run("{ print ($2 > 1) ? 10 : 20 }\n", "20\n10\n10\n"),
    !.

test(paren_condition_in_printf, [condition(clang_available)]) :-
    run("{ printf \"%d\\n\", ($2 > 1) ? 10 : 20 }\n", "20\n10\n10\n"),
    !.

% --- string-comparison conditions, all six operators ----------------------

test(string_equality_condition, [condition(clang_available)]) :-
    run("{ x = $1 == \"a\" ? 1 : 2; print x }\n", "1\n2\n2\n"),
    !.

test(string_inequality_condition, [condition(clang_available)]) :-
    run("{ x = $1 != \"a\" ? 1 : 2; print x }\n", "2\n1\n1\n"),
    !.

test(string_less_than_condition, [condition(clang_available)]) :-
    run("{ x = $1 < \"b\" ? 1 : 2; print x }\n", "1\n2\n2\n"),
    !.

test(string_less_equal_condition, [condition(clang_available)]) :-
    run("{ x = $1 <= \"b\" ? 1 : 2; print x }\n", "1\n1\n2\n"),
    !.

test(string_greater_than_condition, [condition(clang_available)]) :-
    run("{ x = $1 > \"b\" ? 1 : 2; print x }\n", "2\n2\n1\n"),
    !.

test(string_greater_equal_condition, [condition(clang_available)]) :-
    run("{ x = $1 >= \"b\" ? 1 : 2; print x }\n", "2\n1\n1\n"),
    !.

% A string condition in the print and printf contexts too, not just assignment.
test(string_condition_in_print, [condition(clang_available)]) :-
    run("{ print $1 == \"b\" ? 1 : 0 }\n", "0\n1\n0\n"),
    !.

test(string_condition_in_printf, [condition(clang_available)]) :-
    run("{ printf \"%d\\n\", $1 == \"b\" ? 1 : 0 }\n", "0\n1\n0\n"),
    !.

% `$0` against a string literal used to DECLINE here, after an earlier attempt
% made it a miscompile (0/0/0 where gawk gives 0/1/0) because the only comparator
% projected a field slice and answers false for index 0. A whole-record strcmp
% now exists, so the condition compiles and is CORRECT -- covered in
% tests/test_plawk_ternary_record_cond.pl. Asserted here as no-longer-declining.
test(whole_record_string_condition_compiles, [condition(clang_available)]) :-
    build_status("{ x = $0 == \"b 2\" ? 1 : 0; print x }\n", 0),
    !.

% --- regressions: the forms that already worked ---------------------------

test(bare_numeric_condition_unchanged, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? 10 : 20; print x }\n", "20\n10\n10\n"),
    !.

test(bare_nr_condition_unchanged, [condition(clang_available)]) :-
    run("{ x = NR == 2 ? 1 : 0; print x }\n", "0\n1\n0\n"),
    !.

test(bare_condition_in_print_unchanged, [condition(clang_available)]) :-
    run("{ print $2 > 1 ? 10 : 20 }\n", "20\n10\n10\n"),
    !.

% Arithmetic operands in the condition and the branches still work.
test(arithmetic_operands_unchanged, [condition(clang_available)]) :-
    run("{ x = $2 + 1 > 2 ? $2 * 10 : 0; print x }\n", "0\n20\n30\n"),
    !.

% --- clean declines ------------------------------------------------------

% A REVERSED comparison (the literal on the left) used to decline. The parser now
% mirrors it to the field-first form, so it compiles and means the same thing --
% covered in tests/test_plawk_reversed_cmp.pl. Asserted here as no-longer-declining.
test(string_on_the_left_compiles, [condition(clang_available)]) :-
    build_status("{ x = \"a\" == $1 ? 1 : 2; print x }\n", 0),
    !.

test(two_string_literals_decline) :-
    build_status("{ x = \"a\" == \"b\" ? 1 : 2; print x }\n", 3),
    !.

% String-valued BRANCHES used to decline (this suite pinned that). They now
% compile in both the assignment and print contexts -- covered in
% tests/test_plawk_ternary_str_branches.pl. Asserted here as no-longer-declining,
% the same way the reversed-comparison pin above was flipped.
test(string_valued_branches_compile, [condition(clang_available)]) :-
    build_status("{ x = $2 > 1 ? \"hi\" : \"lo\"; print x }\n", 0),
    build_status("{ print $1 == \"b\" ? \"yes\" : \"no\" }\n", 0),
    !.

% MIXED branches used to decline; an INTEGER literal arm now folds to its decimal
% text at compile time, so this compiles -- covered in
% tests/test_plawk_ternary_mixed.pl. A non-literal numeric arm still declines.
test(mixed_string_and_int_branches_compile, [condition(clang_available)]) :-
    build_status("{ x = $2 > 1 ? \"hi\" : 3; print x }\n", 0),
    build_status("{ x = $2 > 1 ? \"hi\" : $1; print x }\n", 3),
    !.

% END has no current record, so a field condition cannot be lowered there.
test(ternary_in_end_declines) :-
    build_status("{ n++ } END { print n > 1 ? 1 : 0 }\n", 3),
    !.

% --- structure -----------------------------------------------------------

% Both grammar forms produce the SAME condition term, so nothing downstream can
% tell them apart -- that is what keeps the parenthesised form from needing its
% own codegen path.
test(both_grammar_forms_agree) :-
    plawk_parse_string("{ x = ($2 > 1) ? 10 : 20 }\n",
        program([], [rule(always, [set(var(x), Paren)])], [])),
    plawk_parse_string("{ x = $2 > 1 ? 10 : 20 }\n",
        program([], [rule(always, [set(var(x), Bare)])], [])),
    assertion(Paren == Bare),
    assertion(Paren = ternary(cmp(field(2), gt, int(1)), int(10), int(20))),
    !.

% The shared condition gate accepts both forms and rejects what neither path can
% lower. One predicate, asked by every ternary gate.
test(shared_condition_gate) :-
    % i64 vs i64
    assertion(plawk_native_codegen:plawk_ternary_cond_ok(field(2), gt, int(1))),
    assertion(plawk_native_codegen:plawk_ternary_cond_ok(special('NR'), eq,
        int(2))),
    % field vs string literal, every operator
    forall(member(Op, [eq, ne, lt, le, gt, ge]),
        assertion(plawk_native_codegen:plawk_ternary_cond_ok(field(1), Op,
            string("a")))),
    % reversed, and two literals: neither is lowerable
    assertion(\+ plawk_native_codegen:plawk_ternary_cond_ok(string("a"), eq,
        field(1))),
    assertion(\+ plawk_native_codegen:plawk_ternary_cond_ok(string("a"), eq,
        string("b"))),
    !.

% Both gates that admit a ternary go through that one predicate.
test(both_gates_admit_a_string_condition) :-
    Ternary = ternary(cmp(field(1), eq, string("a")), int(1), int(2)),
    assertion(plawk_native_codegen:plawk_rule_body_print_field(Ternary)),
    assertion(plawk_native_codegen:plawk_scalar_action_update(
        set(var(x), Ternary), x, set(Ternary))),
    !.

% --- IR shape ------------------------------------------------------------

% A string condition uses the field-vs-literal comparator and still selects, with
% no i64 icmp for the condition.
test(string_condition_ir_uses_the_str_comparator) :-
    plawk_parse_string("{ x = $1 == \"a\" ? 1 : 2; print x }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '@wam_atom_field_str_cmp_value'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'select i1'))),
    !.

% A numeric condition still lowers to an icmp, unchanged.
test(numeric_condition_ir_uses_icmp) :-
    plawk_parse_string("{ x = $2 > 1 ? 10 : 20; print x }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'icmp sgt i64'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_field_str_cmp_value')),
    !.

:- end_tests(plawk_ternary_cond).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ternary_cond', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'tc_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'tc', Prog0),
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
% surface -- a clean decline, distinct from 2 = parse error and 4 = clang failure).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'tc_decline', Prog0),
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
