:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: MIXED ternary branches -- `cond ? "hi" : 3`.
%
% A string-valued ternary required BOTH arms to be string literals; a mixed pair
% declined. awk's value is dual, and printing it gives the number's decimal
% spelling, so the mixed form is ordinary awk:
%
%   x = $2 > 1 ? "hi" : 3     ->  3 / hi / hi
%
% Handled by CONSTANT-FOLDING the integer literal to its decimal string at the
% gate: `c ? "hi" : 3` is `c ? "hi" : "3"`, exactly, because awk's number->string
% conversion of an integer is its decimal digits.
%
% Folding at the GATE is the structural point. Both string-ternary emitters --
% the print-context pointer select and the assignment-context interned-id select
% -- keep seeing `string(_)` arms only, so neither learns about numbers and they
% cannot disagree about the conversion. plawk_ternary_str_branch_value/2 is the
% ONLY place that knows how a number is spelled.
%
% ---------------------------------------------------------------------------
% THE BOUND ON WHAT THIS CLAIMS
%
% The fold is verified for STRING readings of the value: printing it, and
% comparing it as a string (`x == "3"`, which gawk also answers true).
%
% Reading the result back as a NUMBER declines:
%
%   { x = $2 > 1 ? "hi" : 3; print x + 1 }     DECLINES   (gawk: 4/1/1)
%
% so the fold is never exercised in a numeric context and cannot be wrong there.
% That decline is inherited from string-scalar arithmetic generally, not
% introduced here -- but it is pinned below, because it is what keeps the fold's
% correctness argument ("an integer's awk string form is its decimal digits", a
% statement about STRING context) sufficient.
%
% Three things are declined rather than folded, each for a stated reason:
%
%   a FLOAT literal      awk converts non-integers with CONVFMT (%.6g), so 3.5
%                        would fold exactly but 3.14159265 would silently become
%                        3.14159. Truncating a literal is worse than refusing it.
%   a FIELD arm          `c ? $1 : "hi"` needs a runtime conversion.
%   a SCALAR arm         `c ? n : "hi"` likewise.
%
% And two integer literals must keep taking the ORDINARY i64 path -- routing
% `c ? 1 : 0` here would turn it into a string-valued expression. The gate
% requires at least one genuine string arm.
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

:- begin_tests(plawk_ternary_mixed).

% --- the mixed form, both orders ------------------------------------------

test(string_then_int, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? \"hi\" : 3; print x }\n", "3\nhi\nhi\n"),
    !.

test(int_then_string, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? 3 : \"hi\"; print x }\n", "hi\n3\n3\n"),
    !.

% ZERO is the case a truthiness-based shortcut would get wrong: it must print
% "0", not empty.
test(zero_int_arm, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? \"hi\" : 0; print x }\n", "0\nhi\nhi\n"),
    !.

% A NEGATIVE literal keeps its sign through the fold.
test(negative_int_arm, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? \"hi\" : -3; print x }\n", "-3\nhi\nhi\n"),
    !.

% An EMPTY string beside an int arm: both spellings are just text after folding.
test(empty_string_beside_int, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? \"\" : 0; print \"[\" x \"]\" }\n",
        "[0]\n[]\n[]\n"),
    !.

% --- every condition form, since the condition emitter is shared -----------

test(string_condition, [condition(clang_available)]) :-
    run("{ x = $1 == \"b\" ? \"yes\" : 0; print x }\n", "0\nyes\n0\n"),
    !.

test(combinator_condition, [condition(clang_available)]) :-
    run("{ x = (NR == 2 && $1 == \"b\") ? \"hit\" : 0; print x }\n",
        "0\nhit\n0\n"),
    !.

test(whole_record_condition, [condition(clang_available)]) :-
    run("{ x = $0 == \"b 2\" ? \"hit\" : 0; print x }\n", "0\nhit\n0\n"),
    !.

% --- the print / printf context ------------------------------------------

test(printf_context, [condition(clang_available)]) :-
    run("{ printf \"%s\\n\", $1 == \"b\" ? \"yes\" : 0 }\n", "0\nyes\n0\n"),
    !.

test(print_list_context, [condition(clang_available)]) :-
    run("{ print $1, ($1 == \"b\" ? \"yes\" : 0) }\n", "a 0\nb yes\nc 0\n"),
    !.

% --- the folded value is a STRING, and compares as one --------------------

% gawk answers this true as well: the folded arm is the text "3".
test(folded_value_compares_as_a_string, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? \"hi\" : 3; if (x == \"3\") print \"streq\" }\n",
        "streq\n"),
    !.

% --- the bound: a NUMERIC reading declines -------------------------------

% gawk gives 4/1/1 here. plawk declines, so the fold is never used in a numeric
% context and cannot produce a wrong number there. Pinned because it is what
% makes the fold's string-context correctness argument sufficient; if this ever
% starts compiling, the numeric semantics need checking against gawk first.
test(numeric_reading_of_the_result_declines) :-
    build_status("{ x = $2 > 1 ? \"hi\" : 3; print x + 1 }\n", 3),
    !.

% --- clean declines: what is NOT folded ----------------------------------

% A FLOAT literal. awk would use CONVFMT (%.6g); folding would be exact for 3.5
% but lossy for 3.14159265, so it is refused rather than silently truncated.
test(float_literal_arm_declines) :-
    build_status("{ x = $2 > 1 ? \"hi\" : 3.5; print x }\n", 3),
    !.

% A FIELD arm and a SCALAR arm both need a runtime conversion.
test(field_arm_declines) :-
    build_status("{ x = $2 > 1 ? \"hi\" : $1; print x }\n", 3),
    !.

test(scalar_arm_declines) :-
    build_status("{ n++; x = $2 > 1 ? \"hi\" : n; print x }\n", 3),
    !.

% --- regressions: the two pure forms -------------------------------------

% TWO integer literals must keep taking the i64 path. If the string path claimed
% them, `c ? 1 : 0` would become a string-valued expression.
test(pure_i64_branches_unchanged, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? 10 : 20; print x }\n", "20\n10\n10\n"),
    !.

test(pure_string_branches_unchanged, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? \"hi\" : \"lo\"; print x }\n", "lo\nhi\nhi\n"),
    !.

test(whole_record_i64_ternary_unchanged, [condition(clang_available)]) :-
    run("{ x = $0 == \"b 2\" ? 1 : 0; print x }\n", "0\n1\n0\n"),
    !.

% --- structure: one folding predicate ------------------------------------

% The fold, directly. An integer becomes its decimal digits; a string is itself.
test(folding_predicate) :-
    plawk_native_codegen:plawk_ternary_str_branch_value(int(3), T3),
    plawk_native_codegen:plawk_ternary_str_branch_value(int(0), T0),
    plawk_native_codegen:plawk_ternary_str_branch_value(int(-3), TNeg),
    plawk_native_codegen:plawk_ternary_str_branch_value(string("hi"), THi),
    assertion(T3 == "3"),
    assertion(T0 == "0"),
    assertion(TNeg == "-3"),
    assertion(THi == "hi"),
    !.

% What the fold refuses: a float literal, a field, a scalar read.
test(folding_predicate_refuses_non_integers) :-
    assertion(\+ plawk_native_codegen:plawk_ternary_str_branch_value(
        float_const(7, 2), _)),
    assertion(\+ plawk_native_codegen:plawk_ternary_str_branch_value(
        field(1), _)),
    assertion(\+ plawk_native_codegen:plawk_ternary_str_branch_value(
        var(n), _)),
    !.

% The branch gate needs at least one GENUINE string, so two integer literals are
% left to the i64 path.
test(gate_requires_one_real_string) :-
    assertion(plawk_native_codegen:plawk_ternary_str_branches_ok(
        string("hi"), int(3))),
    assertion(plawk_native_codegen:plawk_ternary_str_branches_ok(
        int(3), string("hi"))),
    assertion(plawk_native_codegen:plawk_ternary_str_branches_ok(
        string("hi"), string("lo"))),
    assertion(\+ plawk_native_codegen:plawk_ternary_str_branches_ok(
        int(1), int(0))),
    !.

% --- IR shape ------------------------------------------------------------

% The folded arm is emitted as a string CONSTANT, so the mixed form produces the
% same shape as the pure-string form: no integer-to-string call at runtime.
test(mixed_form_emits_a_string_constant) :-
    plawk_parse_string("{ x = $2 > 1 ? \"hi\" : 3; print x }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'c"3\\00"'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@sprintf')),
    !.

:- end_tests(plawk_ternary_mixed).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ternary_mixed', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'tm_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'tm', Prog0),
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

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'tm_reject', Prog0),
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
