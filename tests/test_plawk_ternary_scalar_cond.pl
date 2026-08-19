:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: SCALAR-VARIABLE operands in ternary conditions and branches.
%
%   { n++; x = n > 1 ? 1 : 0 }        DECLINED
%
% A ternary operand could be a field, NR, NF, length, an integer literal or
% arithmetic over those -- but not a scalar variable, even though a scalar
% compares fine in a rule pattern (`n > 2 { … }`) and in an `if` guard. The
% operand resolves to the slot's SSA value the same way those do.
%
% Works in the condition (`n > 1 ? …`), in a branch (`$2 > 1 ? n : 0`), and on
% both sides (`n > m ? …`).
%
% ---------------------------------------------------------------------------
% THE ASYMMETRY THIS INTRODUCED, AND WHY IT IS PINNED HERE
%
% Admitting the operand made the i64-branch form compile while the STRING-branch
% form still declined:
%
%   { n++; x = n > 1 ? 1 : 0 }        compiled
%   { n++; x = n > 1 ? "a" : "b" }    DECLINED     <-- asymmetry
%
% The cause is a fourth walker over the ternary. A string-valued ternary is
% carried by a set_str operation as ternary_str/3, a DIFFERENT FUNCTOR from
% ternary/3, and plawk_substitute_scalar_reads/4 had a row for the latter only.
% So the scalar read in the condition was never substituted to its slot value.
%
% That is the same shape the shared condition emitter exists to prevent, one
% layer up in the substitution pass. Four traversals of one term have now each
% had to learn the same lesson independently:
%
%   plawk_ternary_cond_ir/8              cmp-only          fixed with `&&`/`||`
%   plawk_expr_uses_nr/1                 cmp-only          fixed with `&&`/`||`
%   plawk_substitute_scalar_reads/4      ternary/3 cmp     fixed here
%   plawk_substitute_scalar_reads/4      ternary_str/3     MISSING ENTIRELY
%
% The fix delegates ternary_str/3 to the same condition walker, so the two
% payload spellings cannot diverge. The symmetry is asserted below as a PAIR: if
% one branch type ever accepts an operand the other refuses, a walker has been
% missed again.
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

:- begin_tests(plawk_ternary_scalar_cond).

% --- THE symmetry pair ----------------------------------------------------

% Same scalar-variable condition, the two branch types. These must move together.
test(scalar_condition_with_i64_branches, [condition(clang_available)]) :-
    run("{ n++; x = n > 1 ? 1 : 0; print x }\n", "0\n1\n1\n"),
    !.

test(scalar_condition_with_string_branches, [condition(clang_available)]) :-
    run("{ n++; x = n > 1 ? \"a\" : \"b\"; print x }\n", "b\na\na\n"),
    !.

% --- the scalar operand in each position ----------------------------------

% In a BRANCH rather than the condition.
test(scalar_in_a_branch, [condition(clang_available)]) :-
    run("{ n++; x = $2 > 1 ? n : 0; print x }\n", "0\n2\n3\n"),
    !.

% On BOTH sides of the comparison.
test(scalar_on_both_sides, [condition(clang_available)]) :-
    run("{ n++; m = 2; x = n > m ? 1 : 0; print x }\n", "0\n0\n1\n"),
    !.

% --- composing with the combinators ---------------------------------------

test(scalar_inside_an_and, [condition(clang_available)]) :-
    run("{ n++; x = (n > 1 && $1 != \"a\") ? 1 : 0; print x }\n", "0\n1\n1\n"),
    !.

test(scalar_inside_an_or_with_string_branches, [condition(clang_available)]) :-
    run("{ n++; x = (n == 2 || n == 3) ? \"hit\" : \"no\"; print x }\n",
        "no\nhit\nhit\n"),
    !.

% --- regressions: every other ternary form --------------------------------

test(field_condition_unchanged, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? 10 : 20; print x }\n", "20\n10\n10\n"),
    !.

test(nr_condition_with_string_branches_unchanged, [condition(clang_available)]) :-
    run("{ x = NR == 2 ? \"a\" : \"b\"; print x }\n", "b\na\nb\n"),
    !.

test(whole_record_condition_unchanged, [condition(clang_available)]) :-
    run("{ x = $0 == \"b 2\" ? 1 : 0; print x }\n", "0\n1\n0\n"),
    !.

test(combinator_condition_unchanged, [condition(clang_available)]) :-
    run("{ x = ($1 == \"a\" && $2 > 0) ? 1 : 0; print x }\n", "1\n0\n0\n"),
    !.

test(paren_whole_ternary_unchanged, [condition(clang_available)]) :-
    run("{ x = ($2 > 1 ? \"hi\" : \"lo\"); print x }\n", "lo\nhi\nhi\n"),
    !.

% --- structure: the substitution walker covers BOTH payload spellings -----

% The bug in one assertion: a scalar read inside a ternary_str condition must be
% rewritten to its slot value, exactly as inside a ternary. If this regresses,
% the string-branch form silently declines again.
test(substitution_covers_both_payload_spellings) :-
    Slots = [scalar_counter(n)],
    Values = ['%slot_0'],
    Cond = cmp(var(n), gt, int(1)),
    plawk_native_codegen:plawk_substitute_scalar_reads(
        ternary(Cond, int(1), int(0)), Slots, Values, Plain),
    plawk_native_codegen:plawk_substitute_scalar_reads(
        ternary_str(Cond, string("a"), string("b")), Slots, Values, Str),
    assertion(Plain = ternary(cmp(ssa('%slot_0'), gt, int(1)), _, _)),
    assertion(Str = ternary_str(cmp(ssa('%slot_0'), gt, int(1)), _, _)),
    !.

% ...and through the combinators, so `(n > 1 && …)` substitutes too.
test(substitution_reaches_through_combinators) :-
    Slots = [scalar_counter(n)],
    Values = ['%slot_0'],
    Cond = and(cmp(var(n), gt, int(1)), cmp(field(2), gt, int(0))),
    plawk_native_codegen:plawk_substitute_scalar_reads(
        ternary_str(Cond, string("a"), string("b")), Slots, Values, Out),
    assertion(Out = ternary_str(and(cmp(ssa('%slot_0'), gt, int(1)), _), _, _)),
    !.

% The gate admits a scalar operand in a condition and in a branch.
test(gate_admits_scalar_operands) :-
    assertion(plawk_native_codegen:plawk_ternary_condition_ok(
        cmp(var(n), gt, int(1)))),
    assertion(plawk_native_codegen:plawk_ternary_i64_operand_ok(var(n))),
    !.

:- end_tests(plawk_ternary_scalar_cond).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ternary_scalar_cond', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'tsc_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'tsc', Prog0),
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

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
