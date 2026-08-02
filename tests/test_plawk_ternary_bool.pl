:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: `&&` / `||` in ternary CONDITIONS.
%
%   x = ($1 == "a" && $2 > 1) ? 1 : 0     DECLINED
%
% A ternary condition was a single comparison. Because the condition is now ONE
% emitter (plawk_ternary_cond_ir/8) with three callers, combinators added there
% land for i64 branches, string branches and the `$0` form simultaneously --
% asserted below rather than assumed.
%
% No short-circuit: a ternary condition has no side effects and both arms are
% already evaluated for the `select`, so `and i1` / `or i1` keeps the whole thing
% straight-line and composing anywhere a ternary does.
%
% The one subtle part is SSA placement. Each side's OPERAND setup goes in the
% operand half and each side's CONDITION lines in the condition half, so the
% caller emits setupA, setupB, [branches], condA, condB, combine -- every use
% dominated by its definition.
%
% Two WALKERS had to learn about combinators as well, both of which had baked a
% `cmp`-only assumption into their heads:
%
%   plawk_expr_uses_nr/1 -- without it, `x = (NR == 2 && …) ? 1 : 0` would have
%       referenced an undefined %current_nr (a clang failure), because the record
%       counter is emitted only when a walker SEES the NR and a combinator hid
%       it. Fixed BEFORE probing this time, having cost a debug cycle on the
%       string-branch work.
%   plawk_substitute_scalar_reads/4 -- covered on the follow-up branch.
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

:- begin_tests(plawk_ternary_bool).

% --- the combinators ------------------------------------------------------

test(and_condition, [condition(clang_available)]) :-
    run("{ x = ($1 == \"a\" && $2 > 0) ? 1 : 0; print x }\n", "1\n0\n0\n"),
    !.

test(or_condition, [condition(clang_available)]) :-
    run("{ x = ($2 > 2 || $1 == \"a\") ? 1 : 0; print x }\n", "1\n0\n1\n"),
    !.

% Unparenthesised: the combinator binds looser than the comparisons.
test(bare_and_condition, [condition(clang_available)]) :-
    run("{ x = $1 == \"a\" && $2 > 0 ? 1 : 0; print x }\n", "1\n0\n0\n"),
    !.

% Precedence override: `||` inside parens, then `&&`.
test(parenthesised_or_inside_and, [condition(clang_available)]) :-
    run("{ x = ($1 == \"a\" || $1 == \"b\") && $2 > 1 ? 1 : 0; print x }\n",
        "0\n1\n0\n"),
    !.

% --- NR inside a combinator: the walker fix -------------------------------

% Without the NR-walker fix these reference an undefined %current_nr and clang
% fails. Both operators, because the walker recurses through both.
test(nr_inside_and, [condition(clang_available)]) :-
    run("{ x = (NR == 2 && $1 == \"b\") ? 1 : 0; print x }\n", "0\n1\n0\n"),
    !.

test(nr_inside_or, [condition(clang_available)]) :-
    run("{ x = (NR == 1 || NR == 3) ? \"edge\" : \"mid\"; print x }\n",
        "edge\nmid\nedge\n"),
    !.

% --- one emitter, every branch type and context ---------------------------

% STRING branches with a combinator condition -- free, because the condition
% emitter is shared. (The NR-or test above is also a string-branch case.)
test(combinator_with_string_branches, [condition(clang_available)]) :-
    run("{ print ($2 > 1 && $1 != \"c\") ? \"y\" : \"n\" }\n", "n\ny\nn\n"),
    !.

% `$0` operands inside a combinator -- also free, same reason.
test(combinator_with_whole_record_operands, [condition(clang_available)]) :-
    run("{ x = ($0 == \"b 2\" || $0 == \"c 3\") ? 1 : 0; print x }\n",
        "0\n1\n1\n"),
    !.

test(combinator_in_printf, [condition(clang_available)]) :-
    run("{ printf \"%d\\n\", ($2 > 1 && $2 < 3) ? 1 : 0 }\n", "0\n1\n0\n"),
    !.

% --- regressions: every pre-existing ternary form -------------------------

test(bare_comparison_unchanged, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? 10 : 20; print x }\n", "20\n10\n10\n"),
    !.

test(paren_condition_unchanged, [condition(clang_available)]) :-
    run("{ x = ($2 > 1) ? 10 : 20; print x }\n", "20\n10\n10\n"),
    !.

test(paren_whole_ternary_unchanged, [condition(clang_available)]) :-
    run("{ x = ($2 > 1 ? \"hi\" : \"lo\"); print x }\n", "lo\nhi\nhi\n"),
    !.

test(whole_record_condition_unchanged, [condition(clang_available)]) :-
    run("{ x = $0 == \"b 2\" ? 1 : 0; print x }\n", "0\n1\n0\n"),
    !.

% --- structure ------------------------------------------------------------

% A single comparison still parses to the bare cmp/3 term -- nothing downstream
% changes unless a combinator is actually written.
test(single_comparison_is_still_a_bare_cmp) :-
    plawk_parse_string("{ x = $2 > 1 ? 10 : 20 }\n",
        program([], [rule(always, [set(var(x), Ternary)])], [])),
    assertion(Ternary = ternary(cmp(field(2), gt, int(1)), int(10), int(20))),
    !.

% `&&` binds tighter than `||`.
test(and_binds_tighter_than_or) :-
    plawk_parse_string("{ x = $1 == \"a\" || $1 == \"b\" && $2 > 1 ? 1 : 0 }\n",
        program([], [rule(always, [set(var(x), Ternary)])], [])),
    assertion(Ternary = ternary(or(_, and(_, _)), int(1), int(0))),
    !.

% ...and parentheses override that.
test(parens_override_precedence) :-
    plawk_parse_string(
        "{ x = ($1 == \"a\" || $1 == \"b\") && $2 > 1 ? 1 : 0 }\n",
        program([], [rule(always, [set(var(x), Ternary)])], [])),
    assertion(Ternary = ternary(and(or(_, _), _), int(1), int(0))),
    !.

% The whole-condition gate recurses, so a combinator admits exactly the
% comparisons a bare condition does -- it cannot be accepted at the top level and
% rejected inside an `&&`.
test(condition_gate_recurses) :-
    Cmp = cmp(field(1), eq, string("a")),
    Num = cmp(field(2), gt, int(1)),
    assertion(plawk_native_codegen:plawk_ternary_condition_ok(Cmp)),
    assertion(plawk_native_codegen:plawk_ternary_condition_ok(and(Cmp, Num))),
    assertion(plawk_native_codegen:plawk_ternary_condition_ok(
        or(and(Cmp, Num), Num))),
    % a leaf the bare gate rejects is rejected inside a combinator too
    Bad = cmp(string("a"), eq, string("b")),
    assertion(\+ plawk_native_codegen:plawk_ternary_condition_ok(Bad)),
    assertion(\+ plawk_native_codegen:plawk_ternary_condition_ok(and(Cmp, Bad))),
    !.

% The NR walker sees NR through both combinators -- the fix that keeps
% %current_nr defined.
test(nr_walker_recurses_through_combinators) :-
    Nr = cmp(special('NR'), eq, int(2)),
    Plain = cmp(field(2), gt, int(1)),
    assertion(plawk_native_codegen:plawk_expr_uses_nr(
        ternary(and(Plain, Nr), int(1), int(0)))),
    assertion(plawk_native_codegen:plawk_expr_uses_nr(
        ternary(or(Nr, Plain), int(1), int(0)))),
    assertion(\+ plawk_native_codegen:plawk_expr_uses_nr(
        ternary(and(Plain, Plain), int(1), int(0)))),
    !.

% --- IR shape -------------------------------------------------------------

% NOTE: these pin the COMBINATOR'S OWN sub-condition bases (`_bl_cond` /
% `_br_cond`), not bare `and i1` / `or i1`. Every driver's read loop already
% contains `%line_bad = and i1 …`, so a bare-mnemonic assertion is true no matter
% what the ternary emitted -- and its negation fails for a reason unrelated to
% the feature. (It did; that is why this note exists.)

test(and_condition_emits_a_combined_i1) :-
    plawk_parse_string("{ x = ($1 == \"a\" && $2 > 0) ? 1 : 0; print x }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '_bl_cond'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_br_cond'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'select i1'))),
    !.

test(or_condition_emits_a_combined_i1) :-
    plawk_parse_string("{ x = ($2 > 2 || $1 == \"a\") ? 1 : 0; print x }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '_bl_cond'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_br_cond'))),
    !.

% A single comparison never enters the combinator path, so neither sub-condition
% base appears.
test(single_comparison_emits_no_combinator) :-
    plawk_parse_string("{ x = $2 > 1 ? 10 : 20; print x }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '_bl_cond')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '_br_cond')),
    !.

:- end_tests(plawk_ternary_bool).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ternary_bool', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'tb_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'tb', Prog0),
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
