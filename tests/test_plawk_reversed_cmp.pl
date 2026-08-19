:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: REVERSED comparisons -- the literal on the left (`"a" == $1`, `2 < $2`).
%
% awk accepts either operand order. plawk failed in three contexts, two of them
% parse errors:
%
%   "a" == $1 { print }          PARSE ERROR  -- rule pattern
%   { if ("a" == $1) print }     PARSE ERROR  -- rule-body if
%   { x = "a" == $1 ? 1 : 2 }    DECLINED     -- ternary condition
%
% and the NUMERIC half was inconsistent on its own: the reversed order already
% worked against the specials (`3 < NF`) and against a scalar (`2 < n`), but not
% against a FIELD (`2 < $2`), which was a parse error at the merge base.
%
% The fix is normalisation, not new lowering: each reversed production mirrors the
% operator with the shared swap_cmp_op/2 -- the very helper the already-working
% reversed forms use -- and emits the SAME term the field-first production builds.
% Nothing downstream sees a literal-first comparison at all, so there is no second
% code path to drift and no new comparator.
%
% Mirroring has to be EXACT, not merely accepted: `"b" < $1` and `$1 < "b"` select
% different records. Every operator is checked in BOTH orders below against gawk
% 5.2, which is the oracle throughout.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records: "a 1" / "b 2" / "c 3". $1 is a/b/c, $2 is 1/2/3.
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_reversed_cmp).

% --- string comparisons, both orders, every operator ----------------------

% The pairs differ, which is the point: an operator that was mirrored the wrong
% way would still compile and still print SOMETHING.
test(reversed_string_patterns_mirror_exactly, [condition(clang_available)]) :-
    forall(member(Op-Reversed-Forward,
               [ "=="-"b"-"b"
               , "!="-"a\nc"-"a\nc"
               , "<"-"c"-"a"
               , "<="-"b\nc"-"a\nb"
               , ">"-"a"-"c"
               , ">="-"a\nb"-"b\nc"
               ]),
        ( format(atom(RevSrc), "\"b\" ~w $1 { print $1 }\n", [Op]),
          format(atom(FwdSrc), "$1 ~w \"b\" { print $1 }\n", [Op]),
          string_concat(Reversed, "\n", RevExpect),
          string_concat(Forward, "\n", FwdExpect),
          run(RevSrc, RevExpect),
          run(FwdSrc, FwdExpect)
        )),
    !.

% --- numeric comparisons against a FIELD, both orders ---------------------

% `2 < $2` was a parse error while `3 < NF` and `2 < n` already worked -- the
% reversed order was supported for specials and scalars but not for fields.
test(reversed_numeric_field_patterns_mirror_exactly, [condition(clang_available)]) :-
    forall(member(Op-Reversed-Forward,
               [ "=="-"2"-"2"
               , "!="-"1\n3"-"1\n3"
               , "<"-"3"-"1"
               , "<="-"2\n3"-"1\n2"
               , ">"-"1"-"3"
               , ">="-"1\n2"-"2\n3"
               ]),
        ( format(atom(RevSrc), "2 ~w $2 { print $2 }\n", [Op]),
          format(atom(FwdSrc), "$2 ~w 2 { print $2 }\n", [Op]),
          string_concat(Reversed, "\n", RevExpect),
          string_concat(Forward, "\n", FwdExpect),
          run(RevSrc, RevExpect),
          run(FwdSrc, FwdExpect)
        )),
    !.

% --- the three contexts ---------------------------------------------------

test(reversed_in_rule_pattern, [condition(clang_available)]) :-
    run("\"a\" == $1 { print \"p\" }\n", "p\n"),
    !.

test(reversed_in_rule_body_if, [condition(clang_available)]) :-
    run("{ if (\"a\" == $1) print \"hit\" }\n", "hit\n"),
    !.

test(reversed_in_ternary_condition, [condition(clang_available)]) :-
    run("{ x = \"a\" == $1 ? 1 : 2; print x }\n", "1\n2\n2\n"),
    !.

test(reversed_numeric_in_rule_body_if, [condition(clang_available)]) :-
    run("{ if (2 < $2) print \"hit\" }\n", "hit\n"),
    !.

% Ordering, not just equality, in the ternary.
test(reversed_ordering_in_ternary, [condition(clang_available)]) :-
    run("{ x = \"b\" < $1 ? 1 : 2; print x }\n", "2\n2\n1\n"),
    !.

% --- composition with the pattern combinators ----------------------------

test(reversed_negated, [condition(clang_available)]) :-
    run("!(\"a\" == $1) { print $1 }\n", "b\nc\n"),
    !.

test(reversed_conjoined_with_forward, [condition(clang_available)]) :-
    run("\"a\" == $1 && $2 == \"1\" { print \"both\" }\n", "both\n"),
    !.

% --- regressions: the forward forms ---------------------------------------

test(forward_string_pattern_unchanged, [condition(clang_available)]) :-
    run("$1 == \"b\" { print $1 }\n", "b\n"),
    !.

test(forward_string_ternary_unchanged, [condition(clang_available)]) :-
    run("{ x = $1 == \"a\" ? 1 : 2; print x }\n", "1\n2\n2\n"),
    !.

test(forward_numeric_pattern_unchanged, [condition(clang_available)]) :-
    run("$2 > 2 { print $2 }\n", "3\n"),
    !.

test(regex_pattern_unchanged, [condition(clang_available)]) :-
    run("/b/ { print $1 }\n", "b\n"),
    !.

% The reversed forms that already worked keep working.
test(reversed_special_unchanged, [condition(clang_available)]) :-
    run("2 == NF { print \"two\" }\n", "two\ntwo\ntwo\n"),
    !.

% --- clean declines / unchanged rejections -------------------------------

% `$0` against a string literal used to be a PARSE ERROR in both orders (this
% suite pinned that). Both orders now compile as a rule pattern, via a dedicated
% record_str_cmp term and a whole-record strcmp -- covered in
% tests/test_plawk_whole_record_str_cmp.pl. Asserted here as no-longer-rejected,
% in PAIRS, so the two orders keep moving together.
test(reversed_whole_record_compiles, [condition(clang_available)]) :-
    build_status("\"b 2\" == $0 { print \"p\" }\n", 0),
    !.

test(forward_whole_record_compiles, [condition(clang_available)]) :-
    build_status("$0 == \"b 2\" { print \"p\" }\n", 0),
    !.

% Two literals: neither order is a comparison plawk lowers.
test(two_literals_still_rejected) :-
    build_status("\"a\" == \"b\" { print \"p\" }\n", 2),
    !.

% --- structure: one canonical term ---------------------------------------

% Both orders parse to the SAME term, which is what makes this a normalisation
% rather than a second lowering path. Equality is field_eq; inequality its
% negation; ordering is field_str_cmp with the operator mirrored.
test(both_orders_yield_one_term) :-
    forall(member(Op-Term,
               [ "=="-field_eq(1, "b")
               , "!="-not_pat(field_eq(1, "b"))
               , "<"-field_str_cmp(1, gt, "b")     % "b" < $1  ==  $1 > "b"
               , ">"-field_str_cmp(1, lt, "b")     % "b" > $1  ==  $1 < "b"
               , "<="-field_str_cmp(1, ge, "b")
               , ">="-field_str_cmp(1, le, "b")
               ]),
        ( format(string(Src), "\"b\" ~w $1 { print $1 }\n", [Op]),
          plawk_parse_string(Src, program([], [rule(Parsed, _)], [])),
          ( Parsed == Term
          -> true
          ;  format(user_error, "~n~w parsed to ~q, expected ~q~n",
                 [Src, Parsed, Term]), fail
          )
        )),
    !.

% The reversed numeric field form yields the field-first term too.
test(reversed_numeric_yields_forward_term) :-
    % (a bare `print` with no argument does not parse -- a separate pre-existing
    %  gap -- so these give it one)
    plawk_parse_string("2 < $2 { print $2 }\n",
        program([], [rule(field_cmp(2, gt, 2), _)], [])),
    plawk_parse_string("$2 > 2 { print $2 }\n",
        program([], [rule(field_cmp(2, gt, 2), _)], [])),
    !.

% A reversed ternary condition normalises to the field-first cmp, identical to
% what the forward source parses to.
test(reversed_ternary_normalises) :-
    plawk_parse_string("{ x = \"a\" == $1 ? 1 : 2 }\n",
        program([], [rule(always, [set(var(x), Reversed)])], [])),
    plawk_parse_string("{ x = $1 == \"a\" ? 1 : 2 }\n",
        program([], [rule(always, [set(var(x), Forward)])], [])),
    assertion(Reversed == Forward),
    assertion(Reversed = ternary(cmp(field(1), eq, string("a")), int(1), int(2))),
    !.

% The mirroring table itself: an involution, with the orderings swapped and the
% equalities fixed. This is the shared helper, so getting it wrong would break the
% already-working reversed special / scalar forms too.
test(swap_cmp_op_is_an_involution) :-
    forall(member(Op, [eq, ne, lt, le, gt, ge]),
        ( plawk_parser:swap_cmp_op(Op, Swapped),
          plawk_parser:swap_cmp_op(Swapped, Op)
        )),
    assertion(plawk_parser:swap_cmp_op(lt, gt)),
    assertion(plawk_parser:swap_cmp_op(le, ge)),
    assertion(plawk_parser:swap_cmp_op(eq, eq)),
    assertion(plawk_parser:swap_cmp_op(ne, ne)),
    !.

:- end_tests(plawk_reversed_cmp).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_reversed_cmp', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'rc_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'rc', Prog0),
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

% Build only, asserting the CLI status (2 = parse error, 3 = parses but outside the
% compilable surface).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'rc_reject', Prog0),
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
