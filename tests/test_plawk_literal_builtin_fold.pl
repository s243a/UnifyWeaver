:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% String builtins over a STRING LITERAL: `length("abc")`, `toupper("abc")`,
% `tolower("ABC")`, `substr("hello", 2, 3)`, `index("abcd", "bc")`.
%
% ---------------------------------------------------------------------------
% A GAP CLOSED BY FOLDING, WITH NO CODEGEN CHANGE AT ALL
%
% Every gap this campaign has closed so far was closed by teaching an emitter a new
% cell -- six field kinds across three END walkers, paid for one at a time. This one
% is different in kind, and the difference is the point.
%
% The five builtins accepted a FIELD argument in every route and refused a string
% LITERAL in all of them. But the answer to a builtin over a literal is itself a
% literal, computable at parse time -- `length("abc")` is 3, `toupper("abc")` is
% "ABC" -- and literals were already in the vocabulary of every route: rule-body
% print, END print, the mixed and assoc END walkers, printf arguments,
% concatenation parts, arithmetic operands, `+=` deltas. So the whole family landed
% in every context at once, and `examples/plawk/codegen/plawk_native_codegen.pl` was
% not touched. All 41 golden-corpus programs stayed byte-identical.
%
% That is worth generalising from: when a refused form has a COMPILE-TIME answer,
% ask what the answer's representation costs before adding a row per route. Here it
% cost nothing, because the answer's representation was already universal.
%
% ---------------------------------------------------------------------------
% WHAT THE REFUSAL ACTUALLY WAS -- THREE STATEMENTS OF ONE VOCABULARY
%
% Not a narrow grammar. `field_expr//1` already parsed literals, variables,
% concatenations and nested calls; each builtin production then asserted
% `Field = field(_)` and threw the general argument away. Eight copies of that
% assertion across two nonterminals (`field_expr//1`, and `scalar_delta_expr//1` for
% the `+=` right-hand side, with `length` and `index` written out in both), plus a
% THIRD statement in the arithmetic-operand family that restricted by choosing a
% narrow nonterminal (`simple_field_expr//1`, `$N` only) instead of by a guard.
%
% Eight copies of a refusal is the same defect as eight copies of a capability, and
% the third statement is the reason it is worth naming: widening the guard alone left
% the arithmetic family behind, so `print length("abc")` worked while
% `print length("abc") + 1` still did not. Its old name is why that was easy to miss
% -- `simple_field_expr` described what it accepted rather than the role it filled,
% so nothing connected it to the vocabulary it was a copy of. It is now
% `builtin_string_arg//1`, and plawk_builtin_string_arg/1 is the single authority all
% three families defer to.
%
% The two exit codes told this story before the code did, and both are pinned below:
% `length("abc")` failed as a clean DECLINE (exit 3) because it parsed as a call to a
% Prolog predicate named `length` that codegen refused, while `length(v)` failed as a
% PARSE error (exit 2) because `v` is not a foreign argument either. The generic
% `name(args)` production shadowing a reserved builtin name was a real defect of its
% own; plawk_surface_reserved_name/1 now excludes it.
%
% gawk 5.2 is the oracle for every expectation here, in the POSIX locale this
% container runs (which is what makes byte counts and ASCII-only case mapping the
% agreeing semantics -- see the_fold_is_ascii_only_by_design below).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records; the last is "7 disk". None of the expectations below depend on the
% input, which is the whole point of a fold -- but the programs still run over it so
% that a fold landing in the wrong place would show up as wrong output per record.
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_literal_builtin_fold).

% --- the five builtins, in a rule body -----------------------------------

test(length_of_a_literal, [condition(clang_available)]) :-
    run("{ print length(\"abc\") }\n", "3\n3\n3\n"),
    !.

test(toupper_of_a_literal, [condition(clang_available)]) :-
    run("{ print toupper(\"abc\") }\n", "ABC\nABC\nABC\n"),
    !.

test(tolower_of_a_literal, [condition(clang_available)]) :-
    run("{ print tolower(\"ABC\") }\n", "abc\nabc\nabc\n"),
    !.

test(substr_of_a_literal, [condition(clang_available)]) :-
    run("{ print substr(\"hello\", 2, 3) }\n", "ell\nell\nell\n"),
    !.

test(substr_to_end_of_a_literal, [condition(clang_available)]) :-
    run("{ print substr(\"hello\", 3) }\n", "llo\nllo\nllo\n"),
    !.

test(index_of_a_literal, [condition(clang_available)]) :-
    run("{ print index(\"abcd\", \"bc\") }\n", "2\n2\n2\n"),
    !.

% All five in one print, so a fold that fired for one shape and not another shows up.
test(all_five_together, [condition(clang_available)]) :-
    run("{ print length(\"gh\"), toupper(\"ab\"), tolower(\"CD\"), substr(\"hello\", 2, 3), index(\"abcd\", \"bc\") }\n",
        "2 AB cd ell 2\n2 AB cd ell 2\n2 AB cd ell 2\n"),
    !.

% --- every route the answer can land in ----------------------------------
%
% This is the part that would have been a row per walker had the answer not been a
% literal. Each of these was refused before, in the same way, for the same reason.

test(beside_a_field_read, [condition(clang_available)]) :-
    run("{ print length(\"abc\"), $1 }\n", "3 5\n3 5\n3 7\n"),
    !.

test(as_an_arithmetic_operand, [condition(clang_available)]) :-
    run("{ print length(\"abc\") + 1 }\n", "4\n4\n4\n"),
    !,
    run("{ print length(\"abc\") - 1 }\n", "2\n2\n2\n"),
    !,
    run("{ print length(\"abc\") * 2 }\n", "6\n6\n6\n"),
    !.

% The `+=` delta family -- the second of the three vocabularies, and the one that
% makes `int(N)` the right spelling for a folded numeric answer.
test(as_a_plus_equals_delta, [condition(clang_available)]) :-
    run("{ n += length(\"abcd\") } END { print n }\n", "12\n"),
    !,
    run("{ n += index(\"abcd\",\"bc\") } END { print n }\n", "6\n"),
    !.

test(as_a_printf_argument, [condition(clang_available)]) :-
    run("{ printf \"%s|%d\\n\", toupper(\"ab\"), length(\"xyz\") }\n",
        "AB|3\nAB|3\nAB|3\n"),
    !.

% --- the END routes ------------------------------------------------------
%
% All three END walkers, which is where a missing cell has cost this campaign the
% most. None of them needed a clause.

test(in_the_scalar_end_route, [condition(clang_available)]) :-
    run("{ n++ } END { print length(\"abc\") }\n", "3\n"),
    !,
    run("{ n++ } END { print length(\"abc\"), n }\n", "3 3\n"),
    !.

test(in_the_mixed_end_route, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print length(\"abc\"), c[\"5\"] }\n", "3 2\n"),
    !.

test(in_the_assoc_only_end_route, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print length(\"abc\"), c[\"5\"] }\n", "3 2\n"),
    !.

test(a_string_answer_in_end, [condition(clang_available)]) :-
    run("{ n++ } END { print toupper(\"abc\") }\n", "ABC\n"),
    !,
    run("{ n++ } END { print index(\"abcd\",\"bc\"), toupper(\"x\") }\n", "2 X\n"),
    !.

% --- awk's exact edge semantics ------------------------------------------
%
% Each of these was checked against gawk rather than reasoned about. substr's
% clamping is where a hand-rolled fold is most likely to be subtly wrong.

test(substr_clamps_at_the_end, [condition(clang_available)]) :-
    run("{ print substr(\"hello\", 1, 99) }\n", "hello\nhello\nhello\n"),
    !,
    run("{ print substr(\"hello\", 5, 5) }\n", "o\no\no\n"),
    !.

test(substr_past_the_end_is_empty, [condition(clang_available)]) :-
    run("{ print substr(\"hello\", 6, 1) }\n", "\n\n\n"),
    !.

test(substr_of_zero_length_is_empty, [condition(clang_available)]) :-
    run("{ print substr(\"hello\", 1, 0) }\n", "\n\n\n"),
    !.

test(index_finds_the_leftmost_occurrence, [condition(clang_available)]) :-
    run("{ print index(\"aaa\", \"a\") }\n", "1\n1\n1\n"),
    !.

test(index_is_zero_when_absent, [condition(clang_available)]) :-
    run("{ print index(\"abc\", \"abcd\") }\n", "0\n0\n0\n"),
    !.

test(length_of_the_empty_string, [condition(clang_available)]) :-
    run("{ print length(\"\") }\n", "0\n0\n0\n"),
    !.

% Case mapping touches letters only -- digits and punctuation pass through.
test(case_mapping_leaves_non_letters_alone, [condition(clang_available)]) :-
    run("{ print toupper(\"aBc123!\") }\n", "ABC123!\nABC123!\nABC123!\n"),
    !,
    run("{ print tolower(\"AbC123!\") }\n", "abc123!\nabc123!\nabc123!\n"),
    !.

test(length_counts_spaces, [condition(clang_available)]) :-
    run("{ print length(\"a b\") }\n", "3\n3\n3\n"),
    !.

% --- the fold compiles to what the literal compiles to -------------------
%
% The claim that made this change free is that folding lands in a representation the
% routes ALREADY have. If that is true, the folded program and the hand-written
% literal must produce the same AST -- not merely the same output. Asserting the AST
% (and then the IR) is what distinguishes "reuses the existing path" from "happens to
% agree today", and it is the property that would break first if someone later gave
% the fold its own emitter.
test(a_folded_answer_parses_to_the_same_ast_as_the_literal) :-
    same_ast("{ print length(\"abc\") }\n", "{ print 3 }\n"),
    same_ast("{ n++ } END { print length(\"abc\") }\n", "{ n++ } END { print 3 }\n"),
    same_ast("{ print toupper(\"abc\") }\n", "{ print \"ABC\" }\n"),
    same_ast("{ print substr(\"hello\", 2, 3) }\n", "{ print \"ell\" }\n"),
    same_ast("{ print length(\"abc\") + 1 }\n", "{ print 3 + 1 }\n"),
    same_ast("{ n += length(\"abcd\") } END { print n }\n",
             "{ n += 4 } END { print n }\n"),
    !.

test(a_folded_answer_emits_the_same_ir_as_the_literal) :-
    same_ir("{ print length(\"abc\") }\n", "{ print 3 }\n"),
    same_ir("{ n++ } END { print length(\"abc\") }\n", "{ n++ } END { print 3 }\n"),
    same_ir("{ n++; c[$1]++ } END { print length(\"abc\"), c[\"5\"] }\n",
            "{ n++; c[$1]++ } END { print 3, c[\"5\"] }\n"),
    !.

% A printed integer literal takes its DECIMAL spelling, which is what the grammar
% already does for `print 3` -- and only at the TOP level of a print field list. The
% operands of `print 3 + 1` are one computed field and must stay `int/1`, so this pins
% that the printed-position rewrite does not recurse into the arithmetic term.
test(the_printed_rewrite_does_not_reach_into_an_arithmetic_field) :-
    plawk_parse_string("{ print 3 + 1 }\n", Direct),
    assertion(Direct == program([], [rule(always, [print([add_i64(int(3), int(1))])])], [])),
    plawk_parse_string("{ print length(\"abc\") + 1 }\n", Folded),
    assertion(Folded == Direct),
    !.

% --- the ASCII boundary, and why it is a boundary ------------------------

% plawk's runtime counts and slices BYTES and maps case over ASCII only. In this
% container's POSIX locale gawk agrees, and that agreement was verified directly
% (`length($1)` on "café" is 5 in both, and `toupper($1)` leaves the é alone in
% both). A fold restricted to code points 0..127 is byte-exact by construction; a
% fold that counted code points while the runtime counts bytes would disagree with
% the runtime for exactly the inputs nobody tests.
%
% So a literal with a byte above 127 is left UNFOLDED and declines (exit 3 -- it
% parsed, the fold declined to answer it). Refusing is the honest direction: the
% alternative failure is a silently wrong constant, which is the one class of bug
% this campaign treats as worst.
test(the_fold_is_ascii_only_by_design) :-
    build_status("{ print length(\"café\") }\n", 3),
    build_status("{ print toupper(\"café\") }\n", 3),
    build_status("{ print substr(\"café\", 1, 2) }\n", 3),
    !.

% ...while the same operations on a FIELD holding the same bytes still work, because
% the runtime does them in bytes. This is the pairing that shows the boundary is the
% fold's and not the language's.
%
% Asserted in BYTES, not text. Every other expectation in this campaign is ASCII, so
% comparing decoded strings has always been the same thing as comparing bytes; here it
% is not. The runtime emits 0xC3 0xA9 for the é and gawk emits the identical pair
% (checked with od -c), but reading that through a text-decoding pipe and comparing
% against a source-level "é" compares a decode of the output against a decode of the
% test file, which is a different claim and fails for encoding reasons rather than
% behavioural ones. A byte-oriented runtime needs a byte-oriented assertion.
test(non_ascii_still_works_through_a_field, [condition(clang_available)]) :-
    run_bytes("café\n", "{ print length($1) }\n", [0'5, 0'\n]),
    !,
    run_bytes("café\n", "{ print toupper($1) }\n",
        [0'C, 0'A, 0'F, 0xC3, 0xA9, 0'\n]),
    !.

% --- the vocabulary boundary, pinned with its kind -----------------------
%
% Two DIFFERENT refusals, and the exit code separates them:
%
%   exit 2 (parse)   -- the argument is outside the vocabulary
%                       (plawk_builtin_string_arg/1 takes a field or a literal).
%   exit 3 (decline) -- the argument is IN the vocabulary but the fold cannot
%                       answer it exactly (the non-ASCII case above).
%
% A nested call is unreachable surface today. The fold is written bottom-up so it
% would collapse `length(toupper("ab"))` in one pass if the vocabulary ever admitted
% a call, which is why this is pinned as a vocabulary boundary and not as a fold gap.
test(a_nested_call_is_outside_the_argument_vocabulary) :-
    build_status("{ print length(toupper(\"ab\")) }\n", 2),
    !.

test(a_variable_argument_is_outside_the_argument_vocabulary) :-
    build_status("{ print length(v) }\n", 2),
    build_status("{ v = $2; print length(v) }\n", 2),
    !.

% `int("3.7")` is deliberately NOT in this vocabulary. int() over a string is numeric
% coercion with strtod prefix semantics ("3.7abc" is 3, "abc" is 0), which none of the
% folds implement -- so int_field_expr//1 keeps the field-only guard rather than being
% widened to something no fold answers. A refusal that matches what is implemented.
test(int_over_a_literal_is_still_refused) :-
    build_status("{ print int(\"3.7\") }\n", 2),
    !.

test(int_over_a_field_is_unchanged, [condition(clang_available)]) :-
    run("{ print int($1) }\n", "5\n5\n7\n"),
    !.

% --- a surface builtin name is never a Prolog call -----------------------
%
% The generic `name(args)` production used to capture any reserved builtin name whose
% arguments happened to be foreign-argument shaped. That is what made
% `length("abc")` a DECLINE rather than a parse error, and it meant two paths
% disagreed about the same text: a bare print field reached the `length` production
% first, an arithmetic operand reached the generic call first. Now excluded by
% plawk_surface_reserved_name/1 -- the list that already guarded DYNENTRY names.
%
% Pinned by outcome, not by structure: `length("abc")` must be the BUILTIN in every
% position, so the arithmetic form and the bare form must agree.
test(a_reserved_builtin_name_is_not_parsed_as_a_prolog_call) :-
    plawk_parse_string("{ print length(\"abc\") }\n", Bare),
    assertion(Bare == program([], [rule(always, [print([string("3")])])], [])),
    plawk_parse_string("{ print length(\"abc\") + 1 }\n", Arith),
    assertion(Arith == program([], [rule(always, [print([add_i64(int(3), int(1))])])], [])),
    !.

% --- regressions: the field argument, which always worked ---------------

test(field_arguments_unchanged, [condition(clang_available)]) :-
    run("{ print length($1) }\n", "1\n1\n1\n"),
    !,
    run("{ print length($0) }\n", "6\n7\n6\n"),
    !,
    run("{ print index($2, \"o\") }\n", "2\n0\n0\n"),
    !,
    run("{ print substr($2, 2, 3) }\n", "oot\nrac\nisk\n"),
    !,
    run("{ print toupper($1), tolower($2) }\n", "5 boot\n5 trace\n7 disk\n"),
    !.

test(field_arguments_in_arithmetic_unchanged, [condition(clang_available)]) :-
    run("{ print length($0) - 3 }\n", "3\n4\n3\n"),
    !.

test(bare_length_unchanged, [condition(clang_available)]) :-
    run("{ print length }\n", "6\n7\n6\n"),
    !.

:- end_tests(plawk_literal_builtin_fold).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_literal_builtin_fold', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'lb_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'lb', Prog0),
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

% Like run_with/3 but compares the program's stdout as BYTES. The pipe is switched to
% octet before the first read, so each character of Out is one byte of output.
run_bytes(Input, Src, ExpectedCodes) :-
    odir(Dir),
    directory_file_path(Dir, 'lb_bytes_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'lb_bytes', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', Bin], 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    set_stream(PS, encoding(octet)),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    string_codes(Out, Codes),
    (   Codes == ExpectedCodes
    ->  true
    ;   format(user_error, "~n~w~n  got bytes      ~q~n  expected bytes ~q~n",
            [Src, Codes, ExpectedCodes]), fail
    ).

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'lb_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

same_ast(FoldedSrc, LiteralSrc) :-
    plawk_parse_string(FoldedSrc, Folded),
    plawk_parse_string(LiteralSrc, Literal),
    (   Folded == Literal
    ->  true
    ;   format(user_error, "~nAST differs:~n  ~w -> ~q~n  ~w -> ~q~n",
            [FoldedSrc, Folded, LiteralSrc, Literal]), fail
    ).

same_ir(FoldedSrc, LiteralSrc) :-
    build_ll(FoldedSrc, Folded),
    build_ll(LiteralSrc, Literal),
    (   Folded == Literal
    ->  true
    ;   format(user_error, "~nIR differs for ~w vs ~w~n", [FoldedSrc, LiteralSrc]),
        fail
    ).

build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
