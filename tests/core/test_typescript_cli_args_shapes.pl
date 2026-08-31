:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_typescript_cli_args_shapes.pl
%
% Compile-shape suite for the Prolog constructs that `examples/cli_args/cli_args.pl`
% (step A1 of the transpilation maturity demo) actually uses, pushed through the
% pattern targets typescript / vanilla_js / annotated_js.
%
% It has two halves:
%
%   1. REGRESSION tests for the A3 fixes that landed in typescript_target.pl and
%      annotated_js_target.pl (G-A3-1..G-A3-5, G-A3-7, G-A3-8, G-A3-11,
%      G-A3-13..G-A3-15, G-A3-17). These assert CORRECT lowering and must stay
%      green.
%
%   2. GAP PROBES for the constructs the pattern targets still cannot lower
%      (G-A3-6, G-A3-9, G-A3-10, G-A3-12, G-A3-16). Each probe is an executable
%      minimal reproduction that pins the CURRENT behaviour and carries a comment
%      saying what correct lowering would be. They are written so that CLOSING
%      the gap makes the probe FAIL -- that is the cue to come back here and
%      promote the probe into a real assertion.
%
% Where a fix changes the SHAPE of the emitted JavaScript, the test runs
% `node --check` over the whole emitted module (and, where an oracle is cheap,
% runs it and compares against SWI). A3's headline lesson was that substring
% assertions let unparseable output ship: `return const arg2 = ...;;` passed
% every has/2 check in the suite for as long as it existed.
%
% See docs/proposals/A3_PATTERN_TRANSPILE_REPORT.md for the full catalogue.
%
% Run: swipl -q -g test_typescript_cli_args_shapes -t halt
%            tests/core/test_typescript_cli_args_shapes.pl

:- module(test_typescript_cli_args_shapes, [test_typescript_cli_args_shapes/0]).
:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module('../../src/unifyweaver/targets/typescript_target').
:- use_module('../../src/unifyweaver/targets/annotated_js_target', []).
:- use_module('../../src/unifyweaver/targets/vanilla_js_target', []).

test_typescript_cli_args_shapes :-
    run_tests([typescript_cli_args_shapes]).

:- begin_tests(typescript_cli_args_shapes).

has(Code, Substr)   :- once(sub_string(Code, _, _, _, Substr)).
hasnt(Code, Substr) :- \+ sub_string(Code, _, _, _, Substr).

node_available :-
    catch(( process_create(path(node), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

%% node_check(+Src)
%  True when node can PARSE Src as an ES module. The point of every
%  `node --check` in this file: a has/2 assertion cannot tell correct output from
%  output node refuses to load.
node_check(Src) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.mjs', File),
    setup_call_cleanup(
        ( open(File, write, W), write(W, Src), close(W) ),
        ( process_create(path(node), ['--check', File],
                         [stdout(null), stderr(null), process(P)]),
          process_wait(P, exit(0)) ),
        catch(delete_file(File), _, true)).

%% node_run(+Src, +Argv, -Output)
%  Run Src under node with Argv and capture stdout verbatim.
node_run(Src, Argv, Output) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.mjs', File),
    setup_call_cleanup(
        ( open(File, write, W), write(W, Src), close(W) ),
        ( process_create(path(node), [File|Argv],
                         [stdout(pipe(O)), stderr(null), process(P)]),
          read_string(O, _, Output), close(O), process_wait(P, _) ),
        catch(delete_file(File), _, true)).

%% node_run_lines(+Src, +Argv, -Lines)
node_run_lines(Src, Argv, Lines) :-
    node_run(Src, Argv, Out),
    split_string(Out, "\n", "", Raw),
    exclude(==(""), Raw, Lines).

%% native_body(+Pred/Arity, -Code)
%  Compile ONLY through typescript_target's native clause-body path.
%
%  Historically this existed because routing a probe through compile_predicate/3
%  was unsafe: when every native path refused, the dispatcher fell back to
%  compile_facts/3, which EXECUTED the predicate to enumerate its solutions
%  (G-A3-8) -- an instantiation error at best, an unbounded findall that ate the
%  test runner's memory at worst. G-A3-8 is closed and compile_predicate/3 is now
%  safe on any shape; this helper stays because it isolates ONE lowering path,
%  which is what most of these assertions are about.
native_body(Pred/Arity, Code) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    once(typescript_target:native_ts_clause_body(Pred/Arity, Clauses, Code)).

native_structural(Pred/Arity, Code) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    once(typescript_target:native_ts_structural(Pred/Arity, Clauses, Code)).

% ============================================================================
% PART 1 -- regressions for the fixes that landed
% ============================================================================

% ---------------------------------------------------------------------------
% G-A3-2 : statement-block clause bodies were wrapped in `return ...;`
% ---------------------------------------------------------------------------
% Before the fix the simplest possible transform predicate compiled to
%     return const arg2 = (arg1 * 2);
%       return arg2;;
% which node refuses to parse. Nothing cli_args-specific about it -- this is
% the target's flagship batch path.

assert_a3_doub :- assertz((user:a3_doub(X, Y) :- Y is X * 2)).
retract_a3_doub :- retractall(user:a3_doub(_, _)).

test(g_a3_2_statement_block_is_not_wrapped_in_return,
     [setup(assert_a3_doub), cleanup(retract_a3_doub)]) :-
    native_body(a3_doub/2, Code),
    hasnt(Code, "return const"),
    has(Code, "const arg2 = (arg1 * 2);"),
    has(Code, "return arg2;").

% A block that renders no `return` at all means goals were dropped; it must say
% so out loud rather than fall off the end of the function.
assert_a3_dropped :-
    assertz((user:a3_dropped(X, Y) :- string_length(X, _L),
                                      a3_unknown_helper(X, Y))).
retract_a3_dropped :- retractall(user:a3_dropped(_, _)).

test(g_a3_2_returnless_block_throws,
     [setup(assert_a3_dropped), cleanup(retract_a3_dropped)]) :-
    native_body(a3_dropped/2, Code),
    has(Code, "incomplete lowering").

% ---------------------------------------------------------------------------
% G-A3-1 : deterministic string / char builtins
% ---------------------------------------------------------------------------

assert_a3_strings :-
    assertz((user:a3_len(S, L)            :- string_length(S, L))),
    assertz((user:a3_cat(A, B, C)         :- string_concat(A, B, C))),
    assertz((user:a3_chars(S, Cs)         :- string_chars(S, Cs))),
    assertz((user:a3_unchars(Cs, S)       :- string_chars(S, Cs))),
    assertz((user:a3_unchars_mid(Cs, Out) :- string_chars(S, Cs), Out = S)),
    assertz((user:a3_code(C, X)           :- char_code(C, X))),
    assertz((user:a3_sub(S, N, Sub)       :- sub_string(S, 0, N, _, Sub))),
    assertz((user:a3_upper(S, U)          :- string_upper(S, U))).
retract_a3_strings :-
    retractall(user:a3_len(_, _)),
    retractall(user:a3_cat(_, _, _)),
    retractall(user:a3_chars(_, _)),
    retractall(user:a3_unchars(_, _)),
    retractall(user:a3_unchars_mid(_, _)),
    retractall(user:a3_code(_, _)),
    retractall(user:a3_sub(_, _, _)),
    retractall(user:a3_upper(_, _)).

test(g_a3_1_string_length,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_len/2, Code), has(Code, "arg1.length").

test(g_a3_1_string_concat,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_cat/3, Code), has(Code, "(arg1 + arg2)").

test(g_a3_1_string_chars_decompose,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_chars/2, Code), has(Code, "Array.from(arg1)").

% Reverse mode: string_chars(-Text, +Chars) BUILDS the text, and its output is
% the goal's FIRST argument. That direction is chosen when the text variable is
% not yet in the VarMap; see the G-A3-15 probe for the ambiguous case.
test(g_a3_1_string_chars_compose,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_unchars_mid/2, Code), has(Code, ".join(\"\")").

test(g_a3_1_char_code,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_code/2, Code), has(Code, "arg1.charCodeAt(0)").

test(g_a3_1_sub_string,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_sub/3, Code), has(Code, "arg1.slice(0, 0 + arg2)").

test(g_a3_1_string_upper,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_upper/2, Code), has(Code, "arg1.toUpperCase()").

% cli_args' real substring helpers, verbatim.
assert_a3_substr :-
    assertz((user:a3_substring_from(String, Start, Sub) :-
                string_length(String, L), Len is L - Start,
                sub_string(String, Start, Len, 0, Sub))),
    assertz((user:a3_substring_range(String, Start, End, Sub) :-
                Len is End - Start,
                sub_string(String, Start, Len, _, Sub))).
retract_a3_substr :-
    retractall(user:a3_substring_from(_, _, _)),
    retractall(user:a3_substring_range(_, _, _, _)).

test(g_a3_1_cli_args_substring_from,
     [setup(assert_a3_substr), cleanup(retract_a3_substr)]) :-
    native_body(a3_substring_from/3, Code),
    has(Code, "arg1.length"),
    has(Code, "arg1.slice(arg2, arg2 + "),
    has(Code, "return arg3;"),
    hasnt(Code, "incomplete lowering").

test(g_a3_1_cli_args_substring_range,
     [setup(assert_a3_substr), cleanup(retract_a3_substr)]) :-
    native_body(a3_substring_range/4, Code),
    has(Code, "(arg3 - arg2)"),
    has(Code, "arg1.slice(arg2, arg2 + "),
    hasnt(Code, "incomplete lowering").

% ---------------------------------------------------------------------------
% G-A3-3 : `,` / `;` / `->` inside a GUARD position
% ---------------------------------------------------------------------------
% cli_args' two character classifiers are code-point range chains.

assert_a3_flagchar :-
    assertz((user:a3_flag_char(C) :-
                char_code(C, X),
                (   X >= 0'a, X =< 0'z -> true
                ;   X >= 0'A, X =< 0'Z -> true
                ;   X >= 0'0, X =< 0'9 -> true
                ;   X =:= 0'-
                ))).
retract_a3_flagchar :- retractall(user:a3_flag_char(_)).

test(g_a3_3_conjunctive_guard_renders,
     [setup(assert_a3_flagchar), cleanup(retract_a3_flagchar)]) :-
    native_body(a3_flag_char/1, Code),
    has(Code, "arg1.charCodeAt(0)"),
    has(Code, ">= 97"), has(Code, "<= 122"),
    has(Code, ">= 65"), has(Code, "<= 90"),
    has(Code, ">= 48"), has(Code, "<= 57"),
    has(Code, "=== 45"),
    hasnt(Code, "incomplete lowering").

% An if-then-else whose branches produce a value still lowers to a value.
assert_a3_class :-
    assertz((user:a3_class(C, R) :-
                char_code(C, X),
                ( X >= 0'a, X =< 0'z -> R = lower ; R = other ))).
retract_a3_class :- retractall(user:a3_class(_, _)).

test(g_a3_3_ite_over_conjunctive_condition,
     [setup(assert_a3_class), cleanup(retract_a3_class)]) :-
    native_body(a3_class/2, Code),
    has(Code, ">= 97"),
    has(Code, "return \"lower\";"),
    has(Code, "return \"other\";").

% ---------------------------------------------------------------------------
% G-A3-4 : a goal with no rendering must not be silently deleted
% ---------------------------------------------------------------------------
% `strip_brackets/2` used to compile with its whole `drop_brackets/2` call
% erased, leaving a function that read an undefined variable and returned it.

assert_a3_strip :-
    assertz((user:a3_strip(String, Stripped) :-
                string_chars(String, Chars),
                a3_drop(Chars, Kept),
                string_chars(Stripped, Kept))).
retract_a3_strip :- retractall(user:a3_strip(_, _)).

test(g_a3_4_unrendered_user_goal_is_loud,
     [setup(assert_a3_strip), cleanup(retract_a3_strip)]) :-
    native_body(a3_strip/2, Code),
    has(Code, "incomplete lowering: unrendered goal a3_drop/2").

% ...and the fallback must NOT fire for goals that do render: a guard-only
% clause still takes the guard/output split path and yields a plain condition.
assert_a3_qpos :- assertz((user:a3_qpos(X) :- integer(X), X > 0)).
retract_a3_qpos :- retractall(user:a3_qpos(_)).

test(g_a3_4_fallback_does_not_shadow_working_guards,
     [setup(assert_a3_qpos), cleanup(retract_a3_qpos)]) :-
    native_body(a3_qpos/1, Code),
    has(Code, "Number.isInteger(arg1)"),
    has(Code, "arg1 > 0"),
    hasnt(Code, "incomplete lowering").

% ---------------------------------------------------------------------------
% G-A3-5 : the "guarded tail" renderer discarded everything after the guards
% ---------------------------------------------------------------------------
% `starts_with/2` is output, output, guard, output, guard. The old renderer
% stopped at the first guard run and emitted a function that returned the
% PREFIX LENGTH, never touching the substring comparison at all.

assert_a3_starts :-
    assertz((user:a3_starts_with(String, Prefix) :-
                string_length(String, L), string_length(Prefix, N),
                L >= N,
                sub_string(String, 0, N, _, Sub),
                Sub == Prefix)).
retract_a3_starts :- retractall(user:a3_starts_with(_, _)).

test(g_a3_5_goals_after_a_guard_run_are_not_discarded,
     [setup(assert_a3_starts), cleanup(retract_a3_starts)]) :-
    native_body(a3_starts_with/2, Code),
    has(Code, "arg1.slice(0, 0 + "),
    has(Code, "=== arg2").

% ---------------------------------------------------------------------------
% G-A3-17 : the guard/output split path threw away intermediate assignments
% ---------------------------------------------------------------------------
% ts_output_goals/3 used to thread only the VarMap through every non-final
% output goal and discard its `const ...;` line, so the returned expression
% referenced variables that were never declared.

test(g_a3_17_intermediate_assignments_survive,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_unchars_mid/2, Code),
    has(Code, "const v3 = arg1.join(\"\");"),
    has(Code, "return v3;").

% ---------------------------------------------------------------------------
% G-A3-7 : annotated_js mistook `const x = (expr);` for an arrow signature
% ---------------------------------------------------------------------------

test(g_a3_7_annotated_js_handles_parenthesised_assignment) :-
    atomic_list_concat(
        ["function f(arg1: number, arg2: number): string {",
         "    const v5 = (arg1 - arg2);",
         "    return v5;",
         "}", ""], '\n', TS),
    annotated_js_target:ts_to_annotated_js(TS, JS),
    has(JS, "const v5 = (arg1 - arg2);"),
    has(JS, "function f(arg1, arg2)"),
    has(JS, "@param {number} arg1").

% ---------------------------------------------------------------------------
% Inheritance: vanilla_js and annotated_js carry the same lowering
% ---------------------------------------------------------------------------

test(inheritance_vanilla_js_matches_typescript_body,
     [setup(assert_a3_substr), cleanup(retract_a3_substr)]) :-
    typescript_target:compile_predicate_to_typescript(a3_substring_from/3, [], TsCode),
    vanilla_js_target:compile_predicate_to_vanilla_js(a3_substring_from/3, [], JsCode),
    has(TsCode, "arg1.slice(arg2, arg2 + "),
    has(JsCode, "arg1.slice(arg2, arg2 + "),
    % the only difference is the erased type syntax
    has(TsCode, "arg1: number"),
    hasnt(JsCode, "arg1: number").

test(inheritance_annotated_js_matches_typescript_body,
     [setup(assert_a3_substr), cleanup(retract_a3_substr)]) :-
    annotated_js_target:compile_predicate(a3_substring_from/3, [], AjsCode),
    has(AjsCode, "arg1.slice(arg2, arg2 + "),
    has(AjsCode, "@param {number} arg1").

% ---------------------------------------------------------------------------
% End-to-end: the emitted substring functions run under node and agree with SWI
% ---------------------------------------------------------------------------
% This drives the compiled function directly, over more cases than the CLI can
% conveniently express (including "" -producing ones). The compiler's OWN entry
% point is exercised separately by
% g_a3_11_compiler_emitted_cli_entry_drives_substring_from, which no longer
% needs a hand-written driver now that G-A3-11.2 is closed.

test(compiled_substring_from_runs_under_node,
     [setup(assert_a3_substr), cleanup(retract_a3_substr),
      condition(node_available)]) :-
    vanilla_js_target:compile_predicate_to_vanilla_js(a3_substring_from/3, [], C1),
    node_check(C1),
    drop_cli_entry(C1, Body),
    Cases = ["--state=alpha"-2, "hello"-0, "hello"-5, "--"-2, "abcdef"-3],
    findall(Line,
            ( member(S, Cases), S = Str-Idx,
              format(string(Line),
                     "console.log(JSON.stringify(a3_substring_from(~q, ~w)));",
                     [Str, Idx]) ),
            CallLines),
    atomic_list_concat(CallLines, '\n', Driver),
    atomic_list_concat([Body, '\n', Driver, '\n'], Src),
    findall(Expected,
            ( member(S2-I2, Cases), user:a3_substring_from(S2, I2, Expected) ),
            ExpectedList),
    node_lines(Src, GotList),
    GotList == ExpectedList.

drop_cli_entry(Code, Body) :-
    (   sub_string(Code, Before, _, _, "// CLI entry point")
    ->  sub_string(Code, 0, Before, _, Body)
    ;   Body = Code
    ).

node_lines(Src, Lines) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.mjs', File),
    setup_call_cleanup(
        ( open(File, write, W), write(W, Src), close(W) ),
        ( process_create(path(node), [File],
                         [stdout(pipe(O)), stderr(null), process(P)]),
          read_string(O, _, Out), close(O), process_wait(P, _) ),
        catch(delete_file(File), _, true)),
    split_string(Out, "\n", "", Raw0),
    exclude(==(""), Raw0, Raw),
    maplist(unquote_json_string, Raw, Lines).

unquote_json_string(In, Out) :-
    string_length(In, L), L >= 2, L1 is L - 2,
    sub_string(In, 1, L1, 1, Out).

% ---------------------------------------------------------------------------
% G-A3-8 : the last-resort fallback must NEVER execute the predicate
% ---------------------------------------------------------------------------
% compile_predicate/3's last resort was compile_facts/3, which enumerates rows
% with `findall(..., (functor(G,...), call(G), ...), Facts)` -- it RUNS the
% predicate with every argument unbound. Measured over the 17 cli_args
% predicates that reached it: 4 semantically wrong fact tables, 10
% instantiation_error / resource_error(stack), and 3 unbounded findalls killed
% at a 20 s / 1.5 GB cap.
%
% The fallback now applies only to genuine fact predicates (every clause a
% ground fact, decided SYNTACTICALLY from the clause database), and refuses
% loudly otherwise. These are the shapes that used to hang or lie.

% The three that had to be killed, verbatim from cli_args.pl, plus the two that
% produced wrong output.
assert_a3_hazards :-
    assertz(user:a3_flags_put([], K, V, [K-V])),
    assertz((user:a3_flags_put([K0-V0|R], K, V, Out) :-
                (   K0 == K
                ->  Out = [K0-V|R]
                ;   Out = [K0-V0|R1], a3_flags_put(R, K, V, R1)
                ))),
    assertz(user:a3_merge_flags([], Base, Base)),
    assertz((user:a3_merge_flags([K-V|Rest], Base, Merged) :-
                a3_flags_set(Base, K, V, Base1),
                a3_merge_flags(Rest, Base1, Merged))),
    assertz(user:a3_drop_brackets([], [])),
    assertz((user:a3_drop_brackets([C|Cs], Kept) :-
                (   ( C == '[' ; C == ']' )
                ->  Kept = Kept1
                ;   Kept = [C|Kept1]
                ),
                a3_drop_brackets(Cs, Kept1))),
    assertz((user:a3_string_member(S, [X|Xs]) :-
                ( S == X -> true ; a3_string_member(S, Xs) ))),
    assertz((user:a3_is_global_key(Key) :-
                ( a3_globals(G), a3_pair_lookup(G, Key, _) -> true
                ; a3_proto_key(Key) ))).
retract_a3_hazards :-
    retractall(user:a3_flags_put(_, _, _, _)),
    retractall(user:a3_merge_flags(_, _, _)),
    retractall(user:a3_drop_brackets(_, _)),
    retractall(user:a3_string_member(_, _)),
    retractall(user:a3_is_global_key(_)).

% a3_string_member/2 was in this list until G-A3-10 closed. It is a semidet
% list walk whose single clause is `( S == X -> true ; recurse )` -- exactly the
% ITE+recursion shape the structural path now lowers, so it COMPILES rather than
% refuses. Its correctness is asserted in the G-A3-10 section below; the four
% left here still have no lowering path (list-BUILDING recursion, an accumulator
% loop through a helper, and an arity-1 ITE over two helper calls).
a3_hazard_shapes([a3_flags_put/4, a3_merge_flags/3, a3_drop_brackets/2,
                  a3_is_global_key/1]).

%% a3_compile_outcome(+PredSpec, -Outcome)
%  refused(Spec, Shape, Msg) | compiled(Code) | failed. Never lets a refusal
%  escape as an exception, so a test can assert on WHICH outcome happened.
a3_compile_outcome(PredSpec, Outcome) :-
    catch(
        (   typescript_target:compile_predicate(PredSpec, [], Code)
        ->  Outcome = compiled(Code)
        ;   Outcome = failed
        ),
        error(unsupported_lowering(typescript, Spec, Shape), Msg),
        Outcome = refused(Spec, Shape, Msg)).

% The headline: the shapes that hung the compiler now refuse, and do it fast.
% 5 s for all five is three orders of magnitude clear of the 20 s at which the
% original runaways were killed, while staying insensitive to machine speed.
test(g_a3_8_runaway_shapes_refuse_fast,
     [setup(assert_a3_hazards), cleanup(retract_a3_hazards)]) :-
    a3_hazard_shapes(Shapes),
    get_time(T0),
    forall(member(P/A, Shapes),
           (   a3_compile_outcome(P/A, Outcome),
               Outcome = refused(P/A, _Shape, Msg),
               has(Msg, "cannot compile"),
               has(Msg, "must never execute it")
           )),
    get_time(T1),
    Elapsed is T1 - T0,
    Elapsed < 5.0.

% The diagnostic has to be actionable: it names the predicate AND the clause
% shape that disqualified it.
test(g_a3_8_refusal_names_the_predicate_and_the_body_shape,
     [setup(assert_a3_hazards), cleanup(retract_a3_hazards)]) :-
    a3_compile_outcome(a3_drop_brackets/2, refused(Spec, Shape, Msg)),
    Spec == a3_drop_brackets/2,
    has(Shape, "clause 2 of 2 is a RULE, not a fact"),
    has(Shape, "if-then-else"),
    has(Msg, "a3_drop_brackets/2").

% A single-clause rule predicate (no fact clause at all) is refused too.
assert_a3_ruleonly :-
    assertz((user:a3_rule_only(X, Y) :- a3_rule_helper(X, Y))).
retract_a3_ruleonly :- retractall(user:a3_rule_only(_, _)).

test(g_a3_8_rule_only_predicate_is_refused,
     [setup(assert_a3_ruleonly), cleanup(retract_a3_ruleonly)]) :-
    % Ask the guard directly, so the assertion is about the guard rather than
    % about which lowering path happens to claim the predicate first. It decides
    % from the clause database and never calls a3_rule_only/2.
    catch(typescript_target:ts_require_fact_predicate(a3_rule_only, 2),
          error(unsupported_lowering(typescript, Spec, Shape), _),
          true),
    Spec == a3_rule_only/2,
    has(Shape, "is a RULE, not a fact"),
    has(Shape, "a3_rule_helper/2").

% A non-ground fact would make compile_facts/3 emit rows containing internal
% `_G` names, so it is refused as well.
assert_a3_nonground :- assertz(user:a3_nonground(_X, b)).
retract_a3_nonground :- retractall(user:a3_nonground(_, _)).

test(g_a3_8_non_ground_fact_is_refused,
     [setup(assert_a3_nonground), cleanup(retract_a3_nonground)]) :-
    catch(typescript_target:compile_facts(a3_nonground, 2, _),
          error(unsupported_lowering(typescript, Spec, Shape), _),
          true),
    Spec == a3_nonground/2,
    has(Shape, "non-ground fact").

% A built-in's clauses cannot be inspected, so facts cannot be told from rules
% without running it -- refused rather than executed. (compile_facts(atom_length,
% 2) would otherwise have called atom_length/2 with both arguments unbound.)
test(g_a3_8_uninspectable_builtin_is_refused) :-
    catch(typescript_target:compile_facts(atom_length, 2, _),
          error(unsupported_lowering(typescript, Spec, Shape), _),
          true),
    Spec == atom_length/2,
    has(Shape, "cannot be inspected").

% An undefined predicate is refused too: the old fallback called it and got an
% existence_error out of findall/3, or, worse, silently emitted an empty table.
test(g_a3_8_undefined_predicate_is_refused) :-
    catch(typescript_target:compile_facts(a3_no_such_predicate, 2, _),
          error(unsupported_lowering(typescript, Spec, Shape), _),
          true),
    Spec == a3_no_such_predicate/2,
    has(Shape, "no clauses at all").

% ...and a genuine fact predicate is untouched.
assert_a3_realfacts :-
    assertz(user:a3_colour(red)),
    assertz(user:a3_colour(blue)).
retract_a3_realfacts :- retractall(user:a3_colour(_)).

test(g_a3_8_genuine_fact_predicate_still_compiles,
     [setup(assert_a3_realfacts), cleanup(retract_a3_realfacts)]) :-
    typescript_target:compile_facts(a3_colour, 1, Code),
    has(Code, "export const a3_colourFacts"),
    has(Code, "\"red\""),
    has(Code, "\"blue\"").

% Inheritance: annotated_js and vanilla_js re-export typescript_target's
% compile_facts/3 and compile_predicate/3 verbatim, so the guard has to reach
% them too -- checked per target rather than assumed (that assumption is how
% G-A3-7 got in).
test(g_a3_8_vanilla_js_inherits_the_guard,
     [setup(assert_a3_hazards), cleanup(retract_a3_hazards)]) :-
    catch(vanilla_js_target:compile_facts(a3_drop_brackets, 2, _),
          error(unsupported_lowering(typescript, Spec1, _), _), true),
    Spec1 == a3_drop_brackets/2,
    catch(vanilla_js_target:compile_predicate_to_vanilla_js(a3_flags_put/4, [], _),
          error(unsupported_lowering(typescript, Spec2, _), _), true),
    Spec2 == a3_flags_put/4.

test(g_a3_8_annotated_js_inherits_the_guard,
     [setup(assert_a3_hazards), cleanup(retract_a3_hazards)]) :-
    catch(annotated_js_target:compile_facts(a3_drop_brackets, 2, _),
          error(unsupported_lowering(typescript, Spec1, _), _), true),
    Spec1 == a3_drop_brackets/2,
    catch(annotated_js_target:compile_predicate(a3_flags_put/4, [], _),
          error(unsupported_lowering(typescript, Spec2, _), _), true),
    Spec2 == a3_flags_put/4.

% ---------------------------------------------------------------------------
% G-A3-13 : the boolean atoms lower to JS booleans
% ---------------------------------------------------------------------------
% Every other Prolog atom is a JS string in this target, so `true`/`false` used
% to collapse into the strings "true"/"false". cli_args' corpus asserts
% flags["include-key"] === true -- a boolean that must stay distinct from the
% string "true" a `--x=true` value produces.

test(g_a3_13_boolean_atoms_are_js_booleans) :-
    typescript_target:ts_literal(true, LitT),
    typescript_target:ts_literal(false, LitF),
    LitT == 'true',
    LitF == 'false'.

assert_a3_boolout :-
    assertz((user:a3_boolout(X, Y) :- ( X > 0 -> Y = true ; Y = false ))).
retract_a3_boolout :- retractall(user:a3_boolout(_, _)).

test(g_a3_13_boolean_branch_values_are_unquoted,
     [setup(assert_a3_boolout), cleanup(retract_a3_boolout)]) :-
    native_body(a3_boolout/2, Code),
    has(Code, "return true;"),
    has(Code, "return false;"),
    hasnt(Code, "\"true\""),
    hasnt(Code, "\"false\"").

% The distinction only matters if it survives to runtime: JSON.stringify of the
% result must be `true`, not `"true"`.
test(g_a3_13_boolean_result_is_a_boolean_under_node,
     [setup(assert_a3_boolout), cleanup(retract_a3_boolout),
      condition(node_available)]) :-
    vanilla_js_target:compile_predicate_to_vanilla_js(a3_boolout/2, [], Code),
    node_check(Code),
    drop_cli_entry(Code, Body),
    atomic_list_concat(
        [Body, '\nconsole.log(JSON.stringify(a3_boolout(4)));',
               '\nconsole.log(JSON.stringify(a3_boolout(-4)));\n'], Src),
    node_run_lines(Src, [], Lines),
    Lines == ["true", "false"].

% A boolean-valued FACT cell is unquoted too, exactly as a numeric cell already
% was.
assert_a3_boolfact :-
    assertz(user:a3_boolfact(alpha, true)),
    assertz(user:a3_boolfact(beta, false)).
retract_a3_boolfact :- retractall(user:a3_boolfact(_, _)).

test(g_a3_13_boolean_fact_cell_is_unquoted,
     [setup(assert_a3_boolfact), cleanup(retract_a3_boolfact)]) :-
    typescript_target:compile_facts(a3_boolfact, 2, Code),
    has(Code, "arg2: true"),
    has(Code, "arg2: false"),
    hasnt(Code, "arg2: \"true\"").

% ---------------------------------------------------------------------------
% G-A3-14 : no internal `_G` name ever reaches the emitted JavaScript
% ---------------------------------------------------------------------------
% ts_expr/3's last resort for an unmapped variable was term_string/2, which put
% SWI's `_41598` -- declared nowhere, different on every run -- straight into
% the output. The variable that triggered it in the wild was one bound inside an
% if-then-else chain and read afterwards; the mid-sequence ITE renderer threw
% its VarMap away and emitted `return` instead of an assignment.

test(g_a3_14_unmapped_variable_is_refused_not_leaked) :-
    \+ typescript_target:ts_expr(_Free, [], _).

assert_a3_itechain :-
    assertz((user:a3_itechain(X, Y) :-
                ( X > 10 -> T = big ; X > 5 -> T = mid ; T = small ),
                Y = T)).
retract_a3_itechain :- retractall(user:a3_itechain(_, _)).

test(g_a3_14_ite_binding_is_named_and_assigned,
     [setup(assert_a3_itechain), cleanup(retract_a3_itechain)]) :-
    native_body(a3_itechain/2, Code),
    has(Code, "let v3;"),          % declared once ...
    has(Code, "v3 = \"big\";"),    % ... assigned in the branches ...
    has(Code, "const arg2 = v3;"), % ... and read afterwards by NAME
    hasnt(Code, "= _"),            % no internal _NNNNN identifier
    hasnt(Code, "incomplete lowering").

test(g_a3_14_ite_chain_runs_under_node,
     [setup(assert_a3_itechain), cleanup(retract_a3_itechain),
      condition(node_available)]) :-
    vanilla_js_target:compile_predicate_to_vanilla_js(a3_itechain/2, [], Code),
    node_check(Code),
    Inputs = [12, 7, 1],
    findall(Got, ( member(I, Inputs),
                   number_string(I, S),
                   node_run_lines(Code, [S], [Got]) ), GotList),
    findall(Exp, ( member(I2, Inputs),
                   user:a3_itechain(I2, E), atom_string(E, Exp) ), ExpList),
    GotList == ExpList.

% ---------------------------------------------------------------------------
% G-A3-15 : a reversible text builtin picks the right direction
% ---------------------------------------------------------------------------
% `p(Cs, S) :- string_chars(S, Cs).` must BUILD the text. Both variables are
% head arguments so both are mapped, the "output must be a fresh variable" pass
% finds nothing, and the fallback pass used to take the first matching rule --
% decompose -- emitting `const arg1 = Array.from(arg2);`, which assigns over the
% function's own parameter and returns it. The emitted calling convention makes
% arg<N> the RETURN value, not an input, so that slot is the one to prefer.

test(g_a3_15_reversible_builtin_prefers_the_head_output_slot,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_unchars/2, Code),
    has(Code, "const arg2 = arg1.join(\"\");"),
    has(Code, "return arg2;"),
    hasnt(Code, "Array.from(arg2)").

% The forward direction is unaffected: with the output in the head's own slot
% and the input a parameter, decompose is still what `p(S, Cs)` means.
test(g_a3_15_forward_direction_is_unchanged,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_chars/2, Code),
    has(Code, "Array.from(arg1)"),
    hasnt(Code, "join(\"\")").

% When BOTH arguments are genuinely known and neither is an output, the goal is
% a check, not a binding -- so it renders as a comparison rather than assigning
% over a value the clause already holds.
test(g_a3_15_both_known_renders_a_check) :-
    typescript_target:ts_guard_condition([S-"s", Cs-"cs"],
                                         string_chars(S, Cs), Cond),
    has(Cond, "==="),
    has(Cond, "cs"),
    has(Cond, "Array.from(s)").

% ---------------------------------------------------------------------------
% G-A3-11 : the generated scaffolding matches the predicate
% ---------------------------------------------------------------------------
% (1) An arity-1 predicate has NO output argument. build_ts_arg_list(Arity-1)
%     gave it ZERO parameters, so js_alpha/1 compiled to
%     `function js_alpha(): string { ... arg1 ... }` -- a body reading a
%     parameter that does not exist -- and returned the char code it had just
%     computed instead of a boolean.

test(g_a3_11_arity1_takes_its_argument_and_returns_a_boolean,
     [setup(assert_a3_flagchar), cleanup(retract_a3_flagchar)]) :-
    typescript_target:compile_predicate_to_typescript(a3_flag_char/1, [], Code),
    has(Code, "function a3_flag_char(arg1: any): boolean"),
    has(Code, "return true;"),
    has(Code, "return false;"),
    hasnt(Code, "function a3_flag_char()").

test(g_a3_11_arity1_module_parses_and_runs_under_node,
     [setup(assert_a3_flagchar), cleanup(retract_a3_flagchar),
      condition(node_available)]) :-
    vanilla_js_target:compile_predicate_to_vanilla_js(a3_flag_char/1, [], Code),
    node_check(Code),
    % No digit characters here: see the G-A3-11.3 probe in Part 2 -- without
    % parameter types the CLI entry cannot tell the CHARACTER '7' from the
    % NUMBER 7.
    Chars = ['q', 'Q', 'z', '-', '#'],
    findall(Got, ( member(C, Chars), node_run_lines(Code, [C], [Got]) ), GotList),
    findall(Exp, ( member(C2, Chars),
                   ( user:a3_flag_char(C2) -> Exp = "true" ; Exp = "false" ) ),
            ExpList),
    GotList == ExpList.

% The two inheritors carry the new signature too -- checked per target, because
% assuming inheritance is how G-A3-7 got in.
test(g_a3_11_vanilla_js_inherits_the_semidet_signature,
     [setup(assert_a3_flagchar), cleanup(retract_a3_flagchar),
      condition(node_available)]) :-
    vanilla_js_target:compile_predicate_to_vanilla_js(a3_flag_char/1, [], Code),
    has(Code, "function a3_flag_char(arg1)"),
    hasnt(Code, "arg1: any"),
    node_check(Code).

test(g_a3_11_annotated_js_inherits_the_semidet_signature,
     [setup(assert_a3_flagchar), cleanup(retract_a3_flagchar),
      condition(node_available)]) :-
    annotated_js_target:compile_predicate(a3_flag_char/1, [], Code),
    has(Code, "@param {any} arg1"),
    has(Code, "@returns {boolean}"),
    has(Code, "function a3_flag_char(arg1)"),
    node_check(Code).

% (2) The CLI entry point used to pass exactly one argument, always through
%     parseInt, whatever the predicate's arity or argument types -- which is why
%     the end-to-end check below used to need a hand-written driver.

test(g_a3_11_cli_entry_passes_every_argument,
     [setup(assert_a3_substr), cleanup(retract_a3_substr)]) :-
    typescript_target:compile_predicate_to_typescript(a3_substring_from/3, [], Code),
    % a3_substring_from/3 -> two parameters -> argv[2..3]
    has(Code, "process.argv.length >= 4"),
    has(Code, "process.argv.slice(2, 4)"),
    has(Code, "a3_substring_from(...argv)"),
    hasnt(Code, "parseInt(process.argv[2])").

% The A3 report had to disclose that its node run used a hand-written driver.
% It no longer does: this drives the compiled module through the entry point the
% compiler emitted, and compares against the SWI oracle.
test(g_a3_11_compiler_emitted_cli_entry_drives_substring_from,
     [setup(assert_a3_substr), cleanup(retract_a3_substr),
      condition(node_available)]) :-
    vanilla_js_target:compile_predicate_to_vanilla_js(a3_substring_from/3, [], Code),
    node_check(Code),
    Cases = ["--state=alpha"-2, "hello"-0, "abcdef"-3],
    findall(Got,
            ( member(S-I, Cases),
              number_string(I, IS),
              node_run_lines(Code, [S, IS], [Got]) ),
            GotList),
    findall(Exp,
            ( member(S2-I2, Cases), user:a3_substring_from(S2, I2, Exp) ),
            ExpList),
    GotList == ExpList.

% (4) compile_module/3 dispatched on four canned recursion patterns and let
%     findall/3 SILENTLY DROP everything else, so a module of otherwise-fine
%     predicates compiled to a header and two blank lines with no error.

assert_a3_modpreds :-
    assertz(user:a3_mod_colour(red)),
    assertz(user:a3_mod_colour(blue)),
    assertz(user:a3_mod_bad(a, b)),
    assertz((user:a3_mod_bad(X, Y) :- a3_mod_helper(X, Y))).
retract_a3_modpreds :-
    retractall(user:a3_mod_colour(_)),
    retractall(user:a3_mod_bad(_, _)).

test(g_a3_11_compile_module_refuses_an_all_unsupported_module,
     [setup(assert_a3_modpreds), cleanup(retract_a3_modpreds)]) :-
    catch(typescript_target:compile_module([pred(a3_mod_bad, 2, facts)],
                                           [module_name(a3ModA)], _),
          error(unsupported_lowering(typescript, module(Mod), Detail), Msg),
          true),
    Mod == a3ModA,
    has(Detail, "a3_mod_bad/2"),
    has(Msg, "refusing to emit module").

test(g_a3_11_compile_module_emits_the_supported_subset_with_a_warning,
     [setup(assert_a3_modpreds), cleanup(retract_a3_modpreds)]) :-
    typescript_target:compile_module([pred(a3_mod_colour, 1, facts),
                                      pred(a3_mod_bad, 2, facts)],
                                     [module_name(a3ModB)], Code),
    has(Code, "// WARNING: 1 predicate(s) omitted"),
    has(Code, "a3_mod_bad/2"),
    has(Code, "export const a3_mod_colourFacts"),
    has(Code, "\"red\"").

% A module whose predicates ARE all supported carries no warning banner.
test(g_a3_11_supported_module_has_no_warning_banner,
     [setup(assert_a3_modpreds), cleanup(retract_a3_modpreds)]) :-
    typescript_target:compile_module([pred(a3_mod_colour, 1, facts)],
                                     [module_name(a3ModC)], Code),
    has(Code, "export const a3_mod_colourFacts"),
    hasnt(Code, "// WARNING").

% ============================================================================
% PART 2 -- GAP PROBES (executable reproductions, still open)
% ============================================================================
% Each probe pins the CURRENT behaviour. Closing the gap should make the probe
% FAIL: that is the cue to come back and turn it into a real assertion.

% ---------------------------------------------------------------------------
% G-A3-6 (M) : guards computed from body-local variables are hoisted above
%              the assignments that define them.
% ---------------------------------------------------------------------------
% CORRECT lowering emits the const-assignments first and only then tests the
% guard:  const v3 = ...; const v4 = ...; if (v3 >= v4) { ... }
% Today the condition is lifted into the clause's `if (...)` header, ahead of
% the block that declares v3/v4 -- a TDZ ReferenceError under node and a
% "used before its declaration" error under tsc.
test(gap_g_a3_6_guard_hoisted_above_its_own_definitions,
     [setup(assert_a3_starts), cleanup(retract_a3_starts)]) :-
    native_body(a3_starts_with/2, Code),
    once(sub_string(Code, IfAt, _, _, "if (v3 >= v4)")),
    once(sub_string(Code, DeclAt, _, _, "const v3 =")),
    IfAt < DeclAt.          % <-- the bug: the test precedes the declaration

% ---------------------------------------------------------------------------
% G-A3-11.3 (S) : parameter and return types are still hardcoded
% ---------------------------------------------------------------------------
% ts_native_signature/4 emits `arg<N>: number` for every parameter of an
% arity > 1 predicate and `: string` for its return, regardless of what the body
% actually does; the arity-1 shape uses `any` because a semidet test is as
% likely to be over text. The emitted CLI entry inherits the same blindness: it
% coerces an argument to a number when the token parses as one, which is the
% best a compiler with no parameter types can do -- and is why it cannot pass
% the CHARACTER '7' to a predicate that wants a character.
%
% CORRECT lowering needs real type inference from the goals of the body
% (char_code/2 implies text, `is`/2 implies numeric, sub_string/5 implies text +
% integers), which then drives both the signature and the argv conversion.
assert_a3_charclass :-
    assertz((user:a3_charclass(C) :- char_code(C, X), X >= 0'0, X =< 0'9)).
retract_a3_charclass :- retractall(user:a3_charclass(_)).

test(gap_g_a3_11_3_cli_entry_cannot_pass_a_numeric_looking_character,
     [setup(assert_a3_charclass), cleanup(retract_a3_charclass),
      condition(node_available)]) :-
    vanilla_js_target:compile_predicate_to_vanilla_js(a3_charclass/1, [], Code),
    node_check(Code),                       % it PARSES -- the gap is semantic
    user:a3_charclass('7'),                 % SWI: '7' is a digit character
    node_run_lines(Code, ['7'], Got),
    Got \== ["true"].                       % <-- the gap: '7' arrived as 7

% ---------------------------------------------------------------------------
% G-A3-9 (L) : multi-accumulator loops keep only ONE output
% ---------------------------------------------------------------------------
% cli_args' two engines are lenient_loop/5 (2 accumulators) and strict_loop/8
% (3 accumulators + a status). The structural path accepts an arity-5 loop but
% treats argument 5 as the sole output and argument 4 -- the OTHER output -- as
% a required INPUT parameter, so the caller must already know the answer.
assert_a3_twoacc :-
    assertz(user:a3_twoacc([], A, B, A, B)),
    assertz((user:a3_twoacc([X|Xs], A0, B0, A, B) :-
                A1 is A0 + X, B1 is B0 + 1, a3_twoacc(Xs, A1, B1, A, B))).
retract_a3_twoacc :- retractall(user:a3_twoacc(_, _, _, _, _)).

test(gap_g_a3_9_second_output_becomes_an_input,
     [setup(assert_a3_twoacc), cleanup(retract_a3_twoacc)]) :-
    native_structural(a3_twoacc/5, Code),
    % a4 -- the FIRST output argument -- is emitted as a function parameter ...
    has(Code, "a4: any"),
    % ... and is then compared against the accumulator instead of assigned.
    has(Code, "a4 === a2").

% Closing G-A3-10 WIDENS G-A3-9's blast radius, and that is worth pinning
% rather than discovering later: a two-output loop whose step is an
% if-then-else used to be refused (no ITE lowering at all); it now reaches the
% structural path and lands on exactly the same wrong shape. Nothing about
% G-A3-9 changed -- only the set of predicates that get to hit it. cli_args'
% lenient_loop/5 and scan_leading_globals/4 are this shape.
assert_a3_twoacc_ite :-
    assertz(user:a3_twoacc_ite([], A, B, A, B)),
    assertz((user:a3_twoacc_ite([X|Xs], A0, B0, A, B) :-
                (   X > 0
                ->  A1 is A0 + X, a3_twoacc_ite(Xs, A1, B0, A, B)
                ;   B1 is B0 + 1, a3_twoacc_ite(Xs, A0, B1, A, B)
                ))).
retract_a3_twoacc_ite :- retractall(user:a3_twoacc_ite(_, _, _, _, _)).

test(gap_g_a3_9_now_reached_by_ite_loops_too,
     [setup(assert_a3_twoacc_ite), cleanup(retract_a3_twoacc_ite)]) :-
    native_structural(a3_twoacc_ite/5, Code),
    has(Code, "a4: any"),
    has(Code, "a4 === a2"),
    % the if-then-else itself lowered correctly -- both branches tail-call
    aggregate_all(count, sub_string(Code, _, _, _, "a3_twoacc_ite(a1.slice(1)"), N),
    N =:= 2.

% ---------------------------------------------------------------------------
% G-A3-10 (CLOSED) : if-then-else composes with structural recursion
% ---------------------------------------------------------------------------
% Every cli_args loop dispatches on an if-then-else chain. ts_struct_goal/13 had
% no clause for `;`/`->`, so ONE if-then-else anywhere in the body made the
% structural path refuse the whole predicate and the dispatcher dropped to the
% fact fallback.
%
% The fix splits the lowering by POSITION of the if-then-else:
%
%   * VALUE position (goals follow it) -- clause_body_analysis'
%     if_then_else_shared_output_vars/4 names the variables both branches bind;
%     each gets `let _sN;` before the block and an assignment at the end of each
%     branch, so the goals after it read the value by name.
%   * TAIL position (it is the last goal) -- each branch renders its own
%     `return`: a branch ending in the recursive call continues the loop, a
%     branch binding the output exits it. Nested else-if chains compose because
%     the else branch is rendered in tail context too.
%
% Refusals kept: a branch that emits a clause-level guard, branches that bind
% no common variable, a bare `(A ; B)`, a bare `(C -> T)`, a condition
% ts_guard_condition/3 cannot render. Each of these fails the renderer, so the
% structural path declines and the caller reaches the loud G-A3-4/G-A3-8
% refusal instead of receiving JavaScript with the wrong control flow.

%% vanilla_structural(+Pred/Arity, -Code)
%  The same structural lowering, emitted through vanilla_js so node can RUN it:
%  the TypeScript module carries `a1: any[]` annotations node will not parse.
%  Assertions about the emitted TypeScript use native_structural/2; assertions
%  about behaviour use this.
vanilla_structural(PredSpec, Code) :-
    vanilla_js_target:compile_predicate_to_vanilla_js(PredSpec, [], Code).

%% run_struct(+Code, +CallSrc, -Line)
%  Append a driver to a structural module (it has no CLI entry of its own) and
%  run it under node, returning the single line it prints.
run_struct(Code, CallSrc, Line) :-
    format(string(Src), "~w\nconsole.log(JSON.stringify(~w));\n", [Code, CallSrc]),
    node_run_lines(Src, [], [Line]).

%% js_atom_list(+Atoms, -JsArrayLiteral) — [a,b] -> ["a", "b"]
js_atom_list(Atoms, Literal) :-
    findall(Q, ( member(A, Atoms), format(string(Q), "\"~w\"", [A]) ), Quoted),
    atomic_list_concat(Quoted, ', ', Inner),
    format(string(Literal), "[~w]", [Inner]).

% --- the canonical case from the report ------------------------------------
assert_a3_iteloop :-
    assertz(user:a3_iteloop([], Acc, Acc)),
    assertz((user:a3_iteloop([X|Xs], Acc0, Acc) :-
                ( X > 0 -> Acc1 is Acc0 + X ; Acc1 = Acc0 ),
                a3_iteloop(Xs, Acc1, Acc))).
retract_a3_iteloop :- retractall(user:a3_iteloop(_, _, _)).

test(g_a3_10_ite_in_a_recursive_body_lowers,
     [setup(assert_a3_iteloop), cleanup(retract_a3_iteloop)]) :-
    native_structural(a3_iteloop/3, Code),
    % the if-then-else is a VALUE: declared once with `let`, assigned per branch
    has(Code, "let _s"),
    has(Code, "if (a1[0] > 0) {"),
    has(Code, "} else {"),
    % ... and the recursive call comes AFTER the block, reading it by name
    has(Code, "a3_iteloop(a1.slice(1), _s"),
    % no internal SWI variable name leaked, no `undefined` placeholder
    hasnt(Code, "undefined"),
    hasnt(Code, "= _G").

test(g_a3_10_ite_loop_parses_and_matches_swi,
     [setup(assert_a3_iteloop), cleanup(retract_a3_iteloop),
      condition(node_available)]) :-
    vanilla_structural(a3_iteloop/3, Code),
    node_check(Code),
    forall(member(L, [[], [1], [-1], [1,-2,3], [5,5,-5,5], [-1,-2,-3], [0,7,-7,2]]),
           ( user:a3_iteloop(L, 0, R),
             format(string(Expect), "~w", [R]),
             term_string(L, LS),
             format(string(Call), "a3_iteloop(~w, 0)", [LS]),
             run_struct(Code, Call, Got),
             Got == Expect )).

% --- nested else-if chain (value position) ---------------------------------
assert_a3_itechainloop :-
    assertz(user:a3_itechainloop([], Acc, Acc)),
    assertz((user:a3_itechainloop([X|Xs], Acc0, Acc) :-
                ( X > 10 -> Acc1 is Acc0 + 2
                ; X > 0  -> Acc1 is Acc0 + 1
                ; Acc1 = Acc0 ),
                a3_itechainloop(Xs, Acc1, Acc))).
retract_a3_itechainloop :- retractall(user:a3_itechainloop(_, _, _)).

test(g_a3_10_nested_else_if_chain_composes,
     [setup(assert_a3_itechainloop), cleanup(retract_a3_itechainloop),
      condition(node_available)]) :-
    vanilla_structural(a3_itechainloop/3, Code),
    node_check(Code),
    % two `let` slots: the inner chain's value feeds the outer one
    aggregate_all(count, sub_string(Code, _, _, _, "let _s"), Lets),
    Lets >= 2,
    forall(member(L, [[], [11], [5], [-5], [11,5,-5], [20,20,1,0,-1]]),
           ( user:a3_itechainloop(L, 0, R),
             format(string(Expect), "~w", [R]),
             term_string(L, LS),
             format(string(Call), "a3_itechainloop(~w, 0)", [LS]),
             run_struct(Code, Call, Got),
             Got == Expect )).

% --- both branches recurse (tail position) ---------------------------------
assert_a3_bothrec :-
    assertz(user:a3_bothrec([], Acc, Acc)),
    assertz((user:a3_bothrec([X|Xs], Acc0, Acc) :-
                (   X > 0
                ->  Acc1 is Acc0 + X, a3_bothrec(Xs, Acc1, Acc)
                ;   a3_bothrec(Xs, Acc0, Acc)
                ))).
retract_a3_bothrec :- retractall(user:a3_bothrec(_, _, _)).

test(g_a3_10_both_branches_recurse_become_branching_returns,
     [setup(assert_a3_bothrec), cleanup(retract_a3_bothrec),
      condition(node_available)]) :-
    vanilla_structural(a3_bothrec/3, Code),
    node_check(Code),
    % a TAIL if-then-else: each branch returns, so there is no let/assign and no
    % trailing `return` after the block.
    hasnt(Code, "let _s"),
    aggregate_all(count, sub_string(Code, _, _, _, "a3_bothrec(a1.slice(1)"), Calls),
    Calls =:= 2,
    forall(member(L, [[], [1,2], [-1,-2], [1,-2,3,-4], [0,0,5]]),
           ( user:a3_bothrec(L, 0, R),
             format(string(Expect), "~w", [R]),
             term_string(L, LS),
             format(string(Call), "a3_bothrec(~w, 0)", [LS]),
             run_struct(Code, Call, Got),
             Got == Expect )).

% --- one branch exits, the other continues the loop ------------------------
% This is the shape of cli_args' pair_lookup/3 and string_member/2: the then
% branch returns a final value, the else branch tail-calls.
assert_a3_findpos :-
    assertz(user:a3_findpos([], _I, none)),
    assertz((user:a3_findpos([X|Xs], I, R) :-
                ( X > 0 -> R = I ; I1 is I + 1, a3_findpos(Xs, I1, R) ))).
retract_a3_findpos :- retractall(user:a3_findpos(_, _, _)).

test(g_a3_10_exit_branch_returns_recursive_branch_continues,
     [setup(assert_a3_findpos), cleanup(retract_a3_findpos),
      condition(node_available)]) :-
    vanilla_structural(a3_findpos/3, Code),
    node_check(Code),
    has(Code, "return a2;"),                  % the exit branch
    has(Code, "a3_findpos(a1.slice(1)"),      % the continuation branch
    forall(member(L, [[], [-1,-2], [3], [-1,-2,7,-3], [0,0,1]]),
           ( user:a3_findpos(L, 0, R),
             ( R == none -> Expect = "\"none\"" ; format(string(Expect), "~w", [R]) ),
             term_string(L, LS),
             format(string(Call), "a3_findpos(~w, 0)", [LS]),
             run_struct(Code, Call, Got),
             Got == Expect )).

% --- if-then-else followed by MORE goals (composes with let+assign) --------
assert_a3_itethenmore :-
    assertz(user:a3_itethenmore([], Acc, Acc)),
    assertz((user:a3_itethenmore([X|Xs], Acc0, Acc) :-
                ( X > 0 -> D is X * 2 ; D = 0 ),
                Acc1 is Acc0 + D,
                a3_itethenmore(Xs, Acc1, Acc))).
retract_a3_itethenmore :- retractall(user:a3_itethenmore(_, _, _)).

test(g_a3_10_ite_value_is_read_by_the_goals_after_it,
     [setup(assert_a3_itethenmore), cleanup(retract_a3_itethenmore),
      condition(node_available)]) :-
    vanilla_structural(a3_itethenmore/3, Code),
    node_check(Code),
    % the `is/2` AFTER the block reads the slot the block assigned
    has(Code, "let _s1;"),
    has(Code, "const _s2 = (a2 + _s1);"),
    forall(member(L, [[], [3], [-3], [1,-1,2], [4,0,-4,6]]),
           ( user:a3_itethenmore(L, 0, R),
             format(string(Expect), "~w", [R]),
             term_string(L, LS),
             format(string(Call), "a3_itethenmore(~w, 0)", [LS]),
             run_struct(Code, Call, Got),
             Got == Expect )).

% --- a semidet ITE walk: the shape that moved out of G-A3-8's hazard list ---
assert_a3_strmember :-
    assertz((user:a3_strmember(S, [X|Xs]) :-
                ( S == X -> true ; a3_strmember(S, Xs) ))).
retract_a3_strmember :- retractall(user:a3_strmember(_, _)).

test(g_a3_10_semidet_ite_walk_lowers_and_matches_swi,
     [setup(assert_a3_strmember), cleanup(retract_a3_strmember),
      condition(node_available)]) :-
    native_structural(a3_strmember/2, TsCode),
    has(TsCode, "): boolean {"),         % semidet -> boolean, not `any`
    vanilla_structural(a3_strmember/2, Code),
    node_check(Code),
    has(Code, "return true;"),
    has(Code, "return a3_strmember(a1, a2.slice(1));"),
    has(Code, "return false;"),          % no clause matched -> fails
    forall(member(S-L, [b-[a,b,c], z-[a,b,c], a-[a], q-[]]),
           ( ( user:a3_strmember(S, L) -> Expect = "true" ; Expect = "false" ),
             js_atom_list(L, LS),
             format(string(Call), "a3_strmember(\"~w\", ~w)", [S, LS]),
             run_struct(Code, Call, Got),
             Got == Expect )).

% --- the two cli_args predicates this gap actually unblocks -----------------
% Verbatim from examples/cli_args/cli_args.pl (renamed so the suite stays
% self-contained). first_char_index/4 is the index walk behind
% first_equals_index/2, which split_flag_token/3 uses to split `--k=v`.
assert_a3_firstcharindex :-
    assertz(user:a3_first_char_index([], _Target, _I, -1)),
    assertz((user:a3_first_char_index([C|Cs], Target, I, Index) :-
                (   C == Target
                ->  Index = I
                ;   I1 is I + 1,
                    a3_first_char_index(Cs, Target, I1, Index)
                ))).
retract_a3_firstcharindex :- retractall(user:a3_first_char_index(_, _, _, _)).

test(g_a3_10_cli_args_first_char_index_lowers_and_matches_swi,
     [setup(assert_a3_firstcharindex), cleanup(retract_a3_firstcharindex),
      condition(node_available)]) :-
    native_structural(a3_first_char_index/4, TsCode),
    has(TsCode, "return -1;"),                       % base clause
    has(TsCode, "if (a1[0] === a2) {"),              % the ITE condition
    has(TsCode, "return a3;"),                       % exit branch
    has(TsCode, "a3_first_char_index(a1.slice(1), a2, _s"),  % continuation
    vanilla_structural(a3_first_char_index/4, Code),
    node_check(Code),
    forall(member(Cs, [[], [a], ['=' ], [a,'=',b], [a,b,c], ['=','=']]),
           ( user:a3_first_char_index(Cs, '=', 0, R),
             format(string(Expect), "~w", [R]),
             js_atom_list(Cs, LS),
             format(string(Call), "a3_first_char_index(~w, \"=\", 0)", [LS]),
             run_struct(Code, Call, Got),
             Got == Expect )).

% --- REFUSAL: a branch that carries a bare test is not a straight-line block -
% `( X > 0 -> X < 100, Acc1 is Acc0 + X ; Acc1 = Acc0 )` can FAIL inside the
% then branch, which no let/assign block expresses. Lowering it as an assignment
% would silently drop the `X < 100` test, so the structural path declines and
% the caller gets the loud refusal instead.
assert_a3_guardbranch :-
    assertz(user:a3_guardbranch([], Acc, Acc)),
    assertz((user:a3_guardbranch([X|Xs], Acc0, Acc) :-
                ( X > 0 -> X < 100, Acc1 is Acc0 + X ; Acc1 = Acc0 ),
                a3_guardbranch(Xs, Acc1, Acc))).
retract_a3_guardbranch :- retractall(user:a3_guardbranch(_, _, _)).

test(g_a3_10_semidet_branch_still_refuses,
     [setup(assert_a3_guardbranch), cleanup(retract_a3_guardbranch)]) :-
    \+ catch(native_structural(a3_guardbranch/3, _), _, fail),
    % and it refuses LOUDLY through the dispatcher rather than emitting anything
    a3_compile_outcome(a3_guardbranch/3, Outcome),
    Outcome = refused(a3_guardbranch/3, Shape, _),
    has(Shape, "if-then-else").

% --- REFUSAL: branches that bind different variables ------------------------
assert_a3_diffvars :-
    assertz(user:a3_diffvars([], Acc, Acc)),
    assertz((user:a3_diffvars([X|Xs], Acc0, Acc) :-
                ( X > 0 -> Acc1 is Acc0 + X ; _B1 is Acc0 - X ),
                a3_diffvars(Xs, Acc1, Acc))).
retract_a3_diffvars :- retractall(user:a3_diffvars(_, _, _)).

test(g_a3_10_branches_with_no_shared_output_still_refuse,
     [setup(assert_a3_diffvars), cleanup(retract_a3_diffvars)]) :-
    \+ catch(native_structural(a3_diffvars/3, _), _, fail).

% --- REFUSAL: a bare disjunction is not an if-then-else ---------------------
assert_a3_baredisj :-
    assertz(user:a3_baredisj([], Acc, Acc)),
    assertz((user:a3_baredisj([X|Xs], Acc0, Acc) :-
                ( Acc1 is Acc0 + X ; Acc1 = Acc0 ),
                a3_baredisj(Xs, Acc1, Acc))).
retract_a3_baredisj :- retractall(user:a3_baredisj(_, _, _)).

test(g_a3_10_bare_disjunction_still_refuses,
     [setup(assert_a3_baredisj), cleanup(retract_a3_baredisj)]) :-
    \+ catch(native_structural(a3_baredisj/3, _), _, fail).

% --- an unbound variable is refused, never rendered as `undefined` ----------
% cli_args' drop_brackets/2 binds `Kept = Kept1` where Kept1 is an OUTPUT of the
% later recursive call -- list-BUILDING recursion, which this path cannot
% express. It used to be a candidate for `return undefined;`.
test(g_a3_10_unbound_term_is_refused_not_undefined) :-
    \+ typescript_target:ts_term_expr(_Free, [], _),
    \+ typescript_target:ts_arith(_Free2, [], _).

% ---------------------------------------------------------------------------
% G-A3-9 remains OPEN; its probe is above. Everything below this line was a
% probe and is now an assertion.
% ---------------------------------------------------------------------------

% ---------------------------------------------------------------------------
% G-A3-12 (M) : compound terms as values are stringified, not built
% ---------------------------------------------------------------------------
% cli_args returns ok(Positional, Flags) / err(Message) / some(V) / none and
% carries schema(Options, Positionals) entries. ts_literal/2 turns any compound
% into a QUOTED STRING of its Prolog text, so the tag and the payload are lost.
test(gap_g_a3_12_compound_term_becomes_a_string_literal) :-
    typescript_target:ts_literal(ok([a], [b-c]), Lit),
    has(Lit, "\"ok("),                 % a JS string literal, not an object
    hasnt(Lit, "{ tag:").

% ---------------------------------------------------------------------------
% G-A3-16 (M) : a compound / list head argument becomes a STRING comparison
% ---------------------------------------------------------------------------
% ts_head_conditions/4 sends every non-variable head argument through
% ts_literal/2, which stringifies a compound term. A first-argument-indexed
% assoc walk (cli_args' pair_lookup/3, flags_put/4, string_member/2, ...)
% therefore compiles to a comparison against the Prolog SOURCE TEXT of the
% pattern -- internal `_G` variable names included.
%
% CORRECT lowering would destructure: arg1.length > 0, k = arg1[0][0], ...
assert_a3_pairlookup :-
    assertz((user:a3_pair_lookup([K-V|Rest], Key, Value) :-
                ( K == Key -> Value = V ; a3_pair_lookup(Rest, Key, Value) ))).
retract_a3_pairlookup :- retractall(user:a3_pair_lookup(_, _, _)).

test(gap_g_a3_16_list_head_pattern_becomes_a_string_literal,
     [setup(assert_a3_pairlookup), cleanup(retract_a3_pairlookup)]) :-
    native_body(a3_pair_lookup/3, Code),
    has(Code, "arg1 === \"["),      % compared against the pattern's TEXT
    has(Code, "|_").                % including a raw internal variable name

:- end_tests(typescript_cli_args_shapes).
