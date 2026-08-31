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
%      annotated_js_target.pl (G-A3-1..G-A3-5, G-A3-7, G-A3-17). These assert
%      CORRECT lowering and must stay green.
%
%   2. GAP PROBES for the constructs the pattern targets still cannot lower
%      (G-A3-6, G-A3-8..G-A3-16). Each probe is an executable minimal
%      reproduction that pins the CURRENT behaviour and carries a comment
%      saying what correct lowering would be. They are written so that CLOSING
%      the gap makes the probe FAIL -- that is the cue to come back here and
%      promote the probe into a real assertion.
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

%% native_body(+Pred/Arity, -Code)
%  Compile ONLY through typescript_target's native clause-body path.
%
%  Never route a gap probe through compile_predicate/3: when every native path
%  refuses, the dispatcher falls back to compile_facts/3, which EXECUTES the
%  predicate to enumerate its solutions (G-A3-8). For a rule predicate that is
%  an instantiation error, a stack overflow, or an unbounded findall that eats
%  memory until the process dies -- so the probes call the path directly.
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
% NOTE: the emitted CLI entry point only ever passes `process.argv[2]`, so it
% cannot drive a 3-argument predicate (G-A3-11). The driver below is written by
% the test; every line of the FUNCTION under test is compiler output.

test(compiled_substring_from_runs_under_node,
     [setup(assert_a3_substr), cleanup(retract_a3_substr),
      condition(node_available)]) :-
    vanilla_js_target:compile_predicate_to_vanilla_js(a3_substring_from/3, [], C1),
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
% G-A3-8 (M) : with no native path, the dispatcher EXECUTES the predicate
% ---------------------------------------------------------------------------
% compile_predicate_to_typescript/3's last resort is compile_facts/3, which
% enumerates the predicate's solutions by CALLING it. For a rule predicate that
% is an instantiation error at best and, for a generator such as `flags_put/4`
% or `string_member/2`, an unbounded findall that grows until the process dies.
%
% This probe pins the SHAPE the fallback cannot represent: a predicate with no
% fact clauses at all. It deliberately does not call compile_predicate/3 --
% doing so is the bug. The fix is a guard on the fallback (take the facts path
% only when EVERY clause body is `true`), not a change to compile_facts/3.
assert_a3_ruleonly :-
    assertz((user:a3_rule_only(X, Y) :- a3_rule_helper(X, Y))).
retract_a3_ruleonly :- retractall(user:a3_rule_only(_, _)).

test(gap_g_a3_8_fact_fallback_is_offered_a_rule_predicate,
     [setup(assert_a3_ruleonly), cleanup(retract_a3_ruleonly)]) :-
    functor(H, a3_rule_only, 2),
    findall(B, user:clause(H, B), Bodies),
    \+ memberchk(true, Bodies).

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

% ---------------------------------------------------------------------------
% G-A3-10 (M) : bodies containing if-then-else defeat the structural path
% ---------------------------------------------------------------------------
% Every cli_args loop dispatches on an if-then-else chain; ts_struct_goal/13 has
% no clause for `;`/`->`, so the structural path refuses the whole predicate and
% the dispatcher drops to the fact fallback.
assert_a3_iteloop :-
    assertz(user:a3_iteloop([], Acc, Acc)),
    assertz((user:a3_iteloop([X|Xs], Acc0, Acc) :-
                ( X > 0 -> Acc1 is Acc0 + X ; Acc1 = Acc0 ),
                a3_iteloop(Xs, Acc1, Acc))).
retract_a3_iteloop :- retractall(user:a3_iteloop(_, _, _)).

test(gap_g_a3_10_ite_in_a_recursive_body_refuses,
     [setup(assert_a3_iteloop), cleanup(retract_a3_iteloop)]) :-
    \+ catch(native_structural(a3_iteloop/3, _), _, fail).

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
% G-A3-13 (S) : the atom `true` is lowered to the STRING "true"
% ---------------------------------------------------------------------------
% cli_args' corpus asserts flags["include-key"] === true (boolean), which must
% stay distinct from the string "true" that a `--x=true` value would produce.
test(gap_g_a3_13_boolean_atoms_are_stringified) :-
    typescript_target:ts_literal(true, LitT),
    typescript_target:ts_literal(false, LitF),
    LitT == '"true"',
    LitF == '"false"'.

% ---------------------------------------------------------------------------
% G-A3-14 (S) : an unmapped variable is printed as its raw _G name
% ---------------------------------------------------------------------------
% ts_expr/3's last resort for an unbound variable is term_string/2, which emits
% the internal `_12345` name straight into the generated JavaScript.
test(gap_g_a3_14_unmapped_variable_leaks_into_output) :-
    typescript_target:ts_expr(_Free, [], Expr),
    once(sub_string(Expr, 0, 1, _, "_")).

% ---------------------------------------------------------------------------
% G-A3-15 (S) : a reversible text builtin picks the decompose direction when
%               BOTH of its arguments are already in the VarMap
% ---------------------------------------------------------------------------
% `p(Cs, S) :- string_chars(S, Cs).` should BUILD the text (S = Cs.join("")).
% Both variables are head arguments, so both are mapped, the "output must be a
% fresh variable" pass finds nothing, and the fallback pass takes the first
% matching rule -- decompose.
%
% CORRECT lowering would prefer the direction whose output is the clause head's
% own output argument; that needs the head's output slot threaded into
% ts_string_builtin/4.
test(gap_g_a3_15_reversible_builtin_picks_decompose_when_ambiguous,
     [setup(assert_a3_strings), cleanup(retract_a3_strings)]) :-
    native_body(a3_unchars/2, Code),
    has(Code, "Array.from(arg2)"),   % <-- the wrong direction
    hasnt(Code, "join(\"\")").

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
