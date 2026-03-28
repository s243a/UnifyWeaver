:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').

%% =====================================================
%% Test predicates matching TypR's native lowering patterns
%% that Python should also handle.
%% =====================================================

try(Label, Pred/Arity, Checks) :-
    format('~w: ', [Label]),
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(python)], Code), _, fail)
    ->  run_checks(Code, Checks, AllOk),
        (AllOk == true -> writeln(ok) ; true)
    ;   writeln('COMPILE FAIL')
    ).

run_checks(_, [], true).
run_checks(Code, [check(Label, Substr)|Rest], AllOk) :-
    (   sub_string(Code, _, _, _, Substr)
    ->  run_checks(Code, Rest, AllOk)
    ;   format('MISS(~w) ', [Label]),
        AllOk = false,
        run_checks(Code, Rest, _)
    ).

show(Pred/Arity) :-
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(python)], Code), _, fail)
    ->  atom_string(Pred, PredStr),
        split_string(Code, "\n", "", Lines),
        format(atom(Prefix), "def ~w", [PredStr]),
        (   nth1(I, Lines, L), sub_string(L, _, _, _, Prefix)
        ->  End is min(I + 10, 99999),
            forall((between(I, End, J), nth1(J, Lines, LJ),
                    (LJ \= "" ; J =:= I)), writeln(LJ))
        ;   writeln('  (function not found)')
        )
    ;   writeln('  (compile failed)')
    ).

%% --- Pattern: If-then-else as GUARD (not output) ---
:- dynamic safe_log/2.
safe_log(X, Y) :- (X > 0 -> true ; fail), Y is log(X).
%% Expected: if arg1 > 0: return math.log(arg1)

%% --- Pattern: Conditional output with pending guards ---
:- dynamic clamped_sqrt/3.
clamped_sqrt(X, Max, Y) :- X >= 0, Y is sqrt(X), Y =< Max.
%% Expected: v1 = math.sqrt(arg1); if v1 <= arg2: return v1

%% --- Pattern: Multi-result if-then-else (two outputs from branches) ---
:- dynamic divmod_result/4.
divmod_result(X, Y, Q, R) :- Q is X // Y, R is X mod Y.
%% Expected: return (arg1 // arg2, arg1 % arg2)

%% --- Pattern: Nested if-then-else as guard ---
:- dynamic complex_guard/2.
complex_guard(X, Y) :- (X > 0 -> X < 100 ; X > -100), Y is X * 2.
%% Expected: if (arg1 < 100 if arg1 > 0 else arg1 > -100): return arg1 * 2

%% --- Pattern: Mixed output + guard + output sequence ---
:- dynamic bounded_square/2.
bounded_square(X, Y) :- Y is X * X, Y < 1000.
%% Expected: arg2 = arg1 * arg1; if arg2 < 1000: return arg2

%% --- Pattern: Output from binding (Python binding registry) ---
%% (length/2 is registered as Python 'len' binding)
:- dynamic list_size/2.
list_size(L, N) :- length(L, N).
%% Would need binding lookup — deferred

%% --- Pattern: Wildcard clause (catch-all) ---
:- dynamic classify_sign/2.
classify_sign(0, zero).
classify_sign(X, positive) :- X > 0.
classify_sign(_, negative).
%% Expected: if arg1 == 0: return "zero"; elif arg1 > 0: return "positive"; else: return "negative"

%% --- Pattern: 5+ clauses ---
:- dynamic day_name/2.
day_name(1, monday). day_name(2, tuesday). day_name(3, wednesday).
day_name(4, thursday). day_name(5, friday). day_name(6, saturday).
day_name(7, sunday). day_name(_, unknown).
%% Expected: 8-clause if/elif chain

run_tests :-
    try('if-then-else guard', safe_log/2, [check('log', "log")]),
    try('guarded tail + pending', clamped_sqrt/3, [check('sqrt', "sqrt")]),
    try('multi-output arithmetic', divmod_result/4, [check('return tuple', "return (")]),
    try('nested ite guard', complex_guard/2, [check('X*2', "* 2")]),
    try('output then guard', bounded_square/2, [check('< 1000', "< 1000")]),
    try('classify_sign', classify_sign/2, [check('zero', "zero"), check('positive', "positive"), check('negative', "negative")]),
    try('day_name 8-clause', day_name/2, [check('monday', "monday"), check('sunday', "sunday"), check('unknown', "unknown")]),
    nl,
    writeln('--- Showing generated functions ---'), nl,
    writeln('=== safe_log ==='), show(safe_log/2), nl,
    writeln('=== clamped_sqrt ==='), show(clamped_sqrt/3), nl,
    writeln('=== divmod_result ==='), show(divmod_result/4), nl,
    writeln('=== bounded_square ==='), show(bounded_square/2), nl,
    writeln('=== classify_sign ==='), show(classify_sign/2), nl,
    writeln('=== day_name ==='), show(day_name/2), nl,
    writeln('=== TYPR PATTERN TESTS DONE ===').
