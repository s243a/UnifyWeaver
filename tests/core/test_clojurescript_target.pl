:- module(test_clojurescript_target, [test_clojurescript_target/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/clojurescript_target').

test_clojurescript_target :-
    run_tests([clojurescript_target]).

:- begin_tests(clojurescript_target).

% Helper: compile using the public API
compile_cljs(Pred/Arity, Code) :-
    clojurescript_target:compile_predicate_to_clojurescript(Pred/Arity, [], Code).

% Helper: deterministic substring check
has(Code, Substr) :- once(sub_string(Code, _, _, _, Substr)).
hasnt(Code, Substr) :- \+ sub_string(Code, _, _, _, Substr).

% ============================================================================
% Interop rewrite: JVM host calls -> JS host calls
% ============================================================================

test(interop_parse_int) :-
    clojurescript_target:clojurescript_interop_rewrite(
        "(Integer/parseInt (first *command-line-args*))", Out),
    has(Out, "js/parseInt"),
    hasnt(Out, "Integer/parseInt").

test(interop_math_abs) :-
    clojurescript_target:clojurescript_interop_rewrite("(Math/abs x)", Out),
    has(Out, "js/Math.abs"),
    hasnt(Out, "(Math/abs").

test(interop_math_generic) :-
    clojurescript_target:clojurescript_interop_rewrite("(Math/floor x)", Out),
    has(Out, "js/Math.floor").

test(interop_exception) :-
    clojurescript_target:clojurescript_interop_rewrite(
        "(catch Exception e (.getMessage e))", Out),
    has(Out, "(catch :default e"),
    has(Out, "(.-message e"),
    hasnt(Out, "Exception").

test(interop_idempotent_on_clean_code) :-
    clojurescript_target:clojurescript_interop_rewrite("(+ 1 2)", Out),
    has(Out, "(+ 1 2)").

% ============================================================================
% Full predicate compilation: reuse Clojure base, rewrite, add banner
% ============================================================================

test(compile_reuses_clojure_codegen) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_cljs(double/2, Code),
    has(Code, "ClojureScript"),
    has(Code, "(defn double [arg1]"),
    has(Code, "(* arg1 2)"),
    retractall(user:double(_, _)).

test(compile_no_jvm_parseint_leak) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_cljs(positive/2, Code),
    hasnt(Code, "Integer/parseInt"),
    retractall(user:positive(_, _)).

% ============================================================================
% Build config + browser wrapper
% ============================================================================

test(shadow_cljs_edn) :-
    clojurescript_target:generate_shadow_cljs_edn([main_ns('generated.demo')], Shadow),
    has(Shadow, ":target :browser"),
    has(Shadow, "generated.demo/-main").

test(scittle_html_wrapper) :-
    clojurescript_target:generate_scittle_html("Demo", [cljs("(println :hi)")], HTML),
    has(HTML, "application/x-scittle"),
    has(HTML, "(println :hi)").

:- end_tests(clojurescript_target).
