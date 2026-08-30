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

% ============================================================================
% Runtime variants (scittle / nbb / bb) -- runtime(Kind) option
% ============================================================================

% Runtime resolution: default when unspecified/unknown, else the named runtime.
test(runtime_resolves_default) :-
    clojurescript_target:cljs_runtime([], R), R == default.
test(runtime_resolves_nbb) :-
    clojurescript_target:cljs_runtime([runtime(nbb)], R), R == nbb.
test(runtime_resolves_bb) :-
    clojurescript_target:cljs_runtime([runtime(bb)], R), R == bb.
test(runtime_unknown_falls_back_to_default) :-
    clojurescript_target:cljs_runtime([runtime(wat)], R), R == default.

% Default runtime preserves the historical output: JS interop, no shebang.
test(default_runtime_js_interop_no_shebang) :-
    assert(user:(dbl(X, R) :- R is X * 2)),
    clojurescript_target:compile_predicate_to_clojurescript(dbl/2, [], Code),
    has(Code, "js/parseInt"),
    hasnt(Code, "Integer/parseInt"),
    hasnt(Code, "#!/usr/bin/env"),
    retractall(user:dbl(_, _)).

% nbb runtime: JS interop plus an executable shebang.
test(nbb_runtime_shebang_and_js_interop) :-
    assert(user:(dbl(X, R) :- R is X * 2)),
    clojurescript_target:compile_predicate_to_clojurescript(dbl/2, [runtime(nbb)], Code),
    has(Code, "#!/usr/bin/env nbb"),
    has(Code, "js/parseInt"),
    has(Code, "nbb, Node sci"),
    hasnt(Code, "Integer/parseInt"),
    retractall(user:dbl(_, _)).

% bb runtime: Babashka is Clojure-on-JVM -- JVM interop is retained (NO rewrite),
% and a bb shebang is emitted.
test(bb_runtime_keeps_jvm_interop) :-
    assert(user:(dbl(X, R) :- R is X * 2)),
    clojurescript_target:compile_predicate_to_clojurescript(dbl/2, [runtime(bb)], Code),
    has(Code, "#!/usr/bin/env bb"),
    has(Code, "Integer/parseInt"),
    hasnt(Code, "js/parseInt"),
    has(Code, "Babashka/bb"),
    retractall(user:dbl(_, _)).

% bb runtime retains Math/abs (not rewritten to js/Math.abs).
test(bb_runtime_keeps_math_abs) :-
    clojurescript_target:clojurescript_from_clojure("(Math/abs x)", [runtime(bb)], Out),
    has(Out, "(Math/abs x)"),
    hasnt(Out, "js/Math.abs").

% clojurescript_from_clojure/3 with nbb rewrites and adds shebang.
test(from_clojure_3_nbb) :-
    clojurescript_target:clojurescript_from_clojure("(Math/abs x)", [runtime(nbb)], Out),
    has(Out, "#!/usr/bin/env nbb"),
    has(Out, "js/Math.abs").

:- end_tests(clojurescript_target).
