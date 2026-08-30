:- module(test_clojurescript_target, [test_clojurescript_target/0]).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module('../../src/unifyweaver/targets/clojurescript_target').
:- use_module('../../src/unifyweaver/core/component_registry').
% clojure_target is loaded transitively by clojurescript_target; it is NOT
% use_module'd here because it also exports compile_module/3 (the CLJS variant
% is the one imported above). Its collect_declared_component/2 is reached via a
% module-qualified call (clojure_target:...).

test_clojurescript_target :-
    run_tests([clojurescript_target]).

:- begin_tests(clojurescript_target).

% Helper: compile using the public API
compile_cljs(Pred/Arity, Code) :-
    clojurescript_target:compile_predicate_to_clojurescript(Pred/Arity, [], Code).

% Helper: deterministic substring check
has(Code, Substr) :- once(sub_string(Code, _, _, _, Substr)).
hasnt(Code, Substr) :- \+ sub_string(Code, _, _, _, Substr).

% nbb availability gate (Node sci ClojureScript runtime)
nbb_available :-
    catch(( process_create(path(nbb), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

% Write CLJS Code to a temp .cljs file, run it under nbb with Argv, return
% trimmed stdout as an atom.
cljs_write_run(Code, Argv, Out) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.cljs', File),
    setup_call_cleanup(
        ( open(File, write, W), write(W, Code), close(W) ),
        cljs_nbb_exec(File, Argv, Out),
        catch(delete_file(File), _, true)).

cljs_nbb_exec(File, Argv, Out) :-
    process_create(path(nbb), [File|Argv],
                   [stdout(pipe(O)), stderr(null), process(P)]),
    read_string(O, _, Str), close(O), process_wait(P, _),
    normalize_space(atom(Out), Str).

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

% ============================================================================
% G-P6: compile_module/3 -- multiple predicates into ONE CLJS module
%
% The ClojureScript target previously had no compile_module/3 (module decl
% omitted it; the clojure base also lacked one). These tests assert that
% compile_module/3 emits a single (ns ...) namespace form followed by each
% predicate's (defn ...), with the JVM->JS interop applied to the whole module.
% ============================================================================

test(compile_module_multi_predicate) :-
    clojurescript_target:init_clojurescript_target,
    assert(user:(dbl(X, R) :- R is X * 2)),
    assert(user:(trp(X, R) :- R is X * 3)),
    clojurescript_target:compile_module(
        [dbl/2, trp/2],
        [namespace('generated.multi')],
        Code),
    % single ns form for the whole module
    has(Code, "(ns generated.multi)"),
    % each predicate present as its own defn (no per-predicate CLI entry / header)
    has(Code, "(defn dbl [arg1]"),
    has(Code, "(* arg1 2)"),
    has(Code, "(defn trp [arg1]"),
    has(Code, "(* arg1 3)"),
    % ClojureScript banner from clojurescript_from_clojure
    has(Code, "ClojureScript"),
    retractall(user:dbl(_, _)),
    retractall(user:trp(_, _)).

% pred(Name,Arity,Type) spec shape (matches typescript_target's compile_module).
test(compile_module_accepts_pred_terms) :-
    clojurescript_target:init_clojurescript_target,
    assert(user:(dbl(X, R) :- R is X * 2)),
    clojurescript_target:compile_module(
        [pred(dbl, 2, native)],
        [namespace('generated.pt')],
        Code),
    has(Code, "(ns generated.pt)"),
    has(Code, "(defn dbl [arg1]"),
    retractall(user:dbl(_, _)).

% ============================================================================
% G-P5: component emission -- a declared custom_clojure component is emitted
% INTO the CLJS module (and interop-rewritten), where before it was dropped.
% ============================================================================

setup_cljs_component :-
    clojurescript_target:init_clojurescript_target,   % clears collected_component/2
    assert(user:(dbl(X, R) :- R is X * 2)),
    component_registry:declare_component(source, cljs_greet, custom_clojure,
        [ code("(str \"hi \" input)") ]),
    clojure_target:collect_declared_component(source, cljs_greet).

cleanup_cljs_component :-
    retractall(clojure_target:collected_component(_, _)),
    catch(component_registry:retract_component(source, cljs_greet), _, true),
    retractall(user:dbl(_, _)).

test(cljs_component_emission_includes_declared,
     [setup(setup_cljs_component), cleanup(cleanup_cljs_component)]) :-
    clojurescript_target:compile_module(
        [dbl/2],
        [namespace('generated.withcomp')],
        Code),
    % predicate still emitted alongside the component
    has(Code, "(defn dbl [arg1]"),
    % custom_clojure component now present (previously dropped)
    has(Code, "Custom Component: cljs_greet"),
    has(Code, "(defn comp-cljs_greet-invoke"),
    has(Code, "(str \"hi \" input)"),
    % no JVM host call leaked into the CLJS module
    hasnt(Code, "Integer/parseInt").

% A CLJS module that declares NO components stays component-marker-free.
test(cljs_component_free_module_unchanged,
     [setup(clojurescript_target:init_clojurescript_target)]) :-
    assert(user:(dbl(X, R) :- R is X * 2)),
    clojurescript_target:compile_module(
        [dbl/2],
        [namespace('generated.nocomp')],
        Code),
    has(Code, "(ns generated.nocomp)"),
    has(Code, "(defn dbl [arg1]"),
    hasnt(Code, "Custom Component"),
    retractall(user:dbl(_, _)).

% ============================================================================
% G-P7: NEGATION + TYPE-CHECK GUARD CODEGEN (inherited from clojure base)
%
% clojure_guard_condition/3 previously handled ONLY binary comparisons. Guards
% the shared classifier routes to the guard renderer -- negation (\+/not) and
% type-check predicates (integer/1, atom/1, is_list/1, ...) -- had NO clause and
% FAILED at render. CLJS inherits the clojure clause compilation, so these tests
% assert the new clause families render (with the JS interop rewrite) and run
% under nbb matching the SWI oracle.
% ============================================================================

assert_gp7_tc :- assertz((user:tc(X, R) :- (integer(X) -> R = yes ; R = no))).
retract_gp7_tc :- retractall(user:tc(_, _)).

assert_gp7_nm :- assertz((user:nm(X, R) :- (\+ member(X, [1,2,3]) -> R = out ; R = in))).
retract_gp7_nm :- retractall(user:nm(_, _)).

% -- Structural: the guards now render into the CLJS defn ----------------------

test(gp7_typecheck_renders, [setup(assert_gp7_tc), cleanup(retract_gp7_tc)]) :-
    compile_cljs(tc/2, Code),
    % integer/1 -> (integer? arg1) inside the if-then-else
    has(Code, "(if (integer? arg1)"),
    has(Code, "\"yes\""),
    has(Code, "\"no\""),
    hasnt(Code, "Integer/parseInt").   % JVM interop rewritten to js/

test(gp7_negation_member_renders, [setup(assert_gp7_nm), cleanup(retract_gp7_nm)]) :-
    compile_cljs(nm/2, Code),
    % \+ member(X, [1,2,3]) -> (not (some #(= % arg1) [1 2 3]))
    has(Code, "(not (some #(= % arg1) [1 2 3]))"),
    has(Code, "\"out\""),
    has(Code, "\"in\"").

% -- nbb execution vs SWI oracle ----------------------------------------------

test(gp7_typecheck_runs_under_nbb,
     [setup(assert_gp7_tc), cleanup(retract_gp7_tc), condition(nbb_available)]) :-
    compile_cljs(tc/2, Code),
    forall(member(X, [5, 0, -3]), (
        once((user:tc(X, RExp) -> atom_string(RExp, ExpS) ; ExpS = "no")),
        cljs_write_run(Code, [X], Got),
        atom_string(Got, GotS),
        GotS == ExpS
    )).

test(gp7_negation_member_runs_under_nbb,
     [setup(assert_gp7_nm), cleanup(retract_gp7_nm), condition(nbb_available)]) :-
    compile_cljs(nm/2, Code),
    forall(member(X, [1, 2, 4, 7]), (
        once((user:nm(X, RExp) -> atom_string(RExp, ExpS) ; ExpS = "in")),
        cljs_write_run(Code, [X], Got),
        atom_string(Got, GotS),
        GotS == ExpS
    )).

% ============================================================================
% G-P7 follow-up: REGEX match/2[,3] GUARD CODEGEN (inherited from clojure base)
%
% clojure_guard_condition/3 now renders match/2 and match/3 (subject FIRST,
% pattern SECOND, optional 3rd = regex type) as `(re-find (re-pattern "<pat>") x)`.
% re-find is unanchored (mirrors Python re.search) and returns the matched
% substring (truthy) or nil, so it works directly as a boolean guard and under
% (not ...). CLJS inherits the clojure clause compilation; re-find/re-pattern are
% portable, needing no JS interop rewrite. The standalone CLI passes its argv as
% a string (not Integer/parseInt) when the predicate match-tests its first arg.
% Oracle: library(pcre) re_match/2 (match/2 is a DSL marker, not executable).
% ============================================================================

:- use_module(library(pcre)).

assert_gp7m_mfa  :- assertz((user:mfa(X, R) :- (match(X, '^a') -> R = yes ; R = no))).
retract_gp7m_mfa :- retractall(user:mfa(_, _)).

assert_gp7m_mfn  :- assertz((user:mfn(X, R) :- (\+ match(X, '^a') -> R = out ; R = in))).
retract_gp7m_mfn :- retractall(user:mfn(_, _)).

assert_gp7m_m3  :- assertz((user:m3(X, R) :- (match(X, 'foo', pcre) -> R = yes ; R = no))).
retract_gp7m_m3 :- retractall(user:m3(_, _)).

% -- Structural: match renders into the CLJS defn -----------------------------

test(gp7m_match_renders, [setup(assert_gp7m_mfa), cleanup(retract_gp7m_mfa)]) :-
    compile_cljs(mfa/2, Code),
    % match(X, '^a') -> (re-find (re-pattern "^a") arg1)
    has(Code, "(re-find (re-pattern \"^a\") arg1)"),
    has(Code, "\"yes\""),
    has(Code, "\"no\""),
    % CLI passes the argv string through (no int parse), enabling text matching
    has(Code, "(println (mfa (first *command-line-args*)))"),
    hasnt(Code, "Integer/parseInt").

test(gp7m_negation_match_renders, [setup(assert_gp7m_mfn), cleanup(retract_gp7m_mfn)]) :-
    compile_cljs(mfn/2, Code),
    % \+ match(X, '^a') -> (not (re-find (re-pattern "^a") arg1))
    has(Code, "(not (re-find (re-pattern \"^a\") arg1))"),
    has(Code, "\"out\""),
    has(Code, "\"in\"").

test(gp7m_match3_renders, [setup(assert_gp7m_m3), cleanup(retract_gp7m_m3)]) :-
    % match/3 (regex type advisory) renders the same re-find test
    compile_cljs(m3/2, Code),
    has(Code, "(re-find (re-pattern \"foo\") arg1)").

% -- nbb execution vs pcre oracle ---------------------------------------------

test(gp7m_match_runs_under_nbb,
     [setup(assert_gp7m_mfa), cleanup(retract_gp7m_mfa), condition(nbb_available)]) :-
    compile_cljs(mfa/2, Code),
    forall(member(X, ["apple","banana","avocado","cherry"]), (
        ( re_match("^a", X) -> ExpS = "yes" ; ExpS = "no" ),
        cljs_write_run(Code, [X], Got),
        atom_string(Got, GotS),
        GotS == ExpS
    )).

test(gp7m_negation_match_runs_under_nbb,
     [setup(assert_gp7m_mfn), cleanup(retract_gp7m_mfn), condition(nbb_available)]) :-
    compile_cljs(mfn/2, Code),
    forall(member(X, ["apple","banana","avocado","cherry"]), (
        ( \+ re_match("^a", X) -> ExpS = "out" ; ExpS = "in" ),
        cljs_write_run(Code, [X], Got),
        atom_string(Got, GotS),
        GotS == ExpS
    )).

% ============================================================================
% G-P-dedup: UNIQUENESS / ORDER CONSTRAINT HANDLING (inherited from clojure base)
%
% compile_facts_to_clojure previously emitted the raw fact vector regardless of
% any declared unique/unordered constraint. It now consults
% constraint_analyzer:get_constraints/2 (declaration merged over the global
% defaults unique=true/unordered=true) and wraps the vector, mirroring rust/go:
%   - unique(false)           -> raw vector [ ... ]              (no dedup)
%   - unique(true), ordered   -> (vec (distinct [ ... ]))        order-preserving
%   - unique(true), unordered -> (vec (sort (distinct [ ... ]))) sort-based dedup
% CLJS reuses compile_facts_to_clojure/3, so it inherits this with no interop
% rewrite (distinct/sort/vec are portable clojure.core). get-all/-main read
% `facts`, so the whole facts surface inherits the dedup. Oracle: SWI setof/sort.
% ============================================================================

:- use_module('../../src/unifyweaver/core/constraint_analyzer',
              [ declare_constraint/2, clear_constraints/1 ]).

% Facts with a genuine duplicate. compile_facts_to_clojure gathers via clause/2,
% which resolves user:dup here.
assert_dup_facts :-
    assertz(user:dup(a, b)),
    assertz(user:dup(a, b)),
    assertz(user:dup(c, d)).
retract_dup_facts :-
    retractall(user:dup(_, _)),
    clear_constraints(dup/2).

% -- Structural recognition ---------------------------------------------------

test(dedup_default_emits_sort_dedup,
     [setup(assert_dup_facts), cleanup(retract_dup_facts)]) :-
    clojurescript_target:compile_facts_to_clojurescript(dup, 2, Code),
    % default constraints -> dedup + sort
    has(Code, "(vec (sort (distinct ["),
    hasnt(Code, "(str ").   % nothing spurious

test(dedup_unique_false_leaves_raw_vector,
     [setup(assert_dup_facts), cleanup(retract_dup_facts)]) :-
    declare_constraint(dup/2, [unique(false)]),
    clojurescript_target:compile_facts_to_clojurescript(dup, 2, Code),
    hasnt(Code, "distinct"),
    hasnt(Code, "(sort "),
    has(Code, "(def facts").

test(dedup_ordered_emits_distinct_no_sort,
     [setup(assert_dup_facts), cleanup(retract_dup_facts)]) :-
    declare_constraint(dup/2, [unique(true), ordered]),
    clojurescript_target:compile_facts_to_clojurescript(dup, 2, Code),
    has(Code, "(vec (distinct ["),
    hasnt(Code, "(sort ").

% -- nbb execution vs SWI oracle ----------------------------------------------

% Default: -main prints one line per fact ("a:b"), duplicates removed, sorted.
test(dedup_default_runs_under_nbb,
     [setup(assert_dup_facts), cleanup(retract_dup_facts), condition(nbb_available)]) :-
    clojurescript_target:compile_facts_to_clojurescript(dup, 2, Code0),
    string_concat(Code0, "\n(-main)\n", Code),
    cljs_run_lines(Code, GotLines),
    % SWI oracle: distinct pairs, sorted, joined "A:B"
    setof(L, A^B^(user:dup(A, B), format(atom(L), '~w:~w', [A, B])), ExpLines),
    GotLines == ExpLines.

% unique(false): duplicates retained -> three printed lines.
test(dedup_unique_false_runs_under_nbb,
     [setup(assert_dup_facts), cleanup(retract_dup_facts), condition(nbb_available)]) :-
    declare_constraint(dup/2, [unique(false)]),
    clojurescript_target:compile_facts_to_clojurescript(dup, 2, Code0),
    string_concat(Code0, "\n(-main)\n", Code),
    cljs_run_lines(Code, GotLines),
    length(GotLines, N),
    N =:= 3.

% Run CLJS under nbb, returning the non-empty stdout lines as a list of strings.
cljs_run_lines(Code, Lines) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.cljs', File),
    setup_call_cleanup(
        ( open(File, write, W), write(W, Code), close(W) ),
        ( process_create(path(nbb), [File],
                         [stdout(pipe(O)), stderr(null), process(P)]),
          read_string(O, _, Str), close(O), process_wait(P, _),
          split_string(Str, "\n", " \t\r", Parts),
          exclude(==(""), Parts, LinesS),
          maplist(atom_string, Lines, LinesS) ),
        catch(delete_file(File), _, true)).

:- end_tests(clojurescript_target).
