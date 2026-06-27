% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% clojurescript_target.pl - ClojureScript Target for UnifyWeaver
%
% A *variant* of the JVM Clojure target (clojure_target.pl) that emits
% ClojureScript instead of JVM Clojure. Follows the same inheritance pattern
% the Python family uses: this module `use_module`s its base target and only
% overrides the JVM->JS differences. The bulk of the codegen (clause lowering,
% recursion patterns, expression translation) is reused unchanged from
% clojure_target.
%
%   clojurescript_target : clojure_target  ::  python_cython_target : python_target
%
% Override surface (see docs/proposals, "ClojureScript support"):
%   - interop      : Java host calls (Integer/parseInt, Math/abs, Exception)
%                    rewritten to JS host calls (js/parseInt, js/Math.abs,
%                    :default / .-message).
%   - namespace/deps: shadow-cljs.edn (or an inline Scittle <script>) instead
%                    of deps.edn.
%   - build artifact: a JS bundle / browser page instead of a JVM jar.
%
% Runtime: the recommended v1 runtime is Scittle/SCI (borkdude's Small Clojure
% Interpreter) embedded in the SciREPL ClojureScript kernel. The simple /
% native-clause-lowering path and the recursion patterns are the
% browser-supported surface. The stdin/stream pipeline modes (generator /
% pipeline) assume an nbb-style Node runtime rather than browser Scittle and
% are passed through best-effort.
%
% Example:
%   ?- compile_predicate_to_clojurescript(double/2, [], Code).
%   ?- generate_scittle_html("Demo", [main_ns('generated.demo'), cljs(Code)], HTML).

:- module(clojurescript_target, [
    compile_predicate_to_clojurescript/3,   % +Pred/Arity, +Options, -CljsCode
    compile_predicate/3,                     % +Pred/Arity, +Options, -Code (registry dispatch)
    compile_facts_to_clojurescript/3,        % +Pred, +Arity, -CljsCode
    clojurescript_from_clojure/2,            % +ClojureCode, -CljsCode (rewrite + banner)
    clojurescript_interop_rewrite/2,         % +ClojureCode, -CljsCode
    generate_shadow_cljs_edn/2,              % +Options, -ShadowFile
    generate_scittle_html/3,                 % +Title, +Options, -HTML
    write_clojurescript_program/2,           % +CljsCode, +FilePath
    init_clojurescript_target/0,             % Initialize ClojureScript target
    test_clojurescript_pipeline_mode/0       % Smoke tests
]).

:- use_module(library(lists)).

% Inherit the JVM Clojure target. Everything not overridden below is reused.
:- use_module(clojure_target).

%% init_clojurescript_target
%  Initialize the ClojureScript target (delegates to the Clojure base).
init_clojurescript_target :-
    init_clojure_target.

%% ============================================
%% PUBLIC API
%% ============================================

%% compile_predicate_to_clojurescript(+Pred/Arity, +Options, -CljsCode)
%  Compile a predicate to ClojureScript by delegating to the Clojure base
%  target and rewriting the JVM host interop into JS host interop.
compile_predicate_to_clojurescript(PredIndicator, Options, CljsCode) :-
    compile_predicate_to_clojure(PredIndicator, Options, ClojureCode),
    clojurescript_from_clojure(ClojureCode, CljsCode).

%% clojurescript_from_clojure(+ClojureCode, -CljsCode)
%  Turn JVM-Clojure source into ClojureScript: rewrite host interop, then
%  prepend the CLJS banner. Shared by the single-predicate path above and the
%  recursive_compiler's transitive-closure path (which reuses the JVM Clojure
%  templates and post-processes them here), keeping the JVM->JS translation in
%  one place.
clojurescript_from_clojure(ClojureCode, CljsCode) :-
    clojurescript_interop_rewrite(ClojureCode, Rewritten),
    cljs_banner(Banner),
    string_concat(Banner, Rewritten, CljsCode).

%% compile_predicate(+Pred/Arity, +Options, -Code)
%  Thin wrapper so target_registry's compile_to_target/4 can dispatch here.
compile_predicate(PredIndicator, Options, Code) :-
    compile_predicate_to_clojurescript(PredIndicator, Options, Code).

%% compile_facts_to_clojurescript(+Pred, +Arity, -CljsCode)
%  Facts export, reusing the Clojure base then rewriting interop.
compile_facts_to_clojurescript(Pred, Arity, CljsCode) :-
    compile_facts_to_clojure(Pred, Arity, ClojureCode),
    clojurescript_interop_rewrite(ClojureCode, CljsCode).

%% ============================================
%% INTEROP REWRITE (the core JVM -> JS override)
%% ============================================

%% clojurescript_interop_rewrite(+ClojureCode, -CljsCode)
%  Translate the JVM host-interop tokens emitted by clojure_target into their
%  ClojureScript equivalents. Order matters: more specific rewrites precede
%  the generic `Math/` catch-all so they are not double-applied.
clojurescript_interop_rewrite(In, Out) :-
    cljs_interop_rules(Rules),
    foldl(apply_cljs_rule, Rules, In, Out0),
    ( string(Out0) -> Out = Out0 ; atom_string(Out0, Out) ).

apply_cljs_rule(From-To, In, Out) :-
    cljs_replace_all(In, From, To, Out).

%% cljs_interop_rules(-Rules)
%  Ordered list of From-To substring rewrites (JVM host -> JS host).
cljs_interop_rules([
    % --- numeric parsing (Java boxed types -> JS globals) ---
    'Integer/parseInt'-'js/parseInt',
    'Long/parseLong'-'js/parseInt',
    'Double/parseDouble'-'js/parseFloat',
    'Float/parseFloat'-'js/parseFloat',
    % --- Math host object: specific first, then generic prefix ---
    'Math/abs'-'js/Math.abs',
    'Math/rint'-'js/Math.round',   % JS has no Math.rint; round is fine for the integral test
    'Math/'-'js/Math.',
    % --- integer coercion: CLJS has no (long x) ---
    '(long '-'(js/Math.trunc ',
    % --- JVM type hints are meaningless in CLJS ---
    '^String '-'',
    % --- exceptions: JS has no java.lang.Exception ---
    '(catch Exception '-'(catch :default ',
    '(.getMessage '-'(.-message ',
    % --- host IO / json libraries are JVM-only; map to JS analogues ---
    '[clojure.data.json :as json]'-'[clojure.string :as cljs-str]',
    '(json/write-str '-'(cljs-json-write-str ',
    '(json/read-str '-'(cljs-json-read-str '
]).

%% cljs_replace_all(+In, +From, +To, -Out)
%  Global substring replacement using the atomic_list_concat split/join trick.
cljs_replace_all(In, From, To, Out) :-
    ( sub_atom_icasechk_safe(In, From)
    ->  atomic_list_concat(Parts, From, In),
        atomic_list_concat(Parts, To, Out)
    ;   Out = In
    ).

% sub_atom_icasechk_safe/2 - true if From occurs in In (atom or string).
sub_atom_icasechk_safe(In, From) :-
    ( string(In) -> S = In ; atom_string(In, S) ),
    sub_string(S, _, _, _, From).

%% cljs_banner(-Banner)
%  Header noting the output is ClojureScript (runs under Scittle/SCI or nbb).
cljs_banner(";; Target: ClojureScript (runs under Scittle/SCI or nbb)\n;; Generated by UnifyWeaver ClojureScript Target (variant of clojure_target)\n").

%% ============================================
%% NAMESPACE / DEPS OVERRIDE (shadow-cljs.edn)
%% ============================================

%% generate_shadow_cljs_edn(+Options, -ShadowFile)
%  ClojureScript build config, the JS-world analogue of generate_deps_edn/2.
%  Options:
%    - main_ns(NS)    : the namespace whose -main is the module init fn
%    - output_dir(D)  : compiled JS output directory (default "public/js")
generate_shadow_cljs_edn(Options, ShadowFile) :-
    clj_option(main_ns(MainNs), Options, 'generated.app'),
    clj_option(output_dir(OutDir), Options, "public/js"),
    format(string(ShadowFile),
";; Generated by UnifyWeaver ClojureScript Target
{:source-paths [\"src\"]
 :dependencies []
 :builds
 {:app {:target :browser
        :output-dir \"~w\"
        :asset-path \"/js\"
        :modules {:main {:init-fn ~w/-main}}}}}
", [OutDir, MainNs]).

%% ============================================
%% SCITTLE BROWSER PAGE (makes the target demonstrable)
%% ============================================

%% generate_scittle_html(+Title, +Options, -HTML)
%  Wrap generated ClojureScript in a Scittle <script> so it runs in-browser
%  with no build step -- the page the SciREPL `clojurescript` kernel would
%  execute. Mirrors the Pyodide target's generate_pyodide_html/3.
%  Options:
%    - cljs(Code)        : the ClojureScript source to embed (required)
%    - scittle_src(URL)  : Scittle loader URL (default jsdelivr CDN; bundle
%                          locally for offline use per the kernel proposal)
generate_scittle_html(Title, Options, HTML) :-
    ( member(cljs(Code), Options) -> true ; Code = ";; (no cljs supplied)" ),
    clj_option(scittle_src(ScittleSrc), Options,
        "https://cdn.jsdelivr.net/npm/scittle@0.6.22/dist/scittle.js"),
    format(string(HTML),
"<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>~w</title>
  <!-- Bundle Scittle locally for offline SciREPL use (see kernel proposal) -->
  <script src=\"~w\"></script>
</head>
<body>
  <h1>~w</h1>
  <pre id=\"output\"></pre>
  <script type=\"application/x-scittle\">
~w
  </script>
</body>
</html>
", [Title, ScittleSrc, Title, Code]).

%% ============================================
%% UTILITIES
%% ============================================

write_clojurescript_program(CljsCode, FilePath) :-
    write_clojure_program(CljsCode, FilePath).

% Local option/3 (clojure_target's option/3 is not exported).
clj_option(Option, Options, _Default) :-
    member(Option, Options), !.
clj_option(Option, _Options, Default) :-
    Option =.. [_, Default].

%% ============================================
%% TESTS
%% ============================================

test_clojurescript_pipeline_mode :-
    format('~n=== Testing ClojureScript Target ===~n~n'),

    % Test 1: interop rewrite translates Java host calls to JS host calls
    format('Test 1: JVM -> JS interop rewrite~n'),
    clojurescript_interop_rewrite(
        "(println (foo (Integer/parseInt (first *command-line-args*))))\n(Math/abs x)\n(catch Exception e (.getMessage e))",
        Rw),
    ( sub_string(Rw, _, _, _, "js/parseInt"),
      \+ sub_string(Rw, _, _, _, "Integer/parseInt")
    ->  format('  [PASS] Integer/parseInt -> js/parseInt~n')
    ;   format('  [FAIL] parseInt not rewritten~n') ),
    ( sub_string(Rw, _, _, _, "js/Math.abs")
    ->  format('  [PASS] Math/abs -> js/Math.abs~n')
    ;   format('  [FAIL] Math/abs not rewritten~n') ),
    ( sub_string(Rw, _, _, _, "(catch :default e"),
      sub_string(Rw, _, _, _, "(.-message e")
    ->  format('  [PASS] Exception/getMessage -> :default/.-message~n')
    ;   format('  [FAIL] exception interop not rewritten~n') ),

    % Test 2: full predicate compile reuses base codegen + rewrites + banner
    format('~nTest 2: compile_predicate_to_clojurescript reuses Clojure base~n'),
    compile_predicate_to_clojurescript(double/2, [], Code2),
    ( sub_string(Code2, _, _, _, "ClojureScript")
    ->  format('  [PASS] ClojureScript banner present~n')
    ;   format('  [FAIL] missing banner~n') ),
    ( sub_string(Code2, _, _, _, "defn")
    ->  format('  [PASS] reused Clojure (defn ...) codegen~n')
    ;   format('  [INFO] no defn in this predicate~n') ),
    ( \+ sub_string(Code2, _, _, _, "Integer/parseInt")
    ->  format('  [PASS] no leftover JVM parseInt~n')
    ;   format('  [FAIL] JVM parseInt leaked through~n') ),

    % Test 3: shadow-cljs.edn instead of deps.edn
    format('~nTest 3: shadow-cljs.edn build config~n'),
    generate_shadow_cljs_edn([main_ns('generated.demo')], Shadow),
    ( sub_string(Shadow, _, _, _, ":target :browser"),
      sub_string(Shadow, _, _, _, "generated.demo/-main")
    ->  format('  [PASS] Generated shadow-cljs.edn~n')
    ;   format('  [FAIL] Invalid shadow-cljs.edn~n') ),

    % Test 4: Scittle HTML wrapper (browser demo / SciREPL kernel input)
    format('~nTest 4: Scittle HTML wrapper~n'),
    generate_scittle_html("Demo", [cljs("(println :hi)")], HTML),
    ( sub_string(HTML, _, _, _, "application/x-scittle"),
      sub_string(HTML, _, _, _, "(println :hi)")
    ->  format('  [PASS] Generated Scittle page~n')
    ;   format('  [FAIL] Invalid Scittle page~n') ),

    format('~n=== ClojureScript Target Tests Complete ===~n').
