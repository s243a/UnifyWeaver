% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% clojure_target.pl - Clojure Target for UnifyWeaver
% Generates Clojure programs for record/field processing
% Supports lazy sequences, immutable data, and Java interop

:- encoding(utf8).

:- module(clojure_target, [
    compile_predicate_to_clojure/3,    % +Predicate, +Options, -ClojureCode
    compile_clojure_pipeline/3,        % +Predicates, +Options, -ClojureCode
    compile_facts_to_clojure/3,        % +Pred, +Arity, -ClojureCode  -- NEW
    compile_module/3,                  % +Predicates, +Options, -ClojureCode
    clojure_predicate_defn/3,          % +Pred/Arity, +Options, -DefnCode
    collect_declared_component/2,      % +Category, +Name (record component to emit)
    compile_collected_components/1,    % -Code (emit all collected components)
    generate_deps_edn/2,               % +Options, -DepsFile
    write_clojure_program/2,           % +ClojureCode, +FilePath
    init_clojure_target/0,             % Initialize Clojure target
    test_clojure_pipeline_mode/0       % Test pipeline mode
]).

:- use_module(library(lists)).

% Binding system integration
:- use_module('../core/binding_registry').
:- use_module('../core/clause_body_analysis').

% Uniqueness/order constraint handling (G-P-dedup). The constraint analyzer
% gives each predicate's effective unique/unordered constraints (its declaration
% merged over the global defaults, unique=true/unordered=true). The facts export
% honors them the way the mature rust/go targets do: unique(true) deduplicates
% the emitted collection, unordered(true) additionally permits sort-based dedup,
% and unique(false) leaves it untouched. Consumed only — never mutated. Portable
% clojure.core (distinct/sort/vec), so clojurescript_target inherits it via the
% shared compile_facts_to_clojure/3 with no interop rewrite needed.
:- use_module('../core/constraint_analyzer', [get_constraints/2]).

% Component pattern integration (G-P5). Load the component registry and the
% custom_clojure component type. custom_clojure self-registers via a
% ':- initialization(..., now)' directive, so loading it here (empty import
% list) triggers that registration and makes the type available to
% declare_component/4. Without this the module was orphaned and dead.
:- use_module('../core/component_registry').
:- use_module('clojure_runtime/custom_clojure', []).

% Track required imports
:- dynamic required_clojure_require/1.

% Track collected components for module emission (G-P5)
:- dynamic collected_component/2.

%% init_clojure_target
init_clojure_target :-
    retractall(required_clojure_require(_)),
    retractall(collected_component(_, _)).

%% clear_clojure_requires
clear_clojure_requires :-
    retractall(required_clojure_require(_)).

%% collect_clojure_require(+Require)
collect_clojure_require(Require) :-
    (   required_clojure_require(Require)
    ->  true
    ;   assertz(required_clojure_require(Require))
    ).

%% get_clojure_requires(-Requires)
get_clojure_requires(Requires) :-
    findall(R, required_clojure_require(R), Requires).

%% format_clojure_requires(+Requires, -FormattedStr)
format_clojure_requires([], "").
format_clojure_requires(Requires, FormattedStr) :-
    Requires \= [],
    sort(Requires, UniqueRequires),
    findall(Formatted,
        (   member(Require, UniqueRequires),
            format(string(Formatted), "  (:require ~w)~n", [Require])
        ),
        FormattedList),
    atomic_list_concat(FormattedList, '', FormattedStr).

%% ============================================
%% PUBLIC API
%% ============================================

%% compile_predicate_to_clojure(+Predicate, +Options, -ClojureCode)
compile_predicate_to_clojure(PredIndicator, Options, ClojureCode) :-
    (   PredIndicator = _Module:Pred/Arity
    ->  true
    ;   PredIndicator = Pred/Arity
    ),
    format('=== Compiling ~w/~w to Clojure ===~n', [Pred, Arity]),

    clear_clojure_requires,

    % Check mode
    (   option(generator_mode(true), Options)
    ->  format('  Mode: Generator (lazy-seq)~n'),
        compile_generator_mode_clojure(Pred, Arity, Options, ClojureCode)
    ;   option(pipeline_input(true), Options)
    ->  format('  Mode: Pipeline (streaming)~n'),
        compile_pipeline_mode_clojure(Pred, Arity, Options, ClojureCode)
    ;   format('  Mode: Simple predicate~n'),
        compile_simple_mode_clojure(Pred, Arity, Options, ClojureCode)
    ).

%% ============================================
%% SIMPLE MODE
%% ============================================

% A3 whole-program lowering (structural + general clause compilers). A RESCUE
% path, tried FIRST but claiming only a predicate whose historical clause-body
% answer would be defective -- it failed, it leaked an internal variable name,
% it stringified a compound term, or its parentheses do not balance. Every
% predicate the historical path lowers correctly falls straight past this clause
% and compiles exactly as before, byte for byte. A predicate with a fact clause
% is excluded too, so genuine fact tables keep going to compile_facts_to_clojure/3.
compile_simple_mode_clojure(Pred, Arity, _Options, ClojureCode) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    \+ clj_ground_fact_predicate(Clauses),
    clj_clause_body_defective(Pred/Arity, Clauses),
    native_clj_whole(Pred/Arity, Clauses, Defn0),
    !,
    clj_attach_runtime(Defn0, Defn),
    clj_fn_name(Pred, Arity, PredStr),
    clj_a3_cli_entry(Pred, Arity, PredStr, CliEntry),
    format(string(ClojureCode),
';; Generated by UnifyWeaver Clojure Target - A3 Whole-Program Lowering
;; Predicate: ~w/~w

~w

~w', [PredStr, Arity, Defn, CliEntry]).

% Try native clause body lowering first
compile_simple_mode_clojure(Pred, Arity, _Options, ClojureCode) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    native_clojure_clause_body(Pred/Arity, Clauses, FuncBody),
    !,
    atom_string(Pred, PredStr),
    Arity1 is Arity - 1,
    build_clojure_arg_list(Arity1, ArgList),
    clojure_native_cli_entry(Pred, Arity, PredStr, CliEntry),
    format(string(ClojureCode),
';; Generated by UnifyWeaver Clojure Target - Native Clause Lowering
;; Predicate: ~w/~w

(defn ~w [~w]
~w)

~w', [PredStr, Arity, PredStr, ArgList, FuncBody, CliEntry]).

% Fallback stub
compile_simple_mode_clojure(Pred, Arity, _Options, ClojureCode) :-
    format(string(ClojureCode),
";; Generated by UnifyWeaver Clojure Target
;; Predicate: ~w/~w

(ns generated.~w)

(defn ~w
  \"Predicate ~w/~w\"
  [& args]
  ;; TODO: Implement ~w logic
  nil)

(defn -main
  [& args]
  ;; TODO: Add main logic
  (println \"Hello from ~w\"))
", [Pred, Arity, Pred, Pred, Pred, Arity, Pred, Pred]).

%% clojure_native_cli_entry(+Pred, +Arity, +PredStr, -CliEntry)
%  The standalone CLI entry point read from *command-line-args*. When the
%  predicate's first argument is a list (empty or cons in some clause), the
%  single argv string is parsed as a comma-separated integer vector; otherwise
%  it is parsed as a single integer. Integer/parseInt is used so the JVM (bb)
%  path stays valid; the ClojureScript interop rewrite maps it to js/parseInt.
clojure_native_cli_entry(Pred, Arity, PredStr, CliEntry) :-
    (   clojure_pred_list_input(Pred, Arity)
    ->  format(string(CliEntry),
';; CLI entry point
(when *command-line-args*
  (let [s (first *command-line-args*)
        xs (if (or (nil? s) (= s "")) [] (mapv #(Integer/parseInt %) (.split s ",")))]
    (println (~w xs))))
', [PredStr])
    ;   clojure_pred_string_input(Pred, Arity)
    ->  format(string(CliEntry),
';; CLI entry point
(when *command-line-args*
  (println (~w (first *command-line-args*))))
', [PredStr])
    ;   format(string(CliEntry),
';; CLI entry point
(when *command-line-args*
  (println (~w (Integer/parseInt (first *command-line-args*)))))
', [PredStr])
    ).

%% clojure_pred_string_input(+Pred, +Arity)
%  True when some clause of the predicate regex-matches its first argument, so
%  the standalone CLI must pass that argv through as a string rather than
%  parsing it as an integer.
clojure_pred_string_input(Pred, Arity) :-
    functor(Head, Pred, Arity),
    user:clause(Head, Body),
    Head =.. [_|Args],
    Args = [First|_],
    var(First),
    clojure_body_match_subject(Body, First),
    !.

%% clojure_pred_list_input(+Pred, +Arity)
%  True when the predicate takes a list as its first argument (some clause head
%  has [] or a cons as its first argument).
clojure_pred_list_input(Pred, Arity) :-
    functor(Head, Pred, Arity),
    user:clause(Head, _),
    Head =.. [_|Args],
    Args = [First|_],
    ( First == [] ; compound(First), First = [_|_] ),
    !.

%% ============================================
%% COMPILE FACTS TO CLOJURE
%% ============================================

compile_facts_to_clojure(Pred, Arity, ClojureCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),
    
    findall(Args, (clause(Head, true), Head =.. [_|Args]), AllFacts),
    
    (   AllFacts == []
    ->  FactEntries = "   ;; No facts defined"
    ;   findall(Entry, (
            member(Args, AllFacts),
            format_clojure_fact_entry(Args, Entry)
        ), Entries),
        atomic_list_concat(Entries, '\n', FactEntries)
    ),

    % Uniqueness/order constraint handling (G-P-dedup). Wrap the fact vector per
    % the effective constraints, mirroring rust/go dedup semantics:
    %   - unique(false)           -> the raw vector (no dedup, unchanged)
    %   - unique(true), ordered   -> (vec (distinct ...))  order-preserving dedup
    %   - unique(true), unordered -> (vec (sort (distinct ...)))  sort-based dedup
    % Default constraints (unique=true, unordered=true) therefore emit a
    % deduplicated, sorted vector. get-all/contains?/-main all read `facts`, so
    % the whole facts surface inherits the dedup.
    get_constraints(Pred/Arity, Constraints),
    clojure_facts_expr(FactEntries, Constraints, FactsExpr),

    format(string(ClojureCode),
';; Generated by UnifyWeaver Clojure Target - Facts Export
;; Predicate: ~w/~w

(ns generated.~w-facts)

(def facts
  "~w facts as Clojure vectors"
  ~w)

(defn get-all
  "Get all facts as a sequence"
  []
  (seq facts))

(defn contains?
  "Check if a fact exists"
  [& args]
  (some #(= % (vec args)) facts))

(defn -main
  [& args]
  (doseq [f facts]
    (println (clojure.string/join ":" f))))
', [PredStr, Arity, PredStr, PredStr, FactsExpr]).

%% clojure_facts_expr(+FactEntries, +Constraints, -Expr)
%  Build the (def facts ...) value expression from the effective uniqueness/order
%  constraints. Plain clojure.core forms (distinct/sort/vec) so both JVM Clojure
%  and ClojureScript (via clojurescript_target's shared reuse) get the behavior
%  with no interop rewrite.
clojure_facts_expr(FactEntries, Constraints, Expr) :-
    (   memberchk(unique(false), Constraints)
    ->  format(string(Expr), '[~w]', [FactEntries])
    ;   memberchk(unordered(false), Constraints)
    ->  % unique + ordered: order-preserving dedup
        format(string(Expr), '(vec (distinct [~w]))', [FactEntries])
    ;   % unique + unordered (incl. default): dedup then sort (sort-based dedup)
        format(string(Expr), '(vec (sort (distinct [~w])))', [FactEntries])
    ).

format_clojure_fact_entry(Args, Entry) :-
    findall(Formatted, (
        member(Arg, Args),
        format(string(Formatted), '"~w"', [Arg])
    ), FormattedArgs),
    atomic_list_concat(FormattedArgs, ' ', ArgsStr),
    format(string(Entry), '   [~w]', [ArgsStr]).

%% ============================================
%% COMPONENT COLLECTION + MODULE COMPILATION (G-P5 / G-P6)
%% ============================================

%% collect_declared_component(+Category, +Name)
%  Record that a declared component instance is used in this module, so
%  compile_module/3 will emit its compiled code. Mirrors the python/typescript
%  emit-loop model (python_target.pl:~187, typescript_target.pl:~165).
collect_declared_component(Category, Name) :-
    (   collected_component(Category, Name)
    ->  true
    ;   assertz(collected_component(Category, Name))
    ).

%% compile_collected_components(-Code)
%  Compile every collected component to Clojure source by delegating to
%  component_registry:compile_component/4 for each. Returns '' when no
%  components were collected, so component-free modules are unchanged.
compile_collected_components(Code) :-
    findall(CompCode, (
        collected_component(Category, Name),
        component_registry:compile_component(Category, Name, [], CompCode)
    ), CompCodes),
    (   CompCodes = []
    ->  Code = ''
    ;   atomic_list_concat(CompCodes, '\n\n', Code)
    ).

%% clojure_predicate_defn(+Pred/Arity, +Options, -DefnCode)
%  Produce just the top-level (defn ...) form for a predicate (no file header
%  and no CLI entry point), suitable for inclusion inside a multi-predicate
%  module. Uses the same native clause-body lowering as the single-predicate
%  path; falls back to a stub defn when the predicate cannot be lowered.
% A3 whole-program lowering, gated exactly as the single-predicate path above:
% only a predicate whose historical answer would be defective is claimed here.
clojure_predicate_defn(PredIndicator, _Options, DefnCode) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    \+ clj_ground_fact_predicate(Clauses),
    % Inside a module this reads the set compile_module/3 fixed up front; outside
    % one it asks the same per-predicate question the dispatcher asks.
    clj_a3_compiled(Pred, Arity),
    native_clj_whole(Pred/Arity, Clauses, DefnCode),
    !.

clojure_predicate_defn(PredIndicator, _Options, DefnCode) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    native_clojure_clause_body(Pred/Arity, Clauses, FuncBody),
    !,
    atom_string(Pred, PredStr),
    Arity1 is Arity - 1,
    build_clojure_arg_list(Arity1, ArgList),
    format(string(DefnCode),
'(defn ~w [~w]
~w)', [PredStr, ArgList, FuncBody]).
clojure_predicate_defn(PredIndicator, _Options, DefnCode) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    atom_string(Pred, PredStr),
    format(string(DefnCode),
'(defn ~w
  "Predicate ~w/~w"
  [& args]
  ;; TODO: Implement ~w logic
  nil)', [PredStr, Pred, Arity, Pred]).

%% normalize_module_preds(+In, -Out)
%  Accept both `Name/Arity` and `pred(Name, Arity, _Type)` predicate specs in a
%  module list (the latter matches typescript_target's compile_module/3 shape),
%  normalising to a list of Name/Arity.
normalize_module_preds([], []).
normalize_module_preds([pred(Name, Arity, _Type)|T], [Name/Arity|T2]) :- !,
    normalize_module_preds(T, T2).
normalize_module_preds([Name/Arity|T], [Name/Arity|T2]) :- !,
    normalize_module_preds(T, T2).
normalize_module_preds([Other|T], [Other|T2]) :-
    normalize_module_preds(T, T2).

%% compile_module(+Predicates, +Options, -ClojureCode)
%  Compile several predicates into a single Clojure namespace: an (ns ...) form,
%  each predicate's (defn ...), then any declared components. This is the base
%  multi-predicate module compiler; clojurescript_target reuses it (then applies
%  its JVM->JS interop rewrite + banner). Predicates may be `Name/Arity` or
%  `pred(Name, Arity, Type)` terms.
%
%  Options:
%    - namespace(NS) : the module namespace (default 'generated.module')
%  Options (added by the A3 port):
%    - include_dependencies(true) : the module is named by its ENTRY predicate
%      and every predicate it transitively calls is pulled in, callees first
%      (G-A3-6). Off by default, so an explicit predicate list is untouched.
compile_module(Predicates0, Options, Code) :-
    option(namespace(Namespace), Options, 'generated.module'),
    (   option(include_dependencies(true), Options)
    ->  clj_dep_closure(Predicates0, Predicates)
    ;   Predicates = Predicates0
    ),
    normalize_module_preds(Predicates, PredList),
    % ONE MODULE, ONE SET OF NAMES. Which predicates the A3 paths claim is fixed
    % ONCE, up front, and published for the whole compile (clj_emitted_name/3
    % reads it), so a call site, a `(declare ...)` entry and a `(defn ...)`
    % header can never disagree about how a function is spelled.
    clj_module_a3_set(PredList, A3Set),
    clj_set_a3_set(A3Set),
    findall(DefnCode, (
        member(P, PredList),
        clojure_predicate_defn(P, Options, DefnCode)
    ), DefnCodes),
    atomic_list_concat(DefnCodes, '\n\n', PredsSection0),
    % FORWARD DECLARATIONS. This is the one place Clojure needs something
    % JavaScript gets for free: a JS `function` declaration is hoisted to the top
    % of the module scope, so the TS lane may emit a caller before its callee and
    % a mutually recursive pair in either order. A Clojure var must exist before
    % it is referenced, so a multi-predicate module opens with `(declare ...)`
    % naming every function it defines. That single form makes call order
    % irrelevant here too.
    clj_module_declares(PredList, DeclCode),
    % The A3 runtime (the failure sentinel, the char helpers) is carried by each
    % predicate that needs it; a namespace may not define the same var twice, so
    % it is lifted out of the individual defns and emitted once, ahead of them.
    clj_module_runtime(PredsSection0, PredsSection, RuntimeSection),
    % Emit any declared components (G-P5). '' when none were collected, so a
    % component-free module carries no component markers.
    compile_collected_components(ComponentsCode),
    (   ComponentsCode == ''
    ->  Body = PredsSection
    ;   format(string(Body), '~w\n\n~w', [PredsSection, ComponentsCode])
    ),
    format(string(Code),
';; Generated by UnifyWeaver Clojure Target - Module
;; Namespace: ~w

(ns ~w)

~w~w~w
', [Namespace, Namespace, RuntimeSection, DeclCode, Body]).

%% clj_module_a3_set(+PredList, -Set)
%
%  The predicates in this module that the A3 paths compile. It is a CLOSURE, not
%  just "the ones the dispatcher would claim": if an A3-compiled predicate calls
%  one whose A3 calling convention differs from the historical one -- several
%  outputs, no output, the failure sentinel, an overloaded name -- then that
%  callee must be compiled by the A3 path too, or the caller would be speaking a
%  convention the callee does not implement. A callee both paths agree about is
%  left where it was, which is what keeps a module of ordinary shapes byte-for-
%  byte unchanged.
%
%  The closure runs in BOTH directions, because the name problem is symmetric:
%
%    FORWARD   an A3 predicate calls one whose A3 convention differs from the
%              historical one -> the callee joins the set.
%    BACKWARD  a historical-path predicate calls one that is IN the set and whose
%              A3 spelling differs from its raw functor -> the CALLER joins the
%              set, because the historical path would emit the raw name and
%              nothing would define it. `merge_flags/3` calling `merge_flags_/3`
%              is exactly this, and it is how the direction was found: the
%              namespace compiled, and nbb refused it with "Unable to resolve
%              symbol: merge_flags_".
clj_module_a3_set(PredList, Set) :-
    findall(P/A,
            ( member(X, PredList), clj_norm_pa(X, P, A), clj_a3_claims(P, A) ),
            Seed0),
    sort(Seed0, Seed),
    clj_a3_closure(PredList, Seed, Set).

clj_norm_pa(_M:P/A, P, A) :- !.
clj_norm_pa(P/A, P, A).

clj_a3_closure(PredList, Set0, Set) :-
    findall(Q/QA,
            ( member(P/A, Set0),
              clj_pred_callees(P/A, Callees),
              member(Q/QA, Callees),
              \+ memberchk(Q/QA, Set0),
              clj_conv_differs(Q, QA)
            ),
            Fwd),
    findall(P2/A2,
            ( member(X2, PredList), clj_norm_pa(X2, P2, A2),
              \+ memberchk(P2/A2, Set0),
              clj_pred_callees(P2/A2, Callees2),
              member(Q2/QA2, Callees2),
              memberchk(Q2/QA2, Set0),
              clj_fn_name(Q2, QA2, N2), atom_string(Q2, Raw2), N2 \== Raw2
            ),
            Bwd),
    append(Fwd, Bwd, New0),
    sort(New0, New),
    (   New == []
    ->  Set = Set0
    ;   append(Set0, New, S1), sort(S1, S2),
        clj_a3_closure(PredList, S2, Set)
    ).

%% clj_module_declares(+PredList, -DeclCode)
%  '' for a single-predicate module (nothing can be forward-referenced there), so
%  such a module stays byte-for-byte what it always was.
clj_module_declares(PredList, DeclCode) :-
    (   PredList = [_, _|_]
    ->  findall(N, ( member(P, PredList),
                     ( P = _:Pred/Arity -> true ; P = Pred/Arity ),
                     clj_emitted_name(Pred, Arity, N) ), Names),
        atomic_list_concat(Names, ' ', NameStr),
        format(string(DeclCode), '(declare ~w)\n\n', [NameStr])
    ;   DeclCode = ''
    ).

%% clj_module_runtime(+Section0, -Section, -RuntimeSection)
clj_module_runtime(Section0, Section, RuntimeSection) :-
    clj_strip_runtime_blocks(Section0, Section),
    (   clj_runtime_block(Section, Block)
    ->  format(string(RuntimeSection), '~w\n', [Block])
    ;   RuntimeSection = ''
    ).

clj_strip_runtime_blocks(In, Out) :-
    clj_runtime_begin(Begin), clj_runtime_end(End),
    atom_string(In, S),
    (   sub_string(S, Before, _, _, Begin),
        sub_string(S, EndAt, EndLen, After, End)
    ->  sub_string(S, 0, Before, _, Head),
        Skip is EndAt + EndLen,
        sub_string(S, Skip, After, 0, Tail0),
        ( sub_string(Tail0, 0, 1, _, "\n") -> sub_string(Tail0, 1, _, 0, Tail) ; Tail = Tail0 ),
        string_concat(Head, Tail, S1),
        clj_strip_runtime_blocks(S1, Out)
    ;   Out = S
    ).

%% ============================================
%% GENERATOR MODE (Clojure's lazy-seq)
%% ============================================

compile_generator_mode_clojure(Pred, Arity, Options, ClojureCode) :-
    collect_clojure_require('[clojure.data.json :as json]'),
    collect_clojure_require('[clojure.java.io :as io]'),
    
    option(namespace(Namespace), Options, 'generated.pipeline'),
    
    % Gather clauses
    functor(Head, Pred, Arity),
    findall((Head, Body), clause(Head, Body), Clauses),
    
    % Generate process function body
    (   Clauses == []
    ->  ProcessBody = "  ;; No clauses found - yield input unchanged\n  (list record)"
    ;   generate_generator_body_clojure(Clauses, ProcessBody)
    ),
    
    get_clojure_requires(RequireList),
    format_clojure_requires(RequireList, RequiresStr),
    
    format(string(ClojureCode),
";; Generated by UnifyWeaver Clojure Target - Generator Mode
;; Predicate: ~w/~w
;; Uses Clojure's lazy-seq for lazy evaluation

(ns ~w
~w)

(defn process
  \"Process a single record, returning zero or more results (lazy).\"
  [record]
~w)

(defn process-all
  \"Process all records from a sequence, flattening results.\"
  [records]
  (mapcat process records))

(defn parse-json-line
  \"Parse a JSONL line to a map.\"
  [line]
  (try
    (json/read-str line :key-fn keyword)
    (catch Exception e
      (binding [*out* *err*]
        (println \"JSON parse error:\" (.getMessage e)))
      nil)))

(defn to-json
  \"Convert a map to JSON string.\"
  [m]
  (json/write-str m))

(defn run-pipeline
  \"Read JSONL from stdin, process, write JSONL to stdout.\"
  []
  (with-open [rdr (io/reader *in*)]
    (doseq [result (->> (line-seq rdr)
                        (filter seq)
                        (keep parse-json-line)
                        process-all)]
      (println (to-json result)))))

(defn -main
  [& args]
  (run-pipeline))
", [Pred, Arity, Namespace, RequiresStr, ProcessBody]).

%% generate_generator_body_clojure(+Clauses, -Code)
generate_generator_body_clojure(Clauses, Code) :-
    Clauses = [(Head, _)|_],
    functor(Head, Name, _),
    (   is_recursive_predicate_clojure(Name, Clauses)
    ->  partition(is_recursive_clause_clojure(Name), Clauses, RecClauses, BaseClauses),
        compile_generator_recursive_clojure(Name, BaseClauses, RecClauses, Code)
    ;   findall(ClauseCode, 
            (member((H, B), Clauses), translate_generator_clause_clojure(H, B, ClauseCode)),
            ClauseCodes),
        atomic_list_concat(ClauseCodes, '\n', Code)
    ).

translate_generator_clause_clojure(Head, Body, Code) :-
    Head =.. [_Pred|Args],
    generate_input_extraction_clojure(Args, InputCode),
    (   Body == true
    ->  BodyCode = "    ;; Fact - unconditional"
    ;   translate_generator_body_clojure(Body, BodyCode)
    ),
    format(string(Code), "~w\n~w\n    (list record)", [InputCode, BodyCode]).

translate_generator_body_clojure((Goal, Rest), Code) :-
    !,
    translate_generator_goal_clojure(Goal, Code1),
    translate_generator_body_clojure(Rest, Code2),
    format(string(Code), "~w\n~w", [Code1, Code2]).
translate_generator_body_clojure(Goal, Code) :-
    translate_generator_goal_clojure(Goal, Code).

translate_generator_goal_clojure(>(X, Y), Code) :-
    !, expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
    format(string(Code), "    (when-not (> ~w ~w) (return nil))", [CX, CY]).

translate_generator_goal_clojure(<(X, Y), Code) :-
    !, expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
    format(string(Code), "    (when-not (< ~w ~w) (return nil))", [CX, CY]).

translate_generator_goal_clojure(=:=(X, Y), Code) :-
    !, expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
    format(string(Code), "    (when-not (= ~w ~w) (return nil))", [CX, CY]).

translate_generator_goal_clojure(true, "    ;; true") :- !.

translate_generator_goal_clojure(Goal, Code) :-
    format(string(Code), "    ;; TODO: ~w", [Goal]).

compile_generator_recursive_clojure(Name, BaseClauses, _RecClauses, Code) :-
    (   BaseClauses = [(BaseHead, _)|_]
    ->  generate_base_condition_clojure(BaseHead, BaseCondition)
    ;   BaseCondition = "false"
    ),
    
    format(string(Code),
"  ;; Recursive generator: ~w
  (letfn [(iterate [current depth]
            (lazy-seq
              (if (> depth 10000)
                (do (binding [*out* *err*]
                      (println \"Warning: Max depth exceeded for ~w\"))
                    nil)
                (if ~w
                  (list current)
                  (cons current (iterate current (inc depth)))))))]
    (iterate record 0))", [Name, Name, BaseCondition]).

%% ============================================
%% PIPELINE MODE
%% ============================================

compile_pipeline_mode_clojure(Pred, Arity, Options, ClojureCode) :-
    collect_clojure_require('[clojure.data.json :as json]'),
    collect_clojure_require('[clojure.java.io :as io]'),
    
    option(namespace(Namespace), Options, 'generated.pipeline'),
    
    functor(Head, Pred, Arity),
    findall((Head, Body), clause(Head, Body), Clauses),
    
    (   Clauses == []
    ->  ProcessBody = "  ;; No clauses found - pass through\n  record"
    ;   generate_pipeline_process_clojure(Clauses, ProcessBody)
    ),
    
    get_clojure_requires(RequireList),
    format_clojure_requires(RequireList, RequiresStr),
    
    format(string(ClojureCode),
";; Generated by UnifyWeaver Clojure Target - Pipeline Mode
;; Predicate: ~w/~w

(ns ~w
~w)

(defn process
  \"Process a single input record.
   Returns record to keep, nil to filter out.\"
  [record]
~w)

(defn parse-json-line
  \"Parse a JSONL line to a map.\"
  [line]
  (try
    (json/read-str line :key-fn keyword)
    (catch Exception e
      (binding [*out* *err*]
        (println \"JSON parse error:\" (.getMessage e)))
      nil)))

(defn to-json
  \"Convert a map to JSON string.\"
  [m]
  (json/write-str m))

(defn run-pipeline
  \"Read JSONL from stdin, process, write JSONL to stdout.\"
  []
  (with-open [rdr (io/reader *in*)]
    (doseq [result (->> (line-seq rdr)
                        (filter seq)
                        (keep parse-json-line)
                        (keep process))]
      (println (to-json result)))))

(defn -main
  [& args]
  (run-pipeline))
", [Pred, Arity, Namespace, RequiresStr, ProcessBody]).

%% generate_pipeline_process_clojure(+Clauses, -Code)
generate_pipeline_process_clojure([], "  record").
generate_pipeline_process_clojure(Clauses, Code) :-
    Clauses \= [],
    Clauses = [(Head, _)|_],
    functor(Head, Name, _),
    (   is_recursive_predicate_clojure(Name, Clauses)
    ->  partition(is_recursive_clause_clojure(Name), Clauses, RecClauses, BaseClauses),
        compile_recursive_clojure(Name, BaseClauses, RecClauses, Code)
    ;   findall(ClauseCode, 
            (member((H, B), Clauses), translate_clause_clojure(H, B, ClauseCode)),
            ClauseCodes),
        atomic_list_concat(ClauseCodes, '\n', Code)
    ).

%% ============================================
%% RECURSION DETECTION
%% ============================================

is_recursive_predicate_clojure(Name, Clauses) :-
    member((_, Body), Clauses),
    contains_recursive_call_clojure(Body, Name).

is_recursive_clause_clojure(Name, (_, Body)) :-
    contains_recursive_call_clojure(Body, Name).

contains_recursive_call_clojure(Body, Name) :-
    extract_goal_clojure(Body, Goal),
    functor(Goal, Name, _),
    !.

extract_goal_clojure(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_),
    \+ Goal = (_;_).
extract_goal_clojure((A, _), Goal) :- extract_goal_clojure(A, Goal).
extract_goal_clojure((_, B), Goal) :- extract_goal_clojure(B, Goal).

%% ============================================
%% TAIL RECURSION (Clojure's loop/recur)
%% ============================================

compile_recursive_clojure(Name, BaseClauses, _RecClauses, Code) :-
    (   BaseClauses = [(BaseHead, _)|_]
    ->  generate_base_condition_clojure(BaseHead, BaseCondition)
    ;   BaseCondition = "false"
    ),
    
    format(string(Code),
"  ;; Recursive predicate: ~w - using loop/recur
  (loop [current record
         depth 0]
    (cond
      (> depth 10000)
      (do (binding [*out* *err*]
            (println \"Warning: Max depth for ~w\"))
          current)
      
      ~w
      current
      
      :else
      (recur current (inc depth))))", [Name, Name, BaseCondition]).

generate_base_condition_clojure(Head, Condition) :-
    Head =.. [_|Args],
    (   Args = [Arg|_],
        (   number(Arg)
        ->  format(string(Condition), "(= (:arg0 current) ~w)", [Arg])
        ;   atom(Arg)
        ->  format(string(Condition), "(= (:arg0 current) \"~w\")", [Arg])
        ;   Condition = "false"
        )
    ;   Condition = "false"
    ).

%% ============================================
%% CLAUSE TRANSLATION
%% ============================================

translate_clause_clojure(Head, Body, Code) :-
    Head =.. [_Pred|Args],
    generate_input_extraction_clojure(Args, InputCode),
    (   Body == true
    ->  BodyCode = "  ;; Fact - no conditions"
    ;   translate_body_clojure(Body, BodyCode)
    ),
    format(string(Code), "~w\n~w\n  record", [InputCode, BodyCode]).

generate_input_extraction_clojure(Args, Code) :-
    findall(Line, (
        nth0(I, Args, Arg),
        (   var(Arg)
        ->  format(string(Line), "  (let [arg~w (:arg~w record)]", [I, I])
        ;   format(string(Line), "  ;; arg~w = ~w (constant)", [I, Arg])
        )
    ), Lines),
    atomic_list_concat(Lines, '\n', Code).

%% ============================================
%% BODY TRANSLATION
%% ============================================

translate_body_clojure((Goal, Rest), Code) :-
    !,
    translate_goal_clojure(Goal, Code1),
    translate_body_clojure(Rest, Code2),
    format(string(Code), "~w\n~w", [Code1, Code2]).
translate_body_clojure(Goal, Code) :-
    translate_goal_clojure(Goal, Code).

translate_goal_clojure(>(X, Y), Code) :-
    !, expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
    format(string(Code), "  (when-not (> ~w ~w) (return nil))", [CX, CY]).

translate_goal_clojure(<(X, Y), Code) :-
    !, expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
    format(string(Code), "  (when-not (< ~w ~w) (return nil))", [CX, CY]).

translate_goal_clojure(>=(X, Y), Code) :-
    !, expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
    format(string(Code), "  (when-not (>= ~w ~w) (return nil))", [CX, CY]).

translate_goal_clojure(=<(X, Y), Code) :-
    !, expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
    format(string(Code), "  (when-not (<= ~w ~w) (return nil))", [CX, CY]).

translate_goal_clojure(=:=(X, Y), Code) :-
    !, expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
    format(string(Code), "  (when-not (= ~w ~w) (return nil))", [CX, CY]).

translate_goal_clojure(=\\=(X, Y), Code) :-
    !, expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
    format(string(Code), "  (when (= ~w ~w) (return nil))", [CX, CY]).

translate_goal_clojure(is(Var, Expr), Code) :-
    !,
    var_to_clojure(Var, ClojureVar),
    expr_to_clojure(Expr, ClojureExpr),
    format(string(Code), "  (let [~w ~w]", [ClojureVar, ClojureExpr]).

translate_goal_clojure(true, "  ;; true") :- !.

translate_goal_clojure(Goal, Code) :-
    format(string(Code), "  ;; TODO: ~w", [Goal]).

%% ============================================
%% HELPER PREDICATES
%% ============================================

var_to_clojure(Var, ClojureVar) :-
    (   var(Var)
    ->  term_to_atom(Var, VarAtom),
        format(atom(ClojureVar), "var-~w", [VarAtom])
    ;   Var = '$VAR'(N)
    ->  format(atom(ClojureVar), "v~w", [N])
    ;   term_to_atom(Var, ClojureVar)
    ).

expr_to_clojure(Expr, ClojureExpr) :-
    (   number(Expr)
    ->  format(atom(ClojureExpr), "~w", [Expr])
    ;   var(Expr)
    ->  var_to_clojure(Expr, ClojureExpr)
    ;   Expr = '$VAR'(N)
    ->  format(atom(ClojureExpr), "v~w", [N])
    ;   Expr = X + Y
    ->  expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
        format(atom(ClojureExpr), "(+ ~w ~w)", [CX, CY])
    ;   Expr = X - Y
    ->  expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
        format(atom(ClojureExpr), "(- ~w ~w)", [CX, CY])
    ;   Expr = X * Y
    ->  expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
        format(atom(ClojureExpr), "(* ~w ~w)", [CX, CY])
    ;   Expr = X / Y
    ->  expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
        format(atom(ClojureExpr), "(/ ~w ~w)", [CX, CY])
    ;   Expr = X mod Y
    ->  expr_to_clojure(X, CX), expr_to_clojure(Y, CY),
        format(atom(ClojureExpr), "(mod ~w ~w)", [CX, CY])
    ;   format(atom(ClojureExpr), "~w", [Expr])
    ).

%% ============================================
%% DEPS.EDN GENERATION
%% ============================================

generate_deps_edn(Options, DepsFile) :-
    option(main_ns(MainNs), Options, 'generated.pipeline'),
    
    format(string(DepsFile),
";; Generated by UnifyWeaver Clojure Target
{:paths [\"src\"]
 :deps {org.clojure/clojure {:mvn/version \"1.11.1\"}
        org.clojure/data.json {:mvn/version \"2.4.0\"}}
 :aliases
 {:run {:main-opts [\"-m\" \"~w\"]}}}
", [MainNs]).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

write_clojure_program(ClojureCode, FilePath) :-
    open(FilePath, write, Stream),
    write(Stream, ClojureCode),
    close(Stream),
    format('Written Clojure program to: ~w~n', [FilePath]).

option(Option, Options, _Default) :-
    member(Option, Options), !.
option(Option, _Options, Default) :-
    Option =.. [_, Default].

compile_clojure_pipeline(_Steps, _Options, Code) :-
    Code = ";; Multi-step Clojure pipeline - use compile_predicate_to_clojure for now".

%% ============================================
%% NATIVE CLAUSE BODY LOWERING
%% ============================================

%% build_clojure_arg_list(+N, -ArgList)
build_clojure_arg_list(0, "") :- !.
build_clojure_arg_list(N, ArgList) :-
    findall(Arg, (
        between(1, N, I),
        format(string(Arg), 'arg~w', [I])
    ), Args),
    atomic_list_concat(Args, ' ', ArgList).

%% native_clojure_clause_body(+PredSpec, +Clauses, -Code)

% Single clause
native_clojure_clause_body(PredSpec, [Head-Body], Code) :-
    native_clojure_clause(PredSpec, Head, Body, Condition, ClauseCode),
    !,
    (   Condition == "true"
    ->  format(string(Code), '  ~w', [ClauseCode])
    ;   format(string(Code),
'  (if ~w
    ~w
    (throw (ex-info "No matching clause for ~w" {})))', [Condition, ClauseCode, PredSpec])
    ).

% Multi-clause → cond form
native_clojure_clause_body(PredSpec, Clauses, Code) :-
    Clauses = [_|[_|_]],
    maplist(native_clojure_clause_pair(PredSpec), Clauses, Branches),
    Branches \= [],
    branches_to_clojure_cond(Branches, PredSpec, Code).

native_clojure_clause_pair(PredSpec, Head-Body, branch(Condition, ClauseCode)) :-
    native_clojure_clause(PredSpec, Head, Body, Condition, ClauseCode),
    !.

%% native_clojure_clause(+PredSpec, +Head, +Body, -Condition, -Code)

%  Straight-line recursion/computation clause: the output (last) head argument
%  is a variable and the body is a linear sequence of guards and value-binding
%  goals (Var is Expr, Var = Expr, or a predicate call whose last arg is the
%  output var, e.g. a recursive call). This is the path that lowers numeric and
%  list-fold recursion (fib, factorial, sum, listsum) into a properly closed
%  Clojure `(let [...] return)` form. Input head arguments may be plain vars, a
%  cons pattern [H|T] (destructured with first/rest under a non-empty guard),
%  the empty list [] (an emptiness guard), or a literal (an equality guard).
%  If the body is not straight-line (contains if-then-else/disjunction) or the
%  output head arg is not a var, this clause fails and the generic clause below
%  (classify_goal_sequence path) handles it.
native_clojure_clause(_PredSpec, Head, Body, Condition, Code) :-
    Head =.. [_Pred|HeadArgs],
    HeadArgs \= [],
    append(InputHeadArgs, [OutputHeadArg], HeadArgs),
    var(OutputHeadArg),
    build_head_varmap(HeadArgs, 1, VarMap0),
    clojure_input_head_analysis(InputHeadArgs, 1, VarMap0, VarMap1, HeadConds, HeadBinds),
    normalize_goals(Body, Goals),
    Goals \= [],
    clojure_straightline_goals(Goals),
    clause_guard_output_split(Goals, VarMap1, GuardGoals, OutputGoals),
    OutputGoals \= [],
    maplist(clojure_guard_condition(VarMap1), GuardGoals, GuardConds),
    clojure_lower_outputs(OutputGoals, VarMap1, _VarMap2, OutBinds, RetExpr),
    !,
    append(HeadConds, GuardConds, AllConditions),
    combine_clojure_conditions(AllConditions, Condition),
    append(HeadBinds, OutBinds, AllBinds),
    clojure_wrap_let(AllBinds, RetExpr, Code).

native_clojure_clause(_PredSpec, Head, Body, Condition, Code) :-
    Head =.. [_Pred|HeadArgs],
    length(HeadArgs, Arity),
    build_head_varmap(HeadArgs, 1, VarMap),
    (   Arity > 1
    ->  append(_InputHeadArgs, [OutputHeadArg], HeadArgs),
        clojure_head_conditions(HeadArgs, 1, Arity, HeadConditions)
    ;   OutputHeadArg = _,
        clojure_head_conditions(HeadArgs, 1, Arity, HeadConditions)
    ),
    normalize_goals(Body, Goals),
    (   Goals == []
    ->  clojure_resolve_value(VarMap, OutputHeadArg, Code),
        GoalConditions = []
    ;   (   Arity > 1, nonvar(OutputHeadArg)
        ->  clause_guard_output_split(Goals, VarMap, GuardGoals, OutputGoals),
            maplist(clojure_guard_condition(VarMap), GuardGoals, GoalConditions),
            (   OutputGoals == []
            ->  clojure_literal(OutputHeadArg, Code)
            ;   clojure_output_goals(OutputGoals, VarMap, Code)
            )
        ;   native_clojure_goal_sequence(Goals, VarMap, GoalConditions, Code)
        )
    ),
    append(HeadConditions, GoalConditions, AllConditions),
    combine_clojure_conditions(AllConditions, Condition).

%% clojure_head_conditions(+HeadArgs, +Index, +Arity, -Conditions)
clojure_head_conditions([], _, _, []).
clojure_head_conditions([_], _, Arity, []) :- Arity > 1, !.
clojure_head_conditions([HeadArg|Rest], Index, Arity, Conditions) :-
    (   var(HeadArg)
    ->  Conditions = RestConditions
    ;   HeadArg == []
    ->  format(string(Cond), '(empty? arg~w)', [Index]),
        Conditions = [Cond|RestConditions]
    ;   compound(HeadArg), HeadArg = [_|_]
    ->  % cons pattern is handled by clojure_input_head_analysis (destructure);
        % emit only a non-empty guard here for the generic (non-straight-line) path.
        format(string(Cond), '(seq arg~w)', [Index]),
        Conditions = [Cond|RestConditions]
    ;   format(string(ArgName), 'arg~w', [Index]),
        clojure_literal(HeadArg, Literal),
        format(string(Cond), '(= ~w ~w)', [ArgName, Literal]),
        Conditions = [Cond|RestConditions]
    ),
    NextIndex is Index + 1,
    clojure_head_conditions(Rest, NextIndex, Arity, RestConditions).

%% ============================================
%% STRAIGHT-LINE RECURSION / COMPUTATION LOWERING
%% ============================================

%% clojure_input_head_analysis(+InputArgs, +Index, +VarMap0, -VarMap, -Conds, -Binds)
%  Analyse the input (non-output) head arguments, producing guard conditions
%  and `Name-Expr` let-binding pairs. Handles: plain var (no cond/bind), []
%  (emptiness guard), [H|T] (non-empty guard + first/rest destructuring binds),
%  and any other literal (equality guard).
clojure_input_head_analysis([], _, VarMap, VarMap, [], []).
clojure_input_head_analysis([Arg|Rest], Index, VarMap0, VarMap, Conds, Binds) :-
    format(atom(ArgName), 'arg~w', [Index]),
    (   var(Arg)
    ->  C0 = [], B0 = [], VarMap1 = VarMap0
    ;   Arg == []
    ->  format(string(EC), '(empty? ~w)', [ArgName]),
        C0 = [EC], B0 = [], VarMap1 = VarMap0
    ;   compound(Arg), Arg = [H|T]
    ->  format(string(SC), '(seq ~w)', [ArgName]),
        ensure_var(VarMap0, H, HN, VM1),
        ensure_var(VM1, T, TN, VM2),
        format(string(HE), '(first ~w)', [ArgName]),
        format(string(TE), '(rest ~w)', [ArgName]),
        C0 = [SC], B0 = [HN-HE, TN-TE], VarMap1 = VM2
    ;   clojure_literal(Arg, Lit),
        format(string(LC), '(= ~w ~w)', [ArgName, Lit]),
        C0 = [LC], B0 = [], VarMap1 = VarMap0
    ),
    NextIndex is Index + 1,
    clojure_input_head_analysis(Rest, NextIndex, VarMap1, VarMap, RestC, RestB),
    append(C0, RestC, Conds),
    append(B0, RestB, Binds).

%% clojure_straightline_goals(+Goals)
%  True when every goal is either a guard or a single-output value goal
%  (Var is/=; or a predicate call whose last arg is an unbound output var),
%  and none is control flow (if-then-else / if-then / disjunction).
clojure_straightline_goals(Goals) :-
    forall(member(G, Goals), clojure_straightline_goal(G)).

clojure_straightline_goal(G0) :-
    ( G0 = _Module:G -> true ; G = G0 ),
    \+ if_then_else_goal(G, _, _, _),
    \+ if_then_goal(G, _, _),
    \+ disjunction_alternatives(G, [_,_|_]),
    (   is_guard_goal(G, [])
    ->  true
    ;   goal_output_var(G, V), var(V)
    ).

%% clojure_lower_outputs(+OutputGoals, +VarMap0, -VarMap, -Binds, -RetExpr)
%  Lower a sequence of value-binding goals into `Name-Expr` let bindings plus a
%  final return expression (the last goal's value). Bindings introduced by
%  earlier goals are visible to later ones (and to the return expression).
clojure_lower_outputs([Last], VarMap, VarMap, [], RetExpr) :-
    !,
    clojure_output_rhs(Last, VarMap, RetExpr).
clojure_lower_outputs([Goal|Rest], VarMap0, VarMap, [Name-Expr|Binds], RetExpr) :-
    clojure_output_rhs(Goal, VarMap0, Expr),
    goal_output_var(Goal, Var),
    var(Var),
    ensure_var(VarMap0, Var, Name, VarMap1),
    clojure_lower_outputs(Rest, VarMap1, VarMap, Binds, RetExpr).

%% clojure_output_rhs(+Goal, +VarMap, -Expr)
%  The Clojure value expression a value-binding goal computes. Arithmetic and
%  unification bind directly; a predicate call becomes a function application on
%  its input args (all but the output/last arg).
clojure_output_rhs(_Module:Goal, VarMap, Expr) :-
    !,
    clojure_output_rhs(Goal, VarMap, Expr).
clojure_output_rhs(is(_Var, ArithExpr), VarMap, Expr) :-
    !,
    clojure_expr(ArithExpr, VarMap, Expr).
clojure_output_rhs(=(Left, Right), VarMap, Expr) :-
    !,
    ( var(Left) -> clojure_expr(Right, VarMap, Expr)
    ; clojure_expr(Left, VarMap, Expr) ).
clojure_output_rhs(Goal, VarMap, Expr) :-
    compound(Goal),
    Goal =.. [Fn|Args],
    Args \= [],
    append(InArgs, [OutArg], Args),
    var(OutArg),
    maplist(clojure_call_arg(VarMap), InArgs, CInArgs),
    (   CInArgs == []
    ->  format(string(Expr), '(~w)', [Fn])
    ;   atomic_list_concat(CInArgs, ' ', ArgStr),
        format(string(Expr), '(~w ~w)', [Fn, ArgStr]) ).

clojure_call_arg(VarMap, Arg, CArg) :-
    clojure_expr(Arg, VarMap, CArg).

%% clojure_wrap_let(+Binds, +RetExpr, -Code)
%  Wrap a return expression in a single (let [name expr ...] ret) when there are
%  bindings; otherwise the return expression is the whole body.
clojure_wrap_let([], RetExpr, RetExpr) :- !.
clojure_wrap_let(Binds, RetExpr, Code) :-
    clojure_binding_lines(Binds, BindStr),
    format(string(Code), '(let [~w]\n      ~w)', [BindStr, RetExpr]).

clojure_binding_lines([Name-Expr], Str) :-
    !,
    format(string(Str), '~w ~w', [Name, Expr]).
clojure_binding_lines([Name-Expr|Rest], Str) :-
    clojure_binding_lines(Rest, RestStr),
    format(string(Str), '~w ~w\n            ~w', [Name, Expr, RestStr]).

%% native_clojure_goal_sequence(+Goals, +VarMap, -Conditions, -Code)
%  Uses classify_goal_sequence for advanced pattern detection.
%  Falls back to clause_guard_output_split if classification fails.
native_clojure_goal_sequence(Goals, VarMap, Conditions, Code) :-
    classify_goal_sequence(Goals, VarMap, ClassifiedGoals),
    ClassifiedGoals \= [],
    clojure_render_classified_goals(ClassifiedGoals, VarMap, Conditions, Lines),
    Lines \= [],
    atomic_list_concat(Lines, '\n', Code),
    !.
native_clojure_goal_sequence(Goals, VarMap, Conditions, Code) :-
    clause_guard_output_split(Goals, VarMap, GuardGoals, OutputGoals),
    maplist(clojure_guard_condition(VarMap), GuardGoals, Conditions),
    clojure_output_goals(OutputGoals, VarMap, Code).

%% clojure_render_classified_goals(+ClassifiedGoals, +VarMap, -Conditions, -Lines)
clojure_render_classified_goals([], _VarMap, [], []).
clojure_render_classified_goals([Classified], VarMap, Conds, Lines) :-
    !,
    clojure_render_classified_last(Classified, VarMap, Conds, Lines).
%% Guarded tail: output followed by guard(s)
clojure_render_classified_goals([output(Goal, _, _)|Rest], VarMap, [], Lines) :-
    Rest = [guard(_, _)|_],
    !,
    clojure_output_goal(Goal, VarMap, LetBinding, VarMap1),
    clojure_collect_trailing_guards(Rest, VarMap1, GuardGoals, _Remaining),
    maplist(clojure_guard_condition(VarMap1), GuardGoals, GuardConds),
    atomic_list_concat(GuardConds, ' ', GuardExpr),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMap1, OutName)
    ->  true
    ;   OutName = 'nil'
    ),
    format(string(IfLine), '  (if (and ~w)', [GuardExpr]),
    format(string(RetLine), '    ~w', [OutName]),
    CloseLine = '    nil)',
    Lines = [LetBinding, IfLine, RetLine, CloseLine].
clojure_render_classified_goals([Classified|Rest], VarMap, Conds, Lines) :-
    clojure_render_classified_mid(Classified, VarMap, MidConds, MidLines, VarMap1),
    clojure_render_classified_goals(Rest, VarMap1, RestConds, RestLines),
    append(MidConds, RestConds, Conds),
    append(MidLines, RestLines, Lines).

%% clojure_render_classified_mid(+Classified, +VarMap, -Conds, -Lines, -VarMapOut)
clojure_render_classified_mid(guard(Goal, _), VarMap, [Cond], [], VarMap) :-
    clojure_guard_condition(VarMap, Goal, Cond).
clojure_render_classified_mid(output(Goal, _, _), VarMap0, [], [Line], VarMapOut) :-
    clojure_output_goal(Goal, VarMap0, Line, VarMapOut).
clojure_render_classified_mid(output_ite(If, Then, Else, _SharedVars), VarMap0, [], [Line], VarMap0) :-
    clojure_guard_condition(VarMap0, If, Cond),
    clojure_branch_value(Then, VarMap0, ThenExpr),
    clojure_branch_value(Else, VarMap0, ElseExpr),
    format(string(Line), '  (if ~w ~w ~w)', [Cond, ThenExpr, ElseExpr]).
clojure_render_classified_mid(passthrough(Goal), VarMap0, [], [Line], VarMapOut) :-
    clojure_output_goal(Goal, VarMap0, Line, VarMapOut).
clojure_render_classified_mid(_, VarMap, [], [], VarMap).

%% clojure_render_classified_last(+Classified, +VarMap, -Conds, -Lines)
clojure_render_classified_last(guard(Goal, _), VarMap, [Cond], []) :-
    clojure_guard_condition(VarMap, Goal, Cond).
clojure_render_classified_last(output(Goal, _, _), VarMap, [], [Line]) :-
    clojure_output_goal_last(Goal, VarMap, Line).
clojure_render_classified_last(output_ite(If, Then, Else, _), VarMap, [], [Line]) :-
    clojure_guard_condition(VarMap, If, Cond),
    clojure_branch_value(Then, VarMap, ThenExpr),
    clojure_branch_value(Else, VarMap, ElseExpr),
    format(string(Line), '  (if ~w ~w ~w)', [Cond, ThenExpr, ElseExpr]).
clojure_render_classified_last(output_disj(Alternatives, _SharedVars), VarMap, [], Lines) :-
    clojure_disj_cond_chain(Alternatives, VarMap, Lines).
clojure_render_classified_last(passthrough(Goal), VarMap, [], [Line]) :-
    clojure_output_goal_last(Goal, VarMap, Line).
clojure_render_classified_last(_, _, [], []).

%% clojure_collect_trailing_guards(+ClassifiedGoals, +VarMap, -GuardGoals, -Remaining)
clojure_collect_trailing_guards([guard(Goal, _)|Rest], VarMap, [Goal|Guards], Remaining) :-
    !, clojure_collect_trailing_guards(Rest, VarMap, Guards, Remaining).
clojure_collect_trailing_guards(Remaining, _, [], Remaining).

%% clojure_disj_cond_chain(+Alternatives, +VarMap, -Lines)
%  Renders disjunctions as a (cond ...) expression in Clojure.
clojure_disj_cond_chain([], _, []).
clojure_disj_cond_chain(Alternatives, VarMap, Lines) :-
    CondOpen = '  (cond',
    clojure_disj_cond_branches(Alternatives, VarMap, BranchLines),
    CondClose = '  )',
    append([[CondOpen], BranchLines, [CondClose]], Lines).

%% clojure_disj_cond_branches(+Alternatives, +VarMap, -Lines)
clojure_disj_cond_branches([], _, []).
clojure_disj_cond_branches([Alt], VarMap, [KeyLine, ValLine]) :-
    !,
    clojure_branch_value(Alt, VarMap, ValExpr),
    KeyLine = '    :else',
    format(string(ValLine), '    ~w', [ValExpr]).
clojure_disj_cond_branches([Alt|Rest], VarMap, [KeyLine, ValLine|RestLines]) :-
    normalize_goals(Alt, Goals),
    clause_guard_output_split(Goals, VarMap, Guards, _Outputs),
    (   Guards \= []
    ->  maplist(clojure_guard_condition(VarMap), Guards, CondStrs),
        atomic_list_concat(CondStrs, ' ', CondParts),
        format(string(CondExpr), '(and ~w)', [CondParts])
    ;   CondExpr = 'true'
    ),
    clojure_branch_value(Alt, VarMap, ValExpr),
    format(string(KeyLine), '    ~w', [CondExpr]),
    format(string(ValLine), '    ~w', [ValExpr]),
    clojure_disj_cond_branches(Rest, VarMap, RestLines).

%% clojure_guard_condition(+VarMap, +Goal, -Condition)
clojure_guard_condition(VarMap, _Module:Goal, Condition) :-
    !, clojure_guard_condition(VarMap, Goal, Condition).
clojure_guard_condition(VarMap, Goal, Condition) :-
    compound(Goal),
    Goal =.. [Op, Left, Right],
    expr_op(Op, StdOp),
    !,
    clojure_expr(Left, VarMap, CLeft),
    clojure_expr(Right, VarMap, CRight),
    clojure_op(StdOp, COp),
    format(string(Condition), '(~w ~w ~w)', [COp, CLeft, CRight]).
%% Negation-as-failure: \+ Inner / not(Inner) → (not <render Inner>) (G-P7).
%% Recurses into clojure_guard_condition for Inner (comparison / type-check /
%% membership / nested negation). If Inner is a non-guard goal with no guard
%% rendering, the recursive call FAILS, so clojure_guard_condition fails cleanly
%% (no code emitted) rather than emitting wrong code.
clojure_guard_condition(VarMap, \+(Inner), Condition) :-
    !,
    clojure_guard_condition(VarMap, Inner, InnerCond),
    format(string(Condition), '(not ~w)', [InnerCond]).
clojure_guard_condition(VarMap, not(Inner), Condition) :-
    !,
    clojure_guard_condition(VarMap, Inner, InnerCond),
    format(string(Condition), '(not ~w)', [InnerCond]).
%% Membership: member(X, List) → (some #(= % x) list) (G-P7). Positive member
%% is not classified as a guard upstream, so this is reached via `\+ member`.
clojure_guard_condition(VarMap, member(X, List), Condition) :-
    !,
    clojure_expr(X, VarMap, CX),
    clojure_member_list(List, VarMap, CList),
    format(string(Condition), '(some #(= % ~w) ~w)', [CX, CList]).
%% Regex match: match(Var, Pattern) / match(Var, Pattern, Type) (G-P7 follow-up).
%% match/2,3 is UnifyWeaver's regex-match predicate: subject FIRST, pattern
%% SECOND, optional 3rd arg the regex TYPE (auto/ere/pcre/...). The type is
%% advisory here — the generated code uses the host's native regex engine
%% (java.util.regex on the JVM, JS RegExp under ClojureScript) rather than
%% translating dialects. Boolean truthiness mirrors Python's unanchored
%% re.search: re-find returns the matched substring (truthy) or nil (falsy),
%% hence `(re-find (re-pattern "<pattern>") x)`. Anchoring lives in the pattern
%% (e.g. '^a.*'). re-find/re-pattern are portable across JVM and CLJS, so this
%% flows to ClojureScript unchanged. Composes under negation via the \+/not
%% clauses above (\+ match(...) → (not (re-find ...))).
clojure_guard_condition(VarMap, match(Var, Pattern), Condition) :-
    !,
    clojure_match_condition(Var, Pattern, VarMap, Condition).
clojure_guard_condition(VarMap, match(Var, Pattern, _Type), Condition) :-
    !,
    clojure_match_condition(Var, Pattern, VarMap, Condition).
%% Type-check predicates (integer/1, atom/1, is_list/1, ...) (G-P7).
clojure_guard_condition(VarMap, Goal, Condition) :-
    compound(Goal),
    Goal =.. [Pred, Arg],
    clojure_type_check(Pred, Arg, VarMap, Condition),
    !.

%% clojure_member_list(+List, +VarMap, -CljListExpr)
%  Render the second argument of member/2: a proper list becomes a Clojure
%  vector literal, a variable resolves to its bound name (assumed seqable).
clojure_member_list(List, VarMap, CList) :-
    is_list(List),
    !,
    maplist(clojure_member_elem(VarMap), List, Elems),
    atomic_list_concat(Elems, ' ', Inner),
    format(string(CList), '[~w]', [Inner]).
clojure_member_list(Var, VarMap, CList) :-
    var(Var),
    !,
    clojure_expr(Var, VarMap, CList).

clojure_member_elem(VarMap, Elem, CElem) :- clojure_expr(Elem, VarMap, CElem).

%% clojure_match_condition(+Var, +Pattern, +VarMap, -Condition)
%  Render a boolean regex test: (re-find (re-pattern "<escaped>") <subject>).
clojure_match_condition(Var, Pattern, VarMap, Condition) :-
    clojure_expr(Var, VarMap, CVar),
    clojure_regex_pattern_string(Pattern, PatStr),
    format(string(Condition), '(re-find (re-pattern "~w") ~w)', [PatStr, CVar]).

%% clojure_regex_pattern_string(+Pattern, -EscapedForCljStringLiteral)
%  Accept an atom or string regex pattern and escape it for a Clojure
%  double-quoted string literal, preserving regex backslash escapes (\d → "\\d")
%  and quotes. re-pattern then compiles the string into a regex.
clojure_regex_pattern_string(Pattern, Escaped) :-
    ( atom(Pattern) -> atom_string(Pattern, S) ; S = Pattern ),
    string_chars(S, Chars),
    clojure_regex_escape_chars(Chars, EChars),
    string_chars(Escaped, EChars).

clojure_regex_escape_chars([], []).
clojure_regex_escape_chars([C|Cs], Out) :-
    (   C == '\\' -> Out = ['\\','\\'|Rest]
    ;   C == '"'  -> Out = ['\\','"'|Rest]
    ;   Out = [C|Rest]
    ),
    clojure_regex_escape_chars(Cs, Rest).

%% clojure_body_match_subject(+Body, +Var)
%  True when the clause body applies a regex match/2,3 to Var (possibly under
%  \+/not or inside control-flow). Used to decide that a standalone CLI takes a
%  string argv (the regex subject) rather than parsing it as an integer.
clojure_body_match_subject(G, _) :- var(G), !, fail.
clojure_body_match_subject(_Module:G, V) :- !, clojure_body_match_subject(G, V).
clojure_body_match_subject((A, B), V) :- !, ( clojure_body_match_subject(A, V) ; clojure_body_match_subject(B, V) ).
clojure_body_match_subject((A ; B), V) :- !, ( clojure_body_match_subject(A, V) ; clojure_body_match_subject(B, V) ).
clojure_body_match_subject((A -> B), V) :- !, ( clojure_body_match_subject(A, V) ; clojure_body_match_subject(B, V) ).
clojure_body_match_subject(\+(A), V) :- !, clojure_body_match_subject(A, V).
clojure_body_match_subject(not(A), V) :- !, clojure_body_match_subject(A, V).
clojure_body_match_subject(match(S, _), V) :- S == V, !.
clojure_body_match_subject(match(S, _, _), V) :- S == V, !.

%% clojure_type_check(+Pred, +Arg, +VarMap, -Condition)
%  Map Prolog type-check predicates (clause_body_analysis:type_check_pred/1) to
%  Clojure runtime predicates. Atoms are strings in this target, unbound vars
%  are nil, and lists/compounds are collections. Fails for a non type-check
%  predicate so the caller can fail cleanly.
clojure_type_check(integer, Arg, VarMap, Cond) :- !,
    clojure_expr(Arg, VarMap, X), format(string(Cond), '(integer? ~w)', [X]).
clojure_type_check(float, Arg, VarMap, Cond) :- !,
    clojure_expr(Arg, VarMap, X), format(string(Cond), '(float? ~w)', [X]).
clojure_type_check(number, Arg, VarMap, Cond) :- !,
    clojure_expr(Arg, VarMap, X), format(string(Cond), '(number? ~w)', [X]).
clojure_type_check(atom, Arg, VarMap, Cond) :- !,
    clojure_expr(Arg, VarMap, X), format(string(Cond), '(string? ~w)', [X]).
clojure_type_check(atomic, Arg, VarMap, Cond) :- !,
    clojure_expr(Arg, VarMap, X), format(string(Cond), '(not (coll? ~w))', [X]).
clojure_type_check(is_list, Arg, VarMap, Cond) :- !,
    clojure_expr(Arg, VarMap, X), format(string(Cond), '(sequential? ~w)', [X]).
clojure_type_check(compound, Arg, VarMap, Cond) :- !,
    clojure_expr(Arg, VarMap, X), format(string(Cond), '(coll? ~w)', [X]).
clojure_type_check(var, Arg, VarMap, Cond) :- !,
    clojure_expr(Arg, VarMap, X), format(string(Cond), '(nil? ~w)', [X]).
clojure_type_check(nonvar, Arg, VarMap, Cond) :- !,
    clojure_expr(Arg, VarMap, X), format(string(Cond), '(some? ~w)', [X]).
clojure_type_check(ground, Arg, VarMap, Cond) :- !,
    clojure_expr(Arg, VarMap, X), format(string(Cond), '(some? ~w)', [X]).

%% clojure_output_goals(+Goals, +VarMap, -Code)
clojure_output_goals([], _VarMap, 'nil') :- !.
clojure_output_goals([Goal], VarMap, Code) :-
    !, clojure_output_goal_last(Goal, VarMap, Code).
clojure_output_goals([Goal|Rest], VarMap0, Code) :-
    clojure_output_goal(Goal, VarMap0, _Line, VarMap1),
    clojure_output_goals(Rest, VarMap1, Code).

%% clojure_output_goal_last — produce the return expression
clojure_output_goal_last(_Module:Goal, VarMap, Code) :-
    !, clojure_output_goal_last(Goal, VarMap, Code).
clojure_output_goal_last(Goal, VarMap, Code) :-
    if_then_else_goal(Goal, IfGoal, ThenGoal, ElseGoal),
    !,
    clojure_if_then_else_output(IfGoal, ThenGoal, ElseGoal, VarMap, Code).
clojure_output_goal_last(=(Var, Expr), VarMap, Code) :-
    var(Var), !,
    clojure_expr(Expr, VarMap, Code).
clojure_output_goal_last(is(Var, Expr), VarMap, Code) :-
    var(Var), !,
    clojure_expr(Expr, VarMap, Code).

%% clojure_output_goal — produce a let binding (not used as return)
clojure_output_goal(_Module:Goal, VarMap0, Line, VarMapOut) :-
    !, clojure_output_goal(Goal, VarMap0, Line, VarMapOut).
clojure_output_goal(=(Var, Expr), VarMap0, Line, VarMapOut) :-
    var(Var), !,
    ensure_var(VarMap0, Var, VarName, VarMapOut),
    clojure_expr(Expr, VarMap0, CExpr),
    format(string(Line), '(let [~w ~w]', [VarName, CExpr]).
clojure_output_goal(is(Var, Expr), VarMap0, Line, VarMapOut) :-
    var(Var), !,
    ensure_var(VarMap0, Var, VarName, VarMapOut),
    clojure_expr(Expr, VarMap0, CExpr),
    format(string(Line), '(let [~w ~w]', [VarName, CExpr]).

%% clojure_if_then_else_output — generate (cond ...) or (if ...)
clojure_if_then_else_output(IfGoal, ThenGoal, ElseGoal, VarMap, Code) :-
    flatten_clojure_if_branches(IfGoal, ThenGoal, ElseGoal, Branches, DefaultGoal),
    clojure_nested_if_expr(Branches, DefaultGoal, VarMap, Code).

flatten_clojure_if_branches(If, Then, Else, [branch(If, Then)|RestBranches], Default) :-
    if_then_else_goal(Else, If2, Then2, Else2),
    !,
    flatten_clojure_if_branches(If2, Then2, Else2, RestBranches, Default).
flatten_clojure_if_branches(If, Then, Else, [branch(If, Then)], Else).

%% clojure_nested_if_expr — builds (cond ...) for nested branches
clojure_nested_if_expr([branch(If, Then)], DefaultGoal, VarMap, Code) :-
    !,
    clojure_guard_condition(VarMap, If, IfCond),
    clojure_branch_value(Then, VarMap, ThenVal),
    clojure_branch_value(DefaultGoal, VarMap, ElseVal),
    format(string(Code), '(if ~w ~w ~w)', [IfCond, ThenVal, ElseVal]).
clojure_nested_if_expr([branch(If, Then)|Rest], DefaultGoal, VarMap, Code) :-
    clojure_guard_condition(VarMap, If, IfCond),
    clojure_branch_value(Then, VarMap, ThenVal),
    clojure_nested_if_cond_pairs(Rest, DefaultGoal, VarMap, RestPairs),
    format(string(Code), '(cond ~w ~w ~w)', [IfCond, ThenVal, RestPairs]).

clojure_nested_if_cond_pairs([branch(If, Then)], DefaultGoal, VarMap, Code) :-
    !,
    clojure_guard_condition(VarMap, If, IfCond),
    clojure_branch_value(Then, VarMap, ThenVal),
    clojure_branch_value(DefaultGoal, VarMap, ElseVal),
    format(string(Code), '~w ~w :else ~w', [IfCond, ThenVal, ElseVal]).
clojure_nested_if_cond_pairs([branch(If, Then)|Rest], DefaultGoal, VarMap, Code) :-
    clojure_guard_condition(VarMap, If, IfCond),
    clojure_branch_value(Then, VarMap, ThenVal),
    clojure_nested_if_cond_pairs(Rest, DefaultGoal, VarMap, RestCode),
    format(string(Code), '~w ~w ~w', [IfCond, ThenVal, RestCode]).

%% clojure_branch_value — extract result value from a branch
clojure_branch_value(_Module:Goal, VarMap, Value) :-
    !, clojure_branch_value(Goal, VarMap, Value).
clojure_branch_value(Goal, VarMap, Value) :-
    if_then_else_goal(Goal, If, Then, Else),
    !,
    clojure_guard_condition(VarMap, If, Cond),
    clojure_branch_value(Then, VarMap, ThenVal),
    clojure_branch_value(Else, VarMap, ElseVal),
    format(string(Value), '(if ~w ~w ~w)', [Cond, ThenVal, ElseVal]).
clojure_branch_value((A, B), VarMap, Value) :-
    !,
    normalize_goals((A, B), Goals),
    last(Goals, LastGoal),
    clojure_branch_value(LastGoal, VarMap, Value).
clojure_branch_value(=(_, Expr), VarMap, Value) :-
    !, clojure_expr(Expr, VarMap, Value).
clojure_branch_value(is(_, Expr), VarMap, Value) :-
    !, clojure_expr(Expr, VarMap, Value).
clojure_branch_value(Goal, VarMap, Value) :-
    clojure_expr(Goal, VarMap, Value).

% ============================================================================
% MULTIFILE HOOKS — Register Clojure renderers for shared compile_expression
% ============================================================================

clause_body_analysis:render_output_goal(clojure, Goal, VarMap, Line, VarName, VarMapOut) :-
    clojure_output_goal(Goal, VarMap, Line, VarMapOut),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMapOut, VarName)
    ->  true
    ;   VarName = "_"
    ).

clause_body_analysis:render_guard_condition(clojure, Goal, VarMap, CondStr) :-
    clojure_guard_condition(VarMap, Goal, CondStr).

clause_body_analysis:render_branch_value(clojure, Branch, VarMap, ExprStr) :-
    clojure_branch_value(Branch, VarMap, ExprStr).

clause_body_analysis:render_ite_block(clojure, Cond, ThenLines, ElseLines, Indent, _ReturnVars, Lines) :-
    format(string(IfLine), '~w(if ~w', [Indent, Cond]),
    clojure_indent_lines(ThenLines, Indent, IndentedThen),
    (   ElseLines \= []
    ->  clojure_indent_lines(ElseLines, Indent, IndentedElse),
        format(string(CloseParen), '~w)', [Indent]),
        append([IfLine|IndentedThen], IndentedElse, PreClose),
        append(PreClose, [CloseParen], Lines)
    ;   format(string(CloseParen), '~w)', [Indent]),
        append([IfLine|IndentedThen], [CloseParen], Lines)
    ).

clojure_indent_lines([], _, []).
clojure_indent_lines([Line|Rest], Indent, [Indented|RestIndented]) :-
    format(string(Indented), '~w  ~w', [Indent, Line]),
    clojure_indent_lines(Rest, Indent, RestIndented).

%% clojure_expr — convert Prolog expression to Clojure syntax
clojure_expr(Var, VarMap, CExpr) :-
    var(Var), !,
    (   lookup_var(Var, VarMap, Name)
    ->  CExpr = Name
    ;   term_string(Var, CExpr)
    ).
clojure_expr(Expr, VarMap, CExpr) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    expr_op(Op, StdOp),
    !,
    clojure_expr(Left, VarMap, CLeft),
    clojure_expr(Right, VarMap, CRight),
    clojure_op(StdOp, COp),
    format(string(CExpr), '(~w ~w ~w)', [COp, CLeft, CRight]).
clojure_expr(-Expr, VarMap, CExpr) :-
    !,
    clojure_expr(Expr, VarMap, Inner),
    format(string(CExpr), '(- ~w)', [Inner]).
clojure_expr(abs(Expr), VarMap, CExpr) :-
    !,
    clojure_expr(Expr, VarMap, Inner),
    format(string(CExpr), '(Math/abs ~w)', [Inner]).
clojure_expr(Atom, _VarMap, CExpr) :-
    atom(Atom), !,
    clojure_literal(Atom, CExpr).
clojure_expr(Number, _VarMap, CExpr) :-
    number(Number), !,
    format(string(CExpr), '~w', [Number]).
clojure_expr(String, _VarMap, CExpr) :-
    string(String), !,
    format(string(CExpr), '"~w"', [String]).

%% clojure_literal — convert Prolog value to Clojure literal
clojure_literal(Value, 'nil') :- var(Value), !.
clojure_literal(true, 'true') :- !.
clojure_literal(false, 'false') :- !.
clojure_literal(Value, CljLiteral) :-
    number(Value), !,
    format(string(CljLiteral), '~w', [Value]).
clojure_literal(Value, CljLiteral) :-
    atom(Value), !,
    format(string(CljLiteral), '"~w"', [Value]).
clojure_literal(Value, CljLiteral) :-
    string(Value), !,
    format(string(CljLiteral), '"~w"', [Value]).
clojure_literal(Value, CljLiteral) :-
    term_string(Value, S),
    format(string(CljLiteral), '"~w"', [S]).

%% clojure_resolve_value — resolve variable or constant to Clojure expression
clojure_resolve_value(VarMap, Var, CExpr) :-
    var(Var), !,
    lookup_var(Var, VarMap, CExpr).
clojure_resolve_value(_VarMap, Value, CExpr) :-
    clojure_literal(Value, CExpr).

%% clojure_op — map standard operator to Clojure syntax (prefix)
clojure_op('>', '>').
clojure_op('<', '<').
clojure_op('>=', '>=').
clojure_op('<=', '<=').
clojure_op('==', '=').
clojure_op('!=', 'not=').
clojure_op('+', '+').
clojure_op('-', '-').
clojure_op('*', '*').
clojure_op('/', 'quot').
clojure_op('%', 'rem').
clojure_op('&&', 'and').
clojure_op('||', 'or').

%% combine_clojure_conditions — join conditions with and
combine_clojure_conditions([], "true") :- !.
combine_clojure_conditions([Condition], Condition) :- !.
combine_clojure_conditions(Conditions, Combined) :-
    atomic_list_concat(Conditions, ' ', CondList),
    format(string(Combined), '(and ~w)', [CondList]).

%% branches_to_clojure_cond — build Clojure cond form
branches_to_clojure_cond(Branches, PredSpec, Code) :-
    clojure_cond_pairs(Branches, PredSpec, Pairs),
    format(string(Code), '  (cond\n~w)', [Pairs]).

clojure_cond_pairs([], PredSpec, Code) :-
    format(string(Code), '    :else (throw (ex-info "No matching clause for ~w" {}))', [PredSpec]).
clojure_cond_pairs([branch(Condition, ClauseCode)|Rest], PredSpec, Code) :-
    clojure_cond_pairs(Rest, PredSpec, RestCode),
    format(string(Code), '    ~w ~w\n~w', [Condition, ClauseCode, RestCode]).

%% ===========================================================================
%% A3 WHOLE-PROGRAM LOWERING — the Clojure port of typescript_target's
%% structural + general clause compilers (G-A3-6, -9, -10, -12, -16, -18, -19,
%% -20).
%% ===========================================================================
%%
%% This section is a PORT, not a re-derivation: every design decision below was
%% settled in the TypeScript lane and is recorded in
%% docs/proposals/A3_PATTERN_TRANSPILE_REPORT.md. What changes here is the
%% RENDERING, and Clojure changes it in one structural way that simplifies four
%% of the seven gaps at once:
%%
%%   TypeScript is a STATEMENT language, so the TS lowering accumulates a list of
%%   statement strings, threads a `return` through every branch, and needs
%%   `let`-then-assign machinery (G-A3-10's VALUE form), an `ts_assemble/3` pass
%%   to re-nest in-block guards, and a manual indentation walker.
%%
%%   Clojure is an EXPRESSION language. A clause body is ONE expression, built by
%%   folding an ordered item list from the inside out:
%%
%%       bind(Name, Expr)  ->  (let [Name Expr] <rest>)
%%       gopen(Cond)       ->  (if Cond <rest> <Fall>)
%%       the tail          ->  the clause's value
%%
%%   so `clj_fold_items/5` replaces ts_assemble/3, the let+assign VALUE form
%%   collapses to `(let [_s (if C then else)] ...)`, and there is no indentation
%%   pass because nesting IS the structure.
%%
%% THE FOUR REPRESENTATION DECISIONS, and why each is what it is.
%%
%%   TUPLE (G-A3-9).   A predicate with N > 1 outputs returns a Clojure VECTOR
%%       `[out1 ... outN]`, destructured by callers with `(let [[a b] (f ...)])`.
%%       Positional because Prolog's outputs are positional and the only names
%%       available at this layer are the emitted parameter slots. The exact
%%       analogue of the TS lane's positional array.
%%
%%   COMPOUND (G-A3-12/16).  `f(A1..An)` becomes the MAP `{:$ "f" :args [...]}`.
%%       NOT a tagged vector, for the same reason the TS lane rejected a tagged
%%       array: a Prolog list is already a vector here, so `["f" e1]` could not be
%%       told from the list `[f, e1]` — and telling them apart IS the gap. With
%%       the map form the four representations are pairwise distinguishable at
%%       run time and no test throws:
%%
%%           atom / string   a string    (string? x)
%%           true / false    a boolean   (boolean? x)
%%           list            a vector/seq (sequential? x)
%%           compound        a map       (map? x), tag in (:$ x)
%%
%%       `(:$ x)` on a string, number, boolean or vector answers nil rather than
%%       throwing, so the tag test is total even without the `map?` guard.
%%
%%   EQUALITY.  Clojure's `=` is ALREADY structural, so the TS lane's emitted
%%       `_uwEq` helper is UNNECESSARY and is not ported. Verified against every
%%       distinction this program depends on: `(= true "true")` is false (the
%%       boolean flag value stays distinct from the string a `--x=true` produces,
%%       G-A3-13), `(= {:$ "ok" :args [[1] [2]]} {:$ "ok" :args [[1] [2]]})` is
%%       true, and `(= [1 2] (rest [0 1 2]))` is true — vectors and seqs compare
%%       sequentially, which is what lets `first`/`rest`/`cons` be used freely
%%       without a `vec` round-trip at every step.
%%
%%   SENTINEL (G-A3-18).  A predicate that has outputs AND can fail answers with
%%       its value (or the tuple) or with the module-private `uw-fail`, and
%%       callers test `(not (identical? x uw-fail))`. `uw-fail` is a freshly
%%       allocated host object — `(Object.)` on the JVM, rewritten to
%%       `(js/Object.)` by clojurescript_target's interop pass. The forgery
%%       argument is the TS lane's, unchanged: a namespaced keyword would work
%%       TODAY (nothing this target lowers a term to is a keyword), but that is a
%%       property of the current term renderer, not of the convention, and data
%%       crossing the module edge is outside it. A fresh object has reference
%%       identity that no term can produce and no data can forge, so
%%       `identical?` is an exact test rather than a conventional one.
%%
%% RECURSION.  Self-calls are emitted as DIRECT calls `(f ...)`, not `recur`.
%%       This is the faithful analogue of the TS lane's `return pred(...)` and it
%%       carries the same risk, stated rather than hidden: neither JS nor nbb has
%%       tail-call elimination, so recursion depth is bounded by input size.
%%       `recur` would give real TCO, but only for calls the generator can prove
%%       are in tail position with a matching binding vector; emitting it
%%       elsewhere is a Clojure compile error. The bound here is small and
%%       measured — the deepest walks in cli_args are over a token's CHARACTERS
%%       (`long_flag_tail/1`, `first_char_index/4`, `drop_brackets/2`) and over
%%       argv (`lenient_loop/5`), so depth is bounded by max token length and by
%%       argv length. nbb overflows at ~10 000 frames; this program uses ~30.

% ---------------------------------------------------------------------------
% Small shared utilities
% ---------------------------------------------------------------------------

clj_bget(V, [V0-E|_], E) :- V0 == V, !.
clj_bget(V, [_|T], E) :- clj_bget(V, T, E).

clj_var_in(V, T) :- var(T), !, T == V.
clj_var_in(V, T) :- compound(T), arg(_, T, A), clj_var_in(V, A).

clj_count_var(V, Term, N) :- findall(x, clj_var_in(V, Term), L), length(L, N).

clj_var_memberchk(V, [W|_]) :- W == V, !.
clj_var_memberchk(V, [_|T]) :- clj_var_memberchk(V, T).

%% clj_all_goals(+Body, -Goals) — every goal, through ,/;/->/\+ and Module:.
clj_all_goals(V, [V]) :- var(V), !.
clj_all_goals(true, []) :- !.
clj_all_goals((A, B), Gs) :- !, clj_all_goals(A, GA), clj_all_goals(B, GB), append(GA, GB, Gs).
clj_all_goals((A ; B), Gs) :- !, clj_all_goals(A, GA), clj_all_goals(B, GB), append(GA, GB, Gs).
clj_all_goals((A -> B), Gs) :- !, clj_all_goals(A, GA), clj_all_goals(B, GB), append(GA, GB, Gs).
clj_all_goals(\+ A, Gs) :- !, clj_all_goals(A, Gs).
clj_all_goals(_M:G, Gs) :- !, clj_all_goals(G, Gs).
clj_all_goals(G, [G]).

clj_strip_mod(_M:G, G) :- !.
clj_strip_mod(G, G).

clj_control_functor(',').
clj_control_functor(';').
clj_control_functor('->').
clj_control_functor('\\+').
clj_control_functor(not).
clj_control_functor(=).
clj_control_functor(is).

clj_binds_var(Goal, V, Rhs) :- nonvar(Goal), Goal = (L = Rhs), L == V.
clj_binds_var(Goal, V, Rhs) :- nonvar(Goal), Goal = (L is Rhs), L == V.

clj_is_cons(T) :- nonvar(T), T = [_|_].
clj_list_pos_ok(T) :- var(T), !.
clj_list_pos_ok([]) :- !.
clj_list_pos_ok(T) :- nonvar(T), T = [_|_].
clj_islisty(T) :- nonvar(T), ( T = [] ; T = [_|_] ).

clj_distinct_vars([]).
clj_distinct_vars([V|Vs]) :- var(V), forall(member(W, Vs), V \== W), clj_distinct_vars(Vs).

clj_same_var_list([], []).
clj_same_var_list([A|As], [B|Bs]) :- A == B, clj_same_var_list(As, Bs).

clj_split_out_args([], _, _, [], []).
clj_split_out_args([A|As], Idx, Outs, InArgs, OutArgs) :-
    Idx1 is Idx + 1,
    (   memberchk(Idx, Outs)
    ->  InArgs = RestIn, OutArgs = [A|RestOut]
    ;   InArgs = [A|RestIn], OutArgs = RestOut
    ),
    clj_split_out_args(As, Idx1, Outs, RestIn, RestOut).

clj_conj_of([G], G) :- !.
clj_conj_of([G|Gs], (G, Rest)) :- clj_conj_of(Gs, Rest).

clj_nth_arg(HeadArgs, Pos, Arg) :- nth1(Pos, HeadArgs, Arg).

%% clj_is_ite(+Goal, -If, -Then, -Else)
%  Reuses clause_body_analysis' matcher so this path and the historical
%  clause-body renderer agree on what an if-then-else IS.
clj_is_ite(Goal, If, Then, Else) :-
    nonvar(Goal), if_then_else_goal(Goal, If, Then, Else).

clj_branch_goals(Body, Goals) :-
    clj_extract_goals(Body, Goals0), maplist(clj_strip_mod, Goals0, Goals).

clj_extract_goals(V, [V]) :- var(V), !.
clj_extract_goals(true, []) :- !.
clj_extract_goals((A, B), Gs) :- !,
    clj_extract_goals(A, GA), clj_extract_goals(B, GB), append(GA, GB, Gs).
clj_extract_goals(G, [G]).

%% clj_cmp_op(+PrologOp, -ClojureOp) — prefix comparison operators.
%  `==`/`\==` map to `=`/`not=`, which in Clojure are STRUCTURAL, so the TS
%  lane's `_uwEq` split (scalar `===` vs an emitted structural helper) has no
%  analogue here and is not ported.
clj_cmp_op(>, ">").
clj_cmp_op(<, "<").
clj_cmp_op(>=, ">=").
clj_cmp_op(=<, "<=").
clj_cmp_op(=:=, "=").
clj_cmp_op(=\=, "not=").
clj_cmp_op(==, "=").
clj_cmp_op(\==, "not=").

% ---------------------------------------------------------------------------
% Names
% ---------------------------------------------------------------------------

%% clj_fn_name(+Pred, +Arity, -NameStr)
%  Prolog `some_pred` becomes Clojure `some-pred`. ARITY MANGLING, exactly as the
%  TS lane needs it: Clojure has no arity overloading across separate defns, so
%  `parse_args/2` and `parse_args/3` — which is what cli_args is — cannot both be
%  `(defn parse-args ...)`. An overloaded name gets its arity appended
%  (`parse-args-3`); a name that is not overloaded keeps the name it had.
clj_fn_name(Pred, Arity, NameStr) :-
    (   clj_name_overloaded(Pred, Arity)
    ->  format(atom(A), '~w_~w', [Pred, Arity])
    ;   A = Pred
    ),
    clj_hyphenate_atom(A, NameStr).

clj_name_overloaded(Pred, Arity) :-
    atom(Pred), integer(Arity),
    findall(A2,
            ( catch(current_predicate(user:Pred/A2), _, fail),
              A2 =\= Arity,
              functor(H, Pred, A2),
              catch(( user:clause(H, _) -> true ; fail ), _, fail) ),
            [_|_]).

clj_hyphenate_atom(A, S) :-
    atomic_list_concat(Parts, '_', A),
    atomic_list_concat(Parts, '-', B),
    atom_string(B, S).

%% clj_emitted_name(+Pred, +Arity, -Name)
%
%  THE NAME A CALL SITE MUST USE. A module can mix the two lowerings, and they do
%  not spell a function the same way: the A3 paths hyphenate and mangle arity
%  (`merge-flags-`, `parse-args-3`), the historical clause-body path emits the
%  raw Prolog functor (`merge_flags_`, `parse_args`). Calling a predicate by the
%  name the OTHER path would have given it produces a namespace nbb refuses to
%  load -- "Unable to resolve symbol" -- which is how this was found.
%
%  So a call site asks which path will actually compile the callee and uses that
%  path's spelling. Inside a module compile the answer is the A3 set that
%  compile_module/3 fixed up front (clj_module_a3_set/2); outside one, it is the
%  same per-predicate question the dispatcher asks.
clj_emitted_name(Pred, Arity, Name) :-
    (   clj_a3_compiled(Pred, Arity)
    ->  clj_fn_name(Pred, Arity, Name)
    ;   atom_string(Pred, Name)
    ).

clj_a3_compiled(Pred, Arity) :-
    (   catch(b_getval(clj_a3_set, Set), _, fail)
    ->  memberchk(Pred/Arity, Set)
    ;   clj_a3_claims(Pred, Arity)
    ).

%% clj_a3_claims(+Pred, +Arity) — would the dispatcher route this to the A3 path?
clj_a3_claims(Pred, Arity) :-
    functor(H, Pred, Arity),
    catch(findall(H-B, user:clause(H, B), Cs), _, fail),
    Cs \== [],
    \+ clj_ground_fact_predicate(Cs),
    catch(clj_clause_body_defective(Pred/Arity, Cs), _, fail).

clj_set_a3_set(Set) :- b_setval(clj_a3_set, Set).

clj_self_fn_name(Pred, NameStr) :-
    (   catch(b_getval(clj_struct_self_arity, SA), _, fail)
    ->  clj_fn_name(Pred, SA, NameStr)
    ;   atom_string(Pred, NameStr)
    ).

clj_set_self_arity(A) :- b_setval(clj_struct_self_arity, A).

clj_is_self(Goal, SelfPred) :-
    compound(Goal), functor(Goal, SelfPred, GA),
    (   catch(b_getval(clj_struct_self_arity, SA), _, fail)
    ->  GA =:= SA
    ;   true
    ).

clj_set_head_bind(B) :- b_setval(clj_struct_head_bind, B).
clj_head_bind(B) :-
    ( catch(b_getval(clj_struct_head_bind, V), _, fail) -> B = V ; B = [] ).

clj_guard_head_only(Goal) :-
    clj_head_bind(HB), term_variables(Goal, Vs),
    forall(member(V, Vs), clj_bget(V, HB, _)).

% ---------------------------------------------------------------------------
% Literals and terms
% ---------------------------------------------------------------------------

%% clj_string_lit(+Text, -Literal) — a Clojure double-quoted string literal.
clj_string_lit(Text, Lit) :-
    ( atom(Text) -> atom_string(Text, S) ; S = Text ),
    string_chars(S, Cs),
    clj_escape_chars(Cs, Es),
    string_chars(Body, Es),
    format(string(Lit), '"~w"', [Body]).

clj_escape_chars([], []).
clj_escape_chars([C|Cs], Out) :-
    (   C == '\\' -> Out = ['\\','\\'|Rest]
    ;   C == '"'  -> Out = ['\\','"'|Rest]
    ;   C == '\n' -> Out = ['\\','n'|Rest]
    ;   C == '\t' -> Out = ['\\','t'|Rest]
    ;   Out = [C|Rest]
    ),
    clj_escape_chars(Cs, Rest).

%% clj_term_expr(+Term, +Bind, -Expr) — a Prolog term as a Clojure value.
%
%  An unbound variable FAILS here rather than rendering nil (the analogue of the
%  TS lane's refusal to emit `undefined`, G-A3-10/G-A3-14). A variable with no
%  binding is a variable this path cannot lower; emitting nil would produce code
%  that runs and is wrong. Failing makes the path refuse the predicate.
clj_term_expr(V, B, E) :- var(V), !, clj_bget(V, B, E).
clj_term_expr([], _, "[]") :- !.
%  A PROPER list renders as a vector literal — `["a" "b"]` rather than
%  `(cons "a" (cons "b" []))`. Same value, far more readable output.
clj_term_expr(L, B, E) :-
    is_list(L), L \== [], !,
    maplist(clj_term_expr_b(B), L, Es),
    atomic_list_concat(Es, ' ', Inner),
    format(string(E), "[~w]", [Inner]).
clj_term_expr([H|T], B, E) :- !,
    clj_term_expr(H, B, HE), clj_term_expr(T, B, TE),
    format(string(E), "(cons ~w ~w)", [HE, TE]).
clj_term_expr(T, B, E) :- compound(T), !, clj_compound_expr(T, B, E).
clj_term_expr(N, _, E) :- number(N), !, format(string(E), "~w", [N]).
clj_term_expr(true,  _, "true")  :- !.
clj_term_expr(false, _, "false") :- !.
clj_term_expr(A, _, E) :- atom(A), !, clj_string_lit(A, E).
clj_term_expr(S, _, E) :- string(S), !, clj_string_lit(S, E).

clj_term_expr_b(B, T, E) :- clj_term_expr(T, B, E).

%% clj_compound_expr(+Term, +Bind, -Expr) — G-A3-12's representation.
clj_compound_expr(Term, Bind, Expr) :-
    Term =.. [F|Args],
    maplist(clj_term_expr_b(Bind), Args, Es),
    atomic_list_concat(Es, ' ', Inner),
    clj_string_lit(F, FLit),
    format(string(Expr), '{:$ ~w :args [~w]}', [FLit, Inner]).

%% clj_arith(+Expr, +Bind, -CljExpr)
clj_arith(V, B, S) :- var(V), !, clj_bget(V, B, S).
clj_arith(N, _, S) :- number(N), !, format(string(S), "~w", [N]).
clj_arith(A+B, Bi, S) :- !, clj_arith(A,Bi,SA), clj_arith(B,Bi,SB), format(string(S),"(+ ~w ~w)",[SA,SB]).
clj_arith(A-B, Bi, S) :- !, clj_arith(A,Bi,SA), clj_arith(B,Bi,SB), format(string(S),"(- ~w ~w)",[SA,SB]).
clj_arith(A*B, Bi, S) :- !, clj_arith(A,Bi,SA), clj_arith(B,Bi,SB), format(string(S),"(* ~w ~w)",[SA,SB]).
%  `/` is SWI's ordinary division and `//` its INTEGER division; conflating them
%  silently changes an index computation by a fraction. Under ClojureScript both
%  operands are JS doubles, so `/` matches the JS lane exactly and `quot`
%  truncates as `//` does.
clj_arith(A/B, Bi, S) :- !, clj_arith(A,Bi,SA), clj_arith(B,Bi,SB), format(string(S),"(/ ~w ~w)",[SA,SB]).
clj_arith(A//B, Bi, S) :- !, clj_arith(A,Bi,SA), clj_arith(B,Bi,SB), format(string(S),"(quot ~w ~w)",[SA,SB]).
clj_arith(A mod B, Bi, S) :- !, clj_arith(A,Bi,SA), clj_arith(B,Bi,SB), format(string(S),"(mod ~w ~w)",[SA,SB]).
clj_arith(A rem B, Bi, S) :- !, clj_arith(A,Bi,SA), clj_arith(B,Bi,SB), format(string(S),"(rem ~w ~w)",[SA,SB]).
clj_arith(min(A,B), Bi, S) :- !, clj_arith(A,Bi,SA), clj_arith(B,Bi,SB), format(string(S),"(min ~w ~w)",[SA,SB]).
clj_arith(max(A,B), Bi, S) :- !, clj_arith(A,Bi,SA), clj_arith(B,Bi,SB), format(string(S),"(max ~w ~w)",[SA,SB]).
%  JVM host form, so clojurescript_target's existing `Math/abs -> js/Math.abs`
%  interop rule carries it to the JS host along with everything else.
clj_arith(abs(A), Bi, S) :- !, clj_arith(A,Bi,SA), format(string(S),"(Math/abs ~w)",[SA]).
clj_arith(-(A), Bi, S) :- !, clj_arith(A,Bi,SA), format(string(S),"(- ~w)",[SA]).
clj_arith(true,  _, "true")  :- !.
clj_arith(false, _, "false") :- !.
clj_arith(A, _, S) :- atom(A), !, clj_string_lit(A, S).
%  Non-arithmetic operands of a comparison — a string, a list, a compound — are
%  DATA and render through clj_term_expr/3.
clj_arith(T, B, S) :- clj_term_expr(T, B, S).

%% clj_cmp_cond(+Op, +L, +R, +Bind, -Cond)
%  One clause, where the TS lane needs two: Clojure's `=` is structural, so a
%  comparison over a compound or a list needs no separate helper.
clj_cmp_cond(Op, L, R, B, Cond) :-
    clj_cmp_op(Op, CljOp),
    clj_arith(L, B, LS), clj_arith(R, B, RS),
    format(string(Cond), "(~w ~w ~w)", [CljOp, LS, RS]).

% ---------------------------------------------------------------------------
% Head / pattern matching
% ---------------------------------------------------------------------------

%% clj_match(+Expr, +Pattern, +Bind0, -Bind, +Conds0, -Conds)
clj_match(Expr, V, B0, B, C0, C) :- var(V), !,
    (   clj_bget(V, B0, Prev)
    ->  B = B0, format(string(Cond), "(= ~w ~w)", [Expr, Prev]), C = [Cond|C0]
    ;   B = [V-Expr|B0], C = C0
    ).
clj_match(Expr, [], B, B, C0, [Cond|C0]) :- !,
    format(string(Cond), "(empty? ~w)", [Expr]).
clj_match(Expr, [H|T], B0, B, C0, C) :- !,
    format(string(NonEmpty), "(seq ~w)", [Expr]),
    format(string(HeadE), "(first ~w)", [Expr]),
    format(string(TailE), "(rest ~w)", [Expr]),
    clj_match(HeadE, H, B0, B1, [NonEmpty|C0], C1),
    clj_match(TailE, T, B1, B, C1, C).
%  A COMPOUND pattern (G-A3-12): a tag test plus a positional destructure of
%  `:args`. Total — `(map? x)` is false for every other representation and
%  `(:$ x)` never throws.
clj_match(Expr, P, B0, B, C0, C) :- compound(P), !,
    P =.. [F|Args], length(Args, N),
    clj_string_lit(F, FLit),
    format(string(Cond),
           "(and (map? ~w) (= (:$ ~w) ~w) (= (count (:args ~w)) ~w))",
           [Expr, Expr, FLit, Expr, N]),
    clj_match_args(Args, 0, Expr, B0, B, [Cond|C0], C).
clj_match(Expr, N, B, B, C0, [Cond|C0]) :- number(N), !,
    format(string(Cond), "(= ~w ~w)", [Expr, N]).
clj_match(Expr, true, B, B, C0, [Cond|C0]) :- !,
    format(string(Cond), "(true? ~w)", [Expr]).
clj_match(Expr, false, B, B, C0, [Cond|C0]) :- !,
    format(string(Cond), "(false? ~w)", [Expr]).
clj_match(Expr, A, B, B, C0, [Cond|C0]) :- atom(A), !,
    clj_string_lit(A, L), format(string(Cond), "(= ~w ~w)", [Expr, L]).
clj_match(Expr, S, B, B, C0, [Cond|C0]) :- string(S), !,
    clj_string_lit(S, L), format(string(Cond), "(= ~w ~w)", [Expr, L]).

clj_match_args([], _, _, B, B, C, C).
clj_match_args([A|As], Idx, Expr, B0, B, C0, C) :-
    format(string(Sub), "(nth (:args ~w) ~w)", [Expr, Idx]),
    clj_match(Sub, A, B0, B1, C0, C1),
    Idx1 is Idx + 1,
    clj_match_args(As, Idx1, Expr, B1, B, C1, C).

%% clj_head_positions(+HeadArgs, +Idx, +OutPositions, +B0,-B, +C0,-C)
clj_head_positions([], _, _, B, B, C, C).
clj_head_positions([Arg|Rest], Idx, OutPositions, B0, B, C0, C) :-
    (   memberchk(Idx, OutPositions)
    ->  B1 = B0, C1 = C0
    ;   format(string(PName), "a~w", [Idx]),
        clj_match(PName, Arg, B0, B1, C0, C1)
    ),
    Idx1 is Idx + 1,
    clj_head_positions(Rest, Idx1, OutPositions, B1, B, C1, C).

%% clj_unify_match(+L, +R, +B0, -B, -Conds)
clj_unify_match(L, R, B0, B, Conds) :-
    clj_term_expr(L, B0, LE), !,
    clj_match(LE, R, B0, B, [], Cs), reverse(Cs, Conds).
clj_unify_match(L, R, B0, B, Conds) :-
    clj_term_expr(R, B0, RE),
    clj_match(RE, L, B0, B, [], Cs), reverse(Cs, Conds).

% ---------------------------------------------------------------------------
% Deterministic builtins (the Clojure rendering of G-A3-1's table)
% ---------------------------------------------------------------------------
%
% Mode selection is by the bind map exactly as in the TS lane: a rule applies
% only when every INPUT term is already resolvable, so the same builtin lowers in
% either direction. Two passes — `strict` (the output must be a variable the map
% has not seen) then `loose`.
%
% In this target Prolog text is a Clojure string and a char is a ONE-CHARACTER
% STRING, so `string_chars/2` decomposes with `(mapv str (seq s))` rather than
% `(vec s)`: `vec` yields JVM characters on the JVM and one-character strings
% under ClojureScript, and `(mapv str ...)` is the same one-character-string
% answer on both hosts.

clj_struct_builtin(Goal, Bind, Out, Expr) :- clj_list_builtin(Goal, Bind, Out, Expr), !.
clj_struct_builtin(Goal, Bind, Out, Expr) :- clj_string_builtin(Goal, Bind, Out, Expr).

clj_list_builtin(reverse(L, Out), B, Out, Expr) :-
    var(Out), \+ clj_bget(Out, B, _),
    clj_term_expr(L, B, LE), format(string(Expr), "(reverse ~w)", [LE]).
clj_list_builtin(length(L, Out), B, Out, Expr) :-
    var(Out), \+ clj_bget(Out, B, _),
    clj_term_expr(L, B, LE), format(string(Expr), "(count ~w)", [LE]).
clj_list_builtin(append(A, Bl, Out), B, Out, Expr) :-
    var(Out), \+ clj_bget(Out, B, _),
    clj_term_expr(A, B, AE), clj_term_expr(Bl, B, BE),
    format(string(Expr), "(concat ~w ~w)", [AE, BE]).

clj_string_builtin(Goal, VM, Out, Expr) :-
    (   clj_sb_rule(Goal, VM, strict, Out, Expr)
    ->  true
    ;   clj_sb_rule(Goal, VM, loose, Out, Expr)
    ).

clj_sb_in(Term, VM, Expr) :- var(Term), !, clj_bget(Term, VM, Expr).
clj_sb_in(Term, VM, Expr) :- ground(Term), clj_term_expr(Term, VM, Expr).

clj_sb_out(Term, VM, strict) :- var(Term), \+ clj_bget(Term, VM, _).
clj_sb_out(Term, _VM, loose)  :- var(Term).

clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [S, Out]), clj_sb_len_pred(F),
    clj_sb_out(Out, VM, Mode), clj_sb_in(S, VM, SE),
    format(string(Expr), '(count ~w)', [SE]).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [A, B, Out]), clj_sb_concat_pred(F),
    clj_sb_out(Out, VM, Mode), clj_sb_in(A, VM, AE), clj_sb_in(B, VM, BE),
    format(string(Expr), '(str ~w ~w)', [AE, BE]).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [S, Bg, L, _After, Out]), clj_sb_sub_pred(F),
    clj_sb_out(Out, VM, Mode),
    clj_sb_in(S, VM, SE), clj_sb_in(Bg, VM, BE), clj_sb_in(L, VM, LE),
    format(string(Expr), '(subs ~w ~w (+ ~w ~w))', [SE, BE, BE, LE]).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [S, Out]), clj_sb_chars_pred(F),
    clj_sb_out(Out, VM, Mode), clj_sb_in(S, VM, SE),
    format(string(Expr), '(mapv str (seq ~w))', [SE]).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [Out, Cs]), clj_sb_chars_pred(F),
    clj_sb_out(Out, VM, Mode), clj_sb_in(Cs, VM, CE),
    format(string(Expr), '(apply str ~w)', [CE]).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [S, Out]), clj_sb_codes_pred(F),
    clj_sb_out(Out, VM, Mode), clj_sb_in(S, VM, SE),
    format(string(Expr), '(mapv uw-char-code (mapv str (seq ~w)))', [SE]).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [Out, Cs]), clj_sb_codes_pred(F),
    clj_sb_out(Out, VM, Mode), clj_sb_in(Cs, VM, CE),
    format(string(Expr), '(apply str (mapv uw-code-char ~w))', [CE]).
clj_sb_rule(char_code(C, Out), VM, Mode, Out, Expr) :-
    clj_sb_out(Out, VM, Mode), clj_sb_in(C, VM, CE),
    format(string(Expr), '(uw-char-code ~w)', [CE]).
clj_sb_rule(char_code(Out, X), VM, Mode, Out, Expr) :-
    clj_sb_out(Out, VM, Mode), clj_sb_in(X, VM, XE),
    format(string(Expr), '(uw-code-char ~w)', [XE]).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [N, Out]), clj_sb_numtext_pred(F),
    clj_sb_out(Out, VM, Mode), clj_sb_in(N, VM, NE),
    format(string(Expr), '(str ~w)', [NE]).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [Out, S]), clj_sb_numtext_pred(F),
    clj_sb_out(Out, VM, Mode), clj_sb_in(S, VM, SE),
    format(string(Expr), '(uw-parse-num ~w)', [SE]).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [A, Out]), clj_sb_textid_pred(F),
    clj_sb_out(Out, VM, Mode), clj_sb_in(A, VM, Expr).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [Out, S]), clj_sb_textid_pred(F),
    clj_sb_out(Out, VM, Mode), clj_sb_in(S, VM, Expr).
clj_sb_rule(Goal, VM, Mode, Out, Expr) :-
    clj_sb_functor(Goal, F, [S, Out]), clj_sb_case_pred(F, Method),
    clj_sb_out(Out, VM, Mode), clj_sb_in(S, VM, SE),
    format(string(Expr), '(.~w ~w)', [Method, SE]).

clj_sb_functor(_M:Goal, F, Args) :- !, clj_sb_functor(Goal, F, Args).
clj_sb_functor(Goal, F, Args) :- compound(Goal), Goal =.. [F|Args].

clj_sb_len_pred(string_length).
clj_sb_len_pred(atom_length).
clj_sb_concat_pred(string_concat).
clj_sb_concat_pred(atom_concat).
clj_sb_sub_pred(sub_string).
clj_sb_sub_pred(sub_atom).
clj_sb_chars_pred(string_chars).
clj_sb_chars_pred(atom_chars).
clj_sb_codes_pred(string_codes).
clj_sb_codes_pred(atom_codes).
clj_sb_numtext_pred(number_string).
clj_sb_textid_pred(atom_string).
clj_sb_textid_pred(string_to_atom).
clj_sb_case_pred(string_lower, 'toLowerCase').
clj_sb_case_pred(string_upper, 'toUpperCase').
clj_sb_case_pred(downcase_atom, 'toLowerCase').
clj_sb_case_pred(upcase_atom, 'toUpperCase').

%% clj_sb_reversible2(+Goal, -A1, -A2)
clj_sb_reversible2(Goal, A1, A2) :-
    clj_sb_functor(Goal, F, [A1, A2]),
    (   clj_sb_chars_pred(F) ; clj_sb_codes_pred(F)
    ;   clj_sb_numtext_pred(F) ; clj_sb_textid_pred(F) ; F == char_code
    ), !.

clj_known_builtin(F, 2) :- clj_sb_len_pred(F).
clj_known_builtin(F, 3) :- clj_sb_concat_pred(F).
clj_known_builtin(F, 5) :- clj_sb_sub_pred(F).
clj_known_builtin(F, 2) :- clj_sb_chars_pred(F).
clj_known_builtin(F, 2) :- clj_sb_codes_pred(F).
clj_known_builtin(F, 2) :- clj_sb_numtext_pred(F).
clj_known_builtin(F, 2) :- clj_sb_textid_pred(F).
clj_known_builtin(F, 2) :- clj_sb_case_pred(F, _).
clj_known_builtin(char_code, 2).
clj_known_builtin(reverse, 2).
clj_known_builtin(length, 2).
clj_known_builtin(append, 3).

% ===========================================================================
% G-A3-18 — the OUTPUT analysis, a GREATEST FIXPOINT over the call graph
% ===========================================================================
%
% Ported unchanged in substance from the TS lane; see the A3 report for why a
% single visited-set walk is not enough (self recursion, mutual recursion, and a
% predicate calling one whose outputs are still being computed).

:- dynamic clj_out_cache/3.        % Root, ClauseSignature, Table

%% clj_pred_outputs(+Pred, +Arity, -Outs)
%  FAILS for a predicate with no visible clauses. That failure is load-bearing:
%  it is what makes clj_cross_call/5 decline an unknown callee, so the caller
%  refuses out loud instead of emitting a call to a function nothing declares.
clj_pred_outputs(Pred, Arity, Outs) :-
    atom(Pred), integer(Arity), Arity >= 0,
    clj_out_clauses(Pred, Arity, Cs), Cs \== [],
    clj_out_table(Pred/Arity, Table),
    memberchk(Pred/Arity-Outs, Table).

clj_out_table(Root, Table) :-
    clj_out_graph([Root], [], Preds, [], Clauses),
    Preds \== [],
    variant_sha1(Clauses, Sig),
    (   clj_out_cache(Root, Sig, T)
    ->  Table = T
    ;   clj_out_init(Preds, T0),
        clj_out_iterate(Preds, T0, Table),
        retractall(clj_out_cache(Root, _, _)),
        assertz(clj_out_cache(Root, Sig, Table))
    ).

clj_out_graph([], Seen, Seen, Cls, Cls).
clj_out_graph([P/A|Ps], Seen, Preds, Cls0, Cls) :-
    (   memberchk(P/A, Seen)
    ->  clj_out_graph(Ps, Seen, Preds, Cls0, Cls)
    ;   clj_out_clauses(P, A, Cs),
        clj_out_callees(P/A, Cs, Callees),
        append(Ps, Callees, Queue),
        clj_out_graph(Queue, [P/A|Seen], Preds, [P/A-Cs|Cls0], Cls)
    ).

clj_out_clauses(P, A, Cs) :-
    functor(H, P, A),
    ( catch(findall(H-B, user:clause(H, B), Cs0), _, fail) -> Cs = Cs0 ; Cs = [] ).

%% clj_out_callees(+Pred/Arity, +Clauses, -Callees)
%  Self-exclusion is by NAME AND ARITY: `parse_args/2` calls `parse_args/3`.
clj_out_callees(Pred/Arity, Cs, Callees) :-
    findall(Q/QA,
            ( member(_-Body, Cs),
              clj_all_goals(Body, Goals),
              member(G0, Goals),
              clj_strip_mod(G0, G),
              compound(G),
              functor(G, Q, QA),
              \+ ( Q == Pred, QA =:= Arity ),
              \+ clj_control_functor(Q),
              \+ clj_known_builtin(Q, QA),
              functor(QH, Q, QA),
              catch(( user:clause(QH, _) -> true ; fail ), _, fail)
            ),
            Cs0),
    sort(Cs0, Callees).

clj_out_init([], []).
clj_out_init([P/A|Ps], [P/A-Outs|Rest]) :-
    clj_out_clauses(P, A, Cs),
    findall(Q, ( between(1, A, Q),
                 once(( member(H-_, Cs), arg(Q, H, X), \+ ground(X) )) ), Cands),
    clj_trailing_run(A, Cands, Outs),
    clj_out_init(Ps, Rest).

clj_out_iterate(Preds, T0, T) :-
    clj_out_step(Preds, T0, T1),
    ( T1 == T0 -> T = T0 ; clj_out_iterate(Preds, T1, T) ).

clj_out_step([], _, []).
clj_out_step([P/A|Ps], T0, [P/A-Outs|Rest]) :-
    memberchk(P/A-Old, T0),
    clj_out_clauses(P, A, Cs),
    (   Cs == []
    ->  New = []
    ;   once(clj_struct_detect_t(P, Cs, A, _DPos, Mode, T0))
    ->  clj_mode_outputs(Mode, New)
    ;   clj_general_outputs_t(A, Cs, T0, New)
    ),
    clj_out_meet(New, Old, Outs),
    clj_out_step(Ps, T0, Rest).

%% clj_out_meet(+New, +Old, -Meet) — both are suffixes, so the shorter is the
%% intersection; taking it makes every round monotone-decreasing.
clj_out_meet(New, Old, Meet) :-
    length(New, LN), length(Old, LO),
    ( LN =< LO -> Meet = New ; Meet = Old ).

clj_mode_outputs(test, []).
clj_mode_outputs(function(P), [P]).
clj_mode_outputs(function_multi(Ps), Ps).

clj_general_outputs_t(Arity, Clauses, Table, Outs) :-
    findall(P,
            ( between(1, Arity, P),
              forall(member(H-B, Clauses), clj_gout_ok(P, H, B, Table)),
              once(( member(H2-_, Clauses), arg(P, H2, A2), \+ ground(A2) ))
            ),
            Candidates),
    clj_trailing_run(Arity, Candidates, Outs).

clj_trailing_run(Arity, Candidates, Suffix) :- clj_trailing_run_(Arity, Candidates, [], Suffix).
clj_trailing_run_(P, Candidates, Acc, Suffix) :-
    (   P >= 1, memberchk(P, Candidates)
    ->  P1 is P - 1, clj_trailing_run_(P1, Candidates, [P|Acc], Suffix)
    ;   Suffix = Acc
    ).

clj_gout_ok(P, Head, Body, Table) :-
    arg(P, Head, A),
    (   ground(A) -> true
    ;   var(A)     -> clj_gout_var_ok(A, P, Head, Body, Table)
    ;   term_variables(A, Vs),
        forall(member(V, Vs), clj_gout_var_ok(V, P, Head, Body, Table))
    ).

clj_gout_var_ok(A, P, Head, Body, Table) :-
    (   clj_count_var(A, Head, NOcc), NOcc >= 2,
        Head =.. [_|HArgs],
        nth1(Q, HArgs, Earlier), Q < P, clj_var_in(A, Earlier)
    ->  true
    ;   clj_count_var(A, Head, 1),
        clj_all_goals(Body, Goals),
        clj_gout_ctx(P, Head, Goals, Ctx),
        forall(member(G-E, Ctx), clj_gout_goal_ok(A, G, E, Table)),
        once(( member(G2-E2, Ctx), clj_gout_produces(A, G2, E2, Table) ))
    ).

clj_gout_ctx(P, Head, Goals, Ctx) :-
    Head =.. [_|HArgs],
    findall(V, ( nth1(I, HArgs, Ai), I =\= P, term_variables(Ai, Vs), member(V, Vs) ), HVs),
    clj_gout_ctx_(Goals, HVs, Ctx).

clj_gout_ctx_([], _, []).
clj_gout_ctx_([G|Gs], Earlier, [G-Earlier|Rest]) :-
    term_variables(G, GVs), append(Earlier, GVs, E1),
    clj_gout_ctx_(Gs, E1, Rest).

clj_gout_goal_ok(V, Goal, Earlier, Table) :-
    ( \+ clj_var_in(V, Goal) -> true ; clj_gout_produces(V, Goal, Earlier, Table) ).

clj_gout_produces(V, Goal0, Earlier, Table) :-
    clj_strip_mod(Goal0, Goal),
    compound(Goal),
    clj_var_in(V, Goal),
    (   clj_binds_var(Goal, V, Rhs)
    ->  \+ clj_var_in(V, Rhs)
    ;   clj_goal_out_positions(Goal, Earlier, Table, Outs),
        Outs \== [],
        Goal =.. [_|Args],
        forall(nth1(I, Args, A),
               ( clj_var_in(V, A) -> ( memberchk(I, Outs), A == V ) ; true ))
    ).

clj_goal_out_positions(Goal, Earlier, _Table, [1]) :-
    clj_sb_reversible2(Goal, A1, A2),
    var(A1), \+ clj_var_memberchk(A1, Earlier),
    term_variables(A2, A2Vs), A2Vs \== [],
    forall(member(V, A2Vs), clj_var_memberchk(V, Earlier)), !.
clj_goal_out_positions(Goal, _Earlier, _Table, [Arity]) :-
    functor(Goal, F, Arity), clj_known_builtin(F, Arity), !.
clj_goal_out_positions(Goal, _Earlier, Table, Outs) :-
    functor(Goal, F, Arity), \+ clj_control_functor(F),
    memberchk(F/Arity-Outs, Table).

% ===========================================================================
% G-A3-18 — WHICH PREDICATES CAN FAIL (a LEAST fixpoint over the same graph)
% ===========================================================================

:- dynamic clj_fail_cache/3.

clj_pred_can_fail(Pred, Arity) :-
    atom(Pred), integer(Arity), Arity >= 0,
    clj_fail_table(Pred/Arity, Table),
    memberchk(Pred/Arity-true, Table).

clj_fail_table(Root, Table) :-
    clj_out_graph([Root], [], Preds, [], Clauses),
    Preds \== [],
    variant_sha1(Clauses, Sig),
    (   clj_fail_cache(Root, Sig, T)
    ->  Table = T
    ;   clj_fail_init(Preds, F0),
        clj_fail_iterate(Preds, F0, Table),
        retractall(clj_fail_cache(Root, _, _)),
        assertz(clj_fail_cache(Root, Sig, Table))
    ).

clj_fail_init([], []).
clj_fail_init([P|Ps], [P-false|R]) :- clj_fail_init(Ps, R).

clj_fail_iterate(Preds, F0, F) :-
    clj_fail_step(Preds, F0, F1),
    ( F1 == F0 -> F = F0 ; clj_fail_iterate(Preds, F1, F) ).

clj_fail_step([], _, []).
clj_fail_step([P/A|Ps], F0, [P/A-V|R]) :-
    (   memberchk(P/A-true, F0) -> V = true      % least fixpoint: never un-set
    ;   clj_fail_compute(P, A, F0) -> V = true
    ;   V = false
    ),
    clj_fail_step(Ps, F0, R).

clj_fail_compute(P, A, F0) :-
    clj_out_clauses(P, A, Cs),
    (   Cs == [] -> true
    ;   clj_pred_outputs(P, A, Outs),
        (   clj_fail_coverage_gap(A, Cs, Outs) -> true
        ;   member(_-B, Cs), clj_fail_body(B, F0)
        )
    ).

clj_fail_coverage_gap(Arity, Clauses, Outs) :-
    between(1, Arity, P),
    \+ memberchk(P, Outs),
    findall(Tag, ( member(H-_, Clauses), arg(P, H, X), clj_shape_tag(X, Tag) ), Tags0),
    sort(Tags0, Tags),
    \+ memberchk(var, Tags),
    Tags \== [cons, nil],
    !.

clj_shape_tag(X, var)  :- var(X), !.
clj_shape_tag([], nil) :- !.
clj_shape_tag(X, cons) :- nonvar(X), X = [_|_], !.
clj_shape_tag(_, other).

clj_fail_body(V, _) :- var(V), !, fail.
clj_fail_body(true, _) :- !, fail.
clj_fail_body((A, B), F) :- !, ( clj_fail_body(A, F) -> true ; clj_fail_body(B, F) ).
%  The CONDITION is allowed to fail — that is what picks the else branch.
clj_fail_body((_C -> T ; E), F) :- !, ( clj_fail_body(T, F) -> true ; clj_fail_body(E, F) ).
clj_fail_body((A ; B), F) :- !, ( clj_fail_body(A, F) -> true ; clj_fail_body(B, F) ).
clj_fail_body((_C -> T), F) :- !, clj_fail_body(T, F).
clj_fail_body(\+ _, _) :- !.
clj_fail_body(not(_), _) :- !.
clj_fail_body(_M:G, F) :- !, clj_fail_body(G, F).
clj_fail_body(G, _F) :- nonvar(G), G =.. [Op, _, _], clj_cmp_op(Op, _), !.
clj_fail_body(G, F) :- clj_fail_goal(G, F).

clj_fail_goal(G, F) :-
    compound(G), functor(G, Q, QA),
    \+ clj_control_functor(Q),
    \+ clj_known_builtin(Q, QA),
    \+ clj_fact_pred(Q, QA),
    (   memberchk(Q/QA-true, F) -> true
    ;   memberchk(Q/QA-_, F), clj_pred_outputs(Q, QA, [])
    ).

% ===========================================================================
% G-A3-19 — a ground-fact predicate used as a CONSTANT TABLE
% ===========================================================================
%
% `global_options/1`, `default_registry/1`, `js_object_prototype_keys/1`: every
% clause is a ground fact, so the output analysis sees no variable in any head
% and answers "no outputs". The lowering is a MATCH against the table, not a
% call — with one fact each argument is matched against the constant (an unbound
% argument BINDS to it), with several facts and every argument known it is a
% membership test over the emitted rows. Such a predicate is not a module member
% at all, so clj_pred_callees/2 keeps it out of the closure.

clj_fact_pred(Q, QA) :-
    atom(Q), integer(QA), QA >= 1,
    functor(H, Q, QA),
    catch(( user:clause(H, _) -> true ; fail ), _, fail),
    clj_out_clauses(Q, QA, Cs),
    Cs \== [],
    forall(member(CH-CB, Cs), ( CB == true, ground(CH) )).

clj_fact_call(SelfPred, Goal0, B0, B, Conds) :-
    clj_strip_mod(Goal0, Goal),
    compound(Goal),
    functor(Goal, Q, QA),
    \+ clj_is_self(Goal, SelfPred),
    \+ clj_control_functor(Q),
    clj_fact_pred(Q, QA),
    clj_out_clauses(Q, QA, Cs),
    Goal =.. [_|CallArgs],
    (   Cs = [FH-_]
    ->  FH =.. [_|FactArgs],
        clj_fact_match(FactArgs, CallArgs, B0, B, [], Cs0),
        reverse(Cs0, Conds)
    ;   B = B0,
        maplist(clj_term_expr_b(B0), CallArgs, CallEs),
        atomic_list_concat(CallEs, ' ', CallStr),
        findall(RowE,
                ( member(RH-_, Cs), RH =.. [_|RowArgs],
                  maplist(clj_fact_const, RowArgs, RowEs),
                  atomic_list_concat(RowEs, ' ', RowInner),
                  format(string(RowE), "[~w]", [RowInner]) ),
                RowExprs),
        atomic_list_concat(RowExprs, ' ', RowsStr),
        format(string(Cond), "(boolean (some #(= % [~w]) [~w]))", [CallStr, RowsStr]),
        Conds = [Cond]
    ).

clj_fact_match([], [], B, B, C, C).
clj_fact_match([F|Fs], [A|As], B0, B, C0, C) :-
    clj_fact_const(F, FE),
    clj_match(FE, A, B0, B1, C0, C1),
    clj_fact_match(Fs, As, B1, B, C1, C).

clj_fact_const(T, E) :- clj_term_expr(T, [], E).

% ===========================================================================
% THE ITEM FOLD — where Clojure's expression-ness replaces four TS mechanisms
% ===========================================================================
%
% A clause body is an ordered list of ITEMS plus a TAIL expression:
%
%     bind(Name, Expr)   a value the rest of the clause reads
%     gopen(Cond)        a test whose failure means THIS CLAUSE failed
%
% folded from the inside out. `Fall` is the expression to evaluate when a gopen
% fails: the next clause's chain, or the predicate's exit expression. This is
% the direct analogue of the TS lane's fall-through-by-block-exit, and it is
% MORE explicit — in TypeScript a failing in-block guard just reaches no
% `return` and drops off the end of the block, which happens to work; here the
% alternative has to be named, so it is.

%% clj_fold_items(+Items, +Tail, +Fall, +Indent, -Code)
clj_fold_items([], Tail, _Fall, _Ind, Tail) :- !.
clj_fold_items([gopen(Cond)|Rest], Tail, Fall, Ind, Code) :- !,
    clj_indent(Ind, Ind1),
    clj_fold_items(Rest, Tail, Fall, Ind1, Inner),
    format(string(Code), "(if ~w\n~w~w\n~w~w)", [Cond, Ind1, Inner, Ind1, Fall]).
clj_fold_items(Items, Tail, Fall, Ind, Code) :-
    clj_take_binds(Items, Binds, Rest),
    Binds = [_|_],
    clj_indent(Ind, Ind1),
    clj_fold_items(Rest, Tail, Fall, Ind1, Inner),
    clj_bind_pairs(Binds, Ind1, PairStr),
    format(string(Code), "(let [~w]\n~w~w)", [PairStr, Ind1, Inner]).

clj_take_binds([bind(N,E)|R], [N-E|Bs], Rest) :- !, clj_take_binds(R, Bs, Rest).
clj_take_binds(Rest, [], Rest).

clj_bind_pairs([N-E], _Ind, S) :- !, format(string(S), "~w ~w", [N, E]).
clj_bind_pairs([N-E|Rest], Ind, S) :-
    clj_bind_pairs(Rest, Ind, RS),
    format(string(S), "~w ~w\n~w      ~w", [N, E, Ind, RS]).

clj_indent(Ind, Ind1) :- format(string(Ind1), "~w  ", [Ind]).

clj_slots(_, 0, []) :- !.
clj_slots(Idx, K, [Name|Rest]) :-
    format(string(Name), "_s~w", [Idx]),
    Idx1 is Idx + 1, K1 is K - 1,
    clj_slots(Idx1, K1, Rest).

clj_bind_slot(Var, Slot, Bin, [Var-Slot|Bin]).

% ===========================================================================
% MODE DETECTION (G-A3-9) — how many outputs, and which arguments are they
% ===========================================================================

clj_struct_detect(Pred, Clauses, Arity, DPos, Mode) :-
    clj_out_table(Pred/Arity, Table),
    clj_struct_detect_t(Pred, Clauses, Arity, DPos, Mode, Table).

clj_struct_detect_t(Pred, Clauses, Arity, DPos, Mode, Table) :-
    once(( member(_-Body, Clauses), clj_body_calls(Body, Pred) )),
    (   member(RH-RBody, Clauses),
        clj_body_calls(RBody, Pred),
        between(1, Arity, DPos),
        arg(DPos, RH, RA), clj_is_cons(RA)
    ->  true
    ),
    forall(member(H-_, Clauses), ( arg(DPos, H, DA), clj_list_pos_ok(DA) )),
    (   DPos =:= Arity
    ->  Mode = test
    ;   clj_struct_output_positions(Pred, Clauses, Arity, DPos, Table, Outs),
        Outs = [_,_|_]
    ->  Mode = function_multi(Outs)
    ;   Mode = function(Arity)
    ).

clj_body_calls(Body, Pred) :-
    clj_all_goals(Body, Goals),
    member(G0, Goals), clj_strip_mod(G0, G),
    compound(G), functor(G, Pred, _), !.

clj_struct_output_positions(Pred, Clauses, Arity, DPos, Table, Outs) :-
    clj_struct_split_clauses(Pred, Clauses, RecClauses, BaseClauses),
    RecClauses \== [], BaseClauses \== [],
    findall(P,
            ( between(1, Arity, P), P =\= DPos,
              forall(member(RH-RB, RecClauses), clj_out_rec_ok(Pred, P, RH, RB, Table)),
              forall(member(BH-BB, BaseClauses), clj_out_base_ok(P, BH, BB))
            ),
            Outs).

clj_struct_split_clauses(_, [], [], []).
clj_struct_split_clauses(Pred, [H-B|Rest], Rec, Base) :-
    clj_struct_split_clauses(Pred, Rest, Rec0, Base0),
    (   clj_body_calls(B, Pred)
    ->  Rec = [H-B|Rec0], Base = Base0
    ;   Rec = Rec0, Base = [H-B|Base0]
    ).

clj_out_rec_ok(Pred, P, Head, Body, Table) :-
    arg(P, Head, V), var(V), clj_count_var(V, Head, 1),
    clj_all_goals(Body, Goals),
    forall(member(G, Goals), clj_out_goal_ok(Pred, P, V, G, Table)).

clj_out_goal_ok(Pred, P, V, Goal, _Table) :-
    compound(Goal), functor(Goal, Pred, _), !,
    Goal =.. [_|Args],
    forall(nth1(I, Args, A),
           ( I =:= P, A == V -> true ; \+ clj_var_in(V, A) )).
clj_out_goal_ok(_Pred, _P, V, Goal, _Table) :-
    clj_binds_var(Goal, V, Rhs), !, \+ clj_var_in(V, Rhs).
clj_out_goal_ok(_Pred, _P, V, Goal, Table) :-
    clj_gout_produces(V, Goal, [], Table), !.
clj_out_goal_ok(_Pred, _P, V, Goal, _Table) :-
    \+ clj_var_in(V, Goal).

clj_out_base_ok(P, Head, Body) :-
    arg(P, Head, A),
    (   nonvar(A) -> true
    ;   Head =.. [_|HArgs],
        (   nth1(Q, HArgs, Other), Q =\= P, Other == A
        ->  true
        ;   clj_all_goals(Body, Goals), member(G, Goals), clj_binds_var(G, A, _)
        )
    ).

clj_struct_inputs(Arity, test, Ps) :- numlist(1, Arity, Ps).
clj_struct_inputs(Arity, function(Out), Ps) :-
    numlist(1, Arity, All), exclude(==(Out), All, Ps).
clj_struct_inputs(Arity, function_multi(Outs), Ps) :-
    numlist(1, Arity, All), exclude(clj_pos_member(Outs), All, Ps).

clj_pos_member(Outs, P) :- memberchk(P, Outs).

%% clj_struct_ret_expr(+Mode, +OutArg, +Bind, -Expr)
%  THE MULTI-OUTPUT CALLING CONVENTION (G-A3-9): a positional Clojure VECTOR.
clj_struct_ret_expr(function_multi(_), outs(Args), Bind, Expr) :- !,
    maplist(clj_term_expr_b(Bind), Args, Es),
    atomic_list_concat(Es, ' ', Inner),
    format(string(Expr), "[~w]", [Inner]).
clj_struct_ret_expr(_Mode, OutArg, Bind, Expr) :- clj_term_expr(OutArg, Bind, Expr).

% ===========================================================================
% CONDITIONS (G-A3-6 · G-A3-18) — an if-then-else condition that may also BIND
% ===========================================================================
%
% `Pre` is the list of items the enclosing expression must bind BEFORE the
% `if`. In the TS lane this is a `let _tN;` declaration plus an assignment made
% INSIDE the condition, because JavaScript needs a mutable slot to smuggle a
% value out of a condition. Clojure needs no such trick: the call is an ordinary
% `bind` item and the condition reads the name.
%
%     (let [_t0 (pair-lookup a1 a2)]
%       (if (not (identical? _t0 uw-fail)) <then, with _t0 in scope> <else>))
%
% The bindings go into the THEN branch's map only, which is Prolog's scope rule.

clj_cond(Goal, B0, B, Cond) :- clj_cond(Goal, B0, B, Pre, Cond, 0, _), Pre == [].

clj_cond(Goal, B0, B, Pre, Cond, N, N) :-
    clj_guard_condition(B0, Goal, Cond), !, B = B0, Pre = [].
%  A ground-fact CONSTANT TABLE read in condition position (G-A3-19): with one
%  fact it binds and cannot fail, so it contributes the condition "true".
clj_cond(Goal, B0, B, Pre, Cond, N, N) :-
    clj_fact_call('$no_self', Goal, B0, B, Conds), !,
    Pre = [],
    ( Conds == [] -> Cond = "true" ; clj_and_conds(Conds, Cond) ).
%  A SEMIDET CALL WITH OUTPUTS (G-A3-18) — the shape the sentinel exists for.
clj_cond(Goal, B0, B, Pre, Cond, N0, N) :-
    clj_cross_call('$no_self', Goal, Q, Args, Outs),
    Outs = [_|_],
    functor(Goal, _, QA),
    clj_pred_can_fail(Q, QA),
    clj_split_out_args(Args, 1, Outs, InArgs, CallOutArgs),
    clj_distinct_vars(CallOutArgs),
    forall(member(OV, CallOutArgs), \+ clj_bget(OV, B0, _)),
    clj_call_expr(Q, QA, InArgs, B0, Call),
    !,
    format(string(Slot), "_t~w", [N0]), N is N0 + 1,
    Pre = [bind(Slot, Call)],
    format(string(Cond), "(not (identical? ~w uw-fail))", [Slot]),
    length(CallOutArgs, K),
    clj_fail_out_exprs(Slot, K, OutEs),
    foldl(clj_bind_slot, CallOutArgs, OutEs, B0, B).
clj_cond(Goal, B0, B, Pre, Cond, N0, N) :-
    nonvar(Goal), Goal = (A, C), !,
    clj_cond(A, B0, B1, PreA, CA, N0, N1),
    clj_cond(C, B1, B, PreC, CC, N1, N),
    append(PreA, PreC, Pre),
    clj_cond_and(CA, CC, Cond).
clj_cond(Goal, B0, B, [], Cond, N, N) :-
    nonvar(Goal), Goal = (L = R), !,
    clj_unify_match(L, R, B0, B, Conds),
    ( Conds == [] -> Cond = "true" ; clj_and_conds(Conds, Cond) ).
clj_cond(Goal, B, B, [], Cond, N, N) :-
    nonvar(Goal), Goal =.. [Op, L, R], clj_cmp_op(Op, _), !,
    clj_cmp_cond(Op, L, R, B, Cond).
%  Negation. Bindings made inside it are discarded, as in Prolog.
clj_cond(Goal, B, B, [], Cond, N, N) :-
    nonvar(Goal), ( Goal = \+(Inner) ; Goal = not(Inner) ), !,
    clj_cond(Inner, B, _, CI),
    format(string(Cond), "(not ~w)", [CI]).

clj_cond_and("true", C, C) :- !.
clj_cond_and(C, "true", C) :- !.
clj_cond_and(A, B, Cond) :- format(string(Cond), "(and ~w ~w)", [A, B]).

clj_and_conds([C], C) :- !.
clj_and_conds(Cs, Cond) :- atomic_list_concat(Cs, ' ', S), format(string(Cond), "(and ~w)", [S]).

%% clj_guard_condition(+Bind, +Goal, -Cond) — the PURE-test forms.
%  Named to mirror the historical clojure_guard_condition/3 but reading the
%  structural path's Var-Expr bind map instead of a VarMap.
clj_guard_condition(B, _M:Goal, Cond) :- !, clj_guard_condition(B, Goal, Cond).
clj_guard_condition(_B, true, "true") :- !.
clj_guard_condition(_B, fail, "false") :- !.
clj_guard_condition(_B, false, "false") :- !.
clj_guard_condition(B, (A, C), Cond) :- !,
    clj_guard_condition(B, A, CA), clj_guard_condition(B, C, CC),
    clj_cond_and(CA, CC, Cond).
clj_guard_condition(B, (A ; C), Cond) :- !,
    clj_guard_condition(B, A, CA), clj_guard_condition(B, C, CC),
    format(string(Cond), "(or ~w ~w)", [CA, CC]).
clj_guard_condition(B, Goal, Cond) :-
    nonvar(Goal), Goal = (If -> Then ; Else), !,
    clj_guard_condition(B, If, CI),
    clj_guard_condition(B, Then, CT),
    clj_guard_condition(B, Else, CE),
    format(string(Cond), "(if ~w ~w ~w)", [CI, CT, CE]).
clj_guard_condition(B, \+(Inner), Cond) :- !,
    clj_guard_condition(B, Inner, CI), format(string(Cond), "(not ~w)", [CI]).
clj_guard_condition(B, not(Inner), Cond) :- !,
    clj_guard_condition(B, Inner, CI), format(string(Cond), "(not ~w)", [CI]).
clj_guard_condition(B, Goal, Cond) :-
    compound(Goal), Goal =.. [Op, L, R], clj_cmp_op(Op, _), !,
    clj_cmp_cond(Op, L, R, B, Cond).
%  A cross-predicate call with NO outputs: the emitted function answers a
%  boolean, so the call IS the condition. Tried LAST, so no goal that already
%  had a rendering changes shape.
clj_guard_condition(B, Goal, Cond) :-
    clj_cross_call('$no_self', Goal, Q, Args, []),
    functor(Goal, _, QA),
    clj_call_expr(Q, QA, Args, B, Cond).

clj_fail_out_exprs(Slot, 1, [Slot]) :- !.
clj_fail_out_exprs(Slot, K, Exprs) :-
    K1 is K - 1,
    findall(E, ( between(0, K1, I), format(string(E), "(nth ~w ~w)", [Slot, I]) ), Exprs).

clj_cross_call(SelfPred, Goal, Q, Args, Outs) :-
    compound(Goal),
    Goal \= (_:_),
    functor(Goal, Q, QA),
    \+ clj_is_self(Goal, SelfPred),
    \+ clj_control_functor(Q),
    \+ clj_fact_pred(Q, QA),
    clj_pred_outputs(Q, QA, Outs),
    Goal =.. [_|Args].

clj_call_expr(Pred, Arity, Args, Bind, Expr) :-
    maplist(clj_term_expr_b(Bind), Args, Es),
    clj_emitted_name(Pred, Arity, PredStr),
    (   Es == []
    ->  format(string(Expr), "(~w)", [PredStr])
    ;   atomic_list_concat(Es, ' ', ArgStr),
        format(string(Expr), "(~w ~w)", [PredStr, ArgStr])
    ).

% ===========================================================================
% THE BODY SEQUENCE
% ===========================================================================

%% clj_struct_body(+Body,+Pred,+Mode,+OutArg,+Fall,+B0,-B,-Guards,-Items,-Tail)
clj_struct_body(Body, Pred, Mode, OutArg, Fall, B0, B, Guards, Items, Tail) :-
    clj_branch_goals(Body, Goals),
    clj_struct_seq(Goals, Pred, Mode, OutArg, tail, Fall,
                   B0, B, [], GuardsR, [], ItemsR, none, Tail, 0, _),
    reverse(GuardsR, Guards),
    reverse(ItemsR, Items).

%% clj_struct_seq(+Goals,+Pred,+Mode,+OutArg,+Ctx,+Fall,
%%                +B0,-B, +G0,-G, +I0,-I, +T0,-T, +N0,-N)
%
%  Ctx is `tail` when the LAST goal sits in return position and `inner` when it
%  does not. Tail is `none` (the caller computes the return expression from the
%  output argument) or `expr(E)` (the body produced the whole value itself).
clj_struct_seq([], _, _, _, _, _, B, B, G, G, I, I, T, T, N, N).
%  A TAIL if-then-else (G-A3-10). In TypeScript this needs a branching-return
%  lowering with its own `return` per branch; in Clojure `if` IS the value.
clj_struct_seq([Goal|Rest], Pred, Mode, OutArg, Ctx, Fall,
               B0, B, G0, G, I0, I, _T0, T, N0, N) :-
    Rest == [], Ctx == tail,
    clj_is_ite(Goal, If, Then, Else),
    clj_struct_tail_ite(If, Then, Else, Pred, Mode, OutArg, Fall, B0, Expr, N0, N1),
    !,
    B = B0, G = G0, I = I0, T = expr(Expr), N = N1.
%  Multi-output TAIL CALL (G-A3-9): the callee's vector IS this clause's answer,
%  so it flows straight through with nothing unpacked and rebuilt.
clj_struct_seq([Goal|Rest], Pred, Mode, OutArg, Ctx, _Fall,
               B0, B, G0, G, I0, I, _T0, T, N0, N) :-
    Rest == [], Ctx == tail,
    Mode = function_multi(Outs),
    OutArg = outs(OutHeadArgs),
    clj_is_self(Goal, Pred),
    Goal =.. [_|Args],
    clj_split_out_args(Args, 1, Outs, InArgs, CallOutArgs),
    clj_same_var_list(CallOutArgs, OutHeadArgs),
    maplist(clj_term_expr_b(B0), InArgs, InEs),
    !,
    clj_self_fn_name(Pred, PredStr),
    ( InEs == [] -> format(string(Expr), "(~w)", [PredStr])
    ; atomic_list_concat(InEs, ' ', ArgStr),
      format(string(Expr), "(~w ~w)", [PredStr, ArgStr]) ),
    B = B0, G = G0, I = I0, T = expr(Expr), N = N0.
%  A BINDING whose right-hand side a LATER goal produces (G-A3-6/G-A3-20).
%  Prolog's conjunction is not evaluation order for a pure `=`/2, and Clojure has
%  no such hole either, so the goal is DEFERRED: the rest of the sequence is
%  rendered first and the binding is made against what comes out of it. Nothing
%  is emitted for the deferred goal — it only names a value.
clj_struct_seq([Goal|Rest], Pred, Mode, OutArg, Ctx, Fall,
               B0, B, G0, G, I0, I, T0, T, N0, N) :-
    Rest \== [],
    nonvar(Goal), Goal = (V = Term), var(V), nonvar(Term),
    \+ clj_bget(V, B0, _),
    \+ clj_term_expr(Term, B0, _),
    !,
    clj_struct_seq(Rest, Pred, Mode, OutArg, Ctx, Fall, B0, B1, G0, G, I0, I, T0, T, N0, N),
    clj_term_expr(Term, B1, E),
    B = [V-E|B1].
clj_struct_seq([Goal|Rest], Pred, Mode, OutArg, Ctx, Fall,
               B0, B, G0, G, I0, I, T0, T, N0, N) :-
    clj_struct_goal(Goal, Pred, Mode, Fall, B0, B1, G0, G1, I0, I1, T0, T1, N0, N1),
    clj_struct_seq(Rest, Pred, Mode, OutArg, Ctx, Fall, B1, B, G1, G, I1, I, T1, T, N1, N).
%  A DEFERRED if-then-else (G-A3-20) — list-BUILDING recursion. Both branches
%  describe the output in terms of a value the call AFTER them computes, so the
%  if-then-else is rendered AFTER the rest of the sequence. Tried only when the
%  in-place value lowering has already FAILED, and only when the CONDITION is
%  renderable against the bindings available before the rest runs.
clj_struct_seq([Goal|Rest], Pred, Mode, OutArg, Ctx, Fall,
               B0, B, G0, G, I0, I, T0, T, N0, N) :-
    Rest \== [],
    clj_is_ite(Goal, If, Then, Else),
    \+ clj_struct_value_ite(If, Then, Else, Pred, Mode, Fall, B0, _, I0, _, N0, _),
    clj_cond(If, B0, _, [], _, N0, _),
    clj_struct_seq(Rest, Pred, Mode, OutArg, Ctx, Fall, B0, B1, G0, G, I0, I1, T0, T, N0, N1),
    clj_struct_value_ite(If, Then, Else, Pred, Mode, Fall, B1, B, I1, I, N1, N).
%  An if-then-else in TAIL context that no VALUE lowering fits, because a branch
%  opens with a failable test (G-A3-20). Prolog COMMITS to a branch, so the goals
%  that follow belong to whichever branch was taken: the continuation is appended
%  to both branches and the whole thing becomes a TAIL if-then-else. Tried last,
%  so the continuation is duplicated only where the alternative is a refusal.
clj_struct_seq([Goal|Rest], Pred, Mode, OutArg, Ctx, Fall,
               B0, B, G0, G, I0, I, _T0, T, N0, N) :-
    Rest \== [], Ctx == tail,
    clj_is_ite(Goal, If, Then, Else),
    clj_conj_of(Rest, RestConj),
    clj_struct_tail_ite(If, (Then, RestConj), (Else, RestConj),
                        Pred, Mode, OutArg, Fall, B0, Expr, N0, N1),
    !,
    B = B0, G = G0, I = I0, T = expr(Expr), N = N1.

%% clj_struct_tail_ite(+If,+Then,+Else,+Pred,+Mode,+OutArg,+Fall,+B0,-Expr,+N0,-N)
clj_struct_tail_ite(If, Then, Else, Pred, Mode, OutArg, Fall, B0, Expr, N0, N) :-
    clj_cond(If, B0, BThen, Pre, Cond, N0, N1),
    clj_struct_branch_value(Then, Pred, Mode, OutArg, Fall, BThen, ThenExpr, N1, N2),
    clj_struct_branch_value(Else, Pred, Mode, OutArg, Fall, B0, ElseExpr, N2, N),
    format(string(Ite), "(if ~w\n      ~w\n      ~w)", [Cond, ThenExpr, ElseExpr]),
    clj_fold_items(Pre, Ite, Fall, "    ", Expr).

%% clj_struct_branch_value(+Branch,+Pred,+Mode,+OutArg,+Fall,+B0,-Expr,+N0,-N)
%  Render one branch of a tail if-then-else as the value it produces.
%
%  G-A3-18: a branch may open with a FAILABLE TEST — a match that binds
%  (`MaybeAction = some(Action)`) or a comparison (`Action \== ""`). A clause
%  collects such a test into its own condition; a branch has none, so the test
%  becomes an `if` around the branch whose else is the clause's Fall. A branch
%  whose test fails therefore falls through exactly as Prolog's reading requires:
%  the branch failed, so the clause failed.
clj_struct_branch_value(Branch, Pred, Mode, OutArg, Fall, B0, Expr, N0, N) :-
    clj_branch_goals(Branch, Goals),
    clj_struct_seq(Goals, Pred, Mode, OutArg, tail, Fall,
                   B0, B1, [], GuardsR, [], ItemsR, none, Tail, N0, N),
    reverse(ItemsR, Items),
    (   Tail = expr(E) -> RetExpr = E
    ;   Mode == test   -> RetExpr = "true"
    ;   clj_struct_ret_expr(Mode, OutArg, B1, RetExpr)
    ),
    clj_fold_items(Items, RetExpr, Fall, "      ", Body),
    (   GuardsR == []
    ->  Expr = Body
    ;   reverse(GuardsR, Guards),
        clj_and_conds(Guards, GuardStr),
        format(string(Expr), "(if ~w\n        ~w\n        ~w)", [GuardStr, Body, Fall])
    ).

%% clj_struct_value_ite(+If,+Then,+Else,+Pred,+Mode,+Fall,+B0,-B,+I0,-I,+N0,-N)
%
%  G-A3-10's VALUE form, and the clearest place Clojure is simpler than
%  TypeScript. The TS lane declares `let _s0, _s1;` before the block and assigns
%  into them at the end of each branch, because a JS `if` is a statement. Here
%  the `if` is the value:
%
%      (let [_s0 (if C then-val else-val)] <rest>)
%
%  and with several shared outputs it destructures a vector:
%
%      (let [[_s0 _s1] (if C [t0 t1] [e0 e1])] <rest>)
%
%  A branch that would need a gopen (a failable test) is REFUSED here rather
%  than half-lowered: a `let` binding position cannot express "and if that test
%  fails, the whole clause fails". Such an if-then-else is picked up by the tail
%  lowering above, where the alternative can be named.
clj_struct_value_ite(If, Then, Else, Pred, Mode, Fall, B0, B, I0, I, N00, N) :-
    clj_cond(If, B0, BThen, Pre, Cond, N00, N0),
    Pre == [],
    if_then_else_shared_output_vars(Then, Else, B0, SharedVars),
    SharedVars \== [],
    clj_branch_goals(Then, ThenGoals),
    clj_branch_goals(Else, ElseGoals),
    clj_struct_seq(ThenGoals, Pred, Mode, '$no_output', inner, Fall,
                   BThen, BT, [], [], [], ThenItemsR, none, none, N0, N1),
    clj_struct_seq(ElseGoals, Pred, Mode, '$no_output', inner, Fall,
                   B0, BE, [], [], [], ElseItemsR, none, none, N1, N2),
    reverse(ThenItemsR, ThenItems), reverse(ElseItemsR, ElseItems),
    forall(member(It, ThenItems), It = bind(_, _)),
    forall(member(It, ElseItems), It = bind(_, _)),
    length(SharedVars, K),
    clj_slots(N2, K, Slots),
    N is N2 + K,
    maplist(clj_shared_expr(BT), SharedVars, ThenEs),
    maplist(clj_shared_expr(BE), SharedVars, ElseEs),
    (   K =:= 1
    ->  ThenEs = [TE0], ElseEs = [EE0], Slots = [SlotName],
        clj_fold_items(ThenItems, TE0, Fall, "      ", ThenVal),
        clj_fold_items(ElseItems, EE0, Fall, "      ", ElseVal)
    ;   atomic_list_concat(ThenEs, ' ', TI), format(string(TV0), "[~w]", [TI]),
        atomic_list_concat(ElseEs, ' ', EI), format(string(EV0), "[~w]", [EI]),
        clj_fold_items(ThenItems, TV0, Fall, "      ", ThenVal),
        clj_fold_items(ElseItems, EV0, Fall, "      ", ElseVal),
        atomic_list_concat(Slots, ' ', SlotInner),
        format(string(SlotName), "[~w]", [SlotInner])
    ),
    format(string(IteExpr), "(if ~w\n        ~w\n        ~w)", [Cond, ThenVal, ElseVal]),
    I = [bind(SlotName, IteExpr)|I0],
    foldl(clj_bind_slot, SharedVars, Slots, B0, B).

clj_shared_expr(Bind, Var, Expr) :- clj_term_expr(Var, Bind, Expr).

% ===========================================================================
% GOALS
% ===========================================================================

%% clj_struct_goal(+Goal,+Pred,+Mode,+Fall,+B0,-B,+G0,-G,+I0,-I,+T0,-T,+N0,-N)
clj_struct_goal(true, _, _, _, B, B, G, G, I, I, T, T, N, N) :- !.
%  if-then-else in VALUE position (G-A3-10). Cut on recognition: if the value
%  lowering does not apply, this path must refuse rather than try some other
%  reading of the same goal.
clj_struct_goal(Goal, Pred, Mode, Fall, B0, B, G, G, I0, I, T, T, N0, N) :-
    clj_is_ite(Goal, If, Then, Else), !,
    clj_struct_value_ite(If, Then, Else, Pred, Mode, Fall, B0, B, I0, I, N0, N).
%  A comparison. Guard PLACEMENT (G-A3-6): a guard over head-bound variables, or
%  one appearing before any item has been emitted, joins the clause's condition;
%  a guard that reads a value a preceding bind produced becomes an in-place
%  `gopen`, which the fold turns into `(if Cond <rest> <Fall>)`.
clj_struct_goal(Goal, _Pred, _Mode, _Fall, B, B, G0, G, I0, I, T, T, N, N) :-
    Goal =.. [Op, L, R], clj_cmp_op(Op, _), !,
    clj_cmp_cond(Op, L, R, B, Cond),
    (   ( I0 == [] ; clj_guard_head_only(Goal) )
    ->  G = [Cond|G0], I = I0
    ;   G = G0, I = [gopen(Cond)|I0]
    ).
clj_struct_goal(is(V, Expr), _Pred, _Mode, _Fall, B0, [V-Nm|B0], G, G,
                I0, [bind(Nm, ES)|I0], T, T, N0, N1) :-
    var(V), !,
    clj_arith(Expr, B0, ES),
    format(string(Nm), "_s~w", [N0]), N1 is N0 + 1.
%  `V = Term` as a BINDING. The cut is after the render, not before it: when
%  clj_term_expr/3 cannot yet name every variable of Term the goal may still be a
%  MATCH, or a goal whose right-hand side a LATER goal produces.
clj_struct_goal(=(V, Term), _Pred, _Mode, _Fall, B0, [V-E|B0], G, G, I, I, T, T, N, N) :-
    var(V),
    %  V must be UNBOUND here. When the clause already holds a value for V the
    %  goal is a TEST against it, not a second binding.
    \+ clj_bget(V, B0, _),
    clj_term_expr(Term, B0, E), !.
%  `L = R` as a MATCH (G-A3-12/G-A3-16).
clj_struct_goal(=(L, R), _Pred, _Mode, _Fall, B0, B, G0, G, I0, I, T, T, N, N) :-
    clj_unify_match(L, R, B0, B, Conds), !,
    (   Conds == []
    ->  G = G0, I = I0
    ;   clj_and_conds(Conds, Cond),
        ( I0 == [] -> G = [Cond|G0], I = I0 ; G = G0, I = [gopen(Cond)|I0] )
    ).
%  A deterministic string / char / list builtin.
clj_struct_goal(Goal, _Pred, _Mode, _Fall, B0, [Out-Nm|B0], G, G,
                I0, [bind(Nm, Expr)|I0], T, T, N0, N1) :-
    clj_struct_builtin(Goal, B0, Out, Expr), !,
    format(string(Nm), "_s~w", [N0]), N1 is N0 + 1.
%  Negation-as-failure as a body GOAL: a test, so it goes where a comparison does.
clj_struct_goal(Goal, _Pred, _Mode, _Fall, B, B, G0, G, I0, I, T, T, N, N) :-
    nonvar(Goal), ( Goal = \+(_) ; Goal = not(_) ),
    clj_cond(Goal, B, _, Cond), !,
    ( I0 == [] -> G = [Cond|G0], I = I0 ; G = G0, I = [gopen(Cond)|I0] ).
%  A non-tail SELF call in a multi-output predicate: it returns the vector, so it
%  is DESTRUCTURED into one slot per output (G-A3-9).
clj_struct_goal(Goal, Pred, function_multi(Outs), _Fall, B0, B, G, G,
                I0, [bind(SlotName, Call)|I0], T, T, N0, N) :-
    clj_is_self(Goal, Pred), !,
    Goal =.. [_|Args],
    clj_split_out_args(Args, 1, Outs, InArgs, CallOutArgs),
    clj_distinct_vars(CallOutArgs),
    forall(member(OV, CallOutArgs), \+ clj_bget(OV, B0, _)),
    maplist(clj_term_expr_b(B0), InArgs, InEs),
    clj_self_fn_name(Pred, PredStr),
    ( InEs == [] -> format(string(Call), "(~w)", [PredStr])
    ; atomic_list_concat(InEs, ' ', ArgStr),
      format(string(Call), "(~w ~w)", [PredStr, ArgStr]) ),
    length(CallOutArgs, K),
    clj_slots(N0, K, Slots),
    N is N0 + K,
    atomic_list_concat(Slots, ' ', SlotInner),
    format(string(SlotName), "[~w]", [SlotInner]),
    foldl(clj_bind_slot, CallOutArgs, Slots, B0, B).
clj_struct_goal(Goal, Pred, function(_), _Fall, B0, B, G, G,
                I0, [bind(Nm, Call)|I0], T, T, N0, N1) :-
    clj_is_self(Goal, Pred), !,
    Goal =.. [_|Args],
    append(InArgs, [OutArg], Args), var(OutArg),
    maplist(clj_term_expr_b(B0), InArgs, InEs),
    clj_self_fn_name(Pred, PredStr),
    ( InEs == [] -> format(string(Call), "(~w)", [PredStr])
    ; atomic_list_concat(InEs, ' ', ArgStr),
      format(string(Call), "(~w ~w)", [PredStr, ArgStr]) ),
    format(string(Nm), "_s~w", [N0]), N1 is N0 + 1,
    B = [OutArg-Nm|B0].
clj_struct_goal(Goal, Pred, test, _Fall, B, B, G, G, I, I, _T0, expr(R), N, N) :-
    clj_is_self(Goal, Pred), !,
    Goal =.. [_|Args],
    maplist(clj_term_expr_b(B), Args, Es),
    clj_self_fn_name(Pred, PredStr),
    ( Es == [] -> format(string(R), "(~w)", [PredStr])
    ; atomic_list_concat(Es, ' ', ArgStr),
      format(string(R), "(~w ~w)", [PredStr, ArgStr]) ).
%  A call to a GROUND-FACT CONSTANT TABLE (G-A3-19): a MATCH against the table,
%  not a call. Placed ahead of the cross-call clauses because such a predicate has
%  no outputs by the analysis's reading and would otherwise be lowered as a
%  boolean test calling a function nothing declares.
clj_struct_goal(Goal, Pred, _Mode, _Fall, B0, B, G0, G, I0, I, T, T, N, N) :-
    clj_fact_call(Pred, Goal, B0, B, Conds), !,
    (   Conds == []
    ->  G = G0, I = I0
    ;   clj_and_conds(Conds, Cond),
        ( I0 == [] -> G = [Cond|G0], I = I0 ; G = G0, I = [gopen(Cond)|I0] )
    ).
%  A cross-predicate call in body-goal position (G-A3-6), lowered by the CALLEE's
%  output count. 0 outputs -> a boolean test (a `gopen`, so a false answer falls
%  through to the caller's next clause, which is Prolog's semantics exactly).
clj_struct_goal(Goal, Pred, _Mode, _Fall, B, B, G, G, I0, [gopen(Call)|I0], T, T, N, N) :-
    clj_cross_call(Pred, Goal, Q, Args, []), !,
    functor(Goal, _, QA),
    clj_call_expr(Q, QA, Args, B, Call).
%  With outputs: a bind, plus (when the callee is SEMIDET, G-A3-18) a sentinel
%  test so a failing call falls through to the caller's next clause.
clj_struct_goal(Goal, Pred, _Mode, _Fall, B0, B, G, G, I0, I, T, T, N0, N) :-
    clj_cross_call(Pred, Goal, Q, Args, Outs), Outs = [_|_], !,
    clj_split_out_args(Args, 1, Outs, InArgs, CallOutArgs),
    clj_distinct_vars(CallOutArgs),
    forall(member(OV, CallOutArgs), \+ clj_bget(OV, B0, _)),
    functor(Goal, _, GArity),
    clj_call_expr(Q, GArity, InArgs, B0, Call),
    length(CallOutArgs, K),
    (   clj_pred_can_fail(Q, GArity)
    ->  clj_slots(N0, 1, [Slot]), N is N0 + 1,
        format(string(Cond), "(not (identical? ~w uw-fail))", [Slot]),
        clj_fail_out_exprs(Slot, K, OutEs),
        foldl(clj_bind_slot, CallOutArgs, OutEs, B0, B),
        I = [gopen(Cond), bind(Slot, Call)|I0]
    ;   clj_slots(N0, K, Slots), N is N0 + K,
        (   K =:= 1
        ->  Slots = [SlotName]
        ;   atomic_list_concat(Slots, ' ', SlotInner),
            format(string(SlotName), "[~w]", [SlotInner])
        ),
        foldl(clj_bind_slot, CallOutArgs, Slots, B0, B),
        I = [bind(SlotName, Call)|I0]
    ).

% ===========================================================================
% CLAUSE CHAIN AND FUNCTION EMISSION
% ===========================================================================

%% clj_struct_clause(+Pred,+Arity,+Mode,+Fall,+Head-Body,-clause(Cond,Code))
clj_struct_clause(Pred, _Arity, Mode, Fall, Head-Body, clause(CondStr, Code)) :-
    Head =.. [_|HeadArgs],
    (   Mode = function(OutPos0)
    ->  OutPositions = [OutPos0], nth1(OutPos0, HeadArgs, OutArg)
    ;   Mode = function_multi(Outs)
    ->  OutPositions = Outs,
        maplist(clj_nth_arg(HeadArgs), Outs, OutArgs),
        OutArg = outs(OutArgs)
    ;   OutPositions = [], OutArg = '$no_output'
    ),
    clj_head_positions(HeadArgs, 1, OutPositions, [], Bind1, [], Conds0),
    reverse(Conds0, HeadConds),
    clj_set_head_bind(Bind1),
    clj_struct_body(Body, Pred, Mode, OutArg, Fall, Bind1, Bind2, GuardConds, Items, Tail),
    append(HeadConds, GuardConds, AllConds),
    ( AllConds == [] -> CondStr = "true" ; clj_and_conds(AllConds, CondStr) ),
    (   Tail = expr(RetExpr) -> true
    ;   Mode == test         -> RetExpr = "true"
    ;   clj_struct_ret_expr(Mode, OutArg, Bind2, RetExpr)
    ),
    clj_fold_items(Items, RetExpr, Fall, "    ", Code).

%% clj_clause_chain(+Clauses,+Pred,+Arity,+Mode,+Default,-Code)
%
%  Prolog's clause order is preserved and its FIRST-MATCH selection is the
%  chain's shape. Rendered right-to-left so each clause knows the expression to
%  evaluate when it does not match, or when one of its in-place guards fails.
%
%  When a clause body never uses that expression internally — the common case,
%  and every multi-clause predicate in cli_args, whose clauses are distinguished
%  by their first argument's list shape — the successor is inlined and the chain
%  is a plain nest of `if`s. When a body DOES use it (an in-place guard, a
%  failable branch) the successor is bound to a zero-argument thunk first, so it
%  is written once and evaluated only if reached.
clj_clause_chain(Clauses, Pred, Arity, Mode, Default, Code) :-
    clj_chain_(Clauses, Pred, Arity, Mode, Default, 1, Code).

clj_chain_([], _, _, _, Default, _, Default).
clj_chain_([Cl|Rest], Pred, Arity, Mode, Default, K, Code) :-
    K1 is K + 1,
    clj_chain_(Rest, Pred, Arity, Mode, Default, K1, RestCode),
    (   Rest == []
    ->  Fall = Default, FallName = none
    ;   format(string(FallName), "uw-fall-~w", [K]),
        format(string(Fall), "(~w)", [FallName])
    ),
    clj_struct_clause(Pred, Arity, Mode, Fall, Cl, clause(Cond, Body)),
    (   Cond == "true"
    ->  Code0 = Body
    ;   format(string(Code0), "(if ~w\n    ~w\n    ~w)", [Cond, Body, Fall])
    ),
    (   FallName == none
    ->  Code = Code0
    ;   sub_string(Body, _, _, _, FallName)
    ->  format(string(Code), "(let [~w (fn [] ~w)]\n    ~w)", [FallName, RestCode, Code0])
    ;   % The successor is used exactly once, in the `if`'s else position: inline
        % it and keep the chain a readable nest of `if`s.
        format(string(Code), "(if ~w\n    ~w\n    ~w)", [Cond, Body, RestCode])
    ).

%% clj_struct_emit(+Pred, +Arity, +Clauses, +Mode, -Code)
clj_struct_emit(Pred, Arity, Clauses, Mode, Code) :-
    clj_fn_name(Pred, Arity, PredStr),
    clj_set_self_arity(Arity),
    clj_struct_inputs(Arity, Mode, InPositions),
    maplist(clj_param_name, InPositions, Params),
    atomic_list_concat(Params, ' ', ParamList),
    %  G-A3-18: a SEMIDET predicate WITH outputs answers with the sentinel
    %  instead of throwing; a semidet TEST answers false; everything else throws
    %  by name, so a match that should not have failed is a crash naming the
    %  predicate rather than a wrong answer.
    (   Mode == test
    ->  Default = "false"
    ;   clj_pred_can_fail(Pred, Arity)
    ->  Default = "uw-fail"
    ;   format(string(Default),
               "(throw (ex-info \"no matching clause for ~w/~w\" {}))", [PredStr, Arity])
    ),
    clj_clause_chain(Clauses, Pred, Arity, Mode, Default, Body),
    format(string(Code), "(defn ~w [~w]\n  ~w)", [PredStr, ParamList, Body]).

clj_param_name(Pos, Name) :- format(string(Name), "a~w", [Pos]).

%% native_clj_structural(+Pred/Arity, +Clauses, -Code)
%  A multi-output loop that this path cannot lower is REFUSED by name rather
%  than handed to a lowering that would treat one argument as the answer and
%  emit the rest as inputs.
native_clj_structural(Pred/Arity, Clauses, Code) :-
    clj_struct_detect(Pred, Clauses, Arity, _DPos, Mode),
    (   Mode = function_multi(Outs)
    ->  (   clj_struct_emit(Pred, Arity, Clauses, Mode, Code)
        ->  true
        ;   clj_refuse_multi_output(Pred/Arity, Outs)
        )
    ;   clj_struct_emit(Pred, Arity, Clauses, Mode, Code)
    ).

clj_refuse_multi_output(PredSpec, Outs) :-
    length(Outs, K),
    atomic_list_concat(Outs, ', ', PosStr),
    format(string(Shape),
           'a recursive predicate with ~w output arguments (positions ~w)', [K, PosStr]),
    format(string(Msg),
'clojure_target: cannot compile ~w -- it has ~w output arguments (positions ~w), \c
and the structural path could not lower its body under the multi-output calling \c
convention (a vector [out1 ... out~w] returned from every exit and threaded \c
through every recursive call). Refused rather than lowered as a single-output \c
predicate, which would emit the other outputs as INPUT parameters and compare \c
them against the accumulators.', [PredSpec, K, PosStr, K]),
    throw(error(unsupported_lowering(clojure, PredSpec, Shape), Msg)).

%% native_clj_general(+Pred/Arity, +Clauses, -Code)
%  The rescue path: the MODE comes from clj_pred_outputs/3 rather than from a
%  decomposition argument, so a predicate needs neither recursion nor a cons
%  pattern to qualify.
native_clj_general(Pred/Arity, Clauses, Code) :-
    clj_general_mode(Pred, Arity, Mode),
    clj_struct_emit(Pred, Arity, Clauses, Mode, Code).

clj_general_mode(Pred, Arity, Mode) :-
    clj_pred_outputs(Pred, Arity, Outs),
    (   Outs == []  -> Mode = test
    ;   Outs = [P]  -> Mode = function(P)
    ;   Mode = function_multi(Outs)
    ).

%% native_clj_whole(+Pred/Arity, +Clauses, -Code)
%  The two paths in the order the TS lane uses them: structural first (a genuine
%  list recursion), general second.
native_clj_whole(PredSpec, Clauses, Code) :-
    catch(native_clj_structural(PredSpec, Clauses, Code0), Err, throw(Err)),
    !, Code = Code0.
native_clj_whole(PredSpec, Clauses, Code) :-
    native_clj_general(PredSpec, Clauses, Code).

%% clj_a3_cli_entry(+Pred, +Arity, +PredStr, -CliEntry)
%
%  The standalone entry point for a predicate compiled by the A3 paths. It
%  differs from clojure_native_cli_entry/4 (the historical one) in the two ways
%  these predicates force, exactly as the TS lane's ts_struct_cli_entry/4 does:
%
%    * every PARAMETER is passed, not just the first, and each argv token is READ
%      when it reads as data (so `[1,2,3]` and `["a" "b"]` arrive as lists and
%      `12` as a number) and kept as a raw string otherwise -- the historical
%      entry's single `Integer/parseInt` can express neither a list argument nor
%      a string one. `read-string` rather than a JSON parser because it is
%      portable across both hosts and because Clojure reads a comma as
%      whitespace, so JSON array syntax parses unchanged. A token that reads as a
%      SYMBOL (`hello`, `--state`) is kept as the string it was: symbols are not
%      a representation this target produces, so reading one means the token was
%      plain text.
%    * the ANSWER may be a vector (G-A3-9's tuple) or a tagged map (G-A3-12), so
%      it is printed with pr-str rather than println, which flattens both.
%
%  Guarded on argument count, so loading the file as a library never fires it.
clj_a3_cli_entry(Pred, Arity, PredStr, CliEntry) :-
    clj_pred_outputs(Pred, Arity, Outs), !,
    clj_general_mode_or_test(Pred, Arity, Outs, Mode),
    clj_struct_inputs(Arity, Mode, InPositions),
    length(InPositions, ParamCount),
    (   ParamCount =:= 0
    ->  CliEntry = ""
    ;   format(string(CliEntry),
';; CLI entry point
(when (>= (count *command-line-args*) ~w)
  (let [read-arg (fn [s]
                   (let [v (try (read-string s) (catch Exception _e s))]
                     (if (symbol? v) s v)))
        args (mapv read-arg (take ~w *command-line-args*))]
    (println (pr-str (apply ~w args)))))
', [ParamCount, ParamCount, PredStr])
    ).
clj_a3_cli_entry(_, _, _, "").

clj_general_mode_or_test(Pred, Arity, Outs, Mode) :-
    functor(H, Pred, Arity),
    findall(H-B, user:clause(H, B), Cs),
    (   once(clj_struct_detect(Pred, Cs, Arity, _, M))
    ->  Mode = M
    ;   Outs == []  -> Mode = test
    ;   Outs = [P]  -> Mode = function(P)
    ;   Mode = function_multi(Outs)
    ).

% ===========================================================================
% THE RUNTIME BLOCK
% ===========================================================================
%
% Emitted only into modules that use it, so nothing that compiled before grows a
% prelude. Written in JVM-Clojure host form; clojurescript_target's interop pass
% rewrites the three host-specific lines for the JS host (see
% cljs_interop_rules/1), which is the same mechanism the rest of this target
% already uses for Integer/parseInt and Math/abs.

clj_runtime_begin(";; --- UnifyWeaver A3 runtime --- BEGIN").
clj_runtime_end(";; --- UnifyWeaver A3 runtime --- END").

clj_runtime_helper('uw-fail',
';; G-A3-18: a predicate that is SEMIDET and has output arguments answers with
;; its value (or G-A3-9\'s vector) or with this sentinel. A freshly allocated
;; host object has reference identity no Prolog term can produce and no data
;; crossing the module edge can forge, so `identical?` is an exact test.
(def ^:private uw-fail (Object.))').
clj_runtime_helper('uw-char-code',
';; The code of a one-character string. Prolog chars are one-character strings
;; in this target (see clj_sb_rule/5), on both hosts.
(defn ^:private uw-char-code [c] (int (.charAt (str c) 0)))').
clj_runtime_helper('uw-code-char',
'(defn ^:private uw-code-char [x] (str (char x)))').
clj_runtime_helper('uw-parse-num',
'(defn ^:private uw-parse-num [s] (Double/parseDouble (str s)))').

%% clj_runtime_block(+Code, -Block)
%  The helpers Code actually references, in a stable order. Fails when none are.
clj_runtime_block(Code, Block) :-
    % `once/1` around the search: sub_string/5 is NONDETERMINISTIC, so without it
    % findall/3 yields one copy of a helper per OCCURRENCE of its name and the
    % module opens with the sentinel defined thirty times over.
    findall(Src,
            ( clj_runtime_helper(Name, Src),
              once(sub_string(Code, _, _, _, Name)) ),
            Srcs),
    Srcs \== [],
    clj_runtime_begin(Begin), clj_runtime_end(End),
    atomic_list_concat(Srcs, '\n', Body),
    format(string(Block), '~w\n~w\n~w\n', [Begin, Body, End]).

%% clj_attach_runtime(+Code0, -Code)
clj_attach_runtime(Code0, Code) :-
    ( clj_runtime_block(Code0, Block) -> format(string(Code), '~w\n~w', [Block, Code0])
    ; Code = Code0 ).

% ===========================================================================
% IS THE HISTORICAL CLAUSE-BODY PATH'S ANSWER DEFECTIVE?
% ===========================================================================
%
% The new paths are a RESCUE, exactly as native_ts_general/3 is in the TS lane:
% they run only for a predicate whose existing lowering would be defective, so
% every predicate clojure_target already compiles keeps its output byte for byte.
% Defective means one of:
%
%   1. native_clojure_clause_body/3 FAILS or throws outright;
%   2. its output leaks an internal SWI variable name (`_41598`) — clojure_expr/3
%      falls back on term_string/2 for a variable the VarMap does not hold, which
%      puts an identifier nothing binds, different on every run, into the source;
%   3. its output carries a STRINGIFIED compound term — clojure_literal/2's last
%      clause wraps term_string/2 in quotes, so `ok([a],[b-c])` becomes the string
%      "ok([a],[b-c])" and the tag and payload are gone (G-A3-12);
%   4. its parentheses do not balance, so the form cannot be read at all.

%% clj_ground_fact_predicate(+Clauses) — a genuine fact table, which is
%% compile_facts_to_clojure/3's territory and which the A3 paths must not take
%% over.
clj_ground_fact_predicate(Clauses) :-
    forall(member(H-B, Clauses), ( B == true, ground(H) )).

%  ONE DELIBERATE DIVERGENCE FROM THE TS LANE. typescript_target's
%  ts_clause_body_defective/2 treats "some clause is a FACT" as defective,
%  because its clause-body path is guarded against fact clauses upstream and a
%  mixed fact+rule predicate has to reach the general path. Clojure's historical
%  path has no such guard and lowers a mixed predicate correctly -- `rfac(0,1).`
%  plus a rule becomes `(cond (= arg1 0) 1 (> arg1 0) ... )`. Porting the branch
%  verbatim reclassified every base-case-plus-recursive-clause predicate as
%  defective and moved four passing runtime-smoke tests onto the new path. The
%  branch is therefore NOT ported: a fact clause is not by itself a defect here.
clj_clause_body_defective(PredSpec, Clauses) :-
    (   \+ catch(once(native_clojure_clause_body(PredSpec, Clauses, _)), _, fail)
    ->  true
    ;   once(native_clojure_clause_body(PredSpec, Clauses, Code)),
        (   clj_leaks_var_name(Code)
        ;   clj_stringified_term(Code)
        ;   clj_prolog_builtin_call(Code)
        %  ARITY OVERLOADING. The historical path names a function by its raw
        %  Prolog functor, so `parse_args/2` and `parse_args/3` would BOTH be
        %  `(defn parse-args ...)` and the second would silently replace the
        %  first. Only the A3 path mangles the arity into the name, so an
        %  overloaded predicate has to be compiled there.
        ;   ( PredSpec = P0/A0, clj_name_overloaded(P0, A0) )
        %  TWO OR MORE OUTPUTS (G-A3-9). The historical path assumes head
        %  arguments 1..N-1 are parameters and N is the answer. For a predicate
        %  with two or more outputs -- `sum_len/5`, `lenient_loop/5`,
        %  `split_flag_token/3` -- that emits the OTHER outputs as required INPUT
        %  parameters and compares them against the accumulators, so the caller
        %  has to already know half the answer. It compiles, it runs, and it is
        %  wrong; this is the gap G-A3-9 exists for, so such a predicate must
        %  reach the A3 path whatever else is true of it.
        %
        %  Deliberately NOT extended to "the output set is not exactly [Arity]".
        %  An empty output set has two very different causes, and only one of them
        %  is a defect: a genuine semidet TEST (`starts_with/2`), which the other
        %  checks already catch, and a CONSTANT ANSWER (`positive(X, yes) :- X > 0.`)
        %  where the last head argument is ground in every clause. The historical
        %  path renders the constant answer correctly; forcing it onto the A3 path
        %  turned four such predicates into semidet tests that discard their own
        %  answer.
        ;   ( PredSpec = P1/A1,
              clj_pred_outputs(P1, A1, Outs1),
              length(Outs1, K1), K1 >= 2 )
        ;   clj_callee_convention_mismatch(PredSpec)
        ;   \+ clj_parens_balanced(Code)
        )
    ).

%% clj_callee_convention_mismatch(+Pred/Arity)
%
%  The old path renders a CROSS-PREDICATE call as `(<raw prolog functor> Ins)`
%  under one fixed convention: the callee's LAST argument is its only output, its
%  name is its Prolog functor spelled exactly, and it cannot fail. Where any of
%  those three is untrue, the emitted call is wrong -- and in a MODULE it is
%  wrong in a way that does not even load, because the A3 paths name their
%  functions in Clojure style: `merge_flags/3` compiled to
%  `(defn merge_flags [arg1 arg2] (merge_flags_ arg2 arg1))` while the module
%  declared and defined `merge-flags-`, so nbb refused the namespace with
%  "Unable to resolve symbol: merge_flags_".
%
%  So a predicate is defective when some callee's A3 calling convention differs
%  from the old path's assumption:
%
%    * the callee is ARITY-OVERLOADED, so its raw functor names two different
%      functions and only the A3 spelling tells them apart;
%    * the callee does not answer with exactly its LAST argument -- a semidet
%      test with no output at all (`starts_with/2`), or G-A3-9's several outputs
%      (`split_flag_token/3`);
%    * the callee is SEMIDET WITH OUTPUTS, so its answer may be the G-A3-18
%      sentinel and the call site owes it a test.
%
%  A callee whose clauses are not visible counts as a mismatch too: the A3 path
%  is then reached and refuses out loud, which is better than emitting a call to
%  a function nothing defines. Predicates with no cross-predicate callee at all --
%  every purely self-recursive or self-contained shape the historical path was
%  built for -- are untouched by this test.
%
%  A mere SPELLING difference (underscore vs hyphen) is NOT listed here: it is
%  handled at the call site instead, by clj_emitted_name/3, which asks which path
%  will compile the callee and uses that path's spelling. Treating it as a defect
%  would drag every underscore-named predicate onto the A3 path for no reason.
clj_callee_convention_mismatch(Pred/Arity) :-
    clj_pred_callees(Pred/Arity, Callees),
    member(Q/QA, Callees),
    clj_conv_differs(Q, QA),
    !.

%% clj_conv_differs(+Pred, +Arity)
%  True when the A3 calling convention for this predicate differs from the
%  historical one ("the last argument is the single output, and it always
%  succeeds").
clj_conv_differs(Q, QA) :-
    (   \+ clj_pred_outputs(Q, QA, _)        -> true
    ;   clj_name_overloaded(Q, QA)           -> true
    ;   \+ clj_pred_outputs(Q, QA, [QA])     -> true
    ;   clj_pred_can_fail(Q, QA)
    ).

%% clj_prolog_builtin_call(+Code)
%
%  The old path had NO rendering for SWI's deterministic text builtins (this is
%  G-A3-1, in its Clojure incarnation): clojure_output_rhs/3's last clause turns
%  any goal whose last argument is the output into a function application on its
%  functor, so `string_length(S, L)` became the Clojure call `(string_length s)`
%  -- a call to a function no namespace defines. It reads as perfectly good
%  Clojure and dies at run time, which is precisely the class of defect that
%  makes the A3 paths worth reaching for.
%
%  Detected by call HEAD, against an explicit list of the builtin functors this
%  target now has a real rendering for. Deliberately narrow: a user predicate
%  named after one of them is not a thing, and matching anything looser would
%  start reclassifying predicates the old path handles correctly.
clj_prolog_builtin_call(Code) :-
    atom_string(Code, S),
    clj_unrenderable_builtin(Name),
    (   format(string(Open), '(~w ', [Name]), sub_string(S, _, _, _, Open)
    ;   format(string(Close), '(~w)', [Name]), sub_string(S, _, _, _, Close)
    ),
    !.

clj_unrenderable_builtin(N) :- clj_sb_len_pred(N).
clj_unrenderable_builtin(N) :- clj_sb_concat_pred(N).
clj_unrenderable_builtin(N) :- clj_sb_sub_pred(N).
clj_unrenderable_builtin(N) :- clj_sb_chars_pred(N).
clj_unrenderable_builtin(N) :- clj_sb_codes_pred(N).
clj_unrenderable_builtin(N) :- clj_sb_numtext_pred(N).
clj_unrenderable_builtin(N) :- clj_sb_textid_pred(N).
clj_unrenderable_builtin(N) :- clj_sb_case_pred(N, _).
clj_unrenderable_builtin(char_code).
clj_unrenderable_builtin(length).
clj_unrenderable_builtin(append).
clj_unrenderable_builtin(nth0).
clj_unrenderable_builtin(nth1).
clj_unrenderable_builtin(msort).

%% clj_leaks_var_name(+Code) — an internal `_NNN` identifier reached the output.
clj_leaks_var_name(Code) :-
    atom_string(Code, S),
    sub_string(S, Before, 1, _, "_"),
    After is Before + 1,
    sub_string(S, After, 1, _, D),
    char_type(DC, digit(_)), atom_string(DC, D),
    ( Before =:= 0 -> true
    ; B1 is Before - 1, sub_string(S, B1, 1, _, P),
      \+ clj_name_char(P)
    ),
    !.

clj_name_char(C) :-
    atom_string(A, C),
    ( char_type(A, alnum) ; A == '-' ; A == '_' ; A == '?' ; A == '!' ).

%% clj_stringified_term(+Code) — a string literal that is really a Prolog term.
%  clojure_literal/2's fallback is term_string/2 in quotes, and the marks of that
%  are a `(` or a `[` INSIDE a double-quoted literal in emitted code that has no
%  business carrying one.
clj_stringified_term(Code) :-
    atom_string(Code, S),
    split_string(S, "\"", "", Parts),
    nth0(I, Parts, P), I mod 2 =:= 1,
    ( sub_string(P, _, _, _, "(") ; sub_string(P, _, _, _, "|") ),
    !.

clj_parens_balanced(Code) :-
    atom_string(Code, S),
    string_chars(S, Cs),
    clj_paren_walk(Cs, 0, out).

clj_paren_walk([], 0, out).
clj_paren_walk(['\\', _|Cs], D, in)  :- !, clj_paren_walk(Cs, D, in).
clj_paren_walk(['"'|Cs], D, out)     :- !, clj_paren_walk(Cs, D, in).
clj_paren_walk(['"'|Cs], D, in)      :- !, clj_paren_walk(Cs, D, out).
clj_paren_walk([_|Cs], D, in)        :- !, clj_paren_walk(Cs, D, in).
clj_paren_walk(['('|Cs], D, out)     :- !, D1 is D + 1, clj_paren_walk(Cs, D1, out).
clj_paren_walk([')'|Cs], D, out)     :- !, D > 0, D1 is D - 1, clj_paren_walk(Cs, D1, out).
clj_paren_walk([_|Cs], D, out)       :- clj_paren_walk(Cs, D, out).

% ===========================================================================
% MODULE DEPENDENCY CLOSURE (G-A3-6)
% ===========================================================================

clj_dep_closure(Predicates, Expanded) :-
    findall(Name/Arity, clj_pred_spec(Predicates, Name, Arity), Roots),
    clj_dep_walk(Roots, [], Ordered),
    findall(pred(N, A, T),
            ( member(N/A, Ordered),
              ( member(pred(N, A, T0), Predicates) -> T = T0 ; T = facts ) ),
            Expanded).

clj_pred_spec(Ps, N, A) :- member(pred(N, A, _), Ps).
clj_pred_spec(Ps, N, A) :- member(N/A, Ps).

clj_dep_walk([], _, []).
clj_dep_walk([P|Ps], Seen, Ordered) :-
    (   memberchk(P, Seen)
    ->  clj_dep_walk(Ps, Seen, Ordered)
    ;   clj_pred_callees(P, Callees),
        clj_dep_walk(Callees, [P|Seen], Sub),
        append([P|Seen], Sub, Seen1),
        clj_dep_walk(Ps, Seen1, Rest),
        append(Sub, [P|Rest], Ordered)
    ).

%% clj_pred_callees(+Pred/Arity, -Callees)
%  A ground-fact CONSTANT TABLE is inlined at its call site (G-A3-19), so it is
%  NOT a module member and must not be pulled into the closure.
clj_pred_callees(Pred/Arity, Callees) :-
    functor(Head, Pred, Arity),
    ( catch(findall(Body, user:clause(Head, Body), Bodies), _, fail) -> true ; Bodies = [] ),
    findall(Q/QA,
            ( member(Body, Bodies),
              clj_all_goals(Body, Goals),
              member(G0, Goals),
              clj_strip_mod(G0, G),
              compound(G),
              functor(G, Q, QA),
              \+ ( Q == Pred, QA =:= Arity ),
              \+ clj_control_functor(Q),
              \+ clj_known_builtin(Q, QA),
              \+ clj_fact_pred(Q, QA),
              functor(QH, Q, QA),
              catch(( user:clause(QH, _) -> true ; fail ), _, fail)
            ),
            Callees0),
    sort(Callees0, Callees).

%% ============================================
%% TESTS
%% ============================================

test_clojure_pipeline_mode :-
    format('~n=== Testing Clojure Pipeline Mode ===~n~n'),
    
    format('Test 1: Basic pipeline generation~n'),
    compile_predicate_to_clojure(test_pred/2, [pipeline_input(true)], Code1),
    (   sub_atom(Code1, _, _, _, 'run-pipeline')
    ->  format('  [PASS] Generated pipeline code~n')
    ;   format('  [FAIL] Missing pipeline code~n')
    ),
    
    format('~nTest 2: Clojure threading macros~n'),
    (   sub_atom(Code1, _, _, _, '->>')
    ->  format('  [PASS] Uses threading macro~n')
    ;   format('  [INFO] No threading macro in this code~n')
    ),
    
    format('~nTest 3: Lisp syntax~n'),
    (   sub_atom(Code1, _, _, _, '(defn')
    ->  format('  [PASS] Uses defn~n')
    ;   format('  [FAIL] Missing defn~n')
    ),
    
    format('~nTest 4: Generator mode~n'),
    compile_predicate_to_clojure(test_gen/2, [generator_mode(true)], Code2),
    (   sub_atom(Code2, _, _, _, 'lazy-seq')
    ->  format('  [PASS] Uses lazy-seq~n')
    ;   (   sub_atom(Code2, _, _, _, 'mapcat')
        ->  format('  [PASS] Uses mapcat~n')
        ;   format('  [FAIL] Missing lazy sequence~n')
        )
    ),
    
    format('~nTest 5: Deps.edn generation~n'),
    generate_deps_edn([main_ns('generated.pipeline')], DepsCode),
    (   sub_atom(DepsCode, _, _, _, ':deps')
    ->  format('  [PASS] Generated deps.edn~n')
    ;   format('  [FAIL] Invalid deps.edn~n')
    ),
    
    format('~n=== Clojure Pipeline Mode Tests Complete ===~n').

% ============================================================================
% MULTIFILE DISPATCH - Tail Recursion
% ============================================================================

:- use_module('../core/advanced/tail_recursion').
:- multifile tail_recursion:compile_tail_pattern/9.

tail_recursion:compile_tail_pattern(clojure, PredStr, Arity, _BaseClauses, _RecClauses, _AccPos, StepOp, _ExitAfterResult, Code) :-
    step_op_to_clojure(StepOp, CljStepExpr),
    (   Arity =:= 3 ->
        format(string(Code),
';; Generated by UnifyWeaver Clojure Target - Tail Recursion (multifile dispatch)
;; Predicate: ~w/~w

(defn ~w [items]
  (loop [remaining items
         acc 0]
    (if (empty? remaining)
      acc
      (let [item (first remaining)]
        (recur (rest remaining) (~w))))))

(when (seq *command-line-args*)
  (let [items (map #(Integer/parseInt %) (clojure.string/split (first *command-line-args*) #","))]
    (println (~w items))))
', [PredStr, Arity, PredStr, CljStepExpr, PredStr])
    ;   Arity =:= 2 ->
        format(string(Code),
';; Generated by UnifyWeaver Clojure Target - Tail Recursion (binary, multifile dispatch)
;; Predicate: ~w/~w

(defn ~w [items]
  (loop [remaining items
         count 0]
    (if (empty? remaining)
      count
      (recur (rest remaining) (inc count)))))

(when (seq *command-line-args*)
  (let [items (clojure.string/split (first *command-line-args*) #",")]
    (println (~w items))))
', [PredStr, Arity, PredStr, PredStr])
    ;   fail
    ).

step_op_to_clojure(arithmetic(Expr), CljExpr) :- tail_expr_to_clojure(Expr, CljExpr).
step_op_to_clojure(unknown, '+ acc 1').

tail_expr_to_clojure(_ + Const, CljExpr) :- integer(Const), !, format(atom(CljExpr), '+ acc ~w', [Const]).
tail_expr_to_clojure(_ + _, '+ acc item') :- !.
tail_expr_to_clojure(_ - _, '- acc item') :- !.
tail_expr_to_clojure(_ * _, '* acc item') :- !.
tail_expr_to_clojure(_, '+ acc 1').

% ============================================================================
% MULTIFILE DISPATCH - Linear Recursion
% ============================================================================

:- use_module('../core/advanced/linear_recursion').
:- multifile linear_recursion:compile_linear_pattern/8.

linear_recursion:compile_linear_pattern(clojure, PredStr, Arity, BaseClauses, _RecClauses, MemoEnabled, _MemoStrategy, Code) :-
    (   Arity =:= 2 ->
        linear_fold_clojure(PredStr, BaseClauses, MemoEnabled, Code)
    ;   linear_generic_clojure(PredStr, Arity, MemoEnabled, Code)
    ).

linear_fold_clojure(PredStr, BaseClauses, MemoEnabled, Code) :-
    linear_recursion:extract_base_case_info(BaseClauses, BaseInput, BaseOutput),
    linear_recursion:detect_input_type(BaseInput, InputType),
    (   MemoEnabled = true ->
        format(string(MemoDecl), '(def ~w-memo (atom {}))', [PredStr])
    ;   MemoDecl = ";; Memoization disabled"
    ),
    % Extract fold
    atom_string(Pred, PredStr),
    functor(Head, Pred, 2),
    findall(clause(Head, Body), user:clause(Head, Body), AllClauses),
    partition(linear_recursion:is_recursive_clause(Pred), AllClauses, ActualRec, _),
    (   ActualRec = [clause(RHead, RBody)|_] ->
        RHead =.. [_, InputVar, _],
        linear_recursion:find_recursive_call(RBody, RecCall),
        RecCall =.. [_, _, AccVar],
        linear_recursion:find_last_is_expression(RBody, _ is FoldExpr),
        translate_fold_expr_clojure(FoldExpr, InputVar, AccVar, CljOp)
    ;   CljOp = "* current acc"
    ),
    (   InputType = numeric ->
        (   MemoEnabled = true ->
            format(string(MemoCheck_N), '  (if-let [cached (get @~w-memo n)]~n    cached', [PredStr]),
            format(string(MemoStore_N), '      (swap! ~w-memo assoc n result)~n      result', [PredStr])
        ;   MemoCheck_N = "  (let [result", MemoStore_N = "      result"
        ),
        format(string(Code),
';; Generated by UnifyWeaver Clojure Target - Linear Recursion (numeric, multifile dispatch)
;; Predicate: ~w/2

~w

(defn ~w [n]
~w
    (let [result (if (= n ~w) ~w
                   (reduce (fn [acc current] (~w))
                           ~w (range n 0 -1)))]
~w)))

(when (seq *command-line-args*)
  (println (~w (Integer/parseInt (first *command-line-args*)))))
', [PredStr, MemoDecl, PredStr, MemoCheck_N, BaseInput, BaseOutput, CljOp, BaseOutput, MemoStore_N, PredStr])
    ;   InputType = list ->
        (   MemoEnabled = true ->
            format(string(MemoCheck_L), '  (if-let [cached (get @~w-memo lst)]~n    cached', [PredStr]),
            format(string(MemoStore_L), '      (swap! ~w-memo assoc lst result)~n      result', [PredStr])
        ;   MemoCheck_L = "  (let [result", MemoStore_L = "      result"
        ),
        format(string(Code),
';; Generated by UnifyWeaver Clojure Target - Linear Recursion (list, multifile dispatch)
;; Predicate: ~w/2

~w

(defn ~w [lst]
~w
    (let [result (if (empty? lst) ~w
                   (reduce (fn [acc current] (~w))
                           ~w lst))]
~w)))

(when (seq *command-line-args*)
  (let [items (map #(Integer/parseInt %) (clojure.string/split (first *command-line-args*) #","))]
    (println (~w items))))
', [PredStr, MemoDecl, PredStr, MemoCheck_L, BaseOutput, CljOp, BaseOutput, MemoStore_L, PredStr])
    ;   linear_generic_clojure(PredStr, 2, MemoEnabled, Code)
    ).

linear_generic_clojure(PredStr, Arity, MemoEnabled, Code) :-
    (   MemoEnabled = true ->
        format(string(Code),
';; Generated by UnifyWeaver Clojure Target - Linear Recursion (generic, multifile dispatch)
;; Predicate: ~w/~w

(def ~w-memo (atom {}))

(defn ~w [n]
  (if-let [cached (get @~w-memo n)]
    cached
    (let [result (cond
                   (<= n 0) 0
                   (= n 1) 1
                   :else (+ (~w (dec n)) n))]
      (swap! ~w-memo assoc n result)
      result)))

(when (seq *command-line-args*)
  (println (~w (Integer/parseInt (first *command-line-args*)))))
', [PredStr, Arity, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr])
    ;   format(string(Code),
';; Generated by UnifyWeaver Clojure Target - Linear Recursion (generic, multifile dispatch)
;; Predicate: ~w/~w

(defn ~w [n]
  (cond
    (<= n 0) 0
    (= n 1) 1
    :else (+ (~w (dec n)) n)))

(when (seq *command-line-args*)
  (println (~w (Integer/parseInt (first *command-line-args*)))))
', [PredStr, Arity, PredStr, PredStr, PredStr])
    ).

translate_fold_expr_clojure(A * B, IV, AV, E) :- translate_clj_term(A, IV, AV, AT), translate_clj_term(B, IV, AV, BT), format(string(E), '* ~w ~w', [AT, BT]).
translate_fold_expr_clojure(A + B, IV, AV, E) :- translate_clj_term(A, IV, AV, AT), translate_clj_term(B, IV, AV, BT), format(string(E), '+ ~w ~w', [AT, BT]).
translate_fold_expr_clojure(A - B, IV, AV, E) :- translate_clj_term(A, IV, AV, AT), translate_clj_term(B, IV, AV, BT), format(string(E), '- ~w ~w', [AT, BT]).
translate_fold_expr_clojure(T, IV, AV, E) :- translate_clj_term(T, IV, AV, E).

translate_clj_term(T, IV, _, 'current') :- T == IV, !.
translate_clj_term(T, _, AV, 'acc') :- T == AV, !.
translate_clj_term(N, _, _, S) :- integer(N), !, format(string(S), '~w', [N]).
translate_clj_term(T, _, _, 'current') :- var(T), !.
translate_clj_term(A, _, _, S) :- format(string(S), '~w', [A]).

% ============================================================================
% HELPER: Convert Prolog underscore names to Clojure hyphenated names
% ============================================================================

pred_to_clojure_name(Pred, ClojureName) :-
    atom_string(Pred, PredStr),
    atomic_list_concat(Parts, '_', PredStr),
    atomic_list_concat(Parts, '-', ClojureName).

% ============================================================================
% MULTIFILE DISPATCH - Tree Recursion
% ============================================================================

:- use_module('../core/advanced/tree_recursion').
:- multifile tree_recursion:compile_tree_pattern/6.

tree_recursion:compile_tree_pattern(clojure, _Pattern, Pred, _Arity, _UseMemo, CljCode) :-
    atom_string(Pred, PredStr),
    format(string(CljCode),
';; Generated by UnifyWeaver Clojure Target - Tree Recursion (multifile dispatch)
;; Predicate: ~w

(def memo (atom {}))

(defn ~w [n]
  (if-let [cached (@memo n)]
    cached
    (let [result (cond
                   (<= n 0) 0
                   (= n 1) 1
                   :else (+ (~w (- n 1)) (~w (- n 2))))]
      (swap! memo assoc n result)
      result)))

(when *command-line-args*
  (println (~w (Integer/parseInt (first *command-line-args*)))))
', [PredStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% MULTIFILE DISPATCH - Multicall Linear Recursion
% ============================================================================

:- use_module('../core/advanced/multicall_linear_recursion').
:- multifile multicall_linear_recursion:compile_multicall_pattern/6.

multicall_linear_recursion:compile_multicall_pattern(clojure, PredStr, BaseClauses, _RecClauses, _MemoEnabled, CljCode) :-
    findall(BaseCaseLine, (
        member(clause(BHead, _), BaseClauses),
        BHead =.. [_P, BInput, BOutput],
        format(string(BaseCaseLine), '                   (= n ~w) ~w', [BInput, BOutput])
    ), BaseCaseLines0),
    sort(BaseCaseLines0, BaseCaseLines),
    atomic_list_concat(BaseCaseLines, '\n', BaseCaseStr),
    format(string(CljCode),
';; Generated by UnifyWeaver Clojure Target - Multicall Linear Recursion (multifile dispatch)
;; Predicate: ~w

(def memo (atom {}))

(defn ~w [n]
  (if-let [cached (@memo n)]
    cached
    (let [result (cond
~w
                   :else (+ (~w (- n 1)) (~w (- n 2))))]
      (swap! memo assoc n result)
      result)))

(when *command-line-args*
  (println (~w (Integer/parseInt (first *command-line-args*)))))
', [PredStr, PredStr, BaseCaseStr, PredStr, PredStr, PredStr]).

% ============================================================================
% MULTIFILE DISPATCH - Direct Multi-Call Recursion
% ============================================================================

:- use_module('../core/advanced/direct_multi_call_recursion').
:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.

direct_multi_call_recursion:compile_direct_multicall_pattern(clojure, PredStr, BaseClauses, _RecClause, CljCode) :-
    findall(BaseCaseLine, (
        member(clause(BHead, _), BaseClauses),
        BHead =.. [_P, BInput, BOutput],
        format(string(BaseCaseLine), '                   (= n ~w) ~w', [BInput, BOutput])
    ), BaseCaseLines0),
    sort(BaseCaseLines0, BaseCaseLines),
    atomic_list_concat(BaseCaseLines, '\n', BaseCaseStr),
    format(string(CljCode),
';; Generated by UnifyWeaver Clojure Target - Direct Multicall Recursion (multifile dispatch)
;; Predicate: ~w

(def memo (atom {}))

(defn ~w [n]
  (if-let [cached (@memo n)]
    cached
    (let [result (cond
~w
                   :else (+ (~w (- n 1)) (~w (- n 2))))]
      (swap! memo assoc n result)
      result)))

(when *command-line-args*
  (println (~w (Integer/parseInt (first *command-line-args*)))))
', [PredStr, PredStr, BaseCaseStr, PredStr, PredStr, PredStr]).

% ============================================================================
% MULTIFILE DISPATCH - Mutual Recursion
% ============================================================================

:- use_module('../core/advanced/mutual_recursion').
:- use_module('../core/advanced/pattern_matchers', [is_per_path_visited_pattern/4]).
:- multifile mutual_recursion:compile_mutual_pattern/5.

mutual_recursion:compile_mutual_pattern(clojure, Predicates, _MemoEnabled, _MemoStrategy, CljCode) :-
    % Generate forward declarations
    mutual_forward_decls_clj(Predicates, DeclCode),
    % Generate function definitions for each predicate
    mutual_functions_clj(Predicates, Predicates, FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),
    % Generate dispatch / main
    mutual_dispatch_clj(Predicates, DispatchCode),
    format(string(CljCode),
';; Generated by UnifyWeaver Clojure Target - Mutual Recursion (multifile dispatch)

~w

~w

~w
', [DeclCode, FunctionsCode, DispatchCode]).

%% mutual_forward_decls_clj(+Predicates, -DeclCode)
%  Generates (declare is-even is-odd) style forward declarations.
mutual_forward_decls_clj(Predicates, DeclCode) :-
    findall(CljName, (
        member(Pred/_Arity, Predicates),
        pred_to_clojure_name(Pred, CljName)
    ), CljNames),
    atomic_list_concat(CljNames, ' ', NamesStr),
    format(string(DeclCode), '(declare ~w)', [NamesStr]).

%% mutual_functions_clj(+Predicates, +AllPreds, -FuncCodes)
%  Generates each function definition with base and recursive cases.
mutual_functions_clj([], _AllPreds, []).
mutual_functions_clj([Pred/Arity|Rest], AllPreds, [FuncCode|RestCodes]) :-
    pred_to_clojure_name(Pred, CljName),
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(mutual_recursion:is_mutual_recursive_clause(AllPreds), Clauses, RecClauses, BaseClauses),
    % Build base cases as cond branches
    % For arity-1 predicates (boolean), base case success means true
    findall(BaseLine, (
        member(clause(BHead, true), BaseClauses),
        BHead =.. [_P|BArgs],
        (   BArgs = [BValue] ->
            % Arity 1: predicate succeeds = true
            format(string(BaseLine), '    (= n ~w) true', [BValue])
        ;   BArgs = [BInput, BOutput] ->
            format(string(BaseLine), '    (= n ~w) ~w', [BInput, BOutput])
        ;   BArgs = [BValue|_] ->
            format(string(BaseLine), '    (= n ~w) true', [BValue])
        )
    ), BaseLines),
    atomic_list_concat(BaseLines, '\n', BaseStr),
    % Build recursive case with guard extraction
    (   RecClauses = [clause(_RHead, RBody)|_] ->
        extract_mutual_rec_info_clj(RBody, AllPreds, RecCallExpr),
        extract_guard_clj(RBody, GuardExpr),
        extract_step_clj(RBody, AllPreds, StepExpr),
        (   GuardExpr \= none ->
            format(string(RecLine), '    ~w ~w', [GuardExpr, StepExpr])
        ;   format(string(RecLine), '    :else ~w', [RecCallExpr])
        )
    ;   RecLine = '    :else false'
    ),
    format(string(FuncCode),
'(defn ~w [n]
  (cond
~w
~w
    :else false))', [CljName, BaseStr, RecLine]),
    mutual_functions_clj(Rest, AllPreds, RestCodes).

%% extract_mutual_rec_info_clj(+Body, +AllPreds, -RecExpr)
%  Extracts the recursive call expression from a clause body.
%  Searches through conjunctions to find the mutual recursive call.
extract_mutual_rec_info_clj(Body, AllPreds, RecExpr) :-
    find_mutual_call_clj(Body, AllPreds, CalledPred, Args),
    !,
    pred_to_clojure_name(CalledPred, CalledCljName),
    (   Args = [ArgExpr|_] ->
        mutual_arg_to_clj(ArgExpr, CljArg),
        format(string(RecExpr), '(~w ~w)', [CalledCljName, CljArg])
    ;   format(string(RecExpr), '(~w)', [CalledCljName])
    ).
extract_mutual_rec_info_clj(_, _, "nil").

%% find_mutual_call_clj(+Body, +AllPreds, -CalledPred, -Args)
%  Finds the mutual recursive call within a clause body.
find_mutual_call_clj((A, B), AllPreds, CalledPred, Args) :-
    !,
    (   find_mutual_call_clj(A, AllPreds, CalledPred, Args)
    ;   find_mutual_call_clj(B, AllPreds, CalledPred, Args)
    ).
find_mutual_call_clj(Goal, AllPreds, CalledPred, Args) :-
    Goal =.. [CalledPred|Args],
    member(CalledPred/_A, AllPreds).

%% extract_guard_clj(+Body, -GuardExpr)
%  Extracts guard condition (e.g., N > 0) from clause body as Clojure expression.
extract_guard_clj((Goal, _Rest), GuardExpr) :-
    Goal =.. [Op, _, Val],
    memberchk(Op, [>, <, >=, =<]),
    number(Val),
    !,
    clj_comp_op(Op, CljOp),
    format(string(GuardExpr), '(~w n ~w)', [CljOp, Val]).
extract_guard_clj(_, none).

clj_comp_op(>, '>').
clj_comp_op(<, '<').
clj_comp_op(>=, '>=').
clj_comp_op(=<, '<=').

%% extract_step_clj(+Body, +AllPreds, -StepExpr)
%  Extracts the full recursive call with computed argument as Clojure expression.
extract_step_clj(Body, AllPreds, StepExpr) :-
    find_mutual_call_clj(Body, AllPreds, CalledPred, [ArgVar|_]),
    pred_to_clojure_name(CalledPred, CalledCljName),
    % Find the 'is' expression that computes the argument
    find_is_expr_for_var_clj(Body, ArgVar, ComputedExpr),
    !,
    mutual_arg_to_clj(ComputedExpr, CljArg),
    format(string(StepExpr), '(~w ~w)', [CalledCljName, CljArg]).
extract_step_clj(Body, AllPreds, StepExpr) :-
    extract_mutual_rec_info_clj(Body, AllPreds, StepExpr).

%% find_is_expr_for_var_clj(+Body, +Var, -Expr)
%  Finds 'Var is Expr' in the body.
find_is_expr_for_var_clj((A, B), Var, Expr) :-
    !,
    (   find_is_expr_for_var_clj(A, Var, Expr)
    ;   find_is_expr_for_var_clj(B, Var, Expr)
    ).
find_is_expr_for_var_clj(V is Expr, Var, Expr) :-
    V == Var.

%% mutual_arg_to_clj(+Expr, -CljStr)
mutual_arg_to_clj(Expr, CljStr) :-
    (   number(Expr) ->
        format(string(CljStr), '~w', [Expr])
    ;   Expr = _ - Y, number(Y) ->
        format(string(CljStr), '(- n ~w)', [Y])
    ;   Expr = _ + Y, number(Y), Y < 0 ->
        AbsY is abs(Y),
        format(string(CljStr), '(- n ~w)', [AbsY])
    ;   Expr = _ + Y, number(Y) ->
        format(string(CljStr), '(+ n ~w)', [Y])
    ;   CljStr = "n"
    ).

%% mutual_dispatch_clj(+Predicates, -DispatchCode)
%  Generates main entry point that calls the first predicate.
mutual_dispatch_clj(Predicates, DispatchCode) :-
    (   Predicates = [FirstPred/_Arity|_] ->
        pred_to_clojure_name(FirstPred, FirstCljName),
        format(string(DispatchCode),
'(when *command-line-args*
  (println (~w (Integer/parseInt (first *command-line-args*)))))', [FirstCljName])
    ;   DispatchCode = ";; No predicates to dispatch"
    ).

%% ============================================
%% GENERAL RECURSIVE PATTERN (visited-set cycle safety)
%% ============================================

:- multifile advanced_recursive_compiler:compile_general_recursive_pattern/6.

%% No-visited-pattern — plain recursive without cycle detection
advanced_recursive_compiler:compile_general_recursive_pattern(clojure, PredStr, Arity, BaseClauses, RecClauses, Code) :-
    atom_string(Pred, PredStr),
    append(BaseClauses, RecClauses, AllClauses),
    \+ is_per_path_visited_pattern(Pred, Arity, AllClauses, _),
    !,
    (   BaseClauses = [(BaseHead, _)|_],
        BaseHead =.. [_|BaseArgs], last(BaseArgs, BaseResult)
    ->  term_to_atom(BaseResult, BaseValAtom), atom_string(BaseValAtom, BaseValStr),
        BaseArgs = [BaseInput|_], term_to_atom(BaseInput, BaseInAtom), atom_string(BaseInAtom, BaseInStr)
    ;   BaseValStr = "[]", BaseInStr = "0"
    ),
    format(string(Code),
';; General recursive: ~w (plain, no visited pattern)\n\c
\n\c
(defn ~w [arg1]\n\c
  (if (= arg1 ~w) [~w]\n\c
    (~w (str arg1))))\n',
    [PredStr, PredStr, BaseInStr, BaseValStr, PredStr]).

advanced_recursive_compiler:compile_general_recursive_pattern(clojure, PredStr, _Arity, BaseClauses, _RecClauses, Code) :-
    %% Extract base value from first base clause
    (   BaseClauses = [(BaseHead, _BaseBody)|_],
        BaseHead =.. [_|BaseArgs],
        last(BaseArgs, BaseResult)
    ->  term_to_atom(BaseResult, BaseValAtom),
        atom_string(BaseValAtom, BaseValStr)
    ;   BaseValStr = "[]"
    ),
    %% Extract base input from first base clause
    (   BaseClauses = [(BaseHead2, _)|_],
        BaseHead2 =.. [_|BaseArgs2],
        BaseArgs2 = [BaseInput|_]
    ->  term_to_atom(BaseInput, BaseInAtom),
        atom_string(BaseInAtom, BaseInStr)
    ;   BaseInStr = "0"
    ),
    string_concat(PredStr, "-worker", WorkerStr),
    format(string(Code),
';; Generated by UnifyWeaver Clojure Target - General Recursion with Visited Set

(defn ~w [arg1] (~w arg1 #{}))

(defn- ~w [arg1 visited]
  (if (contains? visited arg1) []
    (if (= arg1 ~w) [~w]
      (let [visited (conj visited arg1)]
        (~w (str arg1) visited)))))',
        [PredStr, WorkerStr,
         WorkerStr,
         BaseInStr, BaseValStr,
         WorkerStr]).
