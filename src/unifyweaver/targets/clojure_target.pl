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
compile_module(Predicates, Options, Code) :-
    option(namespace(Namespace), Options, 'generated.module'),
    normalize_module_preds(Predicates, PredList),
    findall(DefnCode, (
        member(P, PredList),
        clojure_predicate_defn(P, Options, DefnCode)
    ), DefnCodes),
    atomic_list_concat(DefnCodes, '\n\n', PredsSection),
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

~w
', [Namespace, Namespace, Body]).

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
