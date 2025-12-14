:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% recursive_compiler.pl - UnifyWeave compiler with recursion support
% Extends stream_compiler to handle recursive predicates
% Uses robust templating system for code generation

:- module(recursive_compiler, [
    compile_recursive/3,
    compile_recursive/2,
    test_recursive_compiler/0
]).

:- use_module('advanced/advanced_recursive_compiler').
:- use_module('advanced/call_graph').
:- use_module(stream_compiler).
:- use_module('../targets/awk_target').
:- use_module('../targets/csharp_query_target').
:- use_module('../targets/csharp_stream_target').
:- use_module('../targets/go_target', [compile_predicate_to_go/3]).
:- use_module('../targets/rust_target', [compile_predicate_to_rust/3]).
:- use_module('../targets/java_target', [compile_predicate_to_java/3, write_java_program/2]).
:- use_module('../targets/kotlin_target', [compile_predicate_to_kotlin/3]).
:- use_module('../targets/scala_target', [compile_predicate_to_scala/3]).
:- use_module('../targets/clojure_target', [compile_predicate_to_clojure/3]).
:- use_module('../targets/jython_target', [compile_predicate_to_jython/3]).
:- use_module(template_system).
:- use_module(library(lists)).
:- use_module(constraint_analyzer).
:- use_module(firewall).
:- use_module(preferences).

%% Main entry point - analyze and compile
compile_recursive(Pred/Arity, Options) :-
    compile_recursive(Pred/Arity, Options, _).

compile_recursive(Pred/Arity, RuntimeOptions, GeneratedCode) :-
    % 1. Get constraints and merge with runtime options
    get_constraints(Pred/Arity, DeclaredConstraints),
    preferences:get_final_options(Pred/Arity, RuntimeOptions, FinalOptions),
    merge_options(FinalOptions, DeclaredConstraints, MergedOptions),

    % 2. Get the firewall policy for the predicate.
    firewall:get_firewall_policy(Pred/Arity, Firewall),

    % 3. Determine target backend (default: bash)
    (   member(target(Target), MergedOptions) ->
        true
    ;   Target = bash  % Default target
    ),

    % 4. Validate the request against the firewall.
    (   firewall:enforce_firewall(Pred/Arity, Target, MergedOptions, Firewall)
    ->  % Validation passed, proceed with compilation.
        format('--- Firewall validation passed for ~w. Proceeding with compilation. ---\n', [Pred/Arity]),
        compile_dispatch(Pred/Arity, MergedOptions, Target, GeneratedCode)
    ;   % Validation failed, stop.
        format(user_error, 'Compilation of ~w halted due to firewall policy violation.~n', [Pred/Arity]),
        !, fail
    ).

%% merge_options(RuntimeOpts, Constraints, Merged)
%  Merges runtime options with declared constraints.
merge_options(RuntimeOpts, Constraints, Merged) :-
    (   member(unique(U), RuntimeOpts) -> MergedUnique = [unique(U)] ; MergedUnique = [] ),
    (   member(unordered(O), RuntimeOpts) -> MergedUnordered = [unordered(O)] ; MergedUnordered = [] ),
    append(MergedUnique, MergedUnordered, RuntimeConstraints),
    append(RuntimeConstraints, Constraints, AllConstraints0),
    findall(Opt,
        (   member(Opt, RuntimeOpts),
            functor(Opt, FunctorName, _),
            FunctorName \= unique,
            FunctorName \= unordered
        ),
        OtherRuntimeOpts),
    append(AllConstraints0, OtherRuntimeOpts, AllConstraints),
    list_to_set(AllConstraints, Merged).

%% compile_dispatch(+Pred/Arity, +FinalOptions, +Target, -GeneratedCode)
%  Target-aware compilation logic, now called after validation.
compile_dispatch(Pred/Arity, FinalOptions, Target, GeneratedCode) :-
    format('=== Analyzing ~w/~w ===~n', [Pred, Arity]),
    classify_predicate(Pred/Arity, Classification),
    format('Classification: ~w~n', [Classification]),

    (   Classification = non_recursive ->
        compile_non_recursive(Target, Pred/Arity, FinalOptions, GeneratedCode)
    ;   Target == csharp ->
        compile_recursive_csharp_query(Pred/Arity, FinalOptions, GeneratedCode)
    ;   Classification = transitive_closure(BasePred) ->
        format('Detected transitive closure over ~w~n', [BasePred]),
        compile_transitive_closure(Target, Pred, Arity, BasePred, FinalOptions, GeneratedCode)
    ;   Classification = mutual_recursion -> % Handle mutual recursion
        % Get the full mutual recursion group (not just one predicate)
        call_graph:predicates_in_group(Pred/Arity, Group),
        advanced_recursive_compiler:compile_mutual_recursion(Group, FinalOptions, GeneratedCode)
    ;   catch(
            compile_advanced(Target, Pred/Arity, FinalOptions, GeneratedCode),
            error(existence_error(procedure, _), _),
            fail
        ) ->
        format('Compiled using advanced patterns with options: ~w~n', [FinalOptions])
    ;   format('Unknown recursion pattern - using memoization~n', []),
        compile_memoized_recursion(Target, Pred, Arity, FinalOptions, GeneratedCode)
    ).

compile_non_recursive(bash, Pred/Arity, FinalOptions, GeneratedCode) :-
    stream_compiler:compile_predicate(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(awk, Pred/Arity, FinalOptions, GeneratedCode) :-
    awk_target:compile_predicate_to_awk(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(csharp, Pred/Arity, FinalOptions, GeneratedCode) :-
    csharp_stream_target:compile_predicate_to_csharp(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(go, Pred/Arity, FinalOptions, GeneratedCode) :-
    compile_predicate_to_go(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(rust, Pred/Arity, FinalOptions, GeneratedCode) :-
    compile_predicate_to_rust(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(java, Pred/Arity, FinalOptions, GeneratedCode) :-
    java_target:compile_predicate_to_java(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(kotlin, Pred/Arity, FinalOptions, GeneratedCode) :-
    kotlin_target:compile_predicate_to_kotlin(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(scala, Pred/Arity, FinalOptions, GeneratedCode) :-
    scala_target:compile_predicate_to_scala(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(clojure, Pred/Arity, FinalOptions, GeneratedCode) :-
    clojure_target:compile_predicate_to_clojure(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(jython, Pred/Arity, FinalOptions, GeneratedCode) :-
    jython_target:compile_predicate_to_jython(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(Target, Pred/Arity, _Options, _GeneratedCode) :-
    format(user_error, 'Target ~w not supported for non-recursive predicate ~w.~n', [Target, Pred/Arity]),
    fail.

compile_advanced(bash, Pred/Arity, FinalOptions, GeneratedCode) :-
    advanced_recursive_compiler:compile_advanced_recursive(
        Pred/Arity, FinalOptions, GeneratedCode
    ).
compile_advanced(java, Pred/Arity, FinalOptions, GeneratedCode) :-
    java_target:compile_predicate_to_java(Pred/Arity, [generator_mode(true)|FinalOptions], GeneratedCode).
compile_advanced(kotlin, Pred/Arity, FinalOptions, GeneratedCode) :-
    kotlin_target:compile_predicate_to_kotlin(Pred/Arity, [generator_mode(true)|FinalOptions], GeneratedCode).
compile_advanced(scala, Pred/Arity, FinalOptions, GeneratedCode) :-
    scala_target:compile_predicate_to_scala(Pred/Arity, [generator_mode(true)|FinalOptions], GeneratedCode).
compile_advanced(clojure, Pred/Arity, FinalOptions, GeneratedCode) :-
    clojure_target:compile_predicate_to_clojure(Pred/Arity, [generator_mode(true)|FinalOptions], GeneratedCode).
compile_advanced(jython, Pred/Arity, FinalOptions, GeneratedCode) :-
    jython_target:compile_predicate_to_jython(Pred/Arity, [generator_mode(true)|FinalOptions], GeneratedCode).
compile_advanced(go, Pred/Arity, _FinalOptions, _GeneratedCode) :-
    format(user_error, 'Advanced recursive compilation for target go not yet implemented (~w).~n',
           [Pred/Arity]),
    fail.
compile_advanced(rust, Pred/Arity, _FinalOptions, _GeneratedCode) :-
    format(user_error, 'Advanced recursive compilation for target rust not yet implemented (~w).~n',
           [Pred/Arity]),
    fail.
compile_advanced(Target, Pred/Arity, _FinalOptions, _GeneratedCode) :-
    format(user_error, 'Advanced recursive compilation for target ~w not implemented (~w).~n',
           [Target, Pred/Arity]),
    fail.

compile_recursive_csharp_query(Pred/Arity, Options, Code) :-
    prepare_csharp_query_options(Options, QueryOptions),
    (   csharp_query_target:build_query_plan(Pred/Arity, QueryOptions, Plan)
    ->  csharp_query_target:render_plan_to_csharp(Plan, Code)
    ;   format(user_error, 'C# query target failed to build plan for ~w.~n', [Pred/Arity]),
        fail
    ).

prepare_csharp_query_options(Options, QueryOptions) :-
    exclude(is_target_option, Options, Rest),
    QueryOptions = [target(csharp_query)|Rest].

is_target_option(target(_)).

%% Classify predicate recursion pattern
classify_predicate(Pred/Arity, Classification) :-
    functor(Head, Pred, Arity),
    findall(Body, clause(Head, Body), Bodies),

    % Check for mutual recursion FIRST (before self-recursion check)
    (   call_graph:predicates_in_group(Pred/Arity, Group),
        length(Group, GroupSize),
        GroupSize > 1 ->
        Classification = mutual_recursion
    ;   % Check if self-recursive
        contains_recursive_call(Pred, Bodies) ->
        analyze_recursion_pattern(Pred, Arity, Bodies, Classification)
    ;   Classification = non_recursive
    ).

%% Check if any body contains a recursive call
contains_recursive_call(Pred, Bodies) :-
    member(Body, Bodies),
    contains_goal(Body, Goal),
    functor(Goal, Pred, _).

%% Check if a goal appears in a body
contains_goal(Goal, Goal) :- 
    compound(Goal),
    \+ Goal = (_,_).
contains_goal((A, _), Goal) :- 
    contains_goal(A, Goal).
contains_goal((_, B), Goal) :- 
    contains_goal(B, Goal).
contains_goal((A; _), Goal) :- 
    contains_goal(A, Goal).
contains_goal((_;B), Goal) :- 
    contains_goal(B, Goal).

%% Analyze recursion pattern
analyze_recursion_pattern(Pred, Arity, Bodies, Pattern) :-
    % Separate base cases from recursive cases
    partition(is_recursive_clause(Pred), Bodies, RecClauses, BaseClauses),
    
    % Check for transitive closure pattern
    (   is_transitive_closure(Pred, Arity, BaseClauses, RecClauses, BasePred) ->
        Pattern = transitive_closure(BasePred)
    ;   is_tail_recursive(Pred, RecClauses) ->
        Pattern = tail_recursion
    ;   is_linear_recursive(Pred, RecClauses) ->
        Pattern = linear_recursion
    ;   Pattern = unknown_recursion
    ).

is_recursive_clause(Pred, Body) :-
    contains_goal(Body, Goal),
    functor(Goal, Pred, _).

%% Check for transitive closure pattern
% Two patterns supported:
% 1. Forward: pred(X,Z) :- base(X,Y), pred(Y,Z).  [e.g., ancestor]
% 2. Reverse: pred(X,Z) :- base(Y,X), pred(Y,Z).  [e.g., descendant]
is_transitive_closure(Pred, 2, BaseClauses, RecClauses, BasePred) :-
    % Check base case is a single predicate call
    member(BaseBody, BaseClauses),
    BaseBody \= true,
    functor(BaseBody, BasePred, 2),
    BasePred \= Pred,

    % Check recursive case matches pattern
    member(RecBody, RecClauses),
    RecBody = (BaseCall, RecCall),
    functor(BaseCall, BasePred, 2),
    functor(RecCall, Pred, 2),

    % Try both forward and reverse patterns
    (   % Pattern 1: Forward transitive closure
        % base(X,Y), recursive(Y,Z) - Y flows from base to recursive
        BaseCall =.. [BasePred, _X, Y],
        RecCall =.. [Pred, Y2, _Z],
        Y == Y2
    ;   % Pattern 2: Reverse transitive closure
        % base(Y,X), recursive(Y,Z) - Y flows from base to recursive (reversed args)
        BaseCall =.. [BasePred, Y, _X],
        RecCall =.. [Pred, Y2, _Z],
        Y == Y2
    ).

is_transitive_closure(_, _, _, _, _) :- fail.

%% Check for tail recursion
is_tail_recursive(Pred, RecClauses) :-
    member(Body, RecClauses),
    last_goal(Body, Goal),
    functor(Goal, Pred, _).

last_goal(Goal, Goal) :- 
    compound(Goal),
    \+ Goal = (_,_).
last_goal((_, B), Goal) :- 
    last_goal(B, Goal).

%% Check for linear recursion
is_linear_recursive(Pred, RecClauses) :-
    member(Body, RecClauses),
    findall(G, contains_goal(Body, G), Goals),
    findall(G, (member(G, Goals), functor(G, Pred, _)), RecGoals),
    length(RecGoals, 1).  % Exactly one recursive call

%% Compile transitive closure pattern
compile_transitive_closure(bash, Pred, _Arity, BasePred, Options, GeneratedCode) :-
    % Use the template_system's new constraint-aware predicate
    template_system:generate_transitive_closure(Pred, BasePred, Options, GeneratedCode),
    !.

%% Java transitive closure - generates BFS iterative implementation
compile_transitive_closure(java, Pred, _Arity, BasePred, Options, GeneratedCode) :-
    atom_string(Pred, PredStr),
    atom_string(BasePred, BaseStr),
    upcase_atom(Pred, PredCap),
    upcase_atom(BasePred, BaseCap),
    option(package(Package), Options, generated),
    
    format(string(GeneratedCode),
'// Generated by UnifyWeaver Java Target - Transitive Closure
// Predicate: ~w/2 (transitive closure of ~w)

package ~w;

import java.io.*;
import java.util.*;

/**
 * ~w - transitive closure of ~w
 * Implements BFS-based iterative fixpoint computation
 */
public class ~w {

    // Base facts: ~w(from, to)
    private static final Map<String, Set<String>> baseRelation = new HashMap<>();

    // Load base facts from input or define inline
    static {
        // Example facts - replace with actual data
    }

    /**
     * Add a base fact: ~w(from, to)
     */
    public static void addFact(String from, String to) {
        baseRelation.computeIfAbsent(from, k -> new HashSet<>()).add(to);
    }

    /**
     * Find all descendants/ancestors of start
     * ~w(start, X) for all X
     */
    public static Set<String> findAll(String start) {
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        Set<String> results = new LinkedHashSet<>();

        queue.add(start);
        visited.add(start);

        while (!queue.isEmpty()) {
            String current = queue.poll();
            Set<String> nexts = baseRelation.getOrDefault(current, Collections.emptySet());
            
            for (String next : nexts) {
                if (!visited.contains(next)) {
                    visited.add(next);
                    queue.add(next);
                    results.add(next);
                }
            }
        }
        
        return results;
    }

    /**
     * Check if target is reachable from start
     * ~w(start, target)
     */
    public static boolean check(String start, String target) {
        if (start.equals(target)) return false;
        
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();

        queue.add(start);
        visited.add(start);

        while (!queue.isEmpty()) {
            String current = queue.poll();
            Set<String> nexts = baseRelation.getOrDefault(current, Collections.emptySet());
            
            for (String next : nexts) {
                if (next.equals(target)) {
                    return true;
                }
                if (!visited.contains(next)) {
                    visited.add(next);
                    queue.add(next);
                }
            }
        }
        
        return false;
    }

    /**
     * Stream mode: read JSONL from stdin, output matches
     */
    public static void runStream() throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String line;
        
        while ((line = reader.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            String[] parts = line.split(\":\");
            if (parts.length >= 2) {
                addFact(parts[0].trim(), parts[1].trim());
            }
        }
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            System.err.println(\"Usage: java ~w <start> [target]\");
            System.err.println(\"  With 1 arg: find all ~w of start\");
            System.err.println(\"  With 2 args: check if start ~w target\");
            System.exit(1);
        }

        // Read base facts from stdin (colon-separated)
        try {
            runStream();
        } catch (IOException e) {
            System.err.println(\"Error reading input: \" + e.getMessage());
        }

        String start = args[0];
        
        if (args.length == 1) {
            // Mode: findAll
            for (String result : findAll(start)) {
                System.out.println(start + \":\" + result);
            }
        } else {
            // Mode: check
            String target = args[1];
            if (check(start, target)) {
                System.out.println(start + \":\" + target);
                System.exit(0);
            } else {
                System.exit(1);
            }
        }
    }
}
', [PredStr, BaseStr, Package, PredCap, BaseCap, PredCap, BaseStr, BaseStr, PredStr, PredStr, 
    PredCap, PredStr, PredStr]),
    !.

%% Kotlin transitive closure - generates BFS with sequence
compile_transitive_closure(kotlin, Pred, _Arity, BasePred, _Options, GeneratedCode) :-
    atom_string(Pred, PredStr),
    atom_string(BasePred, BaseStr),
    upcase_atom(Pred, PredCap),
    
    format(string(GeneratedCode),
'// Generated by UnifyWeaver Kotlin Target - Transitive Closure
// Predicate: ~w/2 (transitive closure of ~w)

package generated

/**
 * ~w - transitive closure of ~w
 */
object ~wQuery {
    private val baseRelation = mutableMapOf<String, MutableSet<String>>()

    fun addFact(from: String, to: String) {
        baseRelation.getOrPut(from) { mutableSetOf() }.add(to)
    }

    fun findAll(start: String): Set<String> {
        val visited = mutableSetOf<String>()
        val queue = ArrayDeque<String>()
        val results = linkedSetOf<String>()

        queue.add(start)
        visited.add(start)

        while (queue.isNotEmpty()) {
            val current = queue.removeFirst()
            baseRelation[current]?.forEach { next ->
                if (next !in visited) {
                    visited.add(next)
                    queue.add(next)
                    results.add(next)
                }
            }
        }
        return results
    }

    fun check(start: String, target: String): Boolean {
        if (start == target) return false
        val visited = mutableSetOf<String>()
        val queue = ArrayDeque<String>()
        queue.add(start)
        visited.add(start)
        while (queue.isNotEmpty()) {
            val current = queue.removeFirst()
            baseRelation[current]?.forEach { next ->
                if (next == target) return true
                if (next !in visited) {
                    visited.add(next)
                    queue.add(next)
                }
            }
        }
        return false
    }
}

fun main(args: Array<String>) {
    generateSequence(::readLine).filter { it.isNotBlank() }.forEach { line ->
        val parts = line.split(\":\")
        if (parts.size >= 2) ~wQuery.addFact(parts[0].trim(), parts[1].trim())
    }
    when {
        args.isEmpty() -> { System.err.println(\"Usage: kotlin ~wQueryKt <start> [target]\"); kotlin.system.exitProcess(1) }
        args.size == 1 -> ~wQuery.findAll(args[0]).forEach { println(\"${args[0]}:$it\") }
        else -> if (~wQuery.check(args[0], args[1])) println(\"${args[0]}:${args[1]}\") else kotlin.system.exitProcess(1)
    }
}
', [PredStr, BaseStr, PredStr, BaseStr, PredCap, PredCap, PredCap, PredCap, PredCap]),
    !.

%% Scala transitive closure - generates BFS with LazyList
compile_transitive_closure(scala, Pred, _Arity, BasePred, _Options, GeneratedCode) :-
    atom_string(Pred, PredStr),
    atom_string(BasePred, BaseStr),
    upcase_atom(Pred, PredCap),
    
    format(string(GeneratedCode),
'// Generated by UnifyWeaver Scala Target - Transitive Closure
// Predicate: ~w/2 (transitive closure of ~w)

package generated

import scala.collection.mutable
import scala.io.StdIn

/**
 * ~w - transitive closure of ~w
 */
object ~wQuery {
  private val baseRelation = mutable.Map[String, mutable.Set[String]]()

  def addFact(from: String, to: String): Unit = {
    baseRelation.getOrElseUpdate(from, mutable.Set()) += to
  }

  def findAll(start: String): Set[String] = {
    val visited = mutable.Set[String]()
    val queue = mutable.Queue[String]()
    val results = mutable.LinkedHashSet[String]()

    queue.enqueue(start)
    visited += start

    while (queue.nonEmpty) {
      val current = queue.dequeue()
      baseRelation.getOrElse(current, Set()).foreach { next =>
        if (!visited.contains(next)) {
          visited += next
          queue.enqueue(next)
          results += next
        }
      }
    }
    results.toSet
  }

  def check(start: String, target: String): Boolean = {
    if (start == target) return false
    val visited = mutable.Set[String]()
    val queue = mutable.Queue[String]()
    queue.enqueue(start)
    visited += start
    while (queue.nonEmpty) {
      val current = queue.dequeue()
      baseRelation.getOrElse(current, Set()).foreach { next =>
        if (next == target) return true
        if (!visited.contains(next)) {
          visited += next
          queue.enqueue(next)
        }
      }
    }
    false
  }

  def main(args: Array[String]): Unit = {
    Iterator.continually(StdIn.readLine()).takeWhile(_ != null).filter(_.nonEmpty).foreach { line =>
      val parts = line.split(\":\")
      if (parts.length >= 2) addFact(parts(0).trim, parts(1).trim)
    }
    args.toList match {
      case Nil => System.err.println(\"Usage: scala ~wQuery <start> [target]\"); sys.exit(1)
      case start :: Nil => findAll(start).foreach(r => println(s\"$start:$r\"))
      case start :: target :: _ => if (check(start, target)) println(s\"$start:$target\") else sys.exit(1)
    }
  }
}
', [PredStr, BaseStr, PredStr, BaseStr, PredCap, PredCap]),
    !.

%% Clojure transitive closure - generates BFS with loop/recur
compile_transitive_closure(clojure, Pred, _Arity, BasePred, _Options, GeneratedCode) :-
    atom_string(Pred, PredStr),
    atom_string(BasePred, BaseStr),
    
    format(string(GeneratedCode),
';; Generated by UnifyWeaver Clojure Target - Transitive Closure
;; Predicate: ~w/2 (transitive closure of ~w)

(ns generated.~w-query)

(def base-relation (atom {}))

(defn add-fact [from to]
  (swap! base-relation update from (fnil conj #{}) to))

(defn find-all [start]
  (loop [visited #{start}
         queue [start]
         results []]
    (if (empty? queue)
      results
      (let [current (first queue)
            nexts (get @base-relation current #{})]
        (recur
          (into visited nexts)
          (into (rest queue) (remove visited nexts))
          (into results (remove visited nexts)))))))

(defn check-path [start target]
  (loop [visited #{start}
         queue [start]]
    (cond
      (empty? queue) false
      :else
      (let [current (first queue)
            nexts (get @base-relation current #{})]
        (if (contains? nexts target)
          true
          (recur
            (into visited nexts)
            (into (rest queue) (remove visited nexts))))))))

(defn -main [& args]
  (doseq [line (line-seq (java.io.BufferedReader. *in*))]
    (when (seq line)
      (let [[from to] (clojure.string/split line #\":\")]
        (when (and from to)
          (add-fact (clojure.string/trim from) (clojure.string/trim to))))))
  (case (count args)
    0 (do (binding [*out* *err*] (println \"Usage: clojure ~w_query.clj <start> [target]\"))
          (System/exit 1))
    1 (doseq [r (find-all (first args))]
        (println (str (first args) \":\" r)))
    (if (check-path (first args) (second args))
      (println (str (first args) \":\" (second args)))
      (System/exit 1))))
', [PredStr, BaseStr, PredStr, PredStr]),
    !.

%% Jython transitive closure - generates BFS with Python collections
compile_transitive_closure(jython, Pred, _Arity, BasePred, _Options, GeneratedCode) :-
    atom_string(Pred, PredStr),
    atom_string(BasePred, BaseStr),
    
    format(string(GeneratedCode),
'# Generated by UnifyWeaver Jython Target - Transitive Closure
# Predicate: ~w/2 (transitive closure of ~w)

from collections import deque
import sys

class ~wQuery:
    def __init__(self):
        self.base_relation = {}

    def add_fact(self, from_node, to_node):
        if from_node not in self.base_relation:
            self.base_relation[from_node] = set()
        self.base_relation[from_node].add(to_node)

    def find_all(self, start):
        visited = set([start])
        queue = deque([start])
        results = []

        while queue:
            current = queue.popleft()
            for next_node in self.base_relation.get(current, set()):
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append(next_node)
                    results.append(next_node)
        return results

    def check(self, start, target):
        if start == target:
            return False
        visited = set([start])
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for next_node in self.base_relation.get(current, set()):
                if next_node == target:
                    return True
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append(next_node)
        return False

query = ~wQuery()

# Read base facts from stdin
for line in sys.stdin:
    line = line.strip()
    if line:
        parts = line.split(\":\")
        if len(parts) >= 2:
            query.add_fact(parts[0].strip(), parts[1].strip())

if len(sys.argv) < 2:
    sys.stderr.write(\"Usage: jython ~w_query.py <start> [target]\\\\n\")
    sys.exit(1)
elif len(sys.argv) == 2:
    for r in query.find_all(sys.argv[1]):
        print(\"%s:%s\" % (sys.argv[1], r))
else:
    if query.check(sys.argv[1], sys.argv[2]):
        print(\"%s:%s\" % (sys.argv[1], sys.argv[2]))
    else:
        sys.exit(1)
', [PredStr, BaseStr, PredStr, PredStr, PredStr]),
    !.

%% C transitive closure - generates BFS with hash table
compile_transitive_closure(c, Pred, _Arity, BasePred, _Options, GeneratedCode) :-
    atom_string(Pred, PredStr),
    atom_string(BasePred, BaseStr),
    upcase_atom(Pred, PredCap),
    
    format(string(GeneratedCode),
'/* Generated by UnifyWeaver C Target - Transitive Closure */
/* Predicate: ~w/2 (transitive closure of ~w) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NODES 10000
#define MAX_EDGES 50000
#define MAX_LINE 1024

/* Simple hash table for adjacency list */
typedef struct Edge {
    char* to;
    struct Edge* next;
} Edge;

typedef struct {
    char* from;
    Edge* edges;
} Node;

static Node nodes[MAX_NODES];
static int node_count = 0;

static int find_or_add_node(const char* name) {
    for (int i = 0; i < node_count; i++) {
        if (strcmp(nodes[i].from, name) == 0) return i;
    }
    if (node_count >= MAX_NODES) return -1;
    nodes[node_count].from = strdup(name);
    nodes[node_count].edges = NULL;
    return node_count++;
}

static void add_edge(const char* from, const char* to) {
    int idx = find_or_add_node(from);
    if (idx < 0) return;
    Edge* e = malloc(sizeof(Edge));
    e->to = strdup(to);
    e->next = nodes[idx].edges;
    nodes[idx].edges = e;
}

/* BFS to find all reachable nodes */
static void find_all(const char* start) {
    char* queue[MAX_NODES];
    int visited[MAX_NODES] = {0};
    int head = 0, tail = 0;
    
    int start_idx = -1;
    for (int i = 0; i < node_count; i++) {
        if (strcmp(nodes[i].from, start) == 0) { start_idx = i; break; }
    }
    if (start_idx < 0) return;
    
    queue[tail++] = nodes[start_idx].from;
    visited[start_idx] = 1;
    
    while (head < tail) {
        char* current = queue[head++];
        int cur_idx = -1;
        for (int i = 0; i < node_count; i++) {
            if (strcmp(nodes[i].from, current) == 0) { cur_idx = i; break; }
        }
        if (cur_idx < 0) continue;
        
        for (Edge* e = nodes[cur_idx].edges; e; e = e->next) {
            int next_idx = -1;
            for (int i = 0; i < node_count; i++) {
                if (strcmp(nodes[i].from, e->to) == 0) { next_idx = i; break; }
            }
            if (next_idx >= 0 && !visited[next_idx]) {
                visited[next_idx] = 1;
                queue[tail++] = e->to;
                printf(\"%s:%s\\n\", start, e->to);
            } else if (next_idx < 0) {
                /* Node not in graph as source, just print */
                printf(\"%s:%s\\n\", start, e->to);
            }
        }
    }
}

/* BFS to check if target is reachable */
static int check_path(const char* start, const char* target) {
    char* queue[MAX_NODES];
    int visited[MAX_NODES] = {0};
    int head = 0, tail = 0;
    
    int start_idx = -1;
    for (int i = 0; i < node_count; i++) {
        if (strcmp(nodes[i].from, start) == 0) { start_idx = i; break; }
    }
    if (start_idx < 0) return 0;
    
    queue[tail++] = nodes[start_idx].from;
    visited[start_idx] = 1;
    
    while (head < tail) {
        char* current = queue[head++];
        int cur_idx = -1;
        for (int i = 0; i < node_count; i++) {
            if (strcmp(nodes[i].from, current) == 0) { cur_idx = i; break; }
        }
        if (cur_idx < 0) continue;
        
        for (Edge* e = nodes[cur_idx].edges; e; e = e->next) {
            if (strcmp(e->to, target) == 0) return 1;
            int next_idx = -1;
            for (int i = 0; i < node_count; i++) {
                if (strcmp(nodes[i].from, e->to) == 0) { next_idx = i; break; }
            }
            if (next_idx >= 0 && !visited[next_idx]) {
                visited[next_idx] = 1;
                queue[tail++] = e->to;
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    char line[MAX_LINE];
    
    /* Read ~w facts from stdin */
    while (fgets(line, sizeof(line), stdin)) {
        char* colon = strchr(line, \\\':\\\');
        if (!colon) continue;
        *colon = \\\'\\\\0\\\';
        char* to = colon + 1;
        size_t len = strlen(to);
        if (len > 0 && to[len-1] == \\\'\\\\n\\\') to[len-1] = \\\'\\\\0\\\';
        add_edge(line, to);
    }
    
    if (argc < 2) {
        fprintf(stderr, \"Usage: %s <start> [target]\\\\n\", argv[0]);
        return 1;
    }
    
    if (argc == 2) {
        find_all(argv[1]);
    } else {
        if (check_path(argv[1], argv[2])) {
            printf(\"%s:%s\\\\n\", argv[1], argv[2]);
        } else {
            return 1;
        }
    }
    return 0;
}
', [PredStr, BaseStr, BaseStr]),
    !.

%% C++ transitive closure - generates BFS with STL containers
compile_transitive_closure(cpp, Pred, _Arity, BasePred, _Options, GeneratedCode) :-
    atom_string(Pred, PredStr),
    atom_string(BasePred, BaseStr),
    upcase_atom(Pred, PredCap),
    
    format(string(GeneratedCode),
'// Generated by UnifyWeaver C++ Target - Transitive Closure
// Predicate: ~w/2 (transitive closure of ~w)

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <sstream>

class ~wQuery {
private:
    std::unordered_map<std::string, std::vector<std::string>> baseRelation;

public:
    void addFact(const std::string& from, const std::string& to) {
        baseRelation[from].push_back(to);
    }

    std::vector<std::string> findAll(const std::string& start) {
        std::vector<std::string> results;
        std::unordered_set<std::string> visited;
        std::queue<std::string> queue;

        queue.push(start);
        visited.insert(start);

        while (!queue.empty()) {
            std::string current = queue.front();
            queue.pop();

            auto it = baseRelation.find(current);
            if (it != baseRelation.end()) {
                for (const auto& next : it->second) {
                    if (visited.find(next) == visited.end()) {
                        visited.insert(next);
                        queue.push(next);
                        results.push_back(next);
                    }
                }
            }
        }
        return results;
    }

    bool check(const std::string& start, const std::string& target) {
        if (start == target) return false;
        std::unordered_set<std::string> visited;
        std::queue<std::string> queue;

        queue.push(start);
        visited.insert(start);

        while (!queue.empty()) {
            std::string current = queue.front();
            queue.pop();

            auto it = baseRelation.find(current);
            if (it != baseRelation.end()) {
                for (const auto& next : it->second) {
                    if (next == target) return true;
                    if (visited.find(next) == visited.end()) {
                        visited.insert(next);
                        queue.push(next);
                    }
                }
            }
        }
        return false;
    }
};

int main(int argc, char** argv) {
    ~wQuery query;
    std::string line;

    // Read ~w facts from stdin
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        size_t pos = line.find(\\\':\\\');
        if (pos != std::string::npos) {
            std::string from = line.substr(0, pos);
            std::string to = line.substr(pos + 1);
            query.addFact(from, to);
        }
    }

    if (argc < 2) {
        std::cerr << \"Usage: \" << argv[0] << \" <start> [target]\" << std::endl;
        return 1;
    }

    if (argc == 2) {
        for (const auto& r : query.findAll(argv[1])) {
            std::cout << argv[1] << \":\" << r << std::endl;
        }
    } else {
        if (query.check(argv[1], argv[2])) {
            std::cout << argv[1] << \":\" << argv[2] << std::endl;
        } else {
            return 1;
        }
    }
    return 0;
}
', [PredStr, BaseStr, PredCap, PredCap, BaseStr]),
    !.

compile_transitive_closure(Target, Pred, Arity, BasePred, _Options, _GeneratedCode) :-
    format(user_error,
           'Transitive closure for target ~w not yet supported (~w/~w via ~w).~n',
           [Target, Pred, Arity, BasePred]),
    fail.

%% Compile with memoization for unknown patterns
compile_memoized_recursion(bash, Pred, Arity, Options, GeneratedCode) :-
    compile_plain_recursion(Pred, Arity, Options, GeneratedCode).
compile_memoized_recursion(Target, Pred, Arity, _Options, _GeneratedCode) :-
    format(user_error,
           'Memoized recursion not implemented for target ~w (~w/~w).~n',
           [Target, Pred, Arity]),
    fail.

%% Compile plain recursion as final fallback
compile_plain_recursion(Pred, Arity, Options, GeneratedCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),  % Currently assumes arity 2 semantics
    
    % Get all clauses
    findall(Body, clause(Head, Body), Bodies),
    
    % Separate base and recursive cases
    partition(is_recursive_clause(Pred), Bodies, RecClauses, BaseClauses),
    
    % Generate base cases
    findall(BaseCode, (
        member(Base, BaseClauses),
        generate_base_case(Pred, Base, BaseCode)
    ), BaseCodes),
    atomic_list_concat(BaseCodes, '\n    ', BaseCodeStr),
    
    % Generate recursive cases  
    findall(RecCode, (
        member(Rec, RecClauses),
        generate_recursive_case(Pred, Rec, RecCode)
    ), RecCodes),
    atomic_list_concat(RecCodes, '\n    ', RecCodeStr),
    
    generate_plain_recursion_template(PredStr, BaseCodeStr, RecCodeStr, Options, GeneratedCode).

%% Generate base case code
generate_base_case(Pred, Body, Code) :-
    atom_string(Pred, PredStr),
    (   Body = true ->
        format(string(Code), '# Base: ~s is a fact
    [[ -n "${~s_data[$key]}" ]] && {
        ~s_memo["$key"]="$key"
        echo "$key"
        return
    }', [PredStr, PredStr, PredStr])
    ;   ( functor(Body, BasePred, 2), atom(BasePred), is_alpha(BasePred) ) ->
        atom_string(BasePred, BaseStr),
        format(string(Code), '# Base: check ~s
    if ~s "$arg1" "$arg2" >/dev/null 2>&1; then
        ~s_memo["$key"]="$key"
        echo "$key"
        return
    fi', [BaseStr, BaseStr, PredStr])
    ;   format(string(Code), '# Complex base case - not implemented', [])
    ).

is_alpha(Atom) :-
    atom_codes(Atom, Codes),
    forall(member(Code, Codes), code_type(Code, alpha)).

%% Generate recursive case code
generate_recursive_case(Pred, Body, Code) :-
    atom_string(Pred, PS),
    % Simplified - just note it's recursive
    format(string(Code), '# Recursive case
    # Body: ~w
    # Would need to decompose and call ~s recursively
    # Implementation depends on specific pattern', [Body, PS]).

%% String replacement helper
string_replace(Input, Find, Replace, Output) :-
    split_string(Input, Find, "", Parts),
    atomic_list_concat(Parts, Replace, Output).

%% Test the recursive compiler
test_recursive_compiler :-
    writeln('=== RECURSIVE COMPILER TEST ==='),
    
    % Setup output directory
    (   exists_directory('output') -> true
    ;   make_directory('output')
    ),
    
    % First ensure base predicates exist
    stream_compiler:test_stream_compiler,  % This sets up parent/2, grandparent/2, etc.
    
    writeln(''),
    writeln('--- Testing Recursive Predicates ---'),
    
    % Clear any existing recursive predicates
    abolish(ancestor/2),
    abolish(descendant/2),
    abolish(reachable/2),
    
    % Define ancestor as transitive closure of parent
    assertz((ancestor(X, Y) :- parent(X, Y))),
    assertz((ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z))),
    
    % Define descendant (reverse of ancestor)
    assertz((descendant(X, Y) :- parent(Y, X))),
    assertz((descendant(X, Z) :- parent(Y, X), descendant(Y, Z))),
    
    % Define reachable (follows related)
    assertz((reachable(X, Y) :- related(X, Y))),
    assertz((reachable(X, Z) :- related(X, Y), reachable(Y, Z))),
    
    % Compile recursive predicates
    compile_recursive(ancestor/2, [], AncestorCode),
    stream_compiler:write_bash_file('output/ancestor.sh', AncestorCode),
    
    compile_recursive(descendant/2, [], DescendantCode),
    stream_compiler:write_bash_file('output/descendant.sh', DescendantCode),
    
    compile_recursive(reachable/2, [], ReachableCode),
    stream_compiler:write_bash_file('output/reachable.sh', ReachableCode),
    
    % Generate extended test script
    generate_recursive_test_script,
    
    writeln('--- Recursive Compilation Complete ---'),
    writeln('Check files in output/'),
    writeln('Run: bash output/test_recursive.sh').

%% Generate test script for recursive predicates
generate_recursive_test_script :-
    TestScript = '#!/bin/bash
# Test script for recursive predicates

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source all files - ORDER MATTERS: dependencies first
source "$SCRIPT_DIR/parent.sh"
source "$SCRIPT_DIR/sibling.sh"      # Must be before related.sh
source "$SCRIPT_DIR/grandparent.sh"  # Also useful to have available
source "$SCRIPT_DIR/ancestor.sh"
source "$SCRIPT_DIR/descendant.sh"
source "$SCRIPT_DIR/related.sh"      # Depends on sibling
source "$SCRIPT_DIR/reachable.sh"

echo "=== Testing ancestor (transitive closure) ==="
echo "All ancestors of charlie:"
ancestor_all charlie

echo ""
echo "Is alice an ancestor of eve?"
ancestor_check alice eve && echo "Yes" || echo "No"

echo ""
echo "Is bob an ancestor of frank?"
ancestor_check bob frank && echo "Yes" || echo "No"

echo ""
echo "=== Testing descendant ==="
echo "All descendants of alice:"
descendant_all alice

echo ""
echo "All descendants of diana (should show eve, emily, frank):"
descendant_all diana

echo ""
echo "All descendants of charlie (should show diana, eve, emily, frank):"
descendant_all charlie

echo ""
echo "=== Testing reachable ==="
echo "All nodes reachable from alice (first 10):"
reachable_all alice | head -10

echo ""
echo "All nodes reachable from eve (first 10):"
reachable_all eve | head -10
',
    stream_compiler:write_bash_file('output/test_recursive.sh', TestScript).

%% ============================================
%% LOCAL TEMPLATE FUNCTIONS
%% ============================================
%% These are specialized for recursive compilation
%% They use template_system:render_template/3 for rendering

%% Generate descendant template
generate_descendant_template(PredStr, BashCode) :-
    Template = '#!/bin/bash
# {{pred}} - finds descendants (children, grandchildren, etc.)

# Iterative BFS implementation
{{pred}}() {
    local start="$1"
    local target="$2"
    
    if [[ -z "$target" ]]; then
        # Mode: {{pred}}(+,-)  Find all descendants
        {{pred}}_all "$start"
    else
        # Mode: {{pred}}(+,+)  Check if descendant
        {{pred}}_check "$start" "$target"
    fi
}

# Find all descendants of start
{{pred}}_all() {
    local start="$1"
    declare -A visited
    declare -A output_seen
    
    # Use work queue for BFS
    local queue_file="/tmp/{{pred}}_queue_$"
    local next_queue="/tmp/{{pred}}_next_$"
    
    trap "rm -f $queue_file $next_queue" EXIT
    
    echo "$start" > "$queue_file"
    visited["$start"]=1
    
    while [[ -s "$queue_file" ]]; do
        > "$next_queue"
        
        while IFS= read -r current; do
            # Find all children of current (forward direction)
            parent_stream | grep "^$current:" | while IFS=":" read -r from to; do
                if [[ "$from" == "$current" && -z "${visited[$to]}" ]]; then
                    visited["$to"]=1
                    echo "$to" >> "$next_queue"
                    
                    # Output the descendant relationship
                    local output_key="$start:$to"
                    if [[ -z "${output_seen[$output_key]}" ]]; then
                        output_seen["$output_key"]=1
                        echo "$output_key"
                    fi
                fi
            done
        done < "$queue_file"
        
        mv "$next_queue" "$queue_file"
    done
    
    rm -f "$queue_file" "$next_queue"
}

# Check if target is descendant of start
{{pred}}_check() {
    local start="$1"
    local target="$2"
    local tmpflag="/tmp/{{pred}}_found_$$"
    local timeout_duration="5s"
    
    # Timeout prevents infinite execution, tee prevents SIGPIPE
    timeout "$timeout_duration" {{pred}}_all "$start" | 
    tee >(grep -q "^$start:$target$" && touch "$tmpflag") >/dev/null
    
    if [[ -f "$tmpflag" ]]; then
        echo "$start:$target"
        rm -f "$tmpflag"
        return 0
    else
        rm -f "$tmpflag"
        return 1
    fi
}

# Stream function
{{pred}}_stream() {
    {{pred}}_all "$1"
}',
    template_system:render_template(Template, [pred=PredStr], BashCode).

%% Generate plain recursion template
generate_plain_recursion_template(PredStr, BaseCodeStr, RecCodeStr, Options, BashCode) :-
    constraint_analyzer:get_dedup_strategy(Options, Strategy),
    (   Strategy == no_dedup ->
        MemoSetup = "",
        MemoCheck = ""
    ;   MemoSetup = "declare -gA {{pred}}_memo",
        MemoCheck = 'if [[ -n "${{{pred}}_memo[$key]}" ]]; then echo "${{{pred}}_memo[$key]}"; return; fi'
    ),
    Template = '# Memoization table
{{memo_setup}}

# Main recursive function
{{pred}}_all() {
    local arg1="$1"
    local arg2="$2"
    local key="$arg1:$arg2"
    
    # Check memoization
    {{memo_check}}
    
    # Try base cases first
    {{base_cases}}
    
    # Try recursive cases
    {{rec_cases}}
    
    # Cache miss and no match
    return 1
}

# Wrapper with named pipe support for complex queries
{{pred}}_stream() {
    local pipe="/tmp/{{pred}}_pipe_$$"
    mkfifo "$pipe" 2>/dev/null || true
    
    # Start recursive computation in background
    {{pred}}_all "$@" > "$pipe" &
    
    # Read results from pipe
    cat "$pipe"
    rm -f "$pipe"
}',
    render_template(Template, [
        pred=PredStr,
        base_cases=BaseCodeStr,
        rec_cases=RecCodeStr,
        memo_setup=MemoSetup,
        memo_check=MemoCheck
    ], MainCode),
    render_named_template('dedup_wrapper', [
        pred = PredStr,
        strategy = Strategy,
        main_code = MainCode
    ], BashCode).
