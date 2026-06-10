% test_wam_clojure_lowered_t4.pl
%
% End-to-end execution test for the Clojure T4 lowering — "multi-clause, all
% clauses" (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from
% Scala/Rust/Go/C++/Haskell/F#.
%
% Clojure had NO multi-clause lowering: a try_me_else predicate emitted a
% no-op stub and the run-wam interpreter executed it from start-pc. T4 is its
% first real multi-clause lowering: when every clause is a supported
% deterministic body, all clauses are emitted inline as threaded
% `(let [s0 state ...] sN)` bodies and tried in order on the SAME input state,
% taking the first whose body reaches :succeeded. Clojure's immutable state
% gives a free per-clause restore, so no choice point is pushed and run-wam
% short-circuits on the returned :succeeded / :failed status without
% interpreting any bytecode.
%
% Pins (BOUND first arg; the payoff is the non-first clauses running natively):
%   * grade/2 — fact chain with a REPEATED first arg (alice in clauses 1 & 3);
%   * rel/2   — RULE chain with a VARIABLE first arg (=/2 body).
%
% Skipped automatically when no Clojure jar is found on the usual paths.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_clojure_target').
:- use_module('../src/unifyweaver/targets/wam_clojure_lowered_emitter').

:- dynamic user:grade/2.
:- dynamic user:rel/2.

user:grade(alice, a).
user:grade(bob,   b).
user:grade(alice, c).

user:rel(X, one) :- X = p.
user:rel(X, two) :- X = q.

clojure_jar('/usr/share/java/clojure.jar').
clojure_jar('/usr/share/java/clojure-1.11.jar').
clojure_jar('/usr/share/java/clojure-1.11.1.jar').

clojure_runnable(Jar) :-
    clojure_jar(Jar), exists_file(Jar), !.

:- begin_tests(wam_clojure_lowered_t4, [condition(clojure_runnable(_))]).

% Both predicates must lower as T4 (multi_clause_n), not multi_clause_1.
test(gate_picks_multi_clause_n) :-
    forall(member(PI, [grade/2, rel/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_clojure_lowerable(PI, W, Reason),
             assertion(Reason == multi_clause_n) )).

test(t4_exec_parity) :-
    clojure_runnable(Jar),
    Dir = 'output/test_wam_clojure_t4_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_clojure_project(
        [user:grade/2, user:rel/2],
        [module_name('t4proj')], Dir),
    atomic_list_concat([Dir, '/test_t4.clj'], TestPath),
    clojure_t4_source(Src),
    setup_call_cleanup(open(TestPath, write, S), write(S, Src), close(S)),
    atomic_list_concat([Dir, '/src'], SrcDir),
    format(atom(Cmd),
        'java -cp "~w:~w" clojure.main ~w 2>&1',
        [Jar, SrcDir, TestPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 10 PASS")
    ->  true
    ;   format(user_error, "~n[clojure t4 test output]~n~w~n", [OutStr]),
        throw(clojure_t4_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_clojure_lowered_t4).

% Calls each lowered predicate wrapper; run-wam returns a boolean. The cases
% exercise the non-first clauses (grade clauses 2 & 3, rel clause 2) — the T4
% payoff — plus no-match cases.
clojure_t4_source(
"(require '[generated.wam.core :as core])
(require '[generated.wam.runtime :as runtime])
(defn a [s] (runtime/normalize-literal-term core/atom-intern-context s))
(def cases
  [[\"grade(alice,a)\" (core/grade (a \"alice\") (a \"a\")) true]
   [\"grade(bob,b)\"   (core/grade (a \"bob\")   (a \"b\")) true]
   [\"grade(alice,c)\" (core/grade (a \"alice\") (a \"c\")) true]
   [\"grade(alice,b)\" (core/grade (a \"alice\") (a \"b\")) false]
   [\"grade(carol,a)\" (core/grade (a \"carol\") (a \"a\")) false]
   [\"grade(bob,c)\"   (core/grade (a \"bob\")   (a \"c\")) false]
   [\"rel(p,one)\" (core/rel (a \"p\") (a \"one\")) true]
   [\"rel(q,two)\" (core/rel (a \"q\") (a \"two\")) true]
   [\"rel(p,two)\" (core/rel (a \"p\") (a \"two\")) false]
   [\"rel(q,one)\" (core/rel (a \"q\") (a \"one\")) false]])
(let [fails (filter (fn [[_ got want]] (not= (boolean got) want)) cases)]
  (doseq [[n got want] fails] (println \"FAIL\" n \"got\" (boolean got) \"want\" want))
  (if (empty? fails) (println \"ALL\" (count cases) \"PASS\")
      (println (count fails) \"FAILURES\")))
").
