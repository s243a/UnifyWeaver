% test_wam_clojure_lowered_ite_exec.pl
%
% End-to-end execution test for Clojure if-then-else / negation / once
% lowering.
%
% Generates a WAM Clojure project, compiles+runs it with the JVM Clojure
% toolchain, and calls each lowered predicate wrapper, asserting the
% boolean (success/failure) outcome. Counterpart to the Go/Rust/C++/
% Haskell/F# exec tests. Pins:
%
%   * sequential ITEs   — cseqite(10,pos,small) must fail;
%   * nested ITEs       — cnestite/2;
%   * negation (\+)      — cneg/1 (commit is the !/0 builtin);
%   * simple ITEs        — cite/2.
%
% Clojure previously emitted a no-op stub for any predicate containing an
% if-then-else (its flat let-threading can't branch), relying on the
% run-wam interpreter. The shared-structurer conversion emits native
% branching. Skipped unless java + a Clojure jar are present.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_clojure_target').

:- dynamic user:cite/2.
:- dynamic user:cneg/1.
:- dynamic user:cseqite/3.
:- dynamic user:cnestite/2.

user:cite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:cneg(X)          :- \+ X > 0.
user:cseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:cnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

clojure_jar('/usr/share/java/clojure.jar').
clojure_jar('/usr/share/java/clojure-1.11.jar').
clojure_jar('/usr/share/java/clojure-1.11.1.jar').

clojure_runnable(Jar) :-
    catch(( process_create(path(java), ['-version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail),
    clojure_jar(Jar), exists_file(Jar), !.

:- begin_tests(wam_clojure_lowered_ite_exec, [condition(clojure_runnable(_))]).

test(ite_exec_parity) :-
    clojure_runnable(Jar),
    Dir = 'output/test_wam_clojure_ite_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the WAM Clojure project (ITE preds now lower natively).
    write_wam_clojure_project(
        [user:cite/2, user:cneg/1, user:cseqite/3, user:cnestite/2],
        [module_name('iteproj')], Dir),
    % 2. Harness that calls each lowered wrapper and checks success/failure.
    atomic_list_concat([Dir, '/test_ite.clj'], TestPath),
    clojure_test_source(Src),
    setup_call_cleanup(open(TestPath, write, S), write(S, Src), close(S)),
    % 3. Compile + run on the JVM.
    atomic_list_concat([Dir, '/src'], SrcDir),
    format(atom(Cmd),
        'java -cp "~w:~w" clojure.main ~w 2>&1',
        [Jar, SrcDir, TestPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 15 PASS")
    ->  true
    ;   format(user_error, "~n[clojure ite test output]~n~w~n", [OutStr]),
        throw(clojure_test_failed(Status))
    ).

:- end_tests(wam_clojure_lowered_ite_exec).

% Calls each lowered predicate wrapper; run-wam returns a boolean. Atoms are
% built through the generated compile-time intern context. cseqite(10,pos,
% small)=false and cnestite are the sequential/nested discriminators.
clojure_test_source(
"(require '[generated.wam.core :as core])
(require '[generated.wam.runtime :as runtime])
(defn a [s] (runtime/normalize-literal-term core/atom-intern-context s))
(def cases
  [[\"cite(5,pos)\"        (core/cite 5 (a \"pos\"))        true]
   [\"cite(5,nonpos)\"     (core/cite 5 (a \"nonpos\"))     false]
   [\"cite(-1,nonpos)\"    (core/cite -1 (a \"nonpos\"))    true]
   [\"cite(-1,pos)\"       (core/cite -1 (a \"pos\"))       false]
   [\"cneg(5)\"            (core/cneg 5)                  false]
   [\"cneg(-1)\"           (core/cneg -1)                 true]
   [\"cneg(0)\"            (core/cneg 0)                  true]
   [\"cseqite(10,pos,big)\"      (core/cseqite 10 (a \"pos\") (a \"big\"))      true]
   [\"cseqite(10,pos,small)\"    (core/cseqite 10 (a \"pos\") (a \"small\"))    false]
   [\"cseqite(3,pos,small)\"     (core/cseqite 3 (a \"pos\") (a \"small\"))     true]
   [\"cseqite(-1,nonpos,small)\" (core/cseqite -1 (a \"nonpos\") (a \"small\")) true]
   [\"cnestite(20,big)\"   (core/cnestite 20 (a \"big\"))   true]
   [\"cnestite(5,small)\"  (core/cnestite 5 (a \"small\"))  true]
   [\"cnestite(-1,neg)\"   (core/cnestite -1 (a \"neg\"))   true]
   [\"cnestite(20,small)\" (core/cnestite 20 (a \"small\")) false]])
(let [fails (filter (fn [[_ got want]] (not= (boolean got) want)) cases)]
  (doseq [[n got want] fails] (println \"FAIL\" n \"got\" (boolean got) \"want\" want))
  (if (empty? fails) (println \"ALL\" (count cases) \"PASS\")
      (println (count fails) \"FAILURES\")))
").
