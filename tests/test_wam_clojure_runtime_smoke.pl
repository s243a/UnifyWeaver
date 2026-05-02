:- encoding(utf8).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_clojure_target').

:- dynamic user:wam_fact/1.
:- dynamic user:wam_foreign_pair/2.
:- dynamic user:wam_foreign_pair_query/1.
:- dynamic user:wam_foreign_stream_pair/2.
:- dynamic user:wam_foreign_stream_pair_query/1.
:- dynamic user:wam_execute_caller/1.
:- dynamic user:wam_call_caller/1.
:- dynamic user:wam_choice_fact/1.
:- dynamic user:wam_choice_caller/1.
:- dynamic user:wam_choice_or_z/1.
:- dynamic user:wam_bind_then_fact/1.
:- dynamic user:wam_bind_after_call/1.
:- dynamic user:wam_bind_before_execute/1.
:- dynamic user:wam_if_then_else/1.
:- dynamic user:wam_struct_fact/1.
:- dynamic user:wam_list_fact/1.
:- dynamic user:wam_use_struct/1.
:- dynamic user:wam_use_list/1.
:- dynamic user:wam_make_struct/1.
:- dynamic user:wam_make_list/1.
:- dynamic user:wam_double_struct_match/1.
:- dynamic user:wam_double_list_match/1.
:- dynamic user:wam_build_backtrack/1.
:- dynamic user:wam_env_build_backtrack/1.
:- dynamic user:wam_list_build_backtrack/1.
:- dynamic user:wam_env1/1.
:- dynamic user:wam_env2/1.
:- dynamic user:wam_env3/1.
:- dynamic user:wam_trail_choice/1.
:- dynamic user:wam_soft_cut_helper/1.
:- dynamic user:wam_soft_cut_outer_ok/1.
:- dynamic user:wam_cut_helper/1.
:- dynamic user:wam_hard_cut_outer_ok/1.

has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

user:wam_fact(a).
user:wam_foreign_pair_query(Y) :- user:wam_foreign_pair(a, Y).
user:wam_foreign_stream_pair_query(Y) :- user:wam_foreign_stream_pair(a, Y), Y = b.
user:wam_execute_caller(X) :- user:wam_fact(X).
user:wam_call_caller(X) :- user:wam_fact(X), user:wam_fact(X).
user:wam_choice_fact(a).
user:wam_choice_fact(b).
user:wam_choice_fact(c).
user:wam_choice_caller(X) :- user:wam_choice_fact(X).
user:wam_choice_or_z(X) :- user:wam_choice_fact(X).
user:wam_choice_or_z(z).
user:wam_bind_then_fact(X) :- Y = X, user:wam_fact(Y).
user:wam_bind_after_call(X) :- user:wam_fact(X), X = a.
user:wam_bind_before_execute(X) :- X = a, user:wam_fact(X).
user:wam_if_then_else(X) :- (X = a -> true ; X = b).
user:wam_struct_fact(f(a)).
user:wam_list_fact([a,b]).
user:wam_use_struct(X) :- user:wam_struct_fact(X).
user:wam_use_list(X) :- user:wam_list_fact(X).
user:wam_make_struct(X) :- X = f(a).
user:wam_make_list(X) :- X = [a,b].
user:wam_double_struct_match(X) :- user:wam_struct_fact(X), user:wam_struct_fact(X).
user:wam_double_list_match(X) :- user:wam_list_fact(X), user:wam_list_fact(X).
user:wam_build_backtrack(X) :- (X = f(a), fail ; X = f(b)).
user:wam_env_build_backtrack(X) :- (Y = f(a), fail ; Y = f(b)), X = Y.
user:wam_list_build_backtrack(X) :- (X = [a,b], fail ; X = [a,c]).
user:wam_env1(X) :- Y = X, Z = a, Y = Z.
user:wam_env2(X) :- user:wam_fact(X), Y = X, user:wam_fact(Y).
user:wam_env3(X) :- (X = a ; X = b), user:wam_fact(X).
user:wam_trail_choice(X) :- (Y = a ; Y = b), X = Y.
user:wam_soft_cut_helper(X) :- (X = a -> fail ; true).
user:wam_soft_cut_outer_ok(X) :- (user:wam_soft_cut_helper(Y), X = Y ; X = b).
user:wam_cut_helper(X) :- X = a, !, fail.
user:wam_hard_cut_outer_ok(X) :- (user:wam_cut_helper(Y), X = Y ; X = b).

:- initialization(main, main).

main :-
    catch(run_smoke, Error, (print_message(error, Error), halt(1))),
    halt(0).

run_smoke :-
    unique_tmp_dir('tmp_wam_clojure_smoke', TmpDir),
    write_wam_clojure_project(
        [ user:wam_execute_caller/1,
          user:wam_call_caller/1,
          user:wam_fact/1,
          user:wam_foreign_pair_query/1,
          user:wam_foreign_pair/2,
          user:wam_foreign_stream_pair_query/1,
          user:wam_foreign_stream_pair/2,
          user:wam_choice_fact/1,
          user:wam_choice_caller/1,
          user:wam_choice_or_z/1,
          user:wam_bind_then_fact/1,
          user:wam_bind_after_call/1,
          user:wam_bind_before_execute/1,
          user:wam_if_then_else/1,
          user:wam_struct_fact/1,
          user:wam_list_fact/1,
          user:wam_use_struct/1,
          user:wam_use_list/1,
          user:wam_make_struct/1,
          user:wam_make_list/1,
          user:wam_double_struct_match/1,
          user:wam_double_list_match/1,
          user:wam_build_backtrack/1,
          user:wam_env_build_backtrack/1,
          user:wam_list_build_backtrack/1,
          user:wam_env1/1,
          user:wam_env2/1,
          user:wam_env3/1,
          user:wam_trail_choice/1,
          user:wam_soft_cut_helper/1,
          user:wam_soft_cut_outer_ok/1,
          user:wam_cut_helper/1,
          user:wam_hard_cut_outer_ok/1
        ],
        [ namespace('generated.wam_exec_test'),
          module_name('wam-clojure-exec-test'),
          foreign_predicates([wam_fact/1, wam_foreign_pair/2, wam_foreign_stream_pair/2]),
          clojure_foreign_handlers([
              handler(wam_fact/1, "(fn [args] (= (first args) \"a\"))"),
              handler(wam_foreign_pair/2, "(fn [args] (if (= (first args) \"a\") {:bindings {2 \"b\"}} false))"),
              handler(wam_foreign_stream_pair/2, "(fn [args] (if (= (first args) \"a\") {:solutions [{:bindings {2 \"a\"}} {:bindings {2 \"b\"}}]} false))")
          ])
        ],
        TmpDir),
    assert_lowered_read_unify_prefix_emitted(TmpDir),
    assert_lowered_env_prefix_emitted(TmpDir),
    assert_lowered_execute_emitted(TmpDir),
    assert_lowered_call_emitted(TmpDir),
    assert_lowered_cut_builtin_emitted(TmpDir),
    verify_output(TmpDir, 'wam_execute_caller/1', 'a', "true"),
    verify_output(TmpDir, 'wam_execute_caller/1', 'b', "false"),
    verify_output(TmpDir, 'wam_call_caller/1', 'a', "true"),
    verify_output(TmpDir, 'wam_call_caller/1', 'b', "false"),
    verify_output(TmpDir, 'wam_foreign_pair_query/1', b, "true"),
    verify_output(TmpDir, 'wam_foreign_pair_query/1', c, "false"),
    verify_output(TmpDir, 'wam_foreign_stream_pair_query/1', a, "false"),
    verify_output(TmpDir, 'wam_foreign_stream_pair_query/1', b, "true"),
    verify_output(TmpDir, 'wam_foreign_stream_pair_query/1', c, "false"),
    verify_output(TmpDir, 'wam_choice_caller/1', 'a', "true"),
    verify_output(TmpDir, 'wam_choice_caller/1', 'b', "true"),
    verify_output(TmpDir, 'wam_choice_caller/1', 'c', "true"),
    verify_output(TmpDir, 'wam_choice_caller/1', 'd', "false"),
    verify_output(TmpDir, 'wam_choice_or_z/1', 'z', "true"),
    verify_output(TmpDir, 'wam_bind_then_fact/1', 'a', "true"),
    verify_output(TmpDir, 'wam_bind_then_fact/1', 'b', "false"),
    verify_output(TmpDir, 'wam_bind_after_call/1', 'a', "true"),
    verify_output(TmpDir, 'wam_bind_after_call/1', 'b', "false"),
    verify_output(TmpDir, 'wam_bind_before_execute/1', 'a', "true"),
    verify_output(TmpDir, 'wam_bind_before_execute/1', 'b', "false"),
    verify_output(TmpDir, 'wam_if_then_else/1', 'a', "true"),
    verify_output(TmpDir, 'wam_if_then_else/1', 'b', "true"),
    verify_output(TmpDir, 'wam_if_then_else/1', 'c', "false"),
    verify_output(TmpDir, 'wam_use_struct/1', 'f(a)', "true"),
    verify_output(TmpDir, 'wam_use_struct/1', 'f(b)', "false"),
    verify_output(TmpDir, 'wam_use_list/1', '[a,b]', "true"),
    verify_output(TmpDir, 'wam_use_list/1', '[a,c]', "false"),
    verify_output(TmpDir, 'wam_double_struct_match/1', 'f(a)', "true"),
    verify_output(TmpDir, 'wam_double_struct_match/1', 'f(b)', "false"),
    verify_output(TmpDir, 'wam_double_list_match/1', '[a,b]', "true"),
    verify_output(TmpDir, 'wam_double_list_match/1', '[a,c]', "false"),
    % Write-mode smoke path. We only assert the generated program runs and
    % succeeds for the canonical constructed term case in this environment.
    verify_output(TmpDir, 'wam_make_struct/1', 'f(a)', "true"),
    verify_output(TmpDir, 'wam_make_list/1', '[a,b]', "true"),
    verify_output(TmpDir, 'wam_build_backtrack/1', 'f(a)', "false"),
    verify_output(TmpDir, 'wam_build_backtrack/1', 'f(b)', "true"),
    verify_output(TmpDir, 'wam_env_build_backtrack/1', 'f(a)', "false"),
    verify_output(TmpDir, 'wam_env_build_backtrack/1', 'f(b)', "true"),
    verify_output(TmpDir, 'wam_list_build_backtrack/1', '[a,b]', "false"),
    verify_output(TmpDir, 'wam_list_build_backtrack/1', '[a,c]', "true"),
    verify_output(TmpDir, 'wam_env1/1', a, "true"),
    verify_output(TmpDir, 'wam_env1/1', b, "false"),
    verify_output(TmpDir, 'wam_env2/1', a, "true"),
    verify_output(TmpDir, 'wam_env2/1', b, "false"),
    verify_output(TmpDir, 'wam_env3/1', a, "true"),
    verify_output(TmpDir, 'wam_env3/1', b, "false"),
    verify_output(TmpDir, 'wam_trail_choice/1', a, "true"),
    verify_output(TmpDir, 'wam_trail_choice/1', b, "true"),
    verify_output(TmpDir, 'wam_trail_choice/1', c, "false"),
    verify_output(TmpDir, 'wam_soft_cut_outer_ok/1', b, "true"),
    verify_output(TmpDir, 'wam_hard_cut_outer_ok/1', b, "true"),
    delete_directory_and_contents(TmpDir),
    writeln('wam_clojure_runtime_smoke: ok').

assert_lowered_read_unify_prefix_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-struct-fact-1"),
    has(CoreCode, "defn lowered-wam-list-fact-1"),
    has(CoreCode, "runtime/enter-unify-mode"),
    has(CoreCode, "runtime/pop-unify-item"),
    has(CoreCode, "runtime/unify-values").

assert_lowered_env_prefix_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-env1-1"),
    has(CoreCode, "update :env-stack conj {}"),
    has(CoreCode, "assoc :cut-bar"),
    has(CoreCode, "update :env-stack #(if (seq %) (pop %) %)"),
    has(CoreCode, "runtime/unify-values"),
    has(CoreCode, "runtime/succeed-state").

assert_lowered_execute_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-execute-caller-1"),
    has(CoreCode, "if-let [target-pc"),
    has(CoreCode, "(get (:labels"),
    has(CoreCode, "\"wam_fact/1\""),
    has(CoreCode, ":pc target-pc").

assert_lowered_call_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-call-caller-1"),
    has(CoreCode, "if-let [target-pc"),
    has(CoreCode, "(get (:labels"),
    has(CoreCode, "\"wam_fact/1\""),
    has(CoreCode, "update :stack conj (inc (:pc"),
    has(CoreCode, ":pc target-pc").

assert_lowered_cut_builtin_emitted(ProjectDir) :-
    directory_file_path(ProjectDir, 'src/generated/wam_exec_test/core.clj', CorePath),
    read_file_to_string(CorePath, CoreCode, []),
    has(CoreCode, "defn lowered-wam-cut-helper-1"),
    has(CoreCode, "update :choice-points"),
    has(CoreCode, "take (:cut-bar").

verify_output(ProjectDir, PredKey, Arg, Expected) :-
    run_clojure_predicate(ProjectDir, PredKey, Arg, Actual),
    (   Actual == Expected
    ->  true
    ;   throw(error(assertion_error(PredKey, Arg, Expected, Actual), _))
    ).

run_clojure_predicate(ProjectDir, PredKey, Arg, Output) :-
    find_clojure_classpath(ClassPath),
    prolog_term_string_to_edn(Arg, EdnArg),
    process_create(path(java),
                   ['-cp', ClassPath, 'clojure.main', '-m',
                    'generated.wam_exec_test.core', PredKey, EdnArg],
                   [cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err))]),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    normalize_space(string(Output), OutStr0),
    (   ErrStr == ""
    ->  true
    ;   throw(error(java_stderr(PredKey, Arg, ErrStr), _))
    ).

prolog_term_string_to_edn(a, "\"a\"") :- !.
prolog_term_string_to_edn(b, "\"b\"") :- !.
prolog_term_string_to_edn(c, "\"c\"") :- !.
prolog_term_string_to_edn(d, "\"d\"") :- !.
prolog_term_string_to_edn(z, "\"z\"") :- !.
prolog_term_string_to_edn("a", "\"a\"") :- !.
prolog_term_string_to_edn("b", "\"b\"") :- !.
prolog_term_string_to_edn("c", "\"c\"") :- !.
prolog_term_string_to_edn("d", "\"d\"") :- !.
prolog_term_string_to_edn("z", "\"z\"") :- !.
prolog_term_string_to_edn('f(a)', "{:tag :struct :functor \"f/1\" :args [\"a\"]}") :- !.
prolog_term_string_to_edn('f(b)', "{:tag :struct :functor \"f/1\" :args [\"b\"]}") :- !.
prolog_term_string_to_edn('[a,b]', "{:tag :struct :functor \"[|]/2\" :args [\"a\" {:tag :struct :functor \"[|]/2\" :args [\"b\" \"[]\"]}]}") :- !.
prolog_term_string_to_edn('[a,c]', "{:tag :struct :functor \"[|]/2\" :args [\"a\" {:tag :struct :functor \"[|]/2\" :args [\"c\" \"[]\"]}]}") :- !.
prolog_term_string_to_edn("f(a)", "{:tag :struct :functor \"f/1\" :args [\"a\"]}") :- !.
prolog_term_string_to_edn("f(b)", "{:tag :struct :functor \"f/1\" :args [\"b\"]}") :- !.
prolog_term_string_to_edn("[a,b]", "{:tag :struct :functor \"[|]/2\" :args [\"a\" {:tag :struct :functor \"[|]/2\" :args [\"b\" \"[]\"]}]}") :- !.
prolog_term_string_to_edn("[a,c]", "{:tag :struct :functor \"[|]/2\" :args [\"a\" {:tag :struct :functor \"[|]/2\" :args [\"c\" \"[]\"]}]}") :- !.
prolog_term_string_to_edn(Atom, Atom).

find_clojure_classpath(ClassPath) :-
    findall(Path,
        ( member(Path,
              [ '/data/data/com.termux/files/home/.m2/repository/org/clojure/clojure/1.11.1/clojure-1.11.1.jar',
                '/data/data/com.termux/files/home/.m2/repository/org/clojure/spec.alpha/0.3.218/spec.alpha-0.3.218.jar',
                '/data/data/com.termux/files/home/.m2/repository/org/clojure/core.specs.alpha/0.2.62/core.specs.alpha-0.2.62.jar'
              ]),
          exists_file(Path)
        ),
        JarPaths),
    JarPaths \= [],
    atomic_list_concat(['src'|JarPaths], :, ClassPath).

unique_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).
